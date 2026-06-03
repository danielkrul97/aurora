#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::similar_names
)]

//! Full thermodynamic-LM assembly + rollout BPTT + AdamW step.
//!
//! The field is a single chain across the sequence: for each token we
//! inject localized forcing, run `K` sub-steps, and read logits off the
//! evolved field (which then continues into the next token). Training
//! teacher-forces and backpropagates through the entire `T·K` unroll. The
//! field is reset per window (Fáze T0) — cross-window carry is a later
//! refinement. All caches are retained (no checkpointing yet): cheap at
//! 2D toy scale.

use rand::Rng;

use crate::optimizer::{
    adamw_apply, bias_correction, muon_apply, AdamGroup, AdamWConfig, MuonConfig, MuonGroup,
    MuonScratch,
};
use crate::thermo::field::{init_field, ThermoConfig};
use crate::transformer::loss::cross_entropy_with_grad;
use crate::thermo::inject::{
    build_profile, inject_backward, inject_forward, InjectCache, InjectGrads, InjectWeights,
};
use crate::thermo::readout::{
    build_horizon_mask, readout_backward, readout_forward, ReadoutCache, ReadoutGrads,
    ReadoutWeights,
};
use crate::thermo::step::{step_backward, step_forward, StepCache, StepGrads, StepWeights};
use crate::thermo::vortex::{build_sink_mask, VortexField};

/// Per-sub-step anneal `T_max·(1 - k/K)^α` (the §4 hot→cold schedule).
#[must_use]
pub fn anneal_at(k: usize, cfg: &ThermoConfig) -> f32 {
    let frac = 1.0 - k as f32 / cfg.k_steps as f32;
    cfg.anneal_tmax * frac.powf(cfg.anneal_alpha)
}

#[derive(Clone, Debug)]
pub struct ThermoLM {
    pub config: ThermoConfig,
    pub step: StepWeights,
    pub inject: InjectWeights,
    pub readout: ReadoutWeights,
    pub adam: ThermoAdam,
    pub muon: ThermoMuon,
    pub step_count: u64,
    /// Re-inject the token (anneal-scaled) before every sub-step instead of
    /// once per token (H31). Set by the trainer; default false.
    pub inject_every: bool,
    /// Deep supervision: teacher-forced CE on the intermediate sub-step
    /// readouts (fields `a_1..a_{K-1}`), weighted by this scalar. `0.0` = off
    /// (final-readout-only, the default). Pressures the field to be predictive
    /// throughout the rollout, giving the deep BPTT unroll a denser gradient.
    /// CPU-only for now (the GPU rollout ignores it).
    pub aux_weight: f32,
    /// Read logits off `field_after − field_before` (the token's own effect)
    /// instead of the absolute field. The field grows in magnitude across a
    /// window (injection sources the conserved channels, never drained), so
    /// the absolute readout's per-token signal shrinks with position; the
    /// delta is scale-stable. `false` = absolute (default). CPU-only for now.
    pub delta_readout: bool,
    /// Vortex / black-hole geometry (the §19 experiment). All default off;
    /// the trainer sets them and calls `rebuild_experimental`. CPU-only — the
    /// GPU rollout ignores them. `vortex_beta` = inward advection strength,
    /// `vortex_swirl` = azimuthal/radial ratio; `inject_ring_radius` injects on
    /// a shell instead of the centre; `readout_horizon` reads off a shell;
    /// `sink_radius`/`sink_strength` destroy field in the core each sub-step.
    pub vortex_beta: f32,
    pub vortex_swirl: f32,
    pub inject_ring_radius: f32,
    pub readout_horizon: f32,
    pub sink_radius: f32,
    pub sink_strength: f32,
}

/// Muon momentum buffers for the matrix params (conv kernels + readout
/// projection). The 1-D gates/biases and the injection table stay on AdamW.
#[derive(Clone, Debug)]
pub struct ThermoMuon {
    pub conv_w1: MuonGroup,
    pub conv_w2: MuonGroup,
    pub ro_w: MuonGroup,
}

#[derive(Clone, Debug)]
pub struct ThermoAdam {
    pub conv_w1: AdamGroup,
    pub conv_b1: AdamGroup,
    pub conv_w2: AdamGroup,
    pub conv_b2: AdamGroup,
    pub diff_k0: AdamGroup,
    pub diff_a: AdamGroup,
    pub diff_b: AdamGroup,
    pub inj: AdamGroup,
    pub ro_w: AdamGroup,
    pub ro_bias: AdamGroup,
}

impl ThermoLM {
    pub fn new<R: Rng>(config: ThermoConfig, rng: &mut R) -> Self {
        let step = StepWeights::new(&config, rng);
        let inject = InjectWeights::new(&config, rng);
        let readout = ReadoutWeights::new(&config, rng);
        let adam = ThermoAdam {
            conv_w1: AdamGroup::zeros(step.conv.w1.len()),
            conv_b1: AdamGroup::zeros(step.conv.b1.len()),
            conv_w2: AdamGroup::zeros(step.conv.w2.len()),
            conv_b2: AdamGroup::zeros(step.conv.b2.len()),
            diff_k0: AdamGroup::zeros(config.c),
            diff_a: AdamGroup::zeros(config.c),
            diff_b: AdamGroup::zeros(config.c),
            inj: AdamGroup::zeros(inject.inj.len()),
            ro_w: AdamGroup::zeros(readout.proj.w.len()),
            ro_bias: AdamGroup::zeros(readout.bias.len()),
        };
        let muon = ThermoMuon {
            conv_w1: MuonGroup::zeros(step.conv.w1.len()),
            conv_w2: MuonGroup::zeros(step.conv.w2.len()),
            ro_w: MuonGroup::zeros(readout.proj.w.len()),
        };
        Self {
            config,
            step,
            inject,
            readout,
            adam,
            muon,
            step_count: 0,
            inject_every: false,
            aux_weight: 0.0,
            delta_readout: false,
            vortex_beta: 0.0,
            vortex_swirl: 0.0,
            inject_ring_radius: 0.0,
            readout_horizon: 0.0,
            sink_radius: 0.0,
            sink_strength: 0.0,
        }
    }

    /// (Re)build the derived, parameter-free vortex/sink/ring/horizon fields
    /// from the current scalar knobs. Idempotent; call after changing any of
    /// them. Off-by-default scalars leave the model identical to a plain run.
    pub fn rebuild_experimental(&mut self) {
        let cfg = self.config.clone();
        self.step.vortex = if self.vortex_beta != 0.0 {
            Some(VortexField::new(&cfg, self.vortex_beta, self.vortex_swirl))
        } else {
            None
        };
        self.step.sink = if self.sink_radius > 0.0 && self.sink_strength > 0.0 {
            Some(build_sink_mask(&cfg, self.sink_radius, self.sink_strength))
        } else {
            None
        };
        self.inject.profile = build_profile(&cfg, self.inject_ring_radius);
        self.readout.horizon = if self.readout_horizon > 0.0 {
            build_horizon_mask(&cfg, self.readout_horizon)
        } else {
            Vec::new()
        };
    }

    pub fn param_count(&self) -> usize {
        self.step.conv.w1.len()
            + self.step.conv.b1.len()
            + self.step.conv.w2.len()
            + self.step.conv.b2.len()
            + 3 * self.config.c
            + self.inject.inj.len()
            + self.readout.proj.w.len()
            + self.readout.bias.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RolloutCache {
    /// `[T][n_inj]` inject caches: `n_inj = 1` (once per token) or `K` (every
    /// sub-step, when `inject_every`).
    pub inj: Vec<Vec<InjectCache>>,
    /// `[T][K]` sub-step caches.
    pub steps: Vec<Vec<StepCache>>,
    pub readout: Vec<ReadoutCache>,
    pub logits: Vec<f32>,
    /// Deep-supervision intermediate readouts `[T][K-1]` and their logits
    /// `[T*(K-1)*vocab]`. Empty unless `aux_weight > 0` (and `K > 1`).
    pub aux_readout: Vec<Vec<ReadoutCache>>,
    pub aux_logits: Vec<f32>,
    pub t: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ThermoGrads {
    pub step: StepGrads,
    pub inject: InjectGrads,
    pub readout: ReadoutGrads,
}

fn sq_sum(xs: &[f32]) -> f32 {
    xs.iter().map(|v| v * v).sum()
}

impl ThermoGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        Self {
            step: StepGrads::zeros(cfg),
            inject: InjectGrads::zeros(cfg),
            readout: ReadoutGrads::zeros(cfg),
        }
    }

    pub fn zero(&mut self) {
        self.step.zero();
        self.inject.zero();
        self.readout.zero();
    }

    /// All gradient buffers, in a fixed order, as mutable slices. The
    /// single source of truth for `add_assign` / `scale` / `l2_norm`.
    fn buffers_mut(&mut self) -> [&mut Vec<f32>; 10] {
        [
            &mut self.step.conv.d_w1,
            &mut self.step.conv.d_b1,
            &mut self.step.conv.d_w2,
            &mut self.step.conv.d_b2,
            &mut self.step.diffusion.d_k0,
            &mut self.step.diffusion.d_a,
            &mut self.step.diffusion.d_b,
            &mut self.inject.d_inj,
            &mut self.readout.d_proj.d_w,
            &mut self.readout.d_bias,
        ]
    }

    fn buffers(&self) -> [&Vec<f32>; 10] {
        [
            &self.step.conv.d_w1,
            &self.step.conv.d_b1,
            &self.step.conv.d_w2,
            &self.step.conv.d_b2,
            &self.step.diffusion.d_k0,
            &self.step.diffusion.d_a,
            &self.step.diffusion.d_b,
            &self.inject.d_inj,
            &self.readout.d_proj.d_w,
            &self.readout.d_bias,
        ]
    }

    pub fn add_assign(&mut self, other: &ThermoGrads) {
        let dst = self.buffers_mut();
        let src = other.buffers();
        for (d, s) in dst.into_iter().zip(src) {
            for (a, b) in d.iter_mut().zip(s) {
                *a += *b;
            }
        }
    }

    pub fn scale(&mut self, f: f32) {
        for buf in self.buffers_mut() {
            for v in buf.iter_mut() {
                *v *= f;
            }
        }
    }

    pub fn l2_norm(&self) -> f32 {
        self.buffers().iter().map(|b| sq_sum(b)).sum::<f32>().sqrt()
    }
}

impl ThermoLM {
    /// Forward a window of `tokens`: evolve the field token-by-token and
    /// read logits at each position. Returns the rollout cache; logits live
    /// in `cache.logits` (`[T*vocab]`).
    pub fn forward(&self, tokens: &[u32]) -> RolloutCache {
        let cfg = &self.config;
        let t = tokens.len();
        let (k, vocab) = (cfg.k_steps, cfg.vocab);

        let aux = self.aux_weight > 0.0 && k > 1;
        let delta = self.delta_readout;

        let mut field = init_field(cfg);
        let mut cache = RolloutCache {
            inj: Vec::with_capacity(t),
            steps: Vec::with_capacity(t),
            readout: Vec::with_capacity(t),
            logits: vec![0.0; t * vocab],
            aux_readout: if aux { Vec::with_capacity(t) } else { Vec::new() },
            aux_logits: if aux { Vec::with_capacity(t * (k - 1) * vocab) } else { Vec::new() },
            t,
        };

        for ti in 0..t {
            let field_before = if delta { field.clone() } else { Vec::new() };
            let mut ics = Vec::with_capacity(k);
            let mut sc = Vec::with_capacity(k);
            let mut aux_rc = Vec::new();
            if !self.inject_every {
                let (f_inj, ic) = inject_forward(&field, tokens[ti], 1.0, &self.inject, cfg);
                field = f_inj;
                ics.push(ic);
            }
            for kk in 0..k {
                if self.inject_every {
                    let (f_inj, ic) = inject_forward(&field, tokens[ti], anneal_at(kk, cfg), &self.inject, cfg);
                    field = f_inj;
                    ics.push(ic);
                }
                let (f_next, c) = step_forward(&field, &self.step, anneal_at(kk, cfg), cfg);
                field = f_next;
                sc.push(c);
                // Deep supervision taps the intermediate fields a_1..a_{K-1};
                // the final field a_K feeds the main readout below.
                if aux && kk + 1 < k {
                    let (lg, rc) = readout_forward(&field, &self.readout, cfg);
                    cache.aux_logits.extend_from_slice(&lg);
                    aux_rc.push(rc);
                }
            }
            cache.inj.push(ics);
            cache.steps.push(sc);
            if aux {
                cache.aux_readout.push(aux_rc);
            }

            let (lg, rc) = if delta {
                let mut dlt = field.clone();
                for i in 0..dlt.len() {
                    dlt[i] -= field_before[i];
                }
                readout_forward(&dlt, &self.readout, cfg)
            } else {
                readout_forward(&field, &self.readout, cfg)
            };
            cache.logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&lg);
            cache.readout.push(rc);
        }

        cache
    }

    /// Viz helper: run the rollout and return `(logits[T*vocab], final
    /// field[H*W*C])` — the field after the last token's K sub-steps. No
    /// backward cache (cheap); used to paint the live heatmaps.
    pub fn forward_field(&self, tokens: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.config;
        let (k, vocab) = (cfg.k_steps, cfg.vocab);
        let t = tokens.len();
        let mut field = init_field(cfg);
        let mut logits = vec![0.0_f32; t * vocab];
        for ti in 0..t {
            if !self.inject_every {
                let (f_inj, _) = inject_forward(&field, tokens[ti], 1.0, &self.inject, cfg);
                field = f_inj;
            }
            for kk in 0..k {
                if self.inject_every {
                    let (f_inj, _) = inject_forward(&field, tokens[ti], anneal_at(kk, cfg), &self.inject, cfg);
                    field = f_inj;
                }
                let (f_next, _) = step_forward(&field, &self.step, anneal_at(kk, cfg), cfg);
                field = f_next;
            }
            let (lg, _) = readout_forward(&field, &self.readout, cfg);
            logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&lg);
        }
        (logits, field)
    }

    /// Teacher-forced cross-entropy (mean over positions) + gradient
    /// accumulation into `grads`. Backpropagates through the full rollout.
    pub fn backward(&self, cache: &RolloutCache, targets: &[u32], grads: &mut ThermoGrads) -> f32 {
        let cfg = &self.config;
        let t = cache.t;
        let (k, vocab) = (cfg.k_steps, cfg.vocab);
        debug_assert_eq!(targets.len(), t);

        let aux = self.aux_weight > 0.0 && k > 1;
        let aux_scale = if aux { self.aux_weight / (t as f32 * (k - 1) as f32) } else { 0.0 };
        let delta = self.delta_readout;

        let mut total_loss = 0.0_f32;
        // d_field carries the gradient of the (single) field chain back
        // through positions. readout taps the field without consuming it,
        // so its gradient adds onto whatever the future positions left.
        let mut d_field = vec![0.0_f32; cfg.field_len()];

        for ti in (0..t).rev() {
            let row = &cache.logits[ti * vocab..(ti + 1) * vocab];
            let (loss, mut d_logits) = cross_entropy_with_grad(row, targets[ti] as usize);
            total_loss += loss;
            let inv_t = 1.0 / t as f32;
            for g in &mut d_logits {
                *g *= inv_t;
            }

            let d_from_readout =
                readout_backward(&self.readout, &cache.readout[ti], &d_logits, &mut grads.readout, cfg);
            for i in 0..d_field.len() {
                d_field[i] += d_from_readout[i];
            }

            for kk in (0..k).rev() {
                d_field = step_backward(&self.step, &cache.steps[ti][kk], &d_field, &mut grads.step, cfg);
                if self.inject_every {
                    d_field = inject_backward(&cache.inj[ti][kk], &d_field, &self.inject, &mut grads.inject, cfg);
                }
                // After step (and per-step inject) backward, d_field is ∂L/∂a_kk;
                // a_kk (kk≥1) was tapped by aux readout aux_readout[ti][kk-1].
                if aux && kk >= 1 {
                    let j = kk - 1;
                    let row = &cache.aux_logits[(ti * (k - 1) + j) * vocab..(ti * (k - 1) + j + 1) * vocab];
                    let (_loss, mut d_logits) = cross_entropy_with_grad(row, targets[ti] as usize);
                    for g in &mut d_logits {
                        *g *= aux_scale;
                    }
                    let d_aux = readout_backward(&self.readout, &cache.aux_readout[ti][j], &d_logits, &mut grads.readout, cfg);
                    for i in 0..d_field.len() {
                        d_field[i] += d_aux[i];
                    }
                }
            }
            if !self.inject_every {
                d_field = inject_backward(&cache.inj[ti][0], &d_field, &self.inject, &mut grads.inject, cfg);
            }
            // delta-readout: logits = readout(field_after − field_before), so
            // the carried field (now in d_field) takes −d_from_readout on top
            // of what flowed back through this token's sub-steps.
            if delta {
                for i in 0..d_field.len() {
                    d_field[i] -= d_from_readout[i];
                }
            }
        }

        total_loss / t as f32
    }

    /// Global-norm gradient clip; returns pre-clip norm.
    pub fn clip_grads(grads: &mut ThermoGrads, max_norm: f32) -> f32 {
        let n = grads.l2_norm();
        if n > max_norm && n > 0.0 {
            grads.scale(max_norm / n);
        }
        n
    }

    /// AdamW step over every parameter tensor. Weight decay on the matrix/
    /// table params (conv kernels, readout projection, injection table),
    /// off for the small 1D gates and biases.
    pub fn step_adamw(&mut self, grads: &ThermoGrads, cfg: &AdamWConfig) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, cfg.beta1, cfg.beta2);
        let apply = |p: &mut [f32], g: &[f32], grp: &mut AdamGroup, wd: bool| {
            adamw_apply(p, g, grp, cfg, bc1, bc2, wd);
        };
        apply(&mut self.step.conv.w1, &grads.step.conv.d_w1, &mut self.adam.conv_w1, true);
        apply(&mut self.step.conv.b1, &grads.step.conv.d_b1, &mut self.adam.conv_b1, false);
        apply(&mut self.step.conv.w2, &grads.step.conv.d_w2, &mut self.adam.conv_w2, true);
        apply(&mut self.step.conv.b2, &grads.step.conv.d_b2, &mut self.adam.conv_b2, false);
        apply(&mut self.step.diffusion.k0, &grads.step.diffusion.d_k0, &mut self.adam.diff_k0, false);
        apply(&mut self.step.diffusion.a, &grads.step.diffusion.d_a, &mut self.adam.diff_a, false);
        apply(&mut self.step.diffusion.b, &grads.step.diffusion.d_b, &mut self.adam.diff_b, false);
        apply(&mut self.inject.inj, &grads.inject.d_inj, &mut self.adam.inj, true);
        apply(&mut self.readout.proj.w, &grads.readout.d_proj.d_w, &mut self.adam.ro_w, true);
        apply(&mut self.readout.bias, &grads.readout.d_bias, &mut self.adam.ro_bias, false);
    }

    /// Hybrid step: Muon (orthogonalized update) for the matrix params —
    /// conv `W1` (`c_h × c·27`), conv `W2` (`c_out × c_h`), readout proj
    /// (`vocab × readout_dim`) — and AdamW for the 1-D gates/biases and the
    /// injection table (à la the transformer's `step_muon_hybrid`).
    pub fn step_muon_hybrid(
        &mut self,
        grads: &ThermoGrads,
        adam_cfg: &AdamWConfig,
        muon_cfg: &MuonConfig,
        scratch: &mut MuonScratch,
    ) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let (c, ch, cout, vocab, rdim) = (
            self.config.c,
            self.config.conv_hidden,
            self.config.c_out(),
            self.config.vocab,
            self.config.readout_dim(),
        );
        muon_apply(&mut self.step.conv.w1, &grads.step.conv.d_w1, &mut self.muon.conv_w1, ch, c * 27, muon_cfg, scratch);
        muon_apply(&mut self.step.conv.w2, &grads.step.conv.d_w2, &mut self.muon.conv_w2, cout, ch, muon_cfg, scratch);
        muon_apply(&mut self.readout.proj.w, &grads.readout.d_proj.d_w, &mut self.muon.ro_w, vocab, rdim, muon_cfg, scratch);
        let apply = |p: &mut [f32], g: &[f32], grp: &mut AdamGroup, wd: bool| {
            adamw_apply(p, g, grp, adam_cfg, bc1, bc2, wd);
        };
        apply(&mut self.step.conv.b1, &grads.step.conv.d_b1, &mut self.adam.conv_b1, false);
        apply(&mut self.step.conv.b2, &grads.step.conv.d_b2, &mut self.adam.conv_b2, false);
        apply(&mut self.step.diffusion.k0, &grads.step.diffusion.d_k0, &mut self.adam.diff_k0, false);
        apply(&mut self.step.diffusion.a, &grads.step.diffusion.d_a, &mut self.adam.diff_a, false);
        apply(&mut self.step.diffusion.b, &grads.step.diffusion.d_b, &mut self.adam.diff_b, false);
        apply(&mut self.inject.inj, &grads.inject.d_inj, &mut self.adam.inj, true);
        apply(&mut self.readout.bias, &grads.readout.d_bias, &mut self.adam.ro_bias, false);
    }
}
