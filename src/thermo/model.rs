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
use crate::thermo::inject::{inject_backward, inject_forward, InjectCache, InjectGrads, InjectWeights};
use crate::thermo::readout::{
    readout_backward, readout_forward, ReadoutCache, ReadoutGrads, ReadoutWeights,
};
use crate::thermo::step::{step_backward, step_forward, StepCache, StepGrads, StepWeights};

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
        }
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

        let mut field = init_field(cfg);
        let mut cache = RolloutCache {
            inj: Vec::with_capacity(t),
            steps: Vec::with_capacity(t),
            readout: Vec::with_capacity(t),
            logits: vec![0.0; t * vocab],
            t,
        };

        for ti in 0..t {
            let mut ics = Vec::with_capacity(k);
            let mut sc = Vec::with_capacity(k);
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
            }
            cache.inj.push(ics);
            cache.steps.push(sc);

            let (lg, rc) = readout_forward(&field, &self.readout, cfg);
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
            }
            if !self.inject_every {
                d_field = inject_backward(&cache.inj[ti][0], &d_field, &self.inject, &mut grads.inject, cfg);
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
    /// conv `W1` (`c_h × c·9`), conv `W2` (`c_out × c_h`), readout proj
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
        muon_apply(&mut self.step.conv.w1, &grads.step.conv.d_w1, &mut self.muon.conv_w1, ch, c * 9, muon_cfg, scratch);
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
