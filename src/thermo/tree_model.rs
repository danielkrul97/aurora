#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments
)]

//! Standalone tree-substrate LM (`docs/13` §20). Kept separate from `ThermoLM`
//! so the grid/GPU path is untouched; reuses the leaf ops (inject, moment
//! readout, AdamW/Muon, cross-entropy) via an internal `d=1,h=1,w=N` config
//! view, with the only tree-specific computation being [`super::tree`]'s
//! graph-conv. v1: free residual update only (no flux/diffusion — both ~neutral
//! per §16/§19); tokens inject at leaves, the `pool=1` readout reads all nodes.

use rand::Rng;

use crate::optimizer::{
    adamw_apply, bias_correction, muon_apply, AdamGroup, AdamWConfig, MuonConfig, MuonGroup,
    MuonScratch,
};
use crate::thermo::field::ThermoConfig;
use crate::thermo::inject::{inject_backward, inject_forward, InjectCache, InjectGrads, InjectWeights};
use crate::thermo::readout::{readout_backward, readout_forward, ReadoutCache, ReadoutGrads, ReadoutWeights};
use crate::thermo::tree::{
    tree_conv_backward, tree_conv_forward, TreeConvCache, TreeConvGrads, TreeConvWeights, TreeTopology,
};
use crate::transformer::loss::cross_entropy_with_grad;

#[derive(Clone, Debug)]
pub struct TreeConfig {
    pub vocab: usize,
    pub arity: usize,
    pub depth: usize,
    pub c: usize,
    pub conv_hidden: usize,
    pub k_steps: usize,
    pub dt: f32,
}

/// Fixed leaf-injection profile `[n]`: 1.0 on leaves (the input layer), 0
/// elsewhere — tokens enter at the bottom and the conv lifts them up the tree.
fn leaf_profile(topo: &TreeTopology) -> Vec<f32> {
    let mut p = vec![0.0_f32; topo.n];
    for &l in &topo.leaves {
        p[l as usize] = 1.0;
    }
    p
}

#[derive(Clone, Debug)]
pub struct TreeLM {
    pub tcfg: TreeConfig,
    pub topo: TreeTopology,
    /// `d=1,h=1,w=N,pool=1,conserved=[],moments-only` view for inject/readout.
    pub cfg: ThermoConfig,
    pub conv: TreeConvWeights,
    pub inject: InjectWeights,
    pub readout: ReadoutWeights,
    pub adam: TreeAdam,
    pub muon: TreeMuon,
    pub step_count: u64,
}

#[derive(Clone, Debug)]
pub struct TreeAdam {
    pub conv_w1: AdamGroup,
    pub conv_b1: AdamGroup,
    pub conv_w2: AdamGroup,
    pub conv_b2: AdamGroup,
    pub inj: AdamGroup,
    pub ro_w: AdamGroup,
    pub ro_bias: AdamGroup,
}

#[derive(Clone, Debug)]
pub struct TreeMuon {
    pub conv_w1: MuonGroup,
    pub conv_w2: MuonGroup,
    pub ro_w: MuonGroup,
}

fn thermo_view(tcfg: &TreeConfig, n: usize) -> ThermoConfig {
    ThermoConfig {
        vocab: tcfg.vocab,
        d: 1,
        h: 1,
        w: n,
        c: tcfg.c,
        conv_hidden: tcfg.conv_hidden,
        conserved: Vec::new(),
        k_steps: tcfg.k_steps,
        pool: 1,
        dt: tcfg.dt,
        t_init: 0.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: false,
        readout_modes: 0,
        use_diffusion: false,
    }
}

impl TreeLM {
    pub fn new<R: Rng>(tcfg: TreeConfig, rng: &mut R) -> Self {
        let topo = TreeTopology::new(tcfg.arity, tcfg.depth);
        let n = topo.n;
        let cfg = thermo_view(&tcfg, n);
        let conv = TreeConvWeights::new(tcfg.c, tcfg.conv_hidden, tcfg.c, topo.taps, tcfg.k_steps, rng);
        let mut inject = InjectWeights::new(&cfg, rng);
        inject.profile = leaf_profile(&topo);
        let readout = ReadoutWeights::new(&cfg, rng);
        let adam = TreeAdam {
            conv_w1: AdamGroup::zeros(conv.w1.len()),
            conv_b1: AdamGroup::zeros(conv.b1.len()),
            conv_w2: AdamGroup::zeros(conv.w2.len()),
            conv_b2: AdamGroup::zeros(conv.b2.len()),
            inj: AdamGroup::zeros(inject.inj.len()),
            ro_w: AdamGroup::zeros(readout.proj.w.len()),
            ro_bias: AdamGroup::zeros(readout.bias.len()),
        };
        let muon = TreeMuon {
            conv_w1: MuonGroup::zeros(conv.w1.len()),
            conv_w2: MuonGroup::zeros(conv.w2.len()),
            ro_w: MuonGroup::zeros(readout.proj.w.len()),
        };
        Self { tcfg, topo, cfg, conv, inject, readout, adam, muon, step_count: 0 }
    }

    pub fn param_count(&self) -> usize {
        self.conv.w1.len()
            + self.conv.b1.len()
            + self.conv.w2.len()
            + self.conv.b2.len()
            + self.inject.inj.len()
            + self.readout.proj.w.len()
            + self.readout.bias.len()
    }

    pub fn forward(&self, tokens: &[u32]) -> TreeRolloutCache {
        let n = self.topo.n;
        let (k, vocab, c, dt) = (self.tcfg.k_steps, self.tcfg.vocab, self.tcfg.c, self.tcfg.dt);
        let t = tokens.len();
        let mut field = vec![0.0_f32; n * c];
        let mut cache = TreeRolloutCache {
            inj: Vec::with_capacity(t),
            steps: Vec::with_capacity(t),
            readout: Vec::with_capacity(t),
            logits: vec![0.0; t * vocab],
            t,
        };
        for ti in 0..t {
            let (f_inj, ic) = inject_forward(&field, tokens[ti], 1.0, &self.inject, &self.cfg);
            field = f_inj;
            let mut sc = Vec::with_capacity(k);
            for _ in 0..k {
                let (raw, cc) = tree_conv_forward(&self.conv, &field, &self.topo);
                for i in 0..field.len() {
                    field[i] += dt * raw[i];
                }
                sc.push(cc);
            }
            let (lg, rc) = readout_forward(&field, &self.readout, &self.cfg);
            cache.logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&lg);
            cache.inj.push(ic);
            cache.steps.push(sc);
            cache.readout.push(rc);
        }
        cache
    }

    pub fn backward(&self, cache: &TreeRolloutCache, targets: &[u32], grads: &mut TreeGrads) -> f32 {
        let (k, vocab, dt) = (self.tcfg.k_steps, self.tcfg.vocab, self.tcfg.dt);
        let t = cache.t;
        let mut total_loss = 0.0_f32;
        let mut d_field = vec![0.0_f32; self.topo.n * self.tcfg.c];
        let mut d_raw = vec![0.0_f32; self.topo.n * self.tcfg.c];
        let inv_t = 1.0 / t as f32;
        for ti in (0..t).rev() {
            let row = &cache.logits[ti * vocab..(ti + 1) * vocab];
            let (loss, mut d_logits) = cross_entropy_with_grad(row, targets[ti] as usize);
            total_loss += loss;
            for g in &mut d_logits {
                *g *= inv_t;
            }
            let d_ro = readout_backward(&self.readout, &cache.readout[ti], &d_logits, &mut grads.readout, &self.cfg);
            for i in 0..d_field.len() {
                d_field[i] += d_ro[i];
            }
            for kk in (0..k).rev() {
                // field_{kk+1} = field_kk + dt·raw_kk → ∂L/∂field_kk = d_field + dt·(conv adjoint).
                d_raw.copy_from_slice(&d_field);
                for v in &mut d_raw {
                    *v *= dt;
                }
                let d_s = tree_conv_backward(&self.conv, &cache.steps[ti][kk], &d_raw, &mut grads.conv, &self.topo);
                for i in 0..d_field.len() {
                    d_field[i] += d_s[i];
                }
            }
            d_field = inject_backward(&cache.inj[ti], &d_field, &self.inject, &mut grads.inject, &self.cfg);
        }
        total_loss / t as f32
    }

    pub fn clip_grads(grads: &mut TreeGrads, max_norm: f32) -> f32 {
        let nrm = grads.l2_norm();
        if nrm > max_norm && nrm > 0.0 {
            grads.scale(max_norm / nrm);
        }
        nrm
    }

    pub fn step_adamw(&mut self, grads: &TreeGrads, cfg: &AdamWConfig) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, cfg.beta1, cfg.beta2);
        let apply = |p: &mut [f32], g: &[f32], grp: &mut AdamGroup, wd: bool| {
            adamw_apply(p, g, grp, cfg, bc1, bc2, wd);
        };
        apply(&mut self.conv.w1, &grads.conv.d_w1, &mut self.adam.conv_w1, true);
        apply(&mut self.conv.b1, &grads.conv.d_b1, &mut self.adam.conv_b1, false);
        apply(&mut self.conv.w2, &grads.conv.d_w2, &mut self.adam.conv_w2, true);
        apply(&mut self.conv.b2, &grads.conv.d_b2, &mut self.adam.conv_b2, false);
        apply(&mut self.inject.inj, &grads.inject.d_inj, &mut self.adam.inj, true);
        apply(&mut self.readout.proj.w, &grads.readout.d_proj.d_w, &mut self.adam.ro_w, true);
        apply(&mut self.readout.bias, &grads.readout.d_bias, &mut self.adam.ro_bias, false);
    }

    pub fn step_muon_hybrid(
        &mut self,
        grads: &TreeGrads,
        adam_cfg: &AdamWConfig,
        muon_cfg: &MuonConfig,
        scratch: &mut MuonScratch,
    ) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let (c_h, c, taps, c_out, vocab, rdim) = (
            self.tcfg.conv_hidden,
            self.tcfg.c,
            self.topo.taps,
            self.tcfg.c,
            self.tcfg.vocab,
            self.cfg.readout_dim(),
        );
        muon_apply(&mut self.conv.w1, &grads.conv.d_w1, &mut self.muon.conv_w1, c_h, c * taps, muon_cfg, scratch);
        muon_apply(&mut self.conv.w2, &grads.conv.d_w2, &mut self.muon.conv_w2, c_out, c_h, muon_cfg, scratch);
        muon_apply(&mut self.readout.proj.w, &grads.readout.d_proj.d_w, &mut self.muon.ro_w, vocab, rdim, muon_cfg, scratch);
        let apply = |p: &mut [f32], g: &[f32], grp: &mut AdamGroup, wd: bool| {
            adamw_apply(p, g, grp, adam_cfg, bc1, bc2, wd);
        };
        apply(&mut self.conv.b1, &grads.conv.d_b1, &mut self.adam.conv_b1, false);
        apply(&mut self.conv.b2, &grads.conv.d_b2, &mut self.adam.conv_b2, false);
        apply(&mut self.inject.inj, &grads.inject.d_inj, &mut self.adam.inj, true);
        apply(&mut self.readout.bias, &grads.readout.d_bias, &mut self.adam.ro_bias, false);
    }
}

#[derive(Clone, Debug, Default)]
pub struct TreeRolloutCache {
    pub inj: Vec<InjectCache>,
    pub steps: Vec<Vec<TreeConvCache>>,
    pub readout: Vec<ReadoutCache>,
    pub logits: Vec<f32>,
    pub t: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TreeGrads {
    pub conv: TreeConvGrads,
    pub inject: InjectGrads,
    pub readout: ReadoutGrads,
}

fn sq_sum(xs: &[f32]) -> f32 {
    xs.iter().map(|v| v * v).sum()
}

impl TreeGrads {
    pub fn zeros(m: &TreeLM) -> Self {
        Self {
            conv: TreeConvGrads::zeros(m.tcfg.c, m.tcfg.conv_hidden, m.tcfg.c, m.topo.taps),
            inject: InjectGrads::zeros(&m.cfg),
            readout: ReadoutGrads::zeros(&m.cfg),
        }
    }

    pub fn zero(&mut self) {
        self.conv.zero();
        self.inject.zero();
        self.readout.zero();
    }

    fn buffers_mut(&mut self) -> [&mut Vec<f32>; 7] {
        [
            &mut self.conv.d_w1,
            &mut self.conv.d_b1,
            &mut self.conv.d_w2,
            &mut self.conv.d_b2,
            &mut self.inject.d_inj,
            &mut self.readout.d_proj.d_w,
            &mut self.readout.d_bias,
        ]
    }

    fn buffers(&self) -> [&Vec<f32>; 7] {
        [
            &self.conv.d_w1,
            &self.conv.d_b1,
            &self.conv.d_w2,
            &self.conv.d_b2,
            &self.inject.d_inj,
            &self.readout.d_proj.d_w,
            &self.readout.d_bias,
        ]
    }

    pub fn add_assign(&mut self, other: &TreeGrads) {
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
