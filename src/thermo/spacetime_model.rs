#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments
)]

//! Spacetime LM (`docs/13` §21): the sequence is the field's axis, processed in
//! parallel (no recurrence). embed-at-time → K causal dilated conv steps →
//! per-position readout. BPTT is only K deep, the readout is time-local.
//! Standalone (grid/GPU untouched); reuses AdamW/Muon + cross-entropy.

use rand::Rng;

use crate::optimizer::{
    adamw_apply, bias_correction, muon_apply, AdamGroup, AdamWConfig, MuonConfig, MuonGroup,
    MuonScratch,
};
use crate::thermo::spacetime::{st_conv_backward, st_conv_forward, StConvCache, StConvGrads, StConvWeights};
use crate::transformer::loss::cross_entropy_with_grad;

#[derive(Clone, Debug)]
pub struct StConfig {
    pub vocab: usize,
    pub c: usize,
    pub conv_hidden: usize,
    pub k_steps: usize,
    /// Causal time taps per conv (present + taps−1 past samples).
    pub taps: usize,
    /// Per-sub-step dilation is `dilation_base^k` (exponential temporal reach).
    pub dilation_base: usize,
    pub dt: f32,
}

#[derive(Clone, Debug)]
pub struct StLM {
    pub cfg: StConfig,
    pub conv: StConvWeights,
    /// Token → time-cell injection table `[vocab*c]` (the embedding).
    pub embed: Vec<f32>,
    /// Per-position readout `[vocab*c]` (row-major by vocab).
    pub ro_w: Vec<f32>,
    pub ro_bias: Vec<f32>,
    pub adam: StAdam,
    pub muon: StMuon,
    pub step_count: u64,
}

#[derive(Clone, Debug)]
pub struct StAdam {
    pub conv_w1: AdamGroup,
    pub conv_b1: AdamGroup,
    pub conv_w2: AdamGroup,
    pub conv_b2: AdamGroup,
    pub embed: AdamGroup,
    pub ro_w: AdamGroup,
    pub ro_bias: AdamGroup,
}

#[derive(Clone, Debug)]
pub struct StMuon {
    pub conv_w1: MuonGroup,
    pub conv_w2: MuonGroup,
    pub ro_w: MuonGroup,
}

impl StLM {
    pub fn new<R: Rng>(cfg: StConfig, rng: &mut R) -> Self {
        let (vocab, c) = (cfg.vocab, cfg.c);
        let conv = StConvWeights::new(c, cfg.conv_hidden, c, cfg.taps, cfg.k_steps, rng);
        let escale = (1.0_f32 / c as f32).sqrt();
        let embed: Vec<f32> = (0..vocab * c).map(|_| (rng.random::<f32>() * 2.0 - 1.0) * escale).collect();
        let roscale = 0.1 / (c as f32).sqrt();
        let ro_w: Vec<f32> = (0..vocab * c).map(|_| (rng.random::<f32>() * 2.0 - 1.0) * roscale).collect();
        let adam = StAdam {
            conv_w1: AdamGroup::zeros(conv.w1.len()),
            conv_b1: AdamGroup::zeros(conv.b1.len()),
            conv_w2: AdamGroup::zeros(conv.w2.len()),
            conv_b2: AdamGroup::zeros(conv.b2.len()),
            embed: AdamGroup::zeros(embed.len()),
            ro_w: AdamGroup::zeros(ro_w.len()),
            ro_bias: AdamGroup::zeros(vocab),
        };
        let muon = StMuon {
            conv_w1: MuonGroup::zeros(conv.w1.len()),
            conv_w2: MuonGroup::zeros(conv.w2.len()),
            ro_w: MuonGroup::zeros(ro_w.len()),
        };
        Self { cfg, conv, embed, ro_w, ro_bias: vec![0.0; vocab], adam, muon, step_count: 0 }
    }

    pub fn param_count(&self) -> usize {
        self.conv.w1.len() + self.conv.b1.len() + self.conv.w2.len() + self.conv.b2.len()
            + self.embed.len() + self.ro_w.len() + self.ro_bias.len()
    }

    #[inline]
    fn dilation(&self, k: usize, l: usize) -> usize {
        self.cfg.dilation_base.checked_pow(k as u32).unwrap_or(usize::MAX).min(l.max(1)).max(1)
    }

    pub fn forward(&self, tokens: &[u32]) -> StCache {
        let (c, vocab, kk, dt) = (self.cfg.c, self.cfg.vocab, self.cfg.k_steps, self.cfg.dt);
        let l = tokens.len();
        let mut field = vec![0.0_f32; l * c];
        for t in 0..l {
            let row = &self.embed[tokens[t] as usize * c..(tokens[t] as usize + 1) * c];
            let dst = &mut field[t * c..(t + 1) * c];
            for ci in 0..c {
                dst[ci] += row[ci];
            }
        }
        let mut caches = Vec::with_capacity(kk);
        for k in 0..kk {
            let dil = self.dilation(k, l);
            let (raw, cc) = st_conv_forward(&self.conv, &field, l, dil);
            for i in 0..field.len() {
                field[i] += dt * raw[i];
            }
            caches.push(cc);
        }
        let mut logits = vec![0.0_f32; l * vocab];
        for t in 0..l {
            let f = &field[t * c..(t + 1) * c];
            for o in 0..vocab {
                let row = &self.ro_w[o * c..(o + 1) * c];
                let mut acc = self.ro_bias[o];
                for ci in 0..c {
                    acc += row[ci] * f[ci];
                }
                logits[t * vocab + o] = acc;
            }
        }
        StCache { caches, final_field: field, logits, l }
    }

    pub fn backward(&self, cache: &StCache, tokens: &[u32], targets: &[u32], g: &mut StGrads) -> f32 {
        let (c, vocab, kk, dt) = (self.cfg.c, self.cfg.vocab, self.cfg.k_steps, self.cfg.dt);
        let l = cache.l;
        let inv_l = 1.0 / l as f32;
        let mut total = 0.0_f32;
        let mut d_field = vec![0.0_f32; l * c];
        for t in 0..l {
            let (loss, mut dlog) = cross_entropy_with_grad(&cache.logits[t * vocab..(t + 1) * vocab], targets[t] as usize);
            total += loss;
            for v in &mut dlog {
                *v *= inv_l;
            }
            let f = &cache.final_field[t * c..(t + 1) * c];
            for o in 0..vocab {
                let go = dlog[o];
                g.d_ro_bias[o] += go;
                let row = &self.ro_w[o * c..(o + 1) * c];
                let drow = &mut g.d_ro_w[o * c..(o + 1) * c];
                let df = &mut d_field[t * c..(t + 1) * c];
                for ci in 0..c {
                    drow[ci] += go * f[ci];
                    df[ci] += go * row[ci];
                }
            }
        }
        for k in (0..kk).rev() {
            let mut d_raw = d_field.clone();
            for v in &mut d_raw {
                *v *= dt;
            }
            let d_s = st_conv_backward(&self.conv, &cache.caches[k], &d_raw, &mut g.conv, l);
            for i in 0..d_field.len() {
                d_field[i] += d_s[i];
            }
        }
        for t in 0..l {
            let drow = &mut g.d_embed[tokens[t] as usize * c..(tokens[t] as usize + 1) * c];
            let df = &d_field[t * c..(t + 1) * c];
            for ci in 0..c {
                drow[ci] += df[ci];
            }
        }
        total / l as f32
    }

    pub fn clip_grads(g: &mut StGrads, max_norm: f32) -> f32 {
        let n = g.l2_norm();
        if n > max_norm && n > 0.0 {
            g.scale(max_norm / n);
        }
        n
    }

    pub fn step_muon_hybrid(&mut self, g: &StGrads, adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig, scratch: &mut MuonScratch) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let (c_h, c, taps, vocab) = (self.cfg.conv_hidden, self.cfg.c, self.cfg.taps, self.cfg.vocab);
        muon_apply(&mut self.conv.w1, &g.conv.d_w1, &mut self.muon.conv_w1, c_h, c * taps, muon_cfg, scratch);
        muon_apply(&mut self.conv.w2, &g.conv.d_w2, &mut self.muon.conv_w2, c, c_h, muon_cfg, scratch);
        muon_apply(&mut self.ro_w, &g.d_ro_w, &mut self.muon.ro_w, vocab, c, muon_cfg, scratch);
        let apply = |p: &mut [f32], gr: &[f32], grp: &mut AdamGroup, wd: bool| adamw_apply(p, gr, grp, adam_cfg, bc1, bc2, wd);
        apply(&mut self.conv.b1, &g.conv.d_b1, &mut self.adam.conv_b1, false);
        apply(&mut self.conv.b2, &g.conv.d_b2, &mut self.adam.conv_b2, false);
        apply(&mut self.embed, &g.d_embed, &mut self.adam.embed, true);
        apply(&mut self.ro_bias, &g.d_ro_bias, &mut self.adam.ro_bias, false);
    }
}

#[derive(Clone, Debug, Default)]
pub struct StCache {
    pub caches: Vec<StConvCache>,
    pub final_field: Vec<f32>,
    pub logits: Vec<f32>,
    pub l: usize,
}

#[derive(Clone, Debug, Default)]
pub struct StGrads {
    pub conv: StConvGrads,
    pub d_embed: Vec<f32>,
    pub d_ro_w: Vec<f32>,
    pub d_ro_bias: Vec<f32>,
}

fn sq_sum(xs: &[f32]) -> f32 {
    xs.iter().map(|v| v * v).sum()
}

impl StGrads {
    pub fn zeros(m: &StLM) -> Self {
        Self {
            conv: StConvGrads::zeros(m.cfg.c, m.cfg.conv_hidden, m.cfg.c, m.cfg.taps),
            d_embed: vec![0.0; m.embed.len()],
            d_ro_w: vec![0.0; m.ro_w.len()],
            d_ro_bias: vec![0.0; m.ro_bias.len()],
        }
    }
    pub fn zero(&mut self) {
        self.conv.zero();
        self.d_embed.fill(0.0);
        self.d_ro_w.fill(0.0);
        self.d_ro_bias.fill(0.0);
    }
    fn buffers_mut(&mut self) -> [&mut Vec<f32>; 7] {
        [
            &mut self.conv.d_w1, &mut self.conv.d_b1, &mut self.conv.d_w2, &mut self.conv.d_b2,
            &mut self.d_embed, &mut self.d_ro_w, &mut self.d_ro_bias,
        ]
    }
    fn buffers(&self) -> [&Vec<f32>; 7] {
        [
            &self.conv.d_w1, &self.conv.d_b1, &self.conv.d_w2, &self.conv.d_b2,
            &self.d_embed, &self.d_ro_w, &self.d_ro_bias,
        ]
    }
    pub fn add_assign(&mut self, other: &StGrads) {
        for (d, s) in self.buffers_mut().into_iter().zip(other.buffers()) {
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
