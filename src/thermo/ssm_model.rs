#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments
)]

//! Selective-SSM LM (`docs/13` §22). embed → N residual selective-SSM blocks →
//! per-position readout. Each block: value `v=x·Win`, content gate
//! `a=σ(x·Wg+bg)`, selective scan `y=scan(a,v)`, out `x += y·Wo`. The gate is
//! input-dependent (Mamba-style selectivity) — the content-dependent temporal
//! memory the static spacetime conv lacks. Parallel over the window; the scan
//! is the only sequential part (O(L), contracting → tame gradients). Standalone
//! (grid/GPU untouched); reuses AdamW/Muon + cross-entropy.

use rand::Rng;

use crate::optimizer::{
    adamw_apply, bias_correction, muon_apply, AdamGroup, AdamWConfig, MuonConfig, MuonGroup, MuonScratch,
};
use crate::thermo::ssm::{selective_scan_backward, selective_scan_forward};
use crate::transformer::loss::cross_entropy_with_grad;

#[derive(Clone, Debug)]
pub struct SsmConfig {
    pub vocab: usize,
    pub c: usize,
    pub n_layers: usize,
}

// ---- matmul helpers: y[t,o] = Σ_i x[t,i]·w[i*co+o] ----
fn mm(x: &[f32], w: &[f32], l: usize, ci: usize, co: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; l * co];
    for t in 0..l {
        let xr = &x[t * ci..(t + 1) * ci];
        let yr = &mut y[t * co..(t + 1) * co];
        for i in 0..ci {
            let xv = xr[i];
            let wr = &w[i * co..(i + 1) * co];
            for o in 0..co {
                yr[o] += xv * wr[o];
            }
        }
    }
    y
}

/// Returns d_x `[l*ci]`, accumulates d_w `[ci*co]`.
fn mm_backward(x: &[f32], w: &[f32], dy: &[f32], l: usize, ci: usize, co: usize, d_w: &mut [f32]) -> Vec<f32> {
    let mut d_x = vec![0.0_f32; l * ci];
    for t in 0..l {
        let xr = &x[t * ci..(t + 1) * ci];
        let dyr = &dy[t * co..(t + 1) * co];
        let dxr = &mut d_x[t * ci..(t + 1) * ci];
        for i in 0..ci {
            let wr = &w[i * co..(i + 1) * co];
            let dwr = &mut d_w[i * co..(i + 1) * co];
            let xv = xr[i];
            let mut acc = 0.0_f32;
            for o in 0..co {
                acc += dyr[o] * wr[o];
                dwr[o] += xv * dyr[o];
            }
            dxr[i] = acc;
        }
    }
    d_x
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[derive(Clone, Debug)]
pub struct BlockWeights {
    pub win: Vec<f32>, // [c*c]
    pub wg: Vec<f32>,  // [c*c]
    pub wo: Vec<f32>,  // [c*c]
    pub bg: Vec<f32>,  // [c] gate bias
}

#[derive(Clone, Debug, Default)]
pub struct BlockGrads {
    pub d_win: Vec<f32>,
    pub d_wg: Vec<f32>,
    pub d_wo: Vec<f32>,
    pub d_bg: Vec<f32>,
}

impl BlockGrads {
    fn zeros(c: usize) -> Self {
        Self { d_win: vec![0.0; c * c], d_wg: vec![0.0; c * c], d_wo: vec![0.0; c * c], d_bg: vec![0.0; c] }
    }
    fn zero(&mut self) {
        self.d_win.fill(0.0);
        self.d_wg.fill(0.0);
        self.d_wo.fill(0.0);
        self.d_bg.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockCache {
    x_in: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    y: Vec<f32>,
}

/// Forward one block; returns the residual term `out` (`x += out` by caller).
fn block_forward(b: &BlockWeights, x: &[f32], l: usize, c: usize) -> (Vec<f32>, BlockCache) {
    let v = mm(x, &b.win, l, c, c);
    let mut a = mm(x, &b.wg, l, c, c);
    for t in 0..l {
        for ci in 0..c {
            a[t * c + ci] = sigmoid(a[t * c + ci] + b.bg[ci]);
        }
    }
    let y = selective_scan_forward(&a, &v, l, c);
    let out = mm(&y, &b.wo, l, c, c);
    (out, BlockCache { x_in: x.to_vec(), v, a, y })
}

/// Backward one block. `d_out` is dL/d(out); returns dL/d(x_in), accumulates g.
fn block_backward(b: &BlockWeights, cache: &BlockCache, d_out: &[f32], g: &mut BlockGrads, l: usize, c: usize) -> Vec<f32> {
    // out = y·Wo
    let d_y = mm_backward(&cache.y, &b.wo, d_out, l, c, c, &mut g.d_wo);
    // y = scan(a, v)
    let (d_a, d_v) = selective_scan_backward(&cache.a, &cache.v, &cache.y, &d_y, l, c);
    // a = sigmoid(g_pre); d_g = d_a · a(1-a); bg grad = Σ_t d_g
    let mut d_g = vec![0.0_f32; l * c];
    for t in 0..l {
        for ci in 0..c {
            let av = cache.a[t * c + ci];
            let dgv = d_a[t * c + ci] * av * (1.0 - av);
            d_g[t * c + ci] = dgv;
            g.d_bg[ci] += dgv;
        }
    }
    // v = x·Win ; g_pre = x·Wg
    let d_x_v = mm_backward(&cache.x_in, &b.win, &d_v, l, c, c, &mut g.d_win);
    let d_x_g = mm_backward(&cache.x_in, &b.wg, &d_g, l, c, c, &mut g.d_wg);
    let mut d_x = d_x_v;
    for i in 0..d_x.len() {
        d_x[i] += d_x_g[i];
    }
    d_x
}

#[derive(Clone, Debug)]
pub struct SsmLM {
    pub cfg: SsmConfig,
    pub blocks: Vec<BlockWeights>,
    pub embed: Vec<f32>,
    pub ro_w: Vec<f32>,
    pub ro_bias: Vec<f32>,
    pub adam: SsmAdam,
    pub muon: SsmMuon,
    pub step_count: u64,
}

#[derive(Clone, Debug)]
pub struct SsmAdam {
    pub bg: Vec<AdamGroup>,
    pub embed: AdamGroup,
    pub ro_bias: AdamGroup,
}

#[derive(Clone, Debug)]
pub struct SsmMuon {
    pub win: Vec<MuonGroup>,
    pub wg: Vec<MuonGroup>,
    pub wo: Vec<MuonGroup>,
    pub ro_w: MuonGroup,
}

impl SsmLM {
    pub fn new<R: Rng>(cfg: SsmConfig, rng: &mut R) -> Self {
        let (vocab, c, nl) = (cfg.vocab, cfg.c, cfg.n_layers);
        let unif = |bound: f32, rng: &mut R| (rng.random::<f32>() * 2.0 - 1.0) * bound;
        let inb = (1.0_f32 / c as f32).sqrt();
        let ob = (1.0_f32 / c as f32).sqrt() / (2.0 * nl as f32).sqrt(); // gentle residual
        let mut blocks = Vec::with_capacity(nl);
        for _ in 0..nl {
            blocks.push(BlockWeights {
                win: (0..c * c).map(|_| unif(inb, rng)).collect(),
                wg: (0..c * c).map(|_| unif(inb, rng)).collect(),
                wo: (0..c * c).map(|_| unif(ob, rng)).collect(),
                bg: vec![0.0; c],
            });
        }
        let embed: Vec<f32> = (0..vocab * c).map(|_| unif(inb, rng)).collect();
        let ro_w: Vec<f32> = (0..vocab * c).map(|_| unif(0.1 * inb, rng)).collect();
        let adam = SsmAdam {
            bg: (0..nl).map(|_| AdamGroup::zeros(c)).collect(),
            embed: AdamGroup::zeros(vocab * c),
            ro_bias: AdamGroup::zeros(vocab),
        };
        let muon = SsmMuon {
            win: (0..nl).map(|_| MuonGroup::zeros(c * c)).collect(),
            wg: (0..nl).map(|_| MuonGroup::zeros(c * c)).collect(),
            wo: (0..nl).map(|_| MuonGroup::zeros(c * c)).collect(),
            ro_w: MuonGroup::zeros(vocab * c),
        };
        Self { cfg, blocks, embed, ro_w, ro_bias: vec![0.0; vocab], adam, muon, step_count: 0 }
    }

    pub fn param_count(&self) -> usize {
        let per_block = 3 * self.cfg.c * self.cfg.c + self.cfg.c;
        per_block * self.cfg.n_layers + self.embed.len() + self.ro_w.len() + self.ro_bias.len()
    }

    pub fn forward(&self, tokens: &[u32]) -> SsmCache {
        let (c, vocab) = (self.cfg.c, self.cfg.vocab);
        let l = tokens.len();
        let mut field = vec![0.0_f32; l * c];
        for t in 0..l {
            let row = &self.embed[tokens[t] as usize * c..(tokens[t] as usize + 1) * c];
            let dst = &mut field[t * c..(t + 1) * c];
            for ci in 0..c {
                dst[ci] += row[ci];
            }
        }
        let mut caches = Vec::with_capacity(self.cfg.n_layers);
        for b in &self.blocks {
            let (out, bc) = block_forward(b, &field, l, c);
            for i in 0..field.len() {
                field[i] += out[i];
            }
            caches.push(bc);
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
        SsmCache { caches, final_field: field, logits, l }
    }

    pub fn backward(&self, cache: &SsmCache, tokens: &[u32], targets: &[u32], g: &mut SsmGrads) -> f32 {
        let (c, vocab) = (self.cfg.c, self.cfg.vocab);
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
        for layer in (0..self.cfg.n_layers).rev() {
            let d_x = block_backward(&self.blocks[layer], &cache.caches[layer], &d_field, &mut g.blocks[layer], l, c);
            for i in 0..d_field.len() {
                d_field[i] += d_x[i]; // residual passthrough + block path
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

    pub fn clip_grads(g: &mut SsmGrads, max_norm: f32) -> f32 {
        let n = g.l2_norm();
        if n > max_norm && n > 0.0 {
            g.scale(max_norm / n);
        }
        n
    }

    pub fn step_muon_hybrid(&mut self, g: &SsmGrads, adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig, scratch: &mut MuonScratch) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let (c, vocab) = (self.cfg.c, self.cfg.vocab);
        for layer in 0..self.cfg.n_layers {
            muon_apply(&mut self.blocks[layer].win, &g.blocks[layer].d_win, &mut self.muon.win[layer], c, c, muon_cfg, scratch);
            muon_apply(&mut self.blocks[layer].wg, &g.blocks[layer].d_wg, &mut self.muon.wg[layer], c, c, muon_cfg, scratch);
            muon_apply(&mut self.blocks[layer].wo, &g.blocks[layer].d_wo, &mut self.muon.wo[layer], c, c, muon_cfg, scratch);
            adamw_apply(&mut self.blocks[layer].bg, &g.blocks[layer].d_bg, &mut self.adam.bg[layer], adam_cfg, bc1, bc2, false);
        }
        muon_apply(&mut self.ro_w, &g.d_ro_w, &mut self.muon.ro_w, vocab, c, muon_cfg, scratch);
        adamw_apply(&mut self.embed, &g.d_embed, &mut self.adam.embed, adam_cfg, bc1, bc2, true);
        adamw_apply(&mut self.ro_bias, &g.d_ro_bias, &mut self.adam.ro_bias, adam_cfg, bc1, bc2, false);
    }
}

#[derive(Clone, Debug, Default)]
pub struct SsmCache {
    pub caches: Vec<BlockCache>,
    pub final_field: Vec<f32>,
    pub logits: Vec<f32>,
    pub l: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SsmGrads {
    pub blocks: Vec<BlockGrads>,
    pub d_embed: Vec<f32>,
    pub d_ro_w: Vec<f32>,
    pub d_ro_bias: Vec<f32>,
}

fn sq_sum(xs: &[f32]) -> f32 {
    xs.iter().map(|v| v * v).sum()
}

impl SsmGrads {
    pub fn zeros(m: &SsmLM) -> Self {
        Self {
            blocks: (0..m.cfg.n_layers).map(|_| BlockGrads::zeros(m.cfg.c)).collect(),
            d_embed: vec![0.0; m.embed.len()],
            d_ro_w: vec![0.0; m.ro_w.len()],
            d_ro_bias: vec![0.0; m.ro_bias.len()],
        }
    }
    pub fn zero(&mut self) {
        for b in &mut self.blocks {
            b.zero();
        }
        self.d_embed.fill(0.0);
        self.d_ro_w.fill(0.0);
        self.d_ro_bias.fill(0.0);
    }
    fn buffers_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v: Vec<&mut Vec<f32>> = Vec::new();
        for b in &mut self.blocks {
            v.push(&mut b.d_win);
            v.push(&mut b.d_wg);
            v.push(&mut b.d_wo);
            v.push(&mut b.d_bg);
        }
        v.push(&mut self.d_embed);
        v.push(&mut self.d_ro_w);
        v.push(&mut self.d_ro_bias);
        v
    }
    fn buffers(&self) -> Vec<&Vec<f32>> {
        let mut v: Vec<&Vec<f32>> = Vec::new();
        for b in &self.blocks {
            v.push(&b.d_win);
            v.push(&b.d_wg);
            v.push(&b.d_wo);
            v.push(&b.d_bg);
        }
        v.push(&self.d_embed);
        v.push(&self.d_ro_w);
        v.push(&self.d_ro_bias);
        v
    }
    pub fn add_assign(&mut self, other: &SsmGrads) {
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
