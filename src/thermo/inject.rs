#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::must_use_candidate)]

//! Token injection encoder (the §3.4 localized forcing, reservoir-computing
//! `W_in`).
//!
//! A token adds energy/momentum to the field through a fixed spatial
//! "nozzle" profile times a learned per-channel amplitude:
//! ```text
//! s[z,i,j,c] += anneal · profile[z,i,j] · inj[token, c]
//! ```
//! `profile` is a fixed Gaussian bump centred on the grid (no params).
//! `inj` is a learned `[vocab × c]` table — exactly the embedding-table
//! pattern, so the backward is a scatter-add into the token's row.

use rand::Rng;

use crate::thermo::field::ThermoConfig;

#[derive(Clone, Debug)]
pub struct InjectWeights {
    /// Learned injection amplitudes `[vocab * c]`, row-major by token.
    pub inj: Vec<f32>,
    /// Fixed spatial profile `[d*h*w]`, peak at grid centre.
    pub profile: Vec<f32>,
    pub vocab: usize,
    pub c: usize,
}

impl InjectWeights {
    pub fn new<R: Rng>(cfg: &ThermoConfig, rng: &mut R) -> Self {
        // Injection must write a clearly-readable token signature into the
        // field (against unit background density + accumulated history),
        // or the readout cannot recover even bigram structure.
        let scale = 0.5_f32;
        let inj = (0..cfg.vocab * cfg.c)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let profile = build_profile(cfg, 0.0);
        Self {
            inj,
            profile,
            vocab: cfg.vocab,
            c: cfg.c,
        }
    }
}

/// Fixed spatial nozzle `[d*h*w]`. `ring_radius <= 0` ⇒ a Gaussian bump at the
/// grid centre (the default "inject into the middle"). `ring_radius > 0` ⇒ a
/// Gaussian shell peaked at that 3D radius from the centre — "inject at the
/// rim", so the vortex has somewhere to draw from before it spirals inward.
pub fn build_profile(cfg: &ThermoConfig, ring_radius: f32) -> Vec<f32> {
    let (d, h, w) = (cfg.d, cfg.h, cfg.w);
    let cz = (d as f32 - 1.0) * 0.5;
    let ci = (h as f32 - 1.0) * 0.5;
    let cj = (w as f32 - 1.0) * 0.5;
    let sigma = (d.max(w.max(h)) as f32) / 6.0;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut p = vec![0.0_f32; d * h * w];
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let dz = z as f32 - cz;
                let di = i as f32 - ci;
                let dj = j as f32 - cj;
                let r2 = dz * dz + di * di + dj * dj;
                p[(z * h + i) * w + j] = if ring_radius <= 0.0 {
                    (-r2 * inv).exp()
                } else {
                    let dr = r2.sqrt() - ring_radius;
                    (-dr * dr * inv).exp()
                };
            }
        }
    }
    p
}

#[derive(Clone, Debug, Default)]
pub struct InjectGrads {
    pub d_inj: Vec<f32>,
}

impl InjectGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        Self {
            d_inj: vec![0.0; cfg.vocab * cfg.c],
        }
    }

    pub fn zero(&mut self) {
        self.d_inj.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct InjectCache {
    pub token: u32,
    pub anneal: f32,
}

/// Forward: add the token's localized forcing to a copy of the field.
pub fn inject_forward(
    s: &[f32],
    token: u32,
    anneal: f32,
    w: &InjectWeights,
    cfg: &ThermoConfig,
) -> (Vec<f32>, InjectCache) {
    let c = cfg.c;
    let row = &w.inj[token as usize * c..(token as usize + 1) * c];
    let mut out = s.to_vec();
    for cell in 0..cfg.cells() {
        let p = anneal * w.profile[cell];
        let dst = &mut out[cell * c..(cell + 1) * c];
        for ch in 0..c {
            dst[ch] += p * row[ch];
        }
    }
    (out, InjectCache { token, anneal })
}

/// Backward. `d_out` is `[h*w*c]`; returns `d_s` (identity: injection is
/// additive) and scatter-accumulates the token row of `d_inj`.
pub fn inject_backward(
    cache: &InjectCache,
    d_out: &[f32],
    w: &InjectWeights,
    g: &mut InjectGrads,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    let c = cfg.c;
    let tok = cache.token as usize;
    let drow = &mut g.d_inj[tok * c..(tok + 1) * c];
    for cell in 0..cfg.cells() {
        let p = cache.anneal * w.profile[cell];
        let src = &d_out[cell * c..(cell + 1) * c];
        for ch in 0..c {
            drow[ch] += p * src[ch];
        }
    }
    // ∂s_new/∂s = identity → upstream passes through unchanged.
    d_out.to_vec()
}
