#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::many_single_char_names
)]

//! Local operator `N_θ = Conv1x1(GELU(Conv3x3(s)))`.
//!
//! The 3×3 stage enforces locality (a finite speed of sound: information
//! travels at most one cell per sub-step); the 1×1 stage mixes channels.
//! Weights are shared across space → translation invariance is free.
//! Boundary uses zero padding ("nothing beyond the wall"). Output has
//! `c_out = c + n_conserved` channels, decoded downstream into free
//! residual deltas and conserved-channel edge fluxes.

use rand::Rng;

use crate::linalg::{gelu, gelu_backward};
use crate::thermo::field::ThermoConfig;

#[derive(Clone, Debug)]
pub struct ConvWeights {
    /// `[c_h * c * 3 * 3]`, index `((co*c + ci)*3 + ki)*3 + kj`.
    pub w1: Vec<f32>,
    /// `[c_h]`
    pub b1: Vec<f32>,
    /// `[c_out * c_h]`, index `co*c_h + ci`.
    pub w2: Vec<f32>,
    /// `[c_out]`
    pub b2: Vec<f32>,
    pub c: usize,
    pub c_h: usize,
    pub c_out: usize,
}

impl ConvWeights {
    pub fn new<R: Rng>(cfg: &ThermoConfig, rng: &mut R) -> Self {
        let (c, c_h, c_out) = (cfg.c, cfg.conv_hidden, cfg.c_out());
        let fan1_in = c * 9;
        let b1_bound = (6.0_f32 / (fan1_in + c_h) as f32).sqrt();
        // Gentle-start scaling on the operator's output projection so the
        // residual update `s + Δt·N_θ(s)` stays near-identity at init and
        // the field doesn't blow up over the K-step rollout.
        let resid_scale = (2.0 * cfg.k_steps as f32).sqrt().recip();
        let b2_bound = (6.0_f32 / (c_h + c_out) as f32).sqrt() * resid_scale;
        let unif = |bound: f32, rng: &mut R| (rng.random::<f32>() * 2.0 - 1.0) * bound;
        Self {
            w1: (0..c_h * c * 9).map(|_| unif(b1_bound, rng)).collect(),
            b1: vec![0.0; c_h],
            w2: (0..c_out * c_h).map(|_| unif(b2_bound, rng)).collect(),
            b2: vec![0.0; c_out],
            c,
            c_h,
            c_out,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConvGrads {
    pub d_w1: Vec<f32>,
    pub d_b1: Vec<f32>,
    pub d_w2: Vec<f32>,
    pub d_b2: Vec<f32>,
}

impl ConvGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        let (c, c_h, c_out) = (cfg.c, cfg.conv_hidden, cfg.c_out());
        Self {
            d_w1: vec![0.0; c_h * c * 9],
            d_b1: vec![0.0; c_h],
            d_w2: vec![0.0; c_out * c_h],
            d_b2: vec![0.0; c_out],
        }
    }

    pub fn zero(&mut self) {
        self.d_w1.fill(0.0);
        self.d_b1.fill(0.0);
        self.d_w2.fill(0.0);
        self.d_b2.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConvCache {
    pub s: Vec<f32>,
    /// Pre-GELU hidden activations `[h*w*c_h]`.
    pub h_pre: Vec<f32>,
    /// Post-GELU hidden activations `[h*w*c_h]`.
    pub h_post: Vec<f32>,
}

#[inline]
fn w1_idx(co: usize, ci: usize, ki: usize, kj: usize, c: usize) -> usize {
    ((co * c + ci) * 3 + ki) * 3 + kj
}

/// Forward. `s` is `[h*w*c]`; returns `raw [h*w*c_out]` and the cache.
pub fn conv_forward(w: &ConvWeights, s: &[f32], cfg: &ThermoConfig) -> (Vec<f32>, ConvCache) {
    let (h, ww, c, c_h, c_out) = (cfg.h, cfg.w, w.c, w.c_h, w.c_out);
    debug_assert_eq!(s.len(), h * ww * c);

    // Conv3x3 → h_pre, then GELU → h_post.
    let mut h_pre = vec![0.0_f32; h * ww * c_h];
    for i in 0..h {
        for j in 0..ww {
            let out_base = (i * ww + j) * c_h;
            for co in 0..c_h {
                let mut acc = w.b1[co];
                for ki in 0..3 {
                    let ii = i as isize + ki as isize - 1;
                    if ii < 0 || ii >= h as isize {
                        continue;
                    }
                    for kj in 0..3 {
                        let jj = j as isize + kj as isize - 1;
                        if jj < 0 || jj >= ww as isize {
                            continue;
                        }
                        let in_base = (ii as usize * ww + jj as usize) * c;
                        for ci in 0..c {
                            acc += w.w1[w1_idx(co, ci, ki, kj, c)] * s[in_base + ci];
                        }
                    }
                }
                h_pre[out_base + co] = acc;
            }
        }
    }
    let h_post: Vec<f32> = h_pre.iter().map(|&v| gelu(v)).collect();

    // Conv1x1 (per-cell linear) → raw.
    let mut raw = vec![0.0_f32; h * ww * c_out];
    for cell in 0..h * ww {
        let hp = &h_post[cell * c_h..(cell + 1) * c_h];
        let out = &mut raw[cell * c_out..(cell + 1) * c_out];
        for co in 0..c_out {
            let row = &w.w2[co * c_h..(co + 1) * c_h];
            let mut acc = w.b2[co];
            for ci in 0..c_h {
                acc += row[ci] * hp[ci];
            }
            out[co] = acc;
        }
    }

    (
        raw,
        ConvCache {
            s: s.to_vec(),
            h_pre,
            h_post,
        },
    )
}

/// Backward. `d_raw` is `[h*w*c_out]`; returns `d_s [h*w*c]` and
/// accumulates weight gradients into `g`.
pub fn conv_backward(
    w: &ConvWeights,
    cache: &ConvCache,
    d_raw: &[f32],
    g: &mut ConvGrads,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    let (h, ww, c, c_h, c_out) = (cfg.h, cfg.w, w.c, w.c_h, w.c_out);
    debug_assert_eq!(d_raw.len(), h * ww * c_out);

    // Conv1x1 backward → d_h_post.
    let mut d_h_post = vec![0.0_f32; h * ww * c_h];
    for cell in 0..h * ww {
        let dr = &d_raw[cell * c_out..(cell + 1) * c_out];
        let hp = &cache.h_post[cell * c_h..(cell + 1) * c_h];
        let dhp = &mut d_h_post[cell * c_h..(cell + 1) * c_h];
        for co in 0..c_out {
            let gco = dr[co];
            g.d_b2[co] += gco;
            let w_row = &w.w2[co * c_h..(co + 1) * c_h];
            let dw_row = &mut g.d_w2[co * c_h..(co + 1) * c_h];
            for ci in 0..c_h {
                dw_row[ci] += gco * hp[ci];
                dhp[ci] += gco * w_row[ci];
            }
        }
    }

    // GELU backward → d_h_pre.
    let mut d_h_pre = vec![0.0_f32; h * ww * c_h];
    for k in 0..d_h_pre.len() {
        d_h_pre[k] = d_h_post[k] * gelu_backward(cache.h_pre[k]);
    }

    // Conv3x3 backward → d_s, d_w1, d_b1.
    let mut d_s = vec![0.0_f32; h * ww * c];
    for i in 0..h {
        for j in 0..ww {
            let dhp = &d_h_pre[(i * ww + j) * c_h..(i * ww + j + 1) * c_h];
            for co in 0..c_h {
                let gco = dhp[co];
                if gco == 0.0 {
                    continue;
                }
                g.d_b1[co] += gco;
                for ki in 0..3 {
                    let ii = i as isize + ki as isize - 1;
                    if ii < 0 || ii >= h as isize {
                        continue;
                    }
                    for kj in 0..3 {
                        let jj = j as isize + kj as isize - 1;
                        if jj < 0 || jj >= ww as isize {
                            continue;
                        }
                        let in_base = (ii as usize * ww + jj as usize) * c;
                        for ci in 0..c {
                            let wi = w1_idx(co, ci, ki, kj, c);
                            g.d_w1[wi] += gco * cache.s[in_base + ci];
                            d_s[in_base + ci] += gco * w.w1[wi];
                        }
                    }
                }
            }
        }
    }

    d_s
}
