#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

//! Local operator `N_θ = Conv1x1(GELU(Conv3x3(s)))`.
//!
//! The 3×3 stage enforces locality (a finite speed of sound: information
//! travels at most one cell per sub-step); the 1×1 stage mixes channels.
//! Weights are shared across space → translation invariance is free.
//! Boundary uses zero padding ("nothing beyond the wall"). Output has
//! `c_out = c + n_conserved` channels, decoded downstream into free
//! residual deltas and conserved-channel edge fluxes.
//!
//! ## Vectorization
//!
//! The stored weight layout (`w1[((co·c+ci)·3+ki)·3+kj]`, `w2[co·c_h+ci]`)
//! is the on-disk / GPU contract and is left untouched. Internally each call
//! repacks the small weight tensors into a layout whose innermost loop is a
//! contiguous element-wise multiply-add (an axpy) over the output channel,
//! never a horizontal sum reduction — that is the shape LLVM auto-vectorizes
//! in safe Rust without `fast-math`. The repack is `O(weights)` (~1.7k
//! floats), negligible against the `O(cells·c_h·c·9)` kernel it feeds.

use rand::Rng;

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
    /// Pre-GELU hidden activations `[h*w*c_h]`. Retained for the GPU parity
    /// harness (which re-derives the GELU backward on-device from it); the
    /// CPU backward uses the cached `gelu_deriv` instead.
    pub h_pre: Vec<f32>,
    /// Post-GELU hidden activations `[h*w*c_h]`.
    pub h_post: Vec<f32>,
    /// `d(h_post)/d(h_pre)` per element, captured during the forward GELU so
    /// the backward never recomputes a `tanh`. Bit-identical to
    /// `gelu_backward(h_pre)`.
    pub gelu_deriv: Vec<f32>,
}

#[inline]
const fn w1_idx(co: usize, ci: usize, ki: usize, kj: usize, c: usize) -> usize {
    ((co * c + ci) * 3 + ki) * 3 + kj
}

/// GELU and its derivative in one pass, sharing the single `tanh`. The two
/// outputs equal `linalg::gelu(x)` and `linalg::gelu_backward(x)` exactly.
#[inline]
fn gelu_and_deriv(x: f32) -> (f32, f32) {
    let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044_715 * x * x * x);
    let t = inner.tanh();
    let post = 0.5 * x * (1.0 + t);
    let sech2 = 1.0 - t * t;
    let d_inner_dx = k * (1.0 + 3.0 * 0.044_715 * x * x);
    let deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner_dx;
    (post, deriv)
}

/// Repack `w1` to `wco[(tap*c + ci)*c_h + co]` (output channel innermost) so
/// the 3×3 forward / `d_w1` inner loop is a contiguous axpy over `co`.
fn pack_w1_co(w1: &[f32], c: usize, c_h: usize) -> Vec<f32> {
    let mut wco = vec![0.0_f32; 9 * c * c_h];
    for co in 0..c_h {
        for ci in 0..c {
            for ki in 0..3 {
                for kj in 0..3 {
                    let tap = ki * 3 + kj;
                    wco[(tap * c + ci) * c_h + co] = w1[w1_idx(co, ci, ki, kj, c)];
                }
            }
        }
    }
    wco
}

/// Repack `w1` to `wci[(co*9 + tap)*c + ci]` (input channel innermost) so the
/// 3×3 backward `d_s` inner loop is a contiguous axpy over `ci`.
fn pack_w1_ci(w1: &[f32], c: usize, c_h: usize) -> Vec<f32> {
    let mut wci = vec![0.0_f32; c_h * 9 * c];
    for co in 0..c_h {
        for ci in 0..c {
            for ki in 0..3 {
                for kj in 0..3 {
                    let tap = ki * 3 + kj;
                    wci[(co * 9 + tap) * c + ci] = w1[w1_idx(co, ci, ki, kj, c)];
                }
            }
        }
    }
    wci
}

/// Transpose `w2` to `w2t[ci*c_out + co]` so the 1×1 forward inner loop is a
/// contiguous axpy over `co`.
fn pack_w2_t(w2: &[f32], c_h: usize, c_out: usize) -> Vec<f32> {
    let mut w2t = vec![0.0_f32; c_h * c_out];
    for co in 0..c_out {
        for ci in 0..c_h {
            w2t[ci * c_out + co] = w2[co * c_h + ci];
        }
    }
    w2t
}

/// Forward. `s` is `[h*w*c]`; returns `raw [h*w*c_out]` and the cache.
pub fn conv_forward(w: &ConvWeights, s: &[f32], cfg: &ThermoConfig) -> (Vec<f32>, ConvCache) {
    let (h, ww, c, c_h, c_out) = (cfg.h, cfg.w, w.c, w.c_h, w.c_out);
    debug_assert_eq!(s.len(), h * ww * c);

    // Conv3x3 → h_pre. Output-stationary: accumulate a full c_h vector per
    // cell, inner loop an axpy over co.
    let wco = pack_w1_co(&w.w1, c, c_h);
    let mut h_pre = vec![0.0_f32; h * ww * c_h];
    for i in 0..h {
        for j in 0..ww {
            let acc = &mut h_pre[(i * ww + j) * c_h..(i * ww + j + 1) * c_h];
            acc.copy_from_slice(&w.b1);
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
                    let tap = ki * 3 + kj;
                    for ci in 0..c {
                        let sc = s[in_base + ci];
                        let wrow = &wco[(tap * c + ci) * c_h..(tap * c + ci) * c_h + c_h];
                        for (a, &wv) in acc.iter_mut().zip(wrow) {
                            *a += sc * wv;
                        }
                    }
                }
            }
        }
    }

    // GELU → h_post, capturing the derivative for the backward (one tanh).
    let mut h_post = vec![0.0_f32; h * ww * c_h];
    let mut gelu_deriv = vec![0.0_f32; h * ww * c_h];
    for k in 0..h_pre.len() {
        let (post, deriv) = gelu_and_deriv(h_pre[k]);
        h_post[k] = post;
        gelu_deriv[k] = deriv;
    }

    // Conv1x1 (per-cell linear) → raw. Inner loop an axpy over co via w2t.
    let w2t = pack_w2_t(&w.w2, c_h, c_out);
    let mut raw = vec![0.0_f32; h * ww * c_out];
    for cell in 0..h * ww {
        let out = &mut raw[cell * c_out..(cell + 1) * c_out];
        out.copy_from_slice(&w.b2);
        let hp = &h_post[cell * c_h..(cell + 1) * c_h];
        for ci in 0..c_h {
            let sc = hp[ci];
            let wrow = &w2t[ci * c_out..(ci + 1) * c_out];
            for (o, &wv) in out.iter_mut().zip(wrow) {
                *o += sc * wv;
            }
        }
    }

    (
        raw,
        ConvCache {
            s: s.to_vec(),
            h_pre,
            h_post,
            gelu_deriv,
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

    // Conv1x1 backward → d_h_post. Inner loops are axpy over ci against the
    // stored `w2[co][ci]` (ci already contiguous), so they vectorize as-is.
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

    // GELU backward via the cached derivative — no tanh recompute.
    let mut d_h_pre = vec![0.0_f32; h * ww * c_h];
    for k in 0..d_h_pre.len() {
        d_h_pre[k] = d_h_post[k] * cache.gelu_deriv[k];
    }

    // Conv3x3 backward → d_s, d_w1, d_b1. `d_s` uses the ci-innermost pack
    // (axpy over ci); `d_w1` accumulates into a co-innermost pack (axpy over
    // co) that is scattered back to the canonical layout once at the end.
    let wci = pack_w1_ci(&w.w1, c, c_h);
    let mut d_wco = vec![0.0_f32; 9 * c * c_h];
    let mut d_s = vec![0.0_f32; h * ww * c];
    for i in 0..h {
        for j in 0..ww {
            let dhp = &d_h_pre[(i * ww + j) * c_h..(i * ww + j + 1) * c_h];
            for (b, &dv) in g.d_b1.iter_mut().zip(dhp) {
                *b += dv;
            }
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
                    let tap = ki * 3 + kj;
                    // d_s[in_base + ci] += Σ_co dhp[co]·w1[co][ci][tap]
                    let dsrow = &mut d_s[in_base..in_base + c];
                    for co in 0..c_h {
                        let gco = dhp[co];
                        let wrow = &wci[(co * 9 + tap) * c..(co * 9 + tap) * c + c];
                        for (ds, &wv) in dsrow.iter_mut().zip(wrow) {
                            *ds += gco * wv;
                        }
                    }
                    // d_w1[co][ci][tap] += dhp[co]·s[in_base + ci]
                    for ci in 0..c {
                        let sc = cache.s[in_base + ci];
                        let dwrow = &mut d_wco[(tap * c + ci) * c_h..(tap * c + ci) * c_h + c_h];
                        for (dw, &dv) in dwrow.iter_mut().zip(dhp) {
                            *dw += sc * dv;
                        }
                    }
                }
            }
        }
    }

    // Scatter the packed d_w1 back to the canonical layout (accumulating).
    for co in 0..c_h {
        for ci in 0..c {
            for ki in 0..3 {
                for kj in 0..3 {
                    let tap = ki * 3 + kj;
                    g.d_w1[w1_idx(co, ci, ki, kj, c)] += d_wco[(tap * c + ci) * c_h + co];
                }
            }
        }
    }

    d_s
}
