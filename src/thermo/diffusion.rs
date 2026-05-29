#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::must_use_candidate)]

//! Temperature-gated diffusion (the §2.2 Laplacian term + the §5.1 BPTT
//! stabilizer).
//!
//! Adds `κ_c(T)·∇²s` to every channel, where the conductivity
//! `κ_c(T) = k0_c · sigmoid(a_c + b_c·T)` grows with the local temperature
//! channel `T`. The 5-point Laplacian uses zero-flux (Neumann) boundaries
//! — a missing neighbour contributes nothing — so `Σ ∇²s = 0` and
//! diffusion conserves each channel total, consistent with the flux-form
//! transport. Diffusion is a contracting operator, which keeps the
//! unrolled-rollout gradients tame.
//!
//! `s` enters the output twice: through `∇²` (self-adjoint stencil) and
//! through `κ(T)` via the `T` channel. The backward handles both.

use crate::thermo::field::{idx, ThermoConfig, CH_T};

#[derive(Clone, Debug)]
pub struct DiffusionWeights {
    /// Per-channel conductivity scale `k0_c` (`[c]`).
    pub k0: Vec<f32>,
    /// Per-channel gate bias `a_c` (`[c]`).
    pub a: Vec<f32>,
    /// Per-channel gate temperature-slope `b_c` (`[c]`).
    pub b: Vec<f32>,
}

impl DiffusionWeights {
    pub fn new(cfg: &ThermoConfig) -> Self {
        Self {
            k0: vec![0.1; cfg.c],
            a: vec![0.0; cfg.c],
            b: vec![1.0; cfg.c],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DiffusionGrads {
    pub d_k0: Vec<f32>,
    pub d_a: Vec<f32>,
    pub d_b: Vec<f32>,
}

impl DiffusionGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        Self {
            d_k0: vec![0.0; cfg.c],
            d_a: vec![0.0; cfg.c],
            d_b: vec![0.0; cfg.c],
        }
    }

    pub fn zero(&mut self) {
        self.d_k0.fill(0.0);
        self.d_a.fill(0.0);
        self.d_b.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct DiffusionCache {
    /// `∇²s` per channel, `[h*w*c]`.
    pub lap: Vec<f32>,
    /// Local temperature `T[i,j]`, `[h*w]`.
    pub tvals: Vec<f32>,
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Accumulate the zero-flux 5-point Laplacian `Σ_neighbours (s[nb]-s[c])` of
/// every channel into `dst`. Self-adjoint, so the backward reuses it on the
/// gated source `gbuf`.
fn laplacian_into(dst: &mut [f32], s: &[f32], cfg: &ThermoConfig) {
    let (h, w, c) = (cfg.h, cfg.w, cfg.c);
    for i in 0..h {
        for j in 0..w {
            for ch in 0..c {
                let center = s[idx(i, j, ch, cfg)];
                let mut acc = 0.0_f32;
                if i > 0 {
                    acc += s[idx(i - 1, j, ch, cfg)] - center;
                }
                if i + 1 < h {
                    acc += s[idx(i + 1, j, ch, cfg)] - center;
                }
                if j > 0 {
                    acc += s[idx(i, j - 1, ch, cfg)] - center;
                }
                if j + 1 < w {
                    acc += s[idx(i, j + 1, ch, cfg)] - center;
                }
                dst[idx(i, j, ch, cfg)] += acc;
            }
        }
    }
}

/// Forward: returns the additive diffusion term `delta[i,j,c] =
/// κ_c(T[i,j])·∇²s[i,j,c]` (`[h*w*c]`) and the cache. The caller scales by
/// `dt`/anneal and adds to the field.
pub fn diffusion_forward(
    s: &[f32],
    w: &DiffusionWeights,
    cfg: &ThermoConfig,
) -> (Vec<f32>, DiffusionCache) {
    let mut lap = vec![0.0_f32; cfg.field_len()];
    laplacian_into(&mut lap, s, cfg);
    let mut tvals = vec![0.0_f32; cfg.cells()];
    for cell in 0..cfg.cells() {
        tvals[cell] = s[cell * cfg.c + CH_T];
    }
    let mut delta = vec![0.0_f32; cfg.field_len()];
    for cell in 0..cfg.cells() {
        let t = tvals[cell];
        for ch in 0..cfg.c {
            let kappa = w.k0[ch] * sigmoid(w.a[ch] + w.b[ch] * t);
            delta[cell * cfg.c + ch] = kappa * lap[cell * cfg.c + ch];
        }
    }
    (delta, DiffusionCache { lap, tvals })
}

/// Backward. Returns `d_s [h*w*c]`, accumulates gate-param grads into `g`.
pub fn diffusion_backward(
    w: &DiffusionWeights,
    cache: &DiffusionCache,
    d_delta: &[f32],
    g: &mut DiffusionGrads,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    let (h, ww, c) = (cfg.h, cfg.w, cfg.c);
    let mut d_s = vec![0.0_f32; h * ww * c];
    // gbuf = κ·d_delta, the per-cell source for the self-adjoint Laplacian
    // backward path.
    let mut gbuf = vec![0.0_f32; h * ww * c];

    for cell in 0..cfg.cells() {
        let t = cache.tvals[cell];
        for ch in 0..c {
            let z = w.a[ch] + w.b[ch] * t;
            let sig = sigmoid(z);
            let sigp = sig * (1.0 - sig);
            let kappa = w.k0[ch] * sig;
            let lap_v = cache.lap[cell * c + ch];
            let dd = d_delta[cell * c + ch];

            // gate-parameter gradients
            g.d_k0[ch] += dd * lap_v * sig;
            g.d_a[ch] += dd * lap_v * w.k0[ch] * sigp;
            g.d_b[ch] += dd * lap_v * w.k0[ch] * sigp * t;

            // κ depends on T = s[cell, CH_T] → contributes to that channel.
            d_s[cell * c + CH_T] += dd * lap_v * (w.k0[ch] * sigp * w.b[ch]);

            gbuf[cell * c + ch] = kappa * dd;
        }
    }

    // d_s += ∇²(gbuf) (Laplacian is self-adjoint).
    laplacian_into(&mut d_s, &gbuf, cfg);

    d_s
}
