#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::must_use_candidate)]

//! Readout head: field → vocab logits.
//!
//! Two feature sources, selectable per `ThermoConfig` (§3.1 + §3.2):
//! * **moments** — `AvgPool(P×P)` of the `H×W×C` field, the coarse-grained
//!   macroscopic moment vector.
//! * **spectral** — projection onto the lowest 2D-DCT standing-wave modes
//!   per channel (`readout_modes²` modes × C). The cosine basis is a fixed
//!   linear map (orthonormal rows), so it needs no FFT dependency and its
//!   backward is the transpose. The doc notes modal readout is empirically
//!   stronger for periodic/structured signals.
//!
//! Concatenated features feed a `LinearProj` (small-init, à la nanoGPT)
//! plus a per-vocab bias (the unigram prior). Feature order is
//! `[moments | spectral]`, spectral channel-major (`c` outer, mode inner).

use rand::Rng;

use crate::attention::{linear_backward, linear_forward, LinearGrads, LinearProj};
use crate::thermo::field::ThermoConfig;

#[derive(Clone, Debug)]
pub struct ReadoutWeights {
    pub proj: LinearProj,
    pub bias: Vec<f32>,
    /// Fixed DCT basis `[modes² * cells]`, unit-norm rows. Empty when the
    /// spectral readout is disabled. Not a trained parameter.
    pub basis: Vec<f32>,
    pub use_moments: bool,
    pub use_spectral: bool,
    pub modes: usize,
}

impl ReadoutWeights {
    pub fn new<R: Rng>(cfg: &ThermoConfig, rng: &mut R) -> Self {
        let basis = if cfg.readout_spectral { build_dct_basis(cfg) } else { Vec::new() };
        Self {
            proj: LinearProj::new_scaled(cfg.readout_dim(), cfg.vocab, 0.1, rng),
            bias: vec![0.0; cfg.vocab],
            basis,
            use_moments: cfg.readout_moments,
            use_spectral: cfg.readout_spectral,
            modes: cfg.readout_modes,
        }
    }

    /// Rebuild the (derived, non-serialized) DCT basis after loading.
    pub fn rebuild_basis(&mut self, cfg: &ThermoConfig) {
        self.basis = if cfg.readout_spectral { build_dct_basis(cfg) } else { Vec::new() };
    }
}

/// Lowest `modes²` 2D cosine standing-wave modes, each row L2-normalized so
/// coefficients are well-scaled. Row `mx*modes+my`, cell `(i,j)`:
/// `cos(π·mx·(i+½)/H)·cos(π·my·(j+½)/W)`.
fn build_dct_basis(cfg: &ThermoConfig) -> Vec<f32> {
    let (h, w, modes) = (cfg.h, cfg.w, cfg.readout_modes);
    let cells = h * w;
    let mut basis = vec![0.0_f32; modes * modes * cells];
    let pi = std::f32::consts::PI;
    for mx in 0..modes {
        for my in 0..modes {
            let row = (mx * modes + my) * cells;
            let mut norm_sq = 0.0_f32;
            for i in 0..h {
                for j in 0..w {
                    let v = (pi * mx as f32 * (i as f32 + 0.5) / h as f32).cos()
                        * (pi * my as f32 * (j as f32 + 0.5) / w as f32).cos();
                    basis[row + i * w + j] = v;
                    norm_sq += v * v;
                }
            }
            let inv = norm_sq.sqrt().max(1e-8).recip();
            for c in &mut basis[row..row + cells] {
                *c *= inv;
            }
        }
    }
    basis
}

#[derive(Clone, Debug, Default)]
pub struct ReadoutGrads {
    pub d_proj: LinearGrads,
    pub d_bias: Vec<f32>,
}

impl ReadoutGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        Self {
            d_proj: LinearGrads::zeros(cfg.readout_dim(), cfg.vocab),
            d_bias: vec![0.0; cfg.vocab],
        }
    }

    pub fn zero(&mut self) {
        self.d_proj.zero();
        self.d_bias.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReadoutCache {
    /// Concatenated readout features `[readout_dim]`.
    pub features: Vec<f32>,
}

fn avg_pool(s: &[f32], cfg: &ThermoConfig) -> Vec<f32> {
    let (h, w, c, p) = (cfg.h, cfg.w, cfg.c, cfg.pool);
    let (ph, pw) = (h / p, w / p);
    let inv = 1.0 / (p * p) as f32;
    let mut out = vec![0.0_f32; ph * pw * c];
    for bi in 0..ph {
        for bj in 0..pw {
            let dst = &mut out[(bi * pw + bj) * c..(bi * pw + bj + 1) * c];
            for di in 0..p {
                for dj in 0..p {
                    let cell = (bi * p + di) * w + (bj * p + dj);
                    let src = &s[cell * c..(cell + 1) * c];
                    for ch in 0..c {
                        dst[ch] += src[ch] * inv;
                    }
                }
            }
        }
    }
    out
}

/// `coeff[c, m] = Σ_cell basis[m, cell] · field[cell, c]`, channel-major.
fn spectral_coeffs(s: &[f32], w: &ReadoutWeights, cfg: &ThermoConfig) -> Vec<f32> {
    let (cells, c) = (cfg.cells(), cfg.c);
    let m = cfg.readout_modes * cfg.readout_modes;
    let mut out = vec![0.0_f32; c * m];
    for ch in 0..c {
        for mode in 0..m {
            let brow = &w.basis[mode * cells..(mode + 1) * cells];
            let mut acc = 0.0_f32;
            for cell in 0..cells {
                acc += brow[cell] * s[cell * c + ch];
            }
            out[ch * m + mode] = acc;
        }
    }
    out
}

pub fn readout_forward(s: &[f32], w: &ReadoutWeights, cfg: &ThermoConfig) -> (Vec<f32>, ReadoutCache) {
    let mut features = Vec::with_capacity(cfg.readout_dim());
    if w.use_moments {
        features.extend_from_slice(&avg_pool(s, cfg));
    }
    if w.use_spectral {
        features.extend_from_slice(&spectral_coeffs(s, w, cfg));
    }
    let mut logits = linear_forward(&w.proj, &features);
    for v in 0..cfg.vocab {
        logits[v] += w.bias[v];
    }
    (logits, ReadoutCache { features })
}

pub fn readout_backward(
    w: &ReadoutWeights,
    cache: &ReadoutCache,
    d_logits: &[f32],
    g: &mut ReadoutGrads,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    for v in 0..cfg.vocab {
        g.d_bias[v] += d_logits[v];
    }
    let d_features = linear_backward(&w.proj, &cache.features, d_logits, &mut g.d_proj);

    let (h, ww, c, p) = (cfg.h, cfg.w, cfg.c, cfg.pool);
    let mut d_s = vec![0.0_f32; cfg.field_len()];
    let mut off = 0;

    if w.use_moments {
        // AvgPool backward: each pooled value spreads 1/P² to its block.
        let (ph, pw) = (h / p, ww / p);
        let inv = 1.0 / (p * p) as f32;
        let d_pooled = &d_features[off..off + ph * pw * c];
        for bi in 0..ph {
            for bj in 0..pw {
                let src = &d_pooled[(bi * pw + bj) * c..(bi * pw + bj + 1) * c];
                for di in 0..p {
                    for dj in 0..p {
                        let cell = (bi * p + di) * ww + (bj * p + dj);
                        let dst = &mut d_s[cell * c..(cell + 1) * c];
                        for ch in 0..c {
                            dst[ch] += src[ch] * inv;
                        }
                    }
                }
            }
        }
        off += ph * pw * c;
    }

    if w.use_spectral {
        // Spectral backward (transpose of the basis projection).
        let cells = cfg.cells();
        let m = cfg.readout_modes * cfg.readout_modes;
        let d_spec = &d_features[off..off + c * m];
        for ch in 0..c {
            for mode in 0..m {
                let g_cm = d_spec[ch * m + mode];
                if g_cm == 0.0 {
                    continue;
                }
                let brow = &w.basis[mode * cells..(mode + 1) * cells];
                for cell in 0..cells {
                    d_s[cell * c + ch] += brow[cell] * g_cm;
                }
            }
        }
    }

    d_s
}
