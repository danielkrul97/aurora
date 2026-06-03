#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

//! Readout head: field → vocab logits.
//!
//! Two feature sources, selectable per `ThermoConfig` (§3.1 + §3.2):
//! * **moments** — `AvgPool(P×P×P)` of the `D×H×W×C` field, the coarse-
//!   grained macroscopic moment vector.
//! * **spectral** — projection onto the lowest 3D-DCT standing-wave modes
//!   per channel (`readout_modes³` modes × C). The cosine basis is a fixed
//!   linear map (orthonormal rows), so it needs no FFT dependency and its
//!   backward is the transpose. The doc notes modal readout is empirically
//!   stronger for periodic/structured signals.
//!
//! Concatenated features feed a `LinearProj` (small-init, à la nanoGPT)
//! plus a per-vocab bias (the unigram prior). Feature order is
//! `[moments | spectral]`, spectral channel-major (`c` outer, mode inner).

use rand::Rng;

use crate::attention::{linear_backward, LinearGrads, LinearProj};
use crate::thermo::field::ThermoConfig;

#[derive(Clone, Debug)]
pub struct ReadoutWeights {
    pub proj: LinearProj,
    pub bias: Vec<f32>,
    /// Fixed DCT basis `[modes² * cells]`, unit-norm rows. Empty when the
    /// spectral readout is disabled. Not a trained parameter.
    pub basis: Vec<f32>,
    /// Per-cell horizon mask `[cells]` gating which cells the head reads (the
    /// §19 "read head at the event horizon"). Empty = read the whole field.
    /// Not a trained parameter; set via `ThermoLM`.
    pub horizon: Vec<f32>,
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
            horizon: Vec::new(),
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

/// Gaussian shell `[cells]` peaked at `radius` (3D distance from centre) — the
/// horizon the read head listens on. Cells far from the shell get ≈0, so the
/// readout sees only what the vortex has concentrated near that radius.
pub fn build_horizon_mask(cfg: &ThermoConfig, radius: f32) -> Vec<f32> {
    let (d, h, w) = (cfg.d, cfg.h, cfg.w);
    let (cz, ci, cj) = (
        (d as f32 - 1.0) * 0.5,
        (h as f32 - 1.0) * 0.5,
        (w as f32 - 1.0) * 0.5,
    );
    let sigma = (d.max(w.max(h)) as f32) / 8.0;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut m = vec![0.0_f32; d * h * w];
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let (dz, di, dj) = (z as f32 - cz, i as f32 - ci, j as f32 - cj);
                let dr = (dz * dz + di * di + dj * dj).sqrt() - radius;
                m[(z * h + i) * w + j] = (-dr * dr * inv).exp();
            }
        }
    }
    m
}

/// Lowest `modes³` 3D cosine standing-wave modes, each row L2-normalized so
/// coefficients are well-scaled. Row `(mz*modes+mx)*modes+my`, cell
/// `(z,i,j)`: `cos(π·mz·(z+½)/D)·cos(π·mx·(i+½)/H)·cos(π·my·(j+½)/W)`.
fn build_dct_basis(cfg: &ThermoConfig) -> Vec<f32> {
    let (d, h, w, modes) = (cfg.d, cfg.h, cfg.w, cfg.readout_modes);
    let cells = d * h * w;
    let mut basis = vec![0.0_f32; modes * modes * modes * cells];
    let pi = std::f32::consts::PI;
    for mz in 0..modes {
        for mx in 0..modes {
            for my in 0..modes {
                let row = ((mz * modes + mx) * modes + my) * cells;
                let mut norm_sq = 0.0_f32;
                for z in 0..d {
                    for i in 0..h {
                        for j in 0..w {
                            let v = (pi * mz as f32 * (z as f32 + 0.5) / d as f32).cos()
                                * (pi * mx as f32 * (i as f32 + 0.5) / h as f32).cos()
                                * (pi * my as f32 * (j as f32 + 0.5) / w as f32).cos();
                            basis[row + (z * h + i) * w + j] = v;
                            norm_sq += v * v;
                        }
                    }
                }
                let inv = norm_sq.sqrt().max(1e-8).recip();
                for cc in &mut basis[row..row + cells] {
                    *cc *= inv;
                }
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
    let (d, h, w, c, p) = (cfg.d, cfg.h, cfg.w, cfg.c, cfg.pool);
    let (pd, ph, pw) = (d / p, h / p, w / p);
    let inv = 1.0 / (p * p * p) as f32;
    let mut out = vec![0.0_f32; pd * ph * pw * c];
    for bd in 0..pd {
        for bi in 0..ph {
            for bj in 0..pw {
                let blk = (bd * ph + bi) * pw + bj;
                let dst = &mut out[blk * c..(blk + 1) * c];
                for dz in 0..p {
                    for di in 0..p {
                        for dj in 0..p {
                            let cell =
                                ((bd * p + dz) * h + (bi * p + di)) * w + (bj * p + dj);
                            let src = &s[cell * c..(cell + 1) * c];
                            for ch in 0..c {
                                dst[ch] += src[ch] * inv;
                            }
                        }
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
    let m = cfg.readout_modes * cfg.readout_modes * cfg.readout_modes;
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

/// Dot product with 8 independent accumulator lanes. The fixed-width inner
/// step lets LLVM emit packed multiply-add; the lanes break the serial
/// dependency that pins a naive `acc += a*b` reduction at scalar latency.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f32; 8];
    let chunks = a.len() / 8;
    for ch in 0..chunks {
        let ar = &a[ch * 8..ch * 8 + 8];
        let br = &b[ch * 8..ch * 8 + 8];
        for l in 0..8 {
            acc[l] += ar[l] * br[l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for k in chunks * 8..a.len() {
        s += a[k] * b[k];
    }
    s
}

pub fn readout_forward(s: &[f32], w: &ReadoutWeights, cfg: &ThermoConfig) -> (Vec<f32>, ReadoutCache) {
    // Horizon read head: gate the field to the shell before extracting features.
    let masked_storage;
    let s: &[f32] = if w.horizon.is_empty() {
        s
    } else {
        let c = cfg.c;
        let mut m = s.to_vec();
        for cell in 0..cfg.cells() {
            let g = w.horizon[cell];
            for ch in 0..c {
                m[cell * c + ch] *= g;
            }
        }
        masked_storage = m;
        &masked_storage
    };

    let mut features = Vec::with_capacity(cfg.readout_dim());
    if w.use_moments {
        features.extend_from_slice(&avg_pool(s, cfg));
    }
    if w.use_spectral {
        features.extend_from_slice(&spectral_coeffs(s, w, cfg));
    }
    let d_in = w.proj.d_in;
    let mut logits = vec![0.0_f32; cfg.vocab];
    for o in 0..cfg.vocab {
        let row = &w.proj.w[o * d_in..(o + 1) * d_in];
        logits[o] = dot(row, &features) + w.bias[o];
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

    let (d, h, ww, c, p) = (cfg.d, cfg.h, cfg.w, cfg.c, cfg.pool);
    let mut d_s = vec![0.0_f32; cfg.field_len()];
    let mut off = 0;

    if w.use_moments {
        // AvgPool backward: each pooled value spreads 1/P³ to its block.
        let (pd, ph, pw) = (d / p, h / p, ww / p);
        let inv = 1.0 / (p * p * p) as f32;
        let d_pooled = &d_features[off..off + pd * ph * pw * c];
        for bd in 0..pd {
            for bi in 0..ph {
                for bj in 0..pw {
                    let blk = (bd * ph + bi) * pw + bj;
                    let src = &d_pooled[blk * c..(blk + 1) * c];
                    for dz in 0..p {
                        for di in 0..p {
                            for dj in 0..p {
                                let cell = ((bd * p + dz) * h + (bi * p + di)) * ww
                                    + (bj * p + dj);
                                let dst = &mut d_s[cell * c..(cell + 1) * c];
                                for ch in 0..c {
                                    dst[ch] += src[ch] * inv;
                                }
                            }
                        }
                    }
                }
            }
        }
        off += pd * ph * pw * c;
    }

    if w.use_spectral {
        // Spectral backward (transpose of the basis projection).
        let cells = cfg.cells();
        let m = cfg.readout_modes * cfg.readout_modes * cfg.readout_modes;
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

    // Adjoint of the horizon gate (features were read off `mask·s`).
    if !w.horizon.is_empty() {
        for cell in 0..cfg.cells() {
            let g = w.horizon[cell];
            for ch in 0..c {
                d_s[cell * c + ch] *= g;
            }
        }
    }

    d_s
}
