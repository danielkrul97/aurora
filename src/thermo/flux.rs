#![allow(clippy::needless_range_loop, clippy::must_use_candidate)]

//! Flux-form transport of conserved channels (the §2.2 structural-
//! conservation trick).
//!
//! Each conserved channel is moved by directed edge fluxes rather than a
//! free delta. `fx[z,i,j,m]` is the flux on the edge from cell `(z,i,j)` to
//! `(z,i,j+1)` (its +width edge); `fy[z,i,j,m]` the flux to `(z,i+1,j)`
//! (+height edge); `fz[z,i,j,m]` the flux to `(z+1,i,j)` (+depth edge). The
//! update is the discrete divergence
//! ```text
//! div[z,i,j,m] = (Fx[z,i,j-1] - Fx[z,i,j])
//!              + (Fy[z,i-1,j] - Fy[z,i,j])
//!              + (Fz[z-1,i,j] - Fz[z,i,j])
//! ```
//! with all boundary edges forced to zero (closed walls). Because every
//! interior edge enters one cell with `+` and its neighbour with `−`, the
//! grid-wide sum telescopes to the boundary and is exactly zero — so
//! `Σ field` over a conserved channel is invariant under any weights.
//! Parameter-free and linear: the backward pass is the adjoint gather.
//! `D=1` ⇒ every `fz` edge is a boundary edge (zero), recovering the 2D
//! divergence.

use crate::thermo::field::ThermoConfig;

/// Stored flux on the +width edge of `(z,i,j)`, zero on the far wall.
#[inline]
fn fx_at(fx: &[f32], z: usize, i: usize, j: usize, m: usize, nc: usize, cfg: &ThermoConfig) -> f32 {
    if j + 1 >= cfg.w {
        0.0
    } else {
        fx[((z * cfg.h + i) * cfg.w + j) * nc + m]
    }
}

/// Stored flux on the +height edge of `(z,i,j)`, zero on the far wall.
#[inline]
fn fy_at(fy: &[f32], z: usize, i: usize, j: usize, m: usize, nc: usize, cfg: &ThermoConfig) -> f32 {
    if i + 1 >= cfg.h {
        0.0
    } else {
        fy[((z * cfg.h + i) * cfg.w + j) * nc + m]
    }
}

/// Stored flux on the +depth edge of `(z,i,j)`, zero on the far wall.
#[inline]
fn fz_at(fz: &[f32], z: usize, i: usize, j: usize, m: usize, nc: usize, cfg: &ThermoConfig) -> f32 {
    if z + 1 >= cfg.d {
        0.0
    } else {
        fz[((z * cfg.h + i) * cfg.w + j) * nc + m]
    }
}

/// Forward: directed edge fluxes → per-cell divergence, all
/// `[d*h*w*n_conserved]`.
pub fn flux_div_forward(fx: &[f32], fy: &[f32], fz: &[f32], cfg: &ThermoConfig) -> Vec<f32> {
    let (d, h, w, nc) = (cfg.d, cfg.h, cfg.w, cfg.n_conserved());
    let n = d * h * w * nc;
    debug_assert_eq!(fx.len(), n);
    debug_assert_eq!(fy.len(), n);
    debug_assert_eq!(fz.len(), n);
    let mut div = vec![0.0_f32; n];
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let cell = (z * h + i) * w + j;
                let out = &mut div[cell * nc..(cell + 1) * nc];
                for m in 0..nc {
                    let in_left = if j == 0 { 0.0 } else { fx_at(fx, z, i, j - 1, m, nc, cfg) };
                    let in_down = if i == 0 { 0.0 } else { fy_at(fy, z, i - 1, j, m, nc, cfg) };
                    let in_front = if z == 0 { 0.0 } else { fz_at(fz, z - 1, i, j, m, nc, cfg) };
                    out[m] = (in_left - fx_at(fx, z, i, j, m, nc, cfg))
                        + (in_down - fy_at(fy, z, i, j, m, nc, cfg))
                        + (in_front - fz_at(fz, z, i, j, m, nc, cfg));
                }
            }
        }
    }
    div
}

/// Backward (adjoint). Returns `(d_fx, d_fy, d_fz)`, all
/// `[d*h*w*n_conserved]`, with boundary edges left at zero.
pub fn flux_div_backward(d_div: &[f32], cfg: &ThermoConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (d, h, w, nc) = (cfg.d, cfg.h, cfg.w, cfg.n_conserved());
    let n = d * h * w * nc;
    debug_assert_eq!(d_div.len(), n);
    let mut d_fx = vec![0.0_f32; n];
    let mut d_fy = vec![0.0_f32; n];
    let mut d_fz = vec![0.0_f32; n];
    let at = |z: usize, i: usize, j: usize, m: usize| d_div[((z * h + i) * w + j) * nc + m];
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let cell = (z * h + i) * w + j;
                for m in 0..nc {
                    // fx[z,i,j] (+width edge, exists for j < w-1) enters
                    // div[z,i,j] with −1 and div[z,i,j+1] with +1.
                    if j + 1 < w {
                        d_fx[cell * nc + m] = -at(z, i, j, m) + at(z, i, j + 1, m);
                    }
                    // fy[z,i,j] (+height edge, exists for i < h-1).
                    if i + 1 < h {
                        d_fy[cell * nc + m] = -at(z, i, j, m) + at(z, i + 1, j, m);
                    }
                    // fz[z,i,j] (+depth edge, exists for z < d-1).
                    if z + 1 < d {
                        d_fz[cell * nc + m] = -at(z, i, j, m) + at(z + 1, i, j, m);
                    }
                }
            }
        }
    }
    (d_fx, d_fy, d_fz)
}
