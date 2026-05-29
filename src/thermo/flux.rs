#![allow(clippy::needless_range_loop, clippy::must_use_candidate)]

//! Flux-form transport of conserved channels (the §2.2 structural-
//! conservation trick).
//!
//! Each conserved channel is moved by directed edge fluxes rather than a
//! free delta. `fx[i,j,m]` is the flux on the edge from cell `(i,j)` to
//! `(i,j+1)` (its right edge); `fy[i,j,m]` the flux to `(i+1,j)` (top
//! edge). The update is the discrete divergence
//! ```text
//! div[i,j,m] = (Fx[i,j-1] - Fx[i,j]) + (Fy[i-1,j] - Fy[i,j])
//! ```
//! with all boundary edges forced to zero (closed walls). Because every
//! interior edge enters one cell with `+` and its neighbour with `−`, the
//! grid-wide sum telescopes to the boundary and is exactly zero — so
//! `Σ field` over a conserved channel is invariant under any weights.
//! Parameter-free and linear: the backward pass is the adjoint gather.

use crate::thermo::field::ThermoConfig;

/// Stored flux on the right edge of `(i,j)`, zero on the right wall.
#[inline]
fn fx_at(fx: &[f32], i: usize, j: usize, m: usize, nc: usize, cfg: &ThermoConfig) -> f32 {
    if j + 1 >= cfg.w {
        0.0
    } else {
        fx[(i * cfg.w + j) * nc + m]
    }
}

/// Stored flux on the top edge of `(i,j)`, zero on the top wall.
#[inline]
fn fy_at(fy: &[f32], i: usize, j: usize, m: usize, nc: usize, cfg: &ThermoConfig) -> f32 {
    if i + 1 >= cfg.h {
        0.0
    } else {
        fy[(i * cfg.w + j) * nc + m]
    }
}

/// Forward: directed edge fluxes → per-cell divergence, both
/// `[h*w*n_conserved]`.
pub fn flux_div_forward(fx: &[f32], fy: &[f32], cfg: &ThermoConfig) -> Vec<f32> {
    let (h, w, nc) = (cfg.h, cfg.w, cfg.n_conserved());
    debug_assert_eq!(fx.len(), h * w * nc);
    debug_assert_eq!(fy.len(), h * w * nc);
    let mut div = vec![0.0_f32; h * w * nc];
    for i in 0..h {
        for j in 0..w {
            let out = &mut div[(i * w + j) * nc..(i * w + j + 1) * nc];
            for m in 0..nc {
                let in_left = if j == 0 { 0.0 } else { fx_at(fx, i, j - 1, m, nc, cfg) };
                let in_down = if i == 0 { 0.0 } else { fy_at(fy, i - 1, j, m, nc, cfg) };
                out[m] = (in_left - fx_at(fx, i, j, m, nc, cfg))
                    + (in_down - fy_at(fy, i, j, m, nc, cfg));
            }
        }
    }
    div
}

/// Backward (adjoint). Returns `(d_fx, d_fy)`, both `[h*w*n_conserved]`,
/// with boundary edges left at zero.
pub fn flux_div_backward(d_div: &[f32], cfg: &ThermoConfig) -> (Vec<f32>, Vec<f32>) {
    let (h, w, nc) = (cfg.h, cfg.w, cfg.n_conserved());
    debug_assert_eq!(d_div.len(), h * w * nc);
    let mut d_fx = vec![0.0_f32; h * w * nc];
    let mut d_fy = vec![0.0_f32; h * w * nc];
    let at = |i: usize, j: usize, m: usize| d_div[(i * w + j) * nc + m];
    for i in 0..h {
        for j in 0..w {
            for m in 0..nc {
                // fx[i,j] (right edge, exists for j < w-1) enters div[i,j]
                // with −1 and div[i,j+1] with +1.
                if j + 1 < w {
                    d_fx[(i * w + j) * nc + m] = -at(i, j, m) + at(i, j + 1, m);
                }
                // fy[i,j] (top edge, exists for i < h-1) enters div[i,j]
                // with −1 and div[i+1,j] with +1.
                if i + 1 < h {
                    d_fy[(i * w + j) * nc + m] = -at(i, j, m) + at(i + 1, j, m);
                }
            }
        }
    }
    (d_fx, d_fy)
}
