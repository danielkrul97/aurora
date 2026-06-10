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
                let base = ((z * h + i) * w + j) * nc;
                // Boundary tests and neighbour bases depend only on (z,i,j),
                // not the conserved channel — hoist them out of the m loop.
                let (has_left, has_down, has_front) = (j > 0, i > 0, z > 0);
                let (has_right, has_up, has_back) = (j + 1 < w, i + 1 < h, z + 1 < d);
                let left = if has_left { base - nc } else { 0 };
                let down = if has_down { base - w * nc } else { 0 };
                let front = if has_front { base - h * w * nc } else { 0 };
                let out = &mut div[base..base + nc];
                for m in 0..nc {
                    let in_left = if has_left { fx[left + m] } else { 0.0 };
                    let in_down = if has_down { fy[down + m] } else { 0.0 };
                    let in_front = if has_front { fz[front + m] } else { 0.0 };
                    let cen_fx = if has_right { fx[base + m] } else { 0.0 };
                    let cen_fy = if has_up { fy[base + m] } else { 0.0 };
                    let cen_fz = if has_back { fz[base + m] } else { 0.0 };
                    out[m] = (in_left - cen_fx) + (in_down - cen_fy) + (in_front - cen_fz);
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
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let base = ((z * h + i) * w + j) * nc;
                let (has_right, has_up, has_back) = (j + 1 < w, i + 1 < h, z + 1 < d);
                let (right, up, back) = (base + nc, base + w * nc, base + h * w * nc);
                for m in 0..nc {
                    let c = d_div[base + m];
                    // fx[z,i,j] (+width edge, exists for j < w-1) enters
                    // div[z,i,j] with −1 and div[z,i,j+1] with +1.
                    if has_right {
                        d_fx[base + m] = -c + d_div[right + m];
                    }
                    // fy[z,i,j] (+height edge, exists for i < h-1).
                    if has_up {
                        d_fy[base + m] = -c + d_div[up + m];
                    }
                    // fz[z,i,j] (+depth edge, exists for z < d-1).
                    if has_back {
                        d_fz[base + m] = -c + d_div[back + m];
                    }
                }
            }
        }
    }
    (d_fx, d_fy, d_fz)
}
