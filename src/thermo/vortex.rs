#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::similar_names
)]

//! Prescribed "vortex" advection + horizon sink — an experimental black-hole
//! geometry layered on the flux-form transport (see `docs/13` §19).
//!
//! A fixed inward+rotational velocity field `v(x)` biases the conserved-channel
//! edge fluxes by an upwind advective term `F += β·s_upwind·v·n̂`. Because the
//! bias is just extra directed edge fluxes, it telescopes across cells exactly
//! like the learned fluxes — conservation stays structurally exact. `v` is
//! parameter-free (derived from the grid geometry), so the only learnable
//! interaction is that the conv now sees a field that is continuously advected
//! toward the centre; the backward is the adjoint upwind gather.
//!
//! The **sink** is the opposite: a conservation-breaking attenuation of the
//! grid core (`field *= 1−strength` inside `radius`), i.e. "information is
//! destroyed at the event horizon". Applied as the last op of a sub-step, its
//! adjoint is the same elementwise mask — a spatial stop-gradient that bounds
//! how far BPTT reaches back through the core.

use crate::thermo::field::ThermoConfig;

#[derive(Clone, Debug)]
pub struct VortexField {
    /// Signed advection coefficient `β·v·n̂` on each +width / +height / +depth
    /// edge, indexed by cell; zero on wall edges (closed box preserved).
    pub cx: Vec<f32>,
    pub cy: Vec<f32>,
    pub cz: Vec<f32>,
    /// Upwind source cell for each edge — the cell whose value is advected.
    /// Fixed because `v` (hence the flow direction per edge) is fixed.
    pub ux: Vec<u32>,
    pub uy: Vec<u32>,
    pub uz: Vec<u32>,
}

/// Inward (radial, 3D) + swirl (azimuthal about the depth axis) unit-ish
/// velocity at a point, relative to the grid centre. Returns `(v_z, v_i, v_j)`
/// over (depth, height, width). `swirl` is the azimuthal/radial ratio.
fn velocity(pz: f32, pi: f32, pj: f32, cz: f32, ci: f32, cj: f32, swirl: f32) -> (f32, f32, f32) {
    let (dz, di, dj) = (pz - cz, pi - ci, pj - cj);
    let r3 = (dz * dz + di * di + dj * dj).sqrt().max(1e-4);
    let rij = (di * di + dj * dj).sqrt().max(1e-4);
    let vz = -dz / r3;
    let vi = -di / r3 + swirl * (-dj) / rij;
    let vj = -dj / r3 + swirl * (di) / rij;
    (vz, vi, vj)
}

impl VortexField {
    pub fn new(cfg: &ThermoConfig, beta: f32, swirl: f32) -> Self {
        let (d, h, w) = (cfg.d, cfg.h, cfg.w);
        let cells = d * h * w;
        let (cz, ci, cj) = (
            (d as f32 - 1.0) * 0.5,
            (h as f32 - 1.0) * 0.5,
            (w as f32 - 1.0) * 0.5,
        );
        let id = |z: usize, i: usize, j: usize| ((z * h + i) * w + j) as u32;
        let mut cx = vec![0.0_f32; cells];
        let mut cy = vec![0.0_f32; cells];
        let mut cz_e = vec![0.0_f32; cells];
        let mut ux = vec![0u32; cells];
        let mut uy = vec![0u32; cells];
        let mut uz = vec![0u32; cells];
        for z in 0..d {
            for i in 0..h {
                for j in 0..w {
                    let cell = (z * h + i) * w + j;
                    let here = id(z, i, j);
                    ux[cell] = here;
                    uy[cell] = here;
                    uz[cell] = here;
                    // +width edge (z,i,j)→(z,i,j+1): width velocity at j+½.
                    if j + 1 < w {
                        let (_, _, vj) = velocity(z as f32, i as f32, j as f32 + 0.5, cz, ci, cj, swirl);
                        cx[cell] = beta * vj;
                        ux[cell] = if vj >= 0.0 { here } else { id(z, i, j + 1) };
                    }
                    // +height edge →(z,i+1,j): height velocity at i+½.
                    if i + 1 < h {
                        let (_, vi, _) = velocity(z as f32, i as f32 + 0.5, j as f32, cz, ci, cj, swirl);
                        cy[cell] = beta * vi;
                        uy[cell] = if vi >= 0.0 { here } else { id(z, i + 1, j) };
                    }
                    // +depth edge →(z+1,i,j): depth velocity at z+½.
                    if z + 1 < d {
                        let (vz, _, _) = velocity(z as f32 + 0.5, i as f32, j as f32, cz, ci, cj, swirl);
                        cz_e[cell] = beta * vz;
                        uz[cell] = if vz >= 0.0 { here } else { id(z + 1, i, j) };
                    }
                }
            }
        }
        Self { cx, cy, cz: cz_e, ux, uy, uz }
    }
}

/// Add the prescribed advective fluxes onto the conv's conserved-channel edge
/// fluxes (in place). `fx/fy/fz` are `[cells*n_conserved]`.
pub fn vortex_flux_add(
    s: &[f32],
    vf: &VortexField,
    cfg: &ThermoConfig,
    fx: &mut [f32],
    fy: &mut [f32],
    fz: &mut [f32],
) {
    let (c, nc) = (cfg.c, cfg.n_conserved());
    for cell in 0..cfg.cells() {
        let (ux, uy, uz) = (vf.ux[cell] as usize, vf.uy[cell] as usize, vf.uz[cell] as usize);
        let (cx, cy, cz) = (vf.cx[cell], vf.cy[cell], vf.cz[cell]);
        for m in 0..nc {
            let ch = cfg.conserved[m];
            fx[cell * nc + m] += cx * s[ux * c + ch];
            fy[cell * nc + m] += cy * s[uy * c + ch];
            fz[cell * nc + m] += cz * s[uz * c + ch];
        }
    }
}

/// Adjoint of `vortex_flux_add`: scatter the edge-flux gradients back onto the
/// upwind source cells. Returns the `d_s` contribution `[field_len]`.
pub fn vortex_flux_backward(
    d_fx: &[f32],
    d_fy: &[f32],
    d_fz: &[f32],
    vf: &VortexField,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    let (c, nc) = (cfg.c, cfg.n_conserved());
    let mut d_s = vec![0.0_f32; cfg.field_len()];
    for cell in 0..cfg.cells() {
        let (ux, uy, uz) = (vf.ux[cell] as usize, vf.uy[cell] as usize, vf.uz[cell] as usize);
        let (cx, cy, cz) = (vf.cx[cell], vf.cy[cell], vf.cz[cell]);
        for m in 0..nc {
            let ch = cfg.conserved[m];
            d_s[ux * c + ch] += cx * d_fx[cell * nc + m];
            d_s[uy * c + ch] += cy * d_fy[cell * nc + m];
            d_s[uz * c + ch] += cz * d_fz[cell * nc + m];
        }
    }
    d_s
}

/// Per-cell horizon attenuation mask `[cells]`: `1−strength` inside `radius`
/// (3D distance from centre), `1.0` outside. `strength=1` ⇒ total destruction.
pub fn build_sink_mask(cfg: &ThermoConfig, radius: f32, strength: f32) -> Vec<f32> {
    let (d, h, w) = (cfg.d, cfg.h, cfg.w);
    let (cz, ci, cj) = (
        (d as f32 - 1.0) * 0.5,
        (h as f32 - 1.0) * 0.5,
        (w as f32 - 1.0) * 0.5,
    );
    let keep = 1.0 - strength;
    let mut m = vec![1.0_f32; d * h * w];
    for z in 0..d {
        for i in 0..h {
            for j in 0..w {
                let (dz, di, dj) = (z as f32 - cz, i as f32 - ci, j as f32 - cj);
                if (dz * dz + di * di + dj * dj).sqrt() <= radius {
                    m[(z * h + i) * w + j] = keep;
                }
            }
        }
    }
    m
}
