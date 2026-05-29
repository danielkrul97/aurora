#![allow(clippy::needless_range_loop, clippy::must_use_candidate)]

//! One sub-step of the local operator: `s_{t+1} = s_t + Δt·N_θ(s_t)`.
//!
//! Composes conv → decode (free deltas + conserved-channel edge fluxes) →
//! flux divergence → temperature-gated diffusion → assemble. Conserved
//! channels move only by flux (mass/energy preserved); free channels take
//! a residual delta; every channel diffuses. The per-sub-step `anneal`
//! scalar modulates the diffusion amplitude (the §4 hot→cold schedule);
//! the local `κ(T)` gate modulates it further per cell.
//!
//! Output-channel layout of the conv (`c_out = (c−n_cons) free + 2·n_cons
//! flux`): `[free deltas | Fx per conserved | Fy per conserved]`.

use rand::Rng;

use crate::thermo::conv::{conv_backward, conv_forward, ConvCache, ConvGrads, ConvWeights};
use crate::thermo::diffusion::{
    diffusion_backward, diffusion_forward, DiffusionCache, DiffusionGrads, DiffusionWeights,
};
use crate::thermo::field::{free_channels_for, ThermoConfig};
use crate::thermo::flux::{flux_div_backward, flux_div_forward};

#[derive(Clone, Debug)]
pub struct StepWeights {
    pub conv: ConvWeights,
    pub diffusion: DiffusionWeights,
}

impl StepWeights {
    pub fn new<R: Rng>(cfg: &ThermoConfig, rng: &mut R) -> Self {
        Self {
            conv: ConvWeights::new(cfg, rng),
            diffusion: DiffusionWeights::new(cfg),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepGrads {
    pub conv: ConvGrads,
    pub diffusion: DiffusionGrads,
}

impl StepGrads {
    pub fn zeros(cfg: &ThermoConfig) -> Self {
        Self {
            conv: ConvGrads::zeros(cfg),
            diffusion: DiffusionGrads::zeros(cfg),
        }
    }

    pub fn zero(&mut self) {
        self.conv.zero();
        self.diffusion.zero();
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepCache {
    pub conv: ConvCache,
    pub diff: DiffusionCache,
    pub anneal: f32,
}

pub fn step_forward(
    s: &[f32],
    w: &StepWeights,
    anneal: f32,
    cfg: &ThermoConfig,
) -> (Vec<f32>, StepCache) {
    let (c, nc, c_out) = (cfg.c, cfg.n_conserved(), cfg.c_out());
    let nfree = c - nc;
    let cells = cfg.cells();

    let (raw, conv_cache) = conv_forward(&w.conv, s, cfg);

    // Decode conserved-channel edge fluxes from the conv output.
    let mut fx = vec![0.0_f32; cells * nc];
    let mut fy = vec![0.0_f32; cells * nc];
    for cell in 0..cells {
        let base = cell * c_out;
        for m in 0..nc {
            fx[cell * nc + m] = raw[base + nfree + m];
            fy[cell * nc + m] = raw[base + nfree + nc + m];
        }
    }
    let div = flux_div_forward(&fx, &fy, cfg);

    let (diff_delta, diff_cache) = if cfg.use_diffusion {
        diffusion_forward(s, &w.diffusion, cfg)
    } else {
        (Vec::new(), DiffusionCache::default())
    };

    let free_chs = free_channels_for(cfg);
    let dt = cfg.dt;
    let mut out = s.to_vec();
    for cell in 0..cells {
        let base = cell * c_out;
        for m in 0..nc {
            let ch = cfg.conserved[m];
            out[cell * c + ch] += dt * div[cell * nc + m];
        }
        for k in 0..nfree {
            out[cell * c + free_chs[k]] += dt * raw[base + k];
        }
    }
    if cfg.use_diffusion {
        for i in 0..out.len() {
            out[i] += dt * anneal * diff_delta[i];
        }
    }

    (out, StepCache { conv: conv_cache, diff: diff_cache, anneal })
}

pub fn step_backward(
    w: &StepWeights,
    cache: &StepCache,
    d_out: &[f32],
    g: &mut StepGrads,
    cfg: &ThermoConfig,
) -> Vec<f32> {
    let (c, nc, c_out) = (cfg.c, cfg.n_conserved(), cfg.c_out());
    let nfree = c - nc;
    let cells = cfg.cells();
    let dt = cfg.dt;
    let anneal = cache.anneal;
    let free_chs = free_channels_for(cfg);

    // Residual path: ∂(s + …)/∂s = identity.
    let mut d_s = d_out.to_vec();

    // Diffusion path (ablatable).
    if cfg.use_diffusion {
        let mut d_diff_delta = vec![0.0_f32; cfg.field_len()];
        for i in 0..d_diff_delta.len() {
            d_diff_delta[i] = dt * anneal * d_out[i];
        }
        let d_s_diff =
            diffusion_backward(&w.diffusion, &cache.diff, &d_diff_delta, &mut g.diffusion, cfg);
        for i in 0..d_s.len() {
            d_s[i] += d_s_diff[i];
        }
    }

    // Transport path → d_div (conserved) and free-delta slots of d_raw.
    let mut d_div = vec![0.0_f32; cells * nc];
    let mut d_raw = vec![0.0_f32; cells * c_out];
    for cell in 0..cells {
        let base = cell * c_out;
        for m in 0..nc {
            d_div[cell * nc + m] = dt * d_out[cell * c + cfg.conserved[m]];
        }
        for k in 0..nfree {
            d_raw[base + k] = dt * d_out[cell * c + free_chs[k]];
        }
    }
    let (d_fx, d_fy) = flux_div_backward(&d_div, cfg);
    for cell in 0..cells {
        let base = cell * c_out;
        for m in 0..nc {
            d_raw[base + nfree + m] = d_fx[cell * nc + m];
            d_raw[base + nfree + nc + m] = d_fy[cell * nc + m];
        }
    }
    let d_s_conv = conv_backward(&w.conv, &cache.conv, &d_raw, &mut g.conv, cfg);
    for i in 0..d_s.len() {
        d_s[i] += d_s_conv[i];
    }

    d_s
}
