#![allow(clippy::cast_precision_loss, clippy::must_use_candidate)]

//! Field layout, configuration, and initialization for the thermodynamic
//! LM. The state is an `H×W×C` grid stored row-major, **channel-last**:
//! a cell `(i,j)` owns a contiguous `[C]` feature vector at offset
//! `(i*W + j)*C`. This mirrors the transformer's `[T*D]` per-token row
//! layout, so the moment readout and per-cell projections reuse the same
//! `LinearProj` machinery.

/// Physical channel indices. Channels `[0, N_PHYS)` carry macroscopic
/// quantities; channels `[N_PHYS, c)` are free hidden state that gives the
/// learned operator extra capacity.
pub const CH_RHO: usize = 0;
pub const CH_UX: usize = 1;
pub const CH_UY: usize = 2;
pub const CH_E: usize = 3;
pub const CH_T: usize = 4;
pub const N_PHYS: usize = 5;

#[derive(Clone, Debug)]
pub struct ThermoConfig {
    pub vocab: usize,
    pub h: usize,
    pub w: usize,
    /// Total field channels (physical + hidden). Must be `>= N_PHYS`.
    pub c: usize,
    /// Hidden channel count of the local conv operator `N_θ`.
    pub conv_hidden: usize,
    /// Indices of structurally-conserved channels (transported by
    /// antisymmetric edge fluxes). Every other channel uses a free
    /// residual update. Each index must be `< c`.
    pub conserved: Vec<usize>,
    /// Sub-steps of the operator per token.
    pub k_steps: usize,
    /// Average-pooling factor for the readout. Must divide `h` and `w`.
    pub pool: usize,
    /// Update step size `Δt` in `s_{t+1} = s_t + Δt·N_θ(s_t)`.
    pub dt: f32,
    /// Warm-start temperature written into the `T` channel at init
    /// (the "start hot" of §4).
    pub t_init: f32,
    /// Per-token anneal schedule `T_max·(1 - k/K)^α` modulating injection
    /// and diffusion amplitude across the K sub-steps.
    pub anneal_tmax: f32,
    pub anneal_alpha: f32,
    /// Readout features: moment pooling (§3.1) and/or spectral DCT modes
    /// (§3.2). At least one must be true.
    pub readout_moments: bool,
    pub readout_spectral: bool,
    /// Per-axis low-frequency mode count for the spectral readout
    /// (`readout_modes²` modes per channel).
    pub readout_modes: usize,
    /// Diffusion stabilizer on/off (ablation §9 Fáze T1).
    pub use_diffusion: bool,
}

impl ThermoConfig {
    /// Tiny config for gradient checks — small grid so finite differencing
    /// (one forward per parameter) stays cheap.
    pub fn toy(vocab: usize) -> Self {
        Self {
            vocab,
            h: 4,
            w: 4,
            c: 6,
            conv_hidden: 6,
            conserved: vec![CH_RHO, CH_E],
            k_steps: 2,
            pool: 2,
            dt: 1.0,
            t_init: 1.0,
            anneal_tmax: 1.0,
            anneal_alpha: 1.0,
            readout_moments: true,
            readout_spectral: true,
            readout_modes: 2,
            use_diffusion: true,
        }
    }

    /// nano-scale 2D toy (the §6 Fáze 0 default: 32×32, ~8 channels).
    pub fn nano(vocab: usize) -> Self {
        Self {
            vocab,
            h: 32,
            w: 32,
            c: 8,
            conv_hidden: 16,
            conserved: vec![CH_RHO, CH_E],
            k_steps: 4,
            pool: 4,
            dt: 0.5,
            t_init: 1.0,
            anneal_tmax: 1.0,
            anneal_alpha: 1.0,
            readout_moments: true,
            readout_spectral: false,
            readout_modes: 4,
            use_diffusion: true,
        }
    }

    #[inline]
    pub fn cells(&self) -> usize {
        self.h * self.w
    }

    #[inline]
    pub fn field_len(&self) -> usize {
        self.h * self.w * self.c
    }

    #[inline]
    pub fn n_conserved(&self) -> usize {
        self.conserved.len()
    }

    /// Output-channel count of the local operator: one free delta per
    /// non-conserved channel, plus `Fx` and `Fy` per conserved channel.
    /// `(c - n_cons) + 2·n_cons = c + n_cons`.
    #[inline]
    pub fn c_out(&self) -> usize {
        self.c + self.conserved.len()
    }

    /// Pooled (moment) feature dimension.
    #[inline]
    pub fn pooled_dim(&self) -> usize {
        (self.h / self.pool) * (self.w / self.pool) * self.c
    }

    /// Spectral feature dimension: `readout_modes²` modes per channel.
    #[inline]
    pub fn spectral_dim(&self) -> usize {
        self.readout_modes * self.readout_modes * self.c
    }

    /// Total readout input dimension (moments + spectral, per flags).
    #[inline]
    pub fn readout_dim(&self) -> usize {
        let m = if self.readout_moments { self.pooled_dim() } else { 0 };
        let s = if self.readout_spectral { self.spectral_dim() } else { 0 };
        m + s
    }

    /// Per-channel decode maps for the GPU assemble/disassemble kernels:
    /// `kind[ch]` is 1 for conserved channels and 0 for free; `slot[ch]` is
    /// the channel's index within its kind (conserved → flux slot, free →
    /// free-delta slot). Single source of truth shared by both GPU
    /// orchestrators so their channel orderings match the CPU step.
    pub fn channel_decode(&self) -> (Vec<u32>, Vec<u32>) {
        let mut kind = vec![0u32; self.c];
        let mut slot = vec![0u32; self.c];
        for (m, &cc) in self.conserved.iter().enumerate() {
            kind[cc] = 1;
            slot[cc] = m as u32;
        }
        for (k, &cc) in free_channels_for(self).iter().enumerate() {
            slot[cc] = k as u32;
        }
        (kind, slot)
    }
}

/// Non-conserved (free) channels in increasing index order. Shared by the
/// CPU step and the GPU orchestrator so their channel decode/assemble
/// orderings match.
#[must_use]
pub fn free_channels_for(cfg: &ThermoConfig) -> Vec<usize> {
    (0..cfg.c).filter(|ch| !cfg.conserved.contains(ch)).collect()
}

/// Flat index of channel `c` at cell `(i,j)`, channel-last row-major.
#[inline]
pub fn idx(i: usize, j: usize, c: usize, cfg: &ThermoConfig) -> usize {
    (i * cfg.w + j) * cfg.c + c
}

/// Warm uniform initial field: unit density, `t_init` temperature, the
/// rest zero. Energy/momentum/hidden start empty and are driven entirely
/// by token injection + dynamics.
pub fn init_field(cfg: &ThermoConfig) -> Vec<f32> {
    let mut f = vec![0.0_f32; cfg.field_len()];
    for i in 0..cfg.h {
        for j in 0..cfg.w {
            f[idx(i, j, CH_RHO, cfg)] = 1.0;
            f[idx(i, j, CH_T, cfg)] = cfg.t_init;
        }
    }
    f
}
