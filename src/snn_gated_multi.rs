#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::if_not_else,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::assigning_clones,
    clippy::struct_field_names,
    clippy::too_long_first_doc_paragraph
)]

//! Multi-layer gated recurrent SNN (Sprint 23).
//!
//! Stacks `n_layers` gated LIF cells. Layer 0 consumes the byte input
//! (sparse one-hot column read on `W_in_0`); layer L > 0 consumes
//! layer L-1's spike output (multi-hot, sparse over the firing
//! neurons of layer L-1). A single output projection `W_out` maps the
//! top layer's spike output to logits.
//!
//! Everything else mirrors single-layer `GatedRecurrentSnn`: per-layer
//! gate (sigmoid forget gate), surrogate-gradient BPTT, Adam,
//! stateful TBPTT-with-detach via per-layer `GatedState`.
//!
//! GPU support is **not** implemented yet (Sprint 24+ work). CPU-only.

use rand::Rng;
use rayon::prelude::*;

use crate::linalg::{active_indices, matvec_transpose};
use crate::optimizer::{adam_apply, bias_correction, AdamConfig, AdamGroup};
use crate::snn_gated::GatedState;
use crate::snn_loss::cross_entropy_with_grad;
use crate::surrogate::SpikeFn;

/// Single layer's worth of weights (no output projection — only the
/// top layer routes through `W_out`).
#[derive(Clone, Debug)]
pub struct GatedLayer {
    pub n_in: usize,
    pub n_hidden: usize,
    pub w_in: Vec<f32>,       // [n_hidden, n_in]
    pub w_rec: Vec<f32>,      // [n_hidden, n_hidden]
    pub w_gate_in: Vec<f32>,  // [n_hidden, n_in]
    pub w_gate_rec: Vec<f32>, // [n_hidden, n_hidden]
    pub b_h: Vec<f32>,        // [n_hidden]
    pub b_gate: Vec<f32>,     // [n_hidden]
    /// Sprint 30: per-layer learnable surrogate-gradient width.
    /// Replaces the global `SpikeFn::alpha` in the backward pass so each
    /// layer can adapt its surrogate to its own membrane-potential
    /// distribution (Lian et al., IJCAI 2023). Forward (heaviside) is
    /// unaffected — this only enters the backward via
    /// `surrogate(v) = 1 / (1 + α|v−θ|)²`.
    pub alpha: f32,
}

const DEFAULT_ALPHA: f32 = 4.0;
/// Sprint 37: hard cap on membrane potential to prevent runaway
/// integration. Observed peaks of 10⁷ in overnight enwik8_10m
/// destroyed f32 precision (gradient noise floor at ~1.0 is bigger
/// than θ=1.0 itself). Clamping to ±10 covers any reasonable spike
/// dynamics — neurons that try to sit ten thresholds above firing
/// simply saturate, lose nothing useful.
pub const MEMBRANE_CLAMP: f32 = 10.0;
/// Sprint 40 AdLIF time constant. `w_t = ρ·w_{t-1} + b·spike_t`.
/// `ρ=0.95` ≈ τ ≈ 20 timesteps half-life — slow accumulator.
pub const ADLIF_RHO: f32 = 0.95;
/// Sprint 40 AdLIF spike-driven gain. Each spike adds `b` to `w`,
/// which then suppresses subsequent membrane by `−w`. b=0.3 was first
/// guess; killed firing entirely (w_eq grows to 3.0 at rate=0.5 →
/// membrane equilibrium drops to −5.0). b=0.1 is gentler — at
/// rate=0.5, w_eq=1.0 → membrane shifts down by 1, still fires.
pub const ADLIF_B: f32 = 0.1;
/// Sprint 31 / Sprint 37 retuning. With `b_gate=0` (β=0.5, not 0.73),
/// equilibrium membrane is `b_h / (1−β) = 2·b_h`. Setting `b_h=0.5`
/// puts `v_eq` at θ=1.0 — same firing onset as Sprint 31 but with
/// **half the recurrent carry**, dramatically reducing runaway risk.
const DEFAULT_B_H_INIT: f32 = 0.5;
/// Sprint 37: forget-gate-bias-zero. Old (b_gate=1) gave β≈0.73, which
/// after Sprint 31's wake-up turned into runaway integration —
/// observed membrane peaks of 10⁷ in overnight enwik8_10m destroyed
/// f32 precision. β=0.5 means each timestep keeps half of the previous
/// membrane and integrates the new input — much more aggressive leak.
const DEFAULT_B_GATE_INIT: f32 = 0.0;

impl GatedLayer {
    pub fn new<R: Rng>(n_in: usize, n_hidden: usize, rng: &mut R) -> Self {
        // He-style for `W_in`; aggressively contractive for `W_rec`.
        // The old `0.9/√H` scale aimed at "spectral radius ~1" but in
        // practice trained weights routinely amplified the recurrent
        // path past stability — combined with β=0.73 carry, membrane
        // hit 10⁷ in 1 epoch on enwik8_10m. 0.3/√H drops the working
        // spectral radius to ~0.17 (uniform-init effective). Much
        // more boring dynamics; can be tuned up if convergence stalls.
        let scale_in = (2.0 / n_in as f32).sqrt();
        let scale_rec = 0.3 / (n_hidden as f32).sqrt();
        let scale_gate_in = 0.1 * scale_in;
        let scale_gate_rec = 0.1 * scale_rec;

        let w_in = (0..n_hidden * n_in)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_in)
            .collect();
        let w_rec = (0..n_hidden * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_rec)
            .collect();
        let w_gate_in = (0..n_hidden * n_in)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_gate_in)
            .collect();
        let w_gate_rec = (0..n_hidden * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_gate_rec)
            .collect();

        Self {
            n_in,
            n_hidden,
            w_in,
            w_rec,
            w_gate_in,
            w_gate_rec,
            b_h: vec![DEFAULT_B_H_INIT; n_hidden],
            b_gate: vec![DEFAULT_B_GATE_INIT; n_hidden],
            alpha: DEFAULT_ALPHA,
        }
    }
}

/// Per-layer gradients.
#[derive(Clone, Debug)]
pub struct LayerGrads {
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_gate_in: Vec<f32>,
    pub w_gate_rec: Vec<f32>,
    pub b_h: Vec<f32>,
    pub b_gate: Vec<f32>,
    /// Sprint 30 LSG: scalar gradient for the per-layer surrogate width α.
    pub d_alpha: f32,
}

impl LayerGrads {
    pub fn zeros(n_in: usize, n_hidden: usize) -> Self {
        Self {
            w_in: vec![0.0; n_hidden * n_in],
            w_rec: vec![0.0; n_hidden * n_hidden],
            w_gate_in: vec![0.0; n_hidden * n_in],
            w_gate_rec: vec![0.0; n_hidden * n_hidden],
            b_h: vec![0.0; n_hidden],
            b_gate: vec![0.0; n_hidden],
            d_alpha: 0.0,
        }
    }

    pub fn zero(&mut self) {
        self.w_in.fill(0.0);
        self.w_rec.fill(0.0);
        self.w_gate_in.fill(0.0);
        self.w_gate_rec.fill(0.0);
        self.b_h.fill(0.0);
        self.b_gate.fill(0.0);
        self.d_alpha = 0.0;
    }

    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.w_in.iter_mut().zip(&other.w_in) {
            *a += b;
        }
        for (a, b) in self.w_rec.iter_mut().zip(&other.w_rec) {
            *a += b;
        }
        for (a, b) in self.w_gate_in.iter_mut().zip(&other.w_gate_in) {
            *a += b;
        }
        for (a, b) in self.w_gate_rec.iter_mut().zip(&other.w_gate_rec) {
            *a += b;
        }
        for (a, b) in self.b_h.iter_mut().zip(&other.b_h) {
            *a += b;
        }
        for (a, b) in self.b_gate.iter_mut().zip(&other.b_gate) {
            *a += b;
        }
        self.d_alpha += other.d_alpha;
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &f32> {
        self.w_in
            .iter()
            .chain(&self.w_rec)
            .chain(&self.w_gate_in)
            .chain(&self.w_gate_rec)
            .chain(&self.b_h)
            .chain(&self.b_gate)
            .chain(std::iter::once(&self.d_alpha))
    }

    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.w_in
            .iter_mut()
            .chain(&mut self.w_rec)
            .chain(&mut self.w_gate_in)
            .chain(&mut self.w_gate_rec)
            .chain(&mut self.b_h)
            .chain(&mut self.b_gate)
            .chain(std::iter::once(&mut self.d_alpha))
    }
}

/// Full network gradients: per-layer + top-level output projection.
#[derive(Clone, Debug, Default)]
pub struct MultiLayerGatedGrads {
    pub layers: Vec<LayerGrads>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
    /// Sprint 38: SDSA gradient bundle. `Some` when network has SDSA
    /// enabled — populated by `backward_lm_tokens_into`, consumed by
    /// `step_adam`. Parallel batch trainers each carry their own.
    pub sdsa: Option<crate::sdsa::SdsaGrads>,
}

impl MultiLayerGatedGrads {
    pub fn zeros(layer_dims: &[usize], n_output: usize) -> Self {
        // layer_dims = [n_input, hidden_0, hidden_1, ..., hidden_{L-1}]
        let n_layers = layer_dims.len() - 1;
        let layers = (0..n_layers)
            .map(|l| LayerGrads::zeros(layer_dims[l], layer_dims[l + 1]))
            .collect();
        let top_hidden = *layer_dims.last().unwrap();
        Self {
            layers,
            w_out: vec![0.0; n_output * top_hidden],
            b_o: vec![0.0; n_output],
            sdsa: None,
        }
    }

    pub fn zero(&mut self) {
        for l in &mut self.layers {
            l.zero();
        }
        self.w_out.fill(0.0);
        self.b_o.fill(0.0);
        if let Some(g) = self.sdsa.as_mut() {
            g.zero();
        }
    }

    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.layers.iter_mut().zip(&other.layers) {
            a.add(b);
        }
        for (a, b) in self.w_out.iter_mut().zip(&other.w_out) {
            *a += b;
        }
        for (a, b) in self.b_o.iter_mut().zip(&other.b_o) {
            *a += b;
        }
        if let (Some(a), Some(b)) = (self.sdsa.as_mut(), other.sdsa.as_ref()) {
            a.add(b);
        }
    }

    pub fn l2_norm(&self) -> f32 {
        let mut sq = 0.0_f32;
        for l in &self.layers {
            for v in l.iter_all() {
                sq += v * v;
            }
        }
        for v in self.w_out.iter().chain(&self.b_o) {
            sq += v * v;
        }
        if let Some(g) = self.sdsa.as_ref() {
            for v in g.iter_all() {
                sq += v * v;
            }
        }
        sq.sqrt()
    }

    pub fn scale(&mut self, factor: f32) {
        for l in &mut self.layers {
            for v in l.iter_all_mut() {
                *v *= factor;
            }
        }
        for v in self.w_out.iter_mut().chain(&mut self.b_o) {
            *v *= factor;
        }
        if let Some(g) = self.sdsa.as_mut() {
            for v in g.iter_all_mut() {
                *v *= factor;
            }
        }
    }
}

/// Per-layer cache (per-timestep) for one chunk's forward pass.
#[derive(Clone, Debug)]
pub struct LayerCache {
    pub h_v_pre: Vec<Vec<f32>>,
    pub h_v_after: Vec<Vec<f32>>,
    pub h_spike: Vec<Vec<f32>>,
    pub gate: Vec<Vec<f32>>,
    /// Active spike indices per timestep — derived from `h_spike`,
    /// memoized at forward time so backward (and downstream sparse
    /// matvecs) can skip O(n_h) re-scans.
    pub h_active: Vec<Vec<usize>>,
    pub initial_h_v_after: Vec<f32>,
    pub initial_h_spike: Vec<f32>,
    pub initial_h_active: Vec<usize>,
}

impl LayerCache {
    pub fn with_capacity(t_steps: usize) -> Self {
        Self {
            h_v_pre: Vec::with_capacity(t_steps),
            h_v_after: Vec::with_capacity(t_steps),
            h_spike: Vec::with_capacity(t_steps),
            gate: Vec::with_capacity(t_steps),
            h_active: Vec::with_capacity(t_steps),
            initial_h_v_after: Vec::new(),
            initial_h_spike: Vec::new(),
            initial_h_active: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiLayerCache {
    pub layers: Vec<LayerCache>,
    pub logits: Vec<Vec<f32>>,
    /// Sprint 38: SDSA forward cache + flat residual `y = h_spike +
    /// sdsa_out` over the chunk's top spikes. Both `Some` together when
    /// the network has SDSA enabled.
    pub sdsa: Option<crate::sdsa::SdsaCache>,
    pub sdsa_y: Option<Vec<f32>>,
}

impl MultiLayerCache {
    /// Sprint 30: per-layer mean firing rate over the cached forward
    /// pass. Useful as a sanity probe — too low (≪ 0.05) means the
    /// network is dead, too high (≫ 0.4) means the active-indices
    /// compaction loses its sparsity advantage. Target band ~0.05–0.20.
    #[must_use]
    pub fn firing_rates(&self) -> Vec<f32> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.h_spike.is_empty() {
                    return 0.0;
                }
                let n_h = layer.h_spike[0].len();
                let total = (layer.h_spike.len() * n_h) as f32;
                let count: f32 = layer.h_spike.iter().flatten().copied().sum();
                if total > 0.0 { count / total } else { 0.0 }
            })
            .collect()
    }

    /// Sprint 42: per-layer per-neuron mean firing rate over the
    /// cached forward pass. Used by the apoptosis EMA. Returns one
    /// `Vec<f32>` of length `n_h` per layer.
    #[must_use]
    pub fn per_neuron_rates(&self) -> Vec<Vec<f32>> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.h_spike.is_empty() {
                    return Vec::new();
                }
                let n_h = layer.h_spike[0].len();
                let mut rate = vec![0.0_f32; n_h];
                for row in &layer.h_spike {
                    for i in 0..n_h {
                        rate[i] += row[i];
                    }
                }
                let inv_t = 1.0 / layer.h_spike.len() as f32;
                for r in &mut rate {
                    *r *= inv_t;
                }
                rate
            })
            .collect()
    }

    /// Sprint 31: per-layer (mean, max) of `h_v_pre`. Compared to
    /// threshold θ, this tells you whether the membrane potential is
    /// even reaching firing range — a dead layer fires 0 not because
    /// the surrogate is broken but because v stays far below θ.
    #[must_use]
    pub fn membrane_stats(&self) -> Vec<(f32, f32)> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.h_v_pre.is_empty() {
                    return (0.0, 0.0);
                }
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                let mut peak = f32::NEG_INFINITY;
                for row in &layer.h_v_pre {
                    for &v in row {
                        sum += v;
                        if v > peak {
                            peak = v;
                        }
                        count += 1;
                    }
                }
                let mean = if count > 0 { sum / count as f32 } else { 0.0 };
                (mean, peak)
            })
            .collect()
    }
}

/// `n_layers` per-layer Adam state pairs (m1, m2) for each of 6
/// parameter groups, plus top-level w_out + b_o. Sprint 30 adds a
/// 1-element group for the per-layer surrogate width α (LSG).
#[derive(Clone, Debug)]
struct LayerAdam {
    m_w_in: AdamGroup,
    m_w_rec: AdamGroup,
    m_w_gate_in: AdamGroup,
    m_w_gate_rec: AdamGroup,
    m_b_h: AdamGroup,
    m_b_gate: AdamGroup,
    m_alpha: AdamGroup,
}

impl LayerAdam {
    fn zeros(n_in: usize, n_hidden: usize) -> Self {
        Self {
            m_w_in: AdamGroup::zeros(n_hidden * n_in),
            m_w_rec: AdamGroup::zeros(n_hidden * n_hidden),
            m_w_gate_in: AdamGroup::zeros(n_hidden * n_in),
            m_w_gate_rec: AdamGroup::zeros(n_hidden * n_hidden),
            m_b_h: AdamGroup::zeros(n_hidden),
            m_b_gate: AdamGroup::zeros(n_hidden),
            m_alpha: AdamGroup::zeros(1),
        }
    }
}

/// Sprint 32: spike-rate regularization. Adds a constant-per-layer
/// term to the `b_h` gradient proportional to `(rate − target)`, which
/// pushes the bias down when the layer over-fires (runaway recurrent
/// integration) and up when it under-fires. Decoupled from the
/// surrogate path so the regularization signal is always non-zero,
/// even when the surrogate has gone flat at saturation.
#[derive(Clone, Copy, Debug)]
pub struct RateReg {
    pub target: f32,
    pub lambda: f32,
}

impl Default for RateReg {
    fn default() -> Self {
        // target=0.1 keeps the active-indices compaction sparse.
        // Sprint 33 per-neuron formulation gives Adam a per-parameter
        // signal that *isn't* cancelled by moment normalization, so
        // even at λ=0.01 a measurable shift in the rate distribution
        // accumulates over a typical epoch (2k+ chunks).
        // Stronger λ (0.03–0.05) shifts rates faster but kills firing
        // on the short-corpus CPU regression path; the right value
        // for production is corpus-dependent and exposed on the
        // network for callers to override.
        Self {
            target: 0.1,
            lambda: 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLayerGatedSnn {
    pub n_input: usize,
    pub n_output: usize,
    pub layers: Vec<GatedLayer>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
    pub spike_fn: SpikeFn,
    pub rate_reg: RateReg,
    /// Sprint 38: optional Spike-Driven Self-Attention block applied
    /// after the top SNN layer, before the W_out projection. When
    /// `None`, behavior is identical to pre-SDSA Aurora (sparse W_out
    /// matvec from spike output). When `Some`, residual `y = h_spike +
    /// sdsa_out` feeds a dense W_out matvec.
    pub sdsa: Option<crate::sdsa::SdsaCell>,
    sdsa_adam: Option<crate::sdsa::SdsaAdam>,
    /// Sprint 40: AdLIF adaptation. When `true`, every layer carries
    /// a per-neuron `w` adaptation current that subtracts from `v_pre`
    /// and updates `w ← ρ·w + b·spike` each timestep. Fixed scalars
    /// for now (no Adam coupling) — make learnable in Sprint 41 if
    /// the A/B test on small CPU run shows quality improvement.
    pub adlif_enabled: bool,
    /// Sprint 42: per-neuron activity-dependent pruning. When `Some`,
    /// neurons with `dead_mask[l][i] == true` produce zero spike in
    /// forward and receive zero gradient in backward — biological
    /// "trophic withdrawal" applied to the SNN.
    pub apoptosis: Option<crate::apoptosis::ApoptosisState>,
    grads: MultiLayerGatedGrads,
    adam_t: usize,
    layer_adam: Vec<LayerAdam>,
    adam_w_out: AdamGroup,
    adam_b_o: AdamGroup,
}

impl MultiLayerGatedSnn {
    /// `hidden_dims` is one entry per layer: e.g. `&[256, 128]` for a
    /// two-layer network whose layer 0 has hidden 256 and layer 1 has
    /// hidden 128. Layer 0's input is `n_input`-wide (byte one-hot);
    /// layer L>0's input is the previous layer's hidden.
    pub fn new<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        rng: &mut R,
    ) -> Self {
        assert!(!hidden_dims.is_empty(), "at least one layer required");
        let mut layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = n_input;
        for &h in hidden_dims {
            layers.push(GatedLayer::new(prev_dim, h, rng));
            prev_dim = h;
        }
        let top_hidden = prev_dim;
        let scale_out = (2.0 / top_hidden as f32).sqrt();
        let w_out = (0..n_output * top_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_out)
            .collect();

        let mut layer_dims = vec![n_input];
        layer_dims.extend_from_slice(hidden_dims);
        let layer_adam = (0..hidden_dims.len())
            .map(|l| LayerAdam::zeros(layer_dims[l], layer_dims[l + 1]))
            .collect();

        Self {
            n_input,
            n_output,
            grads: MultiLayerGatedGrads::zeros(&layer_dims, n_output),
            layers,
            w_out,
            b_o: vec![0.0; n_output],
            spike_fn,
            rate_reg: RateReg::default(),
            sdsa: None,
            sdsa_adam: None,
            adlif_enabled: false,
            apoptosis: None,
            adam_t: 0,
            layer_adam,
            adam_w_out: AdamGroup::zeros(n_output * top_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
        }
    }

    /// Sprint 38: turn on Spike-Driven Self-Attention. Allocates
    /// `SdsaCell` (LayerNorm + Q/K/V projections, dim = top hidden)
    /// + grads + Adam state. Reuses the network's RNG seed so two
    /// calls with the same seed produce identical SDSA params.
    pub fn enable_sdsa<R: Rng>(&mut self, rng: &mut R) {
        let n = self.layers.last().expect("≥1 layer").n_hidden;
        self.sdsa = Some(crate::sdsa::SdsaCell::new(n, rng));
        self.grads.sdsa = Some(crate::sdsa::SdsaGrads::zeros(n));
        self.sdsa_adam = Some(crate::sdsa::SdsaAdam::zeros(n));
    }

    /// Sprint 40: turn on AdLIF adaptation current with default
    /// hardcoded `ρ=0.95`, `b=0.3`. Per-neuron `w` carries between
    /// chunks via `GatedState::adlif_w`. No Adam state — params are
    /// constants this sprint; Sprint 41 makes them learnable if the
    /// CPU A/B shows quality improvement.
    /// Sprint 43: per-neuron trophic signal — `|∂L/∂W_outgoing[*, i]|_1`
    /// — measured on the network's current `grads` buffer. Call **after**
    /// backward but **before** `clip_grads` / `step_adam` so the raw
    /// magnitudes are preserved.
    ///
    /// For neuron `i` in layer `l`:
    /// - hidden (l < top): sum of `|grads.layers[l+1].w_in[*, i]|`
    ///   (column i of the next layer's `W_in`)
    /// - top: sum of `|grads.w_out[*, i]|` (column i of `W_out`)
    ///
    /// This is the closest computational analogue to a target-derived
    /// trophic factor: it measures how strongly the loss "wants" to
    /// modify the connections that consume neuron `i`'s output. Useless
    /// neurons receive zero gradient; load-bearing ones receive a lot.
    #[must_use]
    pub fn trophic_signal_from_grads(&self) -> Vec<Vec<f32>> {
        let n_layers = self.layers.len();
        let mut out: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| vec![0.0_f32; l.n_hidden])
            .collect();
        for l in 0..n_layers {
            let n_h = self.layers[l].n_hidden;
            if l + 1 < n_layers {
                let n_h_next = self.layers[l + 1].n_hidden;
                let g = &self.grads.layers[l + 1].w_in;
                for h in 0..n_h_next {
                    let row = h * n_h;
                    for i in 0..n_h {
                        out[l][i] += g[row + i].abs();
                    }
                }
            } else {
                let n_out = self.n_output;
                let g = &self.grads.w_out;
                for o in 0..n_out {
                    let row = o * n_h;
                    for i in 0..n_h {
                        out[l][i] += g[row + i].abs();
                    }
                }
            }
        }
        out
    }

    /// Sprint 42: turn on activity-dependent neuron pruning. The
    /// network starts with all neurons alive; the apoptosis state's
    /// EMA must be observed by the trainer (see `observe_rates`) and
    /// `maybe_kill` called per chunk for the mechanism to actually
    /// prune anything.
    pub fn enable_apoptosis(&mut self, config: crate::apoptosis::ApoptosisConfig) {
        let layer_sizes: Vec<usize> = self.layers.iter().map(|l| l.n_hidden).collect();
        self.apoptosis = Some(crate::apoptosis::ApoptosisState::new(config, &layer_sizes));
    }

    pub fn enable_adlif(&mut self) {
        self.adlif_enabled = true;
    }

    pub fn fresh_states(&self) -> Vec<GatedState> {
        let n_layers = self.layers.len();
        self.layers
            .iter()
            .enumerate()
            .map(|(idx, l)| {
                let attn_state = if self.sdsa.is_some() && idx + 1 == n_layers {
                    vec![0.0_f32; l.n_hidden]
                } else {
                    Vec::new()
                };
                let adlif_w = if self.adlif_enabled {
                    vec![0.0_f32; l.n_hidden]
                } else {
                    Vec::new()
                };
                GatedState {
                    h_v_after: vec![0.0_f32; l.n_hidden],
                    h_spike: vec![0.0_f32; l.n_hidden],
                    attn_state,
                    adlif_w,
                }
            })
            .collect()
    }

    pub const fn grads(&self) -> &MultiLayerGatedGrads {
        &self.grads
    }

    pub fn alloc_grads(&self) -> MultiLayerGatedGrads {
        let mut layer_dims = vec![self.n_input];
        for l in &self.layers {
            layer_dims.push(l.n_hidden);
        }
        let mut g = MultiLayerGatedGrads::zeros(&layer_dims, self.n_output);
        if self.sdsa.is_some() {
            let n = self.layers.last().expect("≥1 layer").n_hidden;
            g.sdsa = Some(crate::sdsa::SdsaGrads::zeros(n));
        }
        g
    }

    pub fn zero_grad(&mut self) {
        self.grads.zero();
    }

    pub fn clip_grads(&mut self, max_norm: f32) {
        let norm = self.grads.l2_norm();
        if norm <= max_norm || norm == 0.0 {
            return;
        }
        self.grads.scale(max_norm / norm);
    }

    pub fn add_grads(&mut self, other: &MultiLayerGatedGrads) {
        self.grads.add(other);
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let g = &self.grads.layers[l];
            let st = &mut self.layer_adam[l];
            adam_apply(&mut layer.w_in, &g.w_in, &mut st.m_w_in, config, bc1, bc2);
            adam_apply(&mut layer.w_rec, &g.w_rec, &mut st.m_w_rec, config, bc1, bc2);
            adam_apply(
                &mut layer.w_gate_in,
                &g.w_gate_in,
                &mut st.m_w_gate_in,
                config,
                bc1,
                bc2,
            );
            adam_apply(
                &mut layer.w_gate_rec,
                &g.w_gate_rec,
                &mut st.m_w_gate_rec,
                config,
                bc1,
                bc2,
            );
            adam_apply(&mut layer.b_h, &g.b_h, &mut st.m_b_h, config, bc1, bc2);
            adam_apply(
                &mut layer.b_gate,
                &g.b_gate,
                &mut st.m_b_gate,
                config,
                bc1,
                bc2,
            );
            // Sprint 30 LSG: Adam step on per-layer surrogate width α.
            // Treat α as a 1-element parameter vector. softplus
            // reparametrization is unnecessary at this scale; a hard
            // floor in `forward_lm_tokens` would be fine but in
            // practice Adam keeps α positive on its own with default
            // hyperparameters.
            let mut alpha_buf = [layer.alpha];
            let alpha_grad = [g.d_alpha];
            adam_apply(
                &mut alpha_buf,
                &alpha_grad,
                &mut st.m_alpha,
                config,
                bc1,
                bc2,
            );
            // Floor at 0.1 — surrogate becomes ~step function as α→∞,
            // ~constant as α→0. Stay in the useful range.
            layer.alpha = alpha_buf[0].max(0.1);
        }
        adam_apply(
            &mut self.w_out,
            &self.grads.w_out,
            &mut self.adam_w_out,
            config,
            bc1,
            bc2,
        );
        adam_apply(
            &mut self.b_o,
            &self.grads.b_o,
            &mut self.adam_b_o,
            config,
            bc1,
            bc2,
        );
        // Sprint 38: step the SDSA params if enabled.
        if let (Some(cell), Some(g), Some(adam)) = (
            self.sdsa.as_mut(),
            self.grads.sdsa.as_ref(),
            self.sdsa_adam.as_mut(),
        ) {
            crate::sdsa::sdsa_step_adam(cell, g, adam, config, bc1, bc2);
        }
    }

    /// Stateful sparse LM forward over arbitrary token IDs (u32).
    /// Sprint 26: takes token indices into a vocab of size
    /// `self.n_input` — works for byte-level (n_input=256), BPE
    /// (n_input=4096), or any other vocabulary.
    pub fn forward_lm_tokens_stateful(
        &self,
        input_tokens: &[u32],
        states: &mut [GatedState],
    ) -> MultiLayerCache {
        assert_eq!(states.len(), self.layers.len());
        let t_steps = input_tokens.len();
        let n_layers = self.layers.len();
        let mut cache = MultiLayerCache {
            layers: (0..n_layers).map(|_| LayerCache::with_capacity(t_steps)).collect(),
            logits: Vec::with_capacity(t_steps),
            sdsa: None,
            sdsa_y: None,
        };
        for (l, st) in states.iter().enumerate() {
            cache.layers[l].initial_h_active = active_indices(&st.h_spike);
            cache.layers[l].initial_h_v_after = st.h_v_after.clone();
            cache.layers[l].initial_h_spike = st.h_spike.clone();
        }

        let theta = self.spike_fn.threshold;
        let top_hidden = self.layers.last().unwrap().n_hidden;

        for &tok in input_tokens {
            let b = tok as usize;
            debug_assert!(b < self.n_input);

            // Active indices INTO layer l (i.e., for layer 0 it's
            // `[byte]`; for L>0 it's the previous layer's spike active
            // set).
            let mut layer_input_active: Vec<usize> = vec![b];

            for l in 0..n_layers {
                let layer = &self.layers[l];
                let n_h = layer.n_hidden;
                let n_in = layer.n_in;
                let state = &states[l];
                let active_prev = active_indices(&state.h_spike);

                let mut gate_pre = vec![0.0_f32; n_h];
                for hi in 0..n_h {
                    let row_offset_rec = hi * n_h;
                    let row_offset_in = hi * n_in;
                    let mut acc = layer.b_gate[hi];
                    for &c in &layer_input_active {
                        acc += layer.w_gate_in[row_offset_in + c];
                    }
                    for &c in &active_prev {
                        acc += layer.w_gate_rec[row_offset_rec + c];
                    }
                    gate_pre[hi] = acc;
                }
                let gate: Vec<f32> = gate_pre.iter().map(|&v| sigmoid(v)).collect();

                let mut h_v_pre_t = vec![0.0_f32; n_h];
                for hi in 0..n_h {
                    let row_offset_rec = hi * n_h;
                    let row_offset_in = hi * n_in;
                    let mut h_in_val = 0.0_f32;
                    for &c in &layer_input_active {
                        h_in_val += layer.w_in[row_offset_in + c];
                    }
                    let mut h_rec_val = 0.0_f32;
                    for &c in &active_prev {
                        h_rec_val += layer.w_rec[row_offset_rec + c];
                    }
                    let mut acc = state.h_v_after[hi] * gate[hi]
                        + h_in_val
                        + h_rec_val
                        + layer.b_h[hi];
                    if !state.adlif_w.is_empty() {
                        acc -= state.adlif_w[hi];
                    }
                    h_v_pre_t[hi] = acc.clamp(-MEMBRANE_CLAMP, MEMBRANE_CLAMP);
                }
                // Fused spike / h_v_after / active-set scan: three
                // O(n_h) passes (`map → collect`, `zip-map → collect`,
                // `active_indices`) compressed into one tight loop.
                // Sprint 42: dead neurons (apoptosis) have spike forced
                // to 0 — their membrane still integrates but no output
                // reaches downstream layers, output projection, or rec.
                let dead_l: Option<&[bool]> = self
                    .apoptosis
                    .as_ref()
                    .map(|a| a.dead_mask[l].as_slice());
                let mut h_spike_t: Vec<f32> = Vec::with_capacity(n_h);
                let mut h_v_after_t: Vec<f32> = Vec::with_capacity(n_h);
                let mut active_t: Vec<usize> = Vec::new();
                for (hi, &v) in h_v_pre_t.iter().enumerate() {
                    let mut s = self.spike_fn.forward(v);
                    if let Some(d) = dead_l {
                        if d[hi] {
                            s = 0.0;
                        }
                    }
                    h_spike_t.push(s);
                    h_v_after_t.push(v - theta * s);
                    if s != 0.0 {
                        active_t.push(hi);
                    }
                }

                // Sprint 40 AdLIF update: w_t = ρ·w_{t-1} + b·spike_t.
                if !states[l].adlif_w.is_empty() {
                    for hi in 0..n_h {
                        states[l].adlif_w[hi] =
                            ADLIF_RHO * states[l].adlif_w[hi] + ADLIF_B * h_spike_t[hi];
                    }
                }

                // Update state via clone_from (reuses Vec capacity, avoids
                // an allocation per (layer, step)) before moving the
                // originals into the cache.
                states[l].h_v_after.clone_from(&h_v_after_t);
                states[l].h_spike.clone_from(&h_spike_t);

                cache.layers[l].h_v_pre.push(h_v_pre_t);
                cache.layers[l].h_v_after.push(h_v_after_t);
                cache.layers[l].h_spike.push(h_spike_t);
                cache.layers[l].gate.push(gate);
                cache.layers[l].h_active.push(active_t.clone());

                layer_input_active = active_t;
            }

            // Output projection from top layer's spike output.
            // Sprint 38: when SDSA is enabled, defer logits to a
            // post-loop pass over the residual `y = h_spike + sdsa_out`.
            if self.sdsa.is_none() {
                let top_active = layer_input_active; // == active indices of top spike
                let mut logits = Vec::with_capacity(self.n_output);
                for o in 0..self.n_output {
                    let row_offset = o * top_hidden;
                    let mut acc = self.b_o[o];
                    for &c in &top_active {
                        acc += self.w_out[row_offset + c];
                    }
                    logits.push(acc);
                }
                cache.logits.push(logits);
            }
        }

        // Sprint 38 SDSA pass: collect top-layer h_spike across all
        // timesteps, run linear-attention forward, then dense W_out
        // matvec on the residual `y = h_spike + sdsa_out`.
        if let Some(sdsa) = &self.sdsa {
            let top = n_layers - 1;
            let mut h_top_flat: Vec<f32> = Vec::with_capacity(t_steps * top_hidden);
            for row in &cache.layers[top].h_spike {
                h_top_flat.extend_from_slice(row);
            }
            let initial_attn = states[top].attn_state.clone();
            let (y_flat, sdsa_cache) =
                crate::sdsa::sdsa_forward(sdsa, &h_top_flat, &initial_attn);
            states[top].attn_state.clone_from(&sdsa_cache.final_attn_state);

            for t in 0..t_steps {
                let y_t = &y_flat[t * top_hidden..(t + 1) * top_hidden];
                let mut logits = Vec::with_capacity(self.n_output);
                for o in 0..self.n_output {
                    let row_offset = o * top_hidden;
                    let mut acc = self.b_o[o];
                    for h in 0..top_hidden {
                        acc += self.w_out[row_offset + h] * y_t[h];
                    }
                    logits.push(acc);
                }
                cache.logits.push(logits);
            }
            cache.sdsa = Some(sdsa_cache);
            cache.sdsa_y = Some(y_flat);
        }

        cache
    }

    pub fn forward_lm_bytes(&self, input_bytes: &[u8]) -> MultiLayerCache {
        let mut states = self.fresh_states();
        self.forward_lm_bytes_stateful(input_bytes, &mut states)
    }

    pub fn forward_lm_tokens(&self, input_tokens: &[u32]) -> MultiLayerCache {
        let mut states = self.fresh_states();
        self.forward_lm_tokens_stateful(input_tokens, &mut states)
    }

    /// Byte-level wrapper around `forward_lm_tokens_stateful` —
    /// converts `&[u8]` to `Vec<u32>`. Aurora's existing byte-level
    /// training code can keep calling this without changes.
    pub fn forward_lm_bytes_stateful(
        &self,
        input_bytes: &[u8],
        states: &mut [GatedState],
    ) -> MultiLayerCache {
        let tokens: Vec<u32> = input_bytes.iter().map(|&b| u32::from(b)).collect();
        self.forward_lm_tokens_stateful(&tokens, states)
    }

    /// Backward pass over the full chunk, accumulating into `grads`.
    /// Returns total cross-entropy loss (sum over all timesteps).
    /// Sprint 26: takes token IDs (u32). Byte-level wrapper below.
    pub fn backward_lm_tokens_into(
        &self,
        cache: &MultiLayerCache,
        input_tokens: &[u32],
        targets: &[u32],
        grads: &mut MultiLayerGatedGrads,
    ) -> f32 {
        let t_steps = input_tokens.len();
        debug_assert_eq!(targets.len(), t_steps);
        let n_layers = self.layers.len();
        let top_hidden = self.layers.last().unwrap().n_hidden;

        // Sprint 38: when SDSA is on, the W_out matvec ate the dense
        // residual `y` (not sparse `h_spike`), so its gradient is a
        // dense outer product. We pre-compute `d_y[t]` (and accumulate
        // W_out + b_o grads), run SDSA backward over all timesteps in
        // one shot to get the augmented `d_h_spike` for the top layer,
        // and feed that back into the per-t reverse loop in place of
        // the sparse `matvec_transpose(W_out, d_logits)` route.
        let (d_h_top_pre, sdsa_loss) = if self.sdsa.is_some() {
            self.backward_top_through_sdsa(cache, targets, grads)
        } else {
            (None, 0.0)
        };

        // Per-layer carry buffers (carry from t+1 back into t).
        let mut d_h_v_after_carry: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| vec![0.0_f32; l.n_hidden])
            .collect();
        let mut d_h_spike_recurrent: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| vec![0.0_f32; l.n_hidden])
            .collect();

        // Sprint 40: AdLIF gradient carry. Per-layer running gradient
        // on the adaptation current `w`, propagated backward in time:
        //   dw[t-1] = ρ·dw[t] − d_v_pre[t]
        // Plus per-step contribution to d_h_spike via the spike update:
        //   d_h_spike[t] += b · dw[t]
        let adlif_active = self.adlif_enabled;
        let mut d_w_carry: Vec<Vec<f32>> = if adlif_active {
            self.layers
                .iter()
                .map(|l| vec![0.0_f32; l.n_hidden])
                .collect()
        } else {
            Vec::new()
        };

        let mut total_loss = 0.0_f32;

        let top_n_h = self.layers.last().map(|l| l.n_hidden).unwrap_or(0);
        for t in (0..t_steps).rev() {
            let d_h_top_spike_from_out = if let Some(d_flat) = d_h_top_pre.as_ref() {
                // SDSA path: loss + W_out grad already accounted for in
                // `backward_top_through_sdsa`. d_h is the SDSA backward
                // result with residual contribution included. Slice the
                // flat buffer directly — saves a Vec<Vec<f32>> reshape.
                d_flat[t * top_n_h..(t + 1) * top_n_h].to_vec()
            } else {
                let (loss_t, d_logits) =
                    cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
                total_loss += loss_t;

                let top_layer = n_layers - 1;
                let active_top_t = &cache.layers[top_layer].h_active[t];

                for o in 0..self.n_output {
                    let d = d_logits[o];
                    if d == 0.0 {
                        continue;
                    }
                    grads.b_o[o] += d;
                    let row_offset = o * top_hidden;
                    for &h in active_top_t {
                        grads.w_out[row_offset + h] += d;
                    }
                }
                matvec_transpose(&self.w_out, &d_logits, self.n_output, top_hidden)
            };

            // Top-down gradient that becomes the per-layer
            // `d_h_spike_from_above` for the top layer; for layer L<top
            // it's filled in after processing layer L+1.
            let mut d_h_spike_from_above: Vec<f32> = d_h_top_spike_from_out;

            for l in (0..n_layers).rev() {
                let layer = &self.layers[l];
                let n_h = layer.n_hidden;
                let n_in = layer.n_in;
                let layer_grad = &mut grads.layers[l];

                // Total d_h_spike at step t for layer l =
                // (from above) + (recurrent feedback from t+1)
                // (+ Sprint 40 AdLIF: ∂w_t/∂spike_t = b · dw_carry).
                let mut d_h_spike_t: Vec<f32> = (0..n_h)
                    .map(|i| d_h_spike_from_above[i] + d_h_spike_recurrent[l][i])
                    .collect();
                if adlif_active {
                    for i in 0..n_h {
                        d_h_spike_t[i] += ADLIF_B * d_w_carry[l][i];
                    }
                }

                // Sprint 42: zero spike-derivative for dead neurons —
                // kills the surrogate path AND the d_alpha contribution.
                // Combined with the forward spike mask, every weight
                // touching a dead neuron is frozen (proof: see commit
                // sprint 42 doc).
                if let Some(apop) = self.apoptosis.as_ref() {
                    for i in 0..n_h {
                        if apop.dead_mask[l][i] {
                            d_h_spike_t[i] = 0.0;
                        }
                    }
                }

                // d_h_v_pre[l, t] = d_h_spike[l, t] * surrogate'(h_v_pre)
                //                 + d_h_v_after_carry[l] (from t+1).
                // Sprint 30 LSG: surrogate uses per-layer learnable α
                //   surrogate(v − θ) = 1 / (1 + α|v − θ|)²
                //   ∂surrogate/∂v = ±2α / (1 + α|v − θ|)³  (sign of v − θ)
                //   ∂surrogate/∂α = -2|v − θ| / (1 + α|v − θ|)³
                // Both derivatives share the cubic denominator so we
                // compute it once per neuron-step.
                let theta = self.spike_fn.threshold;
                let alpha_l = layer.alpha;
                let mut d_h_v_pre_t = vec![0.0_f32; n_h];
                for i in 0..n_h {
                    let dv = cache.layers[l].h_v_pre[t][i] - theta;
                    let abs_dv = dv.abs();
                    let denom = 1.0 + alpha_l * abs_dv;
                    let cubic = denom * denom * denom;
                    let s_grad = 1.0 / (denom * denom);
                    d_h_v_pre_t[i] = d_h_spike_t[i] * s_grad + d_h_v_after_carry[l][i];
                    // ∂L/∂α += d_h_spike × ∂surrogate/∂α
                    layer_grad.d_alpha += d_h_spike_t[i] * (-2.0 * abs_dv / cubic);
                }

                // Sprint 40 AdLIF carry update: dw[t-1] = ρ·dw[t] − d_v_pre[t].
                if adlif_active {
                    for i in 0..n_h {
                        d_w_carry[l][i] = ADLIF_RHO * d_w_carry[l][i] - d_h_v_pre_t[i];
                    }
                }

                // Layer's own input active list (for grad_w_in /
                // grad_w_gate_in column writes). For layer 0: byte_t.
                // For layer L>0: cached active indices of layer L-1's
                // spike output at THIS timestep.
                let layer_input_one: [usize; 1];
                let layer_input_active: &[usize] = if l == 0 {
                    layer_input_one = [input_tokens[t] as usize];
                    &layer_input_one
                } else {
                    &cache.layers[l - 1].h_active[t]
                };
                // Recurrent: previous timestep's spike actives for
                // this layer (cached at forward time).
                let active_prev: &[usize] = if t == 0 {
                    &cache.layers[l].initial_h_active
                } else {
                    &cache.layers[l].h_active[t - 1]
                };

                // grad_b_h, grad_w_in (col additions for active input
                // indices), grad_w_rec (sparse over active_prev).
                for h in 0..n_h {
                    let d = d_h_v_pre_t[h];
                    if d == 0.0 {
                        continue;
                    }
                    layer_grad.b_h[h] += d;
                    let row_in = h * n_in;
                    for &c in layer_input_active {
                        layer_grad.w_in[row_in + c] += d;
                    }
                    let row_rec = h * n_h;
                    for &j in active_prev {
                        layer_grad.w_rec[row_rec + j] += d;
                    }
                }

                // Gate gradient.
                let h_v_after_prev: &[f32] = if t == 0 {
                    &cache.layers[l].initial_h_v_after
                } else {
                    &cache.layers[l].h_v_after[t - 1]
                };
                let mut d_gate_pre = vec![0.0_f32; n_h];
                for h in 0..n_h {
                    let g = cache.layers[l].gate[t][h];
                    let d_gate_h = d_h_v_pre_t[h] * h_v_after_prev[h];
                    d_gate_pre[h] = d_gate_h * g * (1.0 - g);
                }

                // grad_b_gate, grad_w_gate_in, grad_w_gate_rec.
                for h in 0..n_h {
                    let d = d_gate_pre[h];
                    if d == 0.0 {
                        continue;
                    }
                    layer_grad.b_gate[h] += d;
                    let row_in = h * n_in;
                    for &c in layer_input_active {
                        layer_grad.w_gate_in[row_in + c] += d;
                    }
                    let row_rec = h * n_h;
                    for &j in active_prev {
                        layer_grad.w_gate_rec[row_rec + j] += d;
                    }
                }

                // Carry into step t-1 within this layer.
                for h in 0..n_h {
                    d_h_v_after_carry[l][h] = d_h_v_pre_t[h] * cache.layers[l].gate[t][h];
                }
                let d_from_rec = matvec_transpose(&layer.w_rec, &d_h_v_pre_t, n_h, n_h);
                let d_from_gate_rec =
                    matvec_transpose(&layer.w_gate_rec, &d_gate_pre, n_h, n_h);
                for h in 0..n_h {
                    d_h_spike_recurrent[l][h] = d_from_rec[h] + d_from_gate_rec[h];
                }

                // Cross-layer back-prop: layer l's d_h_v_pre + d_gate
                // through `W_in_l` / `W_gate_in_l` becomes layer l-1's
                // `d_h_spike_from_above`.
                if l > 0 {
                    let d_from_w_in =
                        matvec_transpose(&layer.w_in, &d_h_v_pre_t, n_h, n_in);
                    let d_from_w_gate_in =
                        matvec_transpose(&layer.w_gate_in, &d_gate_pre, n_h, n_in);
                    d_h_spike_from_above = (0..n_in)
                        .map(|i| d_from_w_in[i] + d_from_w_gate_in[i])
                        .collect();
                }
            }
        }

        // Sprint 33 per-neuron spike-rate l2 regularization. Each
        // neuron gets a correction proportional to its own firing rate
        // vs. the target — a per-layer scalar correction (Sprint 32)
        // got absorbed by Adam's moment normalization because all
        // neurons of the layer were nudged identically. Per-neuron
        // gives the optimizer a usable signal: over-firing neurons
        // get pushed down, silent ones get pushed up, the layer-mean
        // can stay where the CE gradient wants it.
        let RateReg { target, lambda } = self.rate_reg;
        if lambda > 0.0 && t_steps > 0 {
            let inv_t = 1.0 / t_steps as f32;
            for (l, layer_cache) in cache.layers.iter().enumerate() {
                let n_h = self.layers[l].n_hidden;
                // Accumulate per-neuron firing counts in cache-friendly
                // order: outer t, inner i — each `row` (size n_h) is
                // walked sequentially. The original loop nesting was
                // `for i { for t { row[i] }}`, which strided through
                // memory with gap n_h and torpedoed the L1 cache for
                // any reasonable hidden width.
                let mut rate = vec![0.0_f32; n_h];
                for row in &layer_cache.h_spike {
                    for i in 0..n_h {
                        rate[i] += row[i];
                    }
                }
                let dead_l: Option<&[bool]> =
                    self.apoptosis.as_ref().map(|a| a.dead_mask[l].as_slice());
                for i in 0..n_h {
                    if let Some(d) = dead_l {
                        if d[i] {
                            continue;
                        }
                    }
                    let correction = 2.0 * lambda * (rate[i] * inv_t - target);
                    grads.layers[l].b_h[i] += correction;
                }
            }
        }

        total_loss + sdsa_loss
    }

    /// Sprint 38: SDSA-aware top-of-stack backward. Walks all
    /// timesteps' logits in one shot, accumulates `W_out` + `b_o`
    /// grads against the dense residual `y`, runs `sdsa_backward`
    /// once over the chunk, returns per-t `d_h_spike` for the top
    /// layer plus the summed CE loss. Caller injects the per-t result
    /// in place of the sparse-W_out path.
    fn backward_top_through_sdsa(
        &self,
        cache: &MultiLayerCache,
        targets: &[u32],
        grads: &mut MultiLayerGatedGrads,
    ) -> (Option<Vec<f32>>, f32) {
        let sdsa = self.sdsa.as_ref().expect("SDSA enabled");
        let sdsa_cache = cache
            .sdsa
            .as_ref()
            .expect("forward must populate cache.sdsa when SDSA on");
        let y_flat = cache
            .sdsa_y
            .as_ref()
            .expect("forward must populate cache.sdsa_y when SDSA on");
        let n_layers = self.layers.len();
        let top = n_layers - 1;
        let n_h = self.layers[top].n_hidden;
        let t_steps = targets.len();

        let mut total_loss = 0.0_f32;
        let mut d_y_flat = vec![0.0_f32; t_steps * n_h];
        for t in 0..t_steps {
            let (loss_t, d_logits) =
                cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;
            let y_t = &y_flat[t * n_h..(t + 1) * n_h];
            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                grads.b_o[o] += d;
                let row = o * n_h;
                for h in 0..n_h {
                    grads.w_out[row + h] += d * y_t[h];
                }
            }
            let d_y_t = matvec_transpose(&self.w_out, &d_logits, self.n_output, n_h);
            d_y_flat[t * n_h..(t + 1) * n_h].copy_from_slice(&d_y_t);
        }

        let mut h_top_flat: Vec<f32> = Vec::with_capacity(t_steps * n_h);
        for row in &cache.layers[top].h_spike {
            h_top_flat.extend_from_slice(row);
        }
        let sdsa_grads = grads
            .sdsa
            .as_mut()
            .expect("MultiLayerGatedGrads.sdsa must be Some when SDSA on");
        let d_h_spike_flat = crate::sdsa::sdsa_backward(
            sdsa,
            sdsa_cache,
            &h_top_flat,
            &d_y_flat,
            sdsa_grads,
        );

        (Some(d_h_spike_flat), total_loss)
    }

    /// Byte-level wrapper around `backward_lm_tokens_into` (Sprint 26).
    pub fn backward_lm_bytes_into(
        &self,
        cache: &MultiLayerCache,
        input_bytes: &[u8],
        targets: &[u8],
        grads: &mut MultiLayerGatedGrads,
    ) -> f32 {
        let toks: Vec<u32> = input_bytes.iter().map(|&b| u32::from(b)).collect();
        let tgts: Vec<u32> = targets.iter().map(|&b| u32::from(b)).collect();
        self.backward_lm_tokens_into(cache, &toks, &tgts, grads)
    }

    pub fn backward_lm_bytes(
        &mut self,
        cache: &MultiLayerCache,
        input_bytes: &[u8],
        targets: &[u8],
    ) -> f32 {
        let mut grads = std::mem::take(&mut self.grads);
        let loss = self.backward_lm_bytes_into(cache, input_bytes, targets, &mut grads);
        self.grads = grads;
        loss
    }

    pub fn backward_lm_tokens(
        &mut self,
        cache: &MultiLayerCache,
        input_tokens: &[u32],
        targets: &[u32],
    ) -> f32 {
        let mut grads = std::mem::take(&mut self.grads);
        let loss = self.backward_lm_tokens_into(cache, input_tokens, targets, &mut grads);
        self.grads = grads;
        loss
    }

    /// Streaming inference for a single token (u32). Sprint 26: works
    /// for arbitrary vocab via token id; the byte-level version is a
    /// thin wrapper.
    pub fn step_lm_token(&self, token: u32, states: &mut [GatedState]) -> Vec<f32> {
        assert_eq!(states.len(), self.layers.len());
        let n_layers = self.layers.len();
        let theta = self.spike_fn.threshold;
        let top_hidden = self.layers.last().unwrap().n_hidden;

        let mut layer_input_active: Vec<usize> = vec![token as usize];

        for l in 0..n_layers {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let active_prev = active_indices(&states[l].h_spike);

            let mut gate_pre = vec![0.0_f32; n_h];
            for hi in 0..n_h {
                let row_off_rec = hi * n_h;
                let row_off_in = hi * n_in;
                let mut acc = layer.b_gate[hi];
                for &c in &layer_input_active {
                    acc += layer.w_gate_in[row_off_in + c];
                }
                for &c in &active_prev {
                    acc += layer.w_gate_rec[row_off_rec + c];
                }
                gate_pre[hi] = acc;
            }
            let gate: Vec<f32> = gate_pre.iter().map(|&v| sigmoid(v)).collect();

            let mut h_v_pre = vec![0.0_f32; n_h];
            for hi in 0..n_h {
                let row_off_rec = hi * n_h;
                let row_off_in = hi * n_in;
                let mut h_in_val = 0.0_f32;
                for &c in &layer_input_active {
                    h_in_val += layer.w_in[row_off_in + c];
                }
                let mut h_rec_val = 0.0_f32;
                for &c in &active_prev {
                    h_rec_val += layer.w_rec[row_off_rec + c];
                }
                h_v_pre[hi] = (states[l].h_v_after[hi] * gate[hi]
                    + h_in_val
                    + h_rec_val
                    + layer.b_h[hi])
                    .clamp(-MEMBRANE_CLAMP, MEMBRANE_CLAMP);
            }
            let mut h_spike: Vec<f32> = h_v_pre
                .iter()
                .map(|&v| self.spike_fn.forward(v))
                .collect();
            if let Some(apop) = self.apoptosis.as_ref() {
                let dead = &apop.dead_mask[l];
                for (s, d) in h_spike.iter_mut().zip(dead.iter()) {
                    if *d {
                        *s = 0.0;
                    }
                }
            }
            let h_v_after: Vec<f32> = h_v_pre
                .iter()
                .zip(h_spike.iter())
                .map(|(&v, &s)| v - theta * s)
                .collect();

            let active_t = active_indices(&h_spike);
            states[l].h_v_after = h_v_after;
            states[l].h_spike = h_spike;
            layer_input_active = active_t;
        }

        let mut logits = Vec::with_capacity(self.n_output);
        for o in 0..self.n_output {
            let row_offset = o * top_hidden;
            let mut acc = self.b_o[o];
            for &c in &layer_input_active {
                acc += self.w_out[row_offset + c];
            }
            logits.push(acc);
        }
        logits
    }

    /// Byte-level wrapper around `step_lm_token`.
    pub fn step_lm_byte(&self, byte: u8, states: &mut [GatedState]) -> Vec<f32> {
        self.step_lm_token(u32::from(byte), states)
    }

    /// Stateless single-batch wrapper.
    pub fn forward_backward_batch(
        &mut self,
        batch_inputs: &[&[u8]],
        batch_targets: &[&[u8]],
    ) -> f32 {
        let mut states: Vec<Vec<GatedState>> = (0..batch_inputs.len())
            .map(|_| self.fresh_states())
            .collect();
        self.forward_backward_batch_stateful(batch_inputs, batch_targets, &mut states)
    }

    /// Rayon-parallel batched stateful trainer. Mirrors single-layer
    /// `GatedRecurrentSnn::forward_backward_batch_stateful`: each batch
    /// slot owns its `Vec<GatedState>` (one per layer); workers
    /// allocate per-thread `MultiLayerGatedGrads`, the reductions sum
    /// into `self.grads`. Caller does `clip_grads` + `step_adam` after.
    ///
    /// # Panics
    /// Panics on length mismatch between `batch_inputs`,
    /// `batch_targets`, and `states`.
    pub fn forward_backward_batch_stateful(
        &mut self,
        batch_inputs: &[&[u8]],
        batch_targets: &[&[u8]],
        states: &mut [Vec<GatedState>],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        if batch_inputs.is_empty() {
            return 0.0;
        }
        let weights = &*self;
        let results: Vec<(f32, MultiLayerGatedGrads)> = batch_inputs
            .par_iter()
            .zip(batch_targets.par_iter())
            .zip(states.par_iter_mut())
            .map(|((inputs, targets), state)| {
                let mut grads = weights.alloc_grads();
                let cache = weights.forward_lm_bytes_stateful(inputs, state);
                let loss =
                    weights.backward_lm_bytes_into(&cache, inputs, targets, &mut grads);
                (loss, grads)
            })
            .collect();
        self.zero_grad();
        let mut total_loss = 0.0_f32;
        for (loss, grads) in &results {
            total_loss += loss;
            self.grads.add(grads);
        }
        total_loss
    }

    /// Sprint 42: same as `forward_backward_batch_tokens_stateful`
    /// but additionally returns per-layer per-neuron mean firing rate
    /// across the chunk + batch. Used by the apoptosis EMA observer.
    /// Cost over the plain variant is one extra O(B·T·N) reduction
    /// per layer per chunk — negligible at our hidden widths.
    ///
    /// # Panics
    /// Panics on length mismatch between `batch_inputs`,
    /// `batch_targets`, and `states`.
    pub fn forward_backward_batch_tokens_stateful_with_rates(
        &mut self,
        batch_inputs: &[&[u32]],
        batch_targets: &[&[u32]],
        states: &mut [Vec<GatedState>],
    ) -> (f32, Vec<Vec<f32>>) {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        let n_layers = self.layers.len();
        let layer_sizes: Vec<usize> =
            self.layers.iter().map(|l| l.n_hidden).collect();
        if batch_inputs.is_empty() {
            let zeros: Vec<Vec<f32>> =
                layer_sizes.iter().map(|&n| vec![0.0; n]).collect();
            return (0.0, zeros);
        }
        let weights = &*self;
        let results: Vec<(f32, MultiLayerGatedGrads, Vec<Vec<f32>>)> = batch_inputs
            .par_iter()
            .zip(batch_targets.par_iter())
            .zip(states.par_iter_mut())
            .map(|((inputs, targets), state)| {
                let mut grads = weights.alloc_grads();
                let cache = weights.forward_lm_tokens_stateful(inputs, state);
                let loss =
                    weights.backward_lm_tokens_into(&cache, inputs, targets, &mut grads);
                let rates = cache.per_neuron_rates();
                (loss, grads, rates)
            })
            .collect();
        self.zero_grad();
        let mut total_loss = 0.0_f32;
        let mut rate_sum: Vec<Vec<f32>> =
            layer_sizes.iter().map(|&n| vec![0.0; n]).collect();
        for (loss, grads, rates) in &results {
            total_loss += loss;
            self.grads.add(grads);
            for l in 0..n_layers {
                if rates[l].is_empty() {
                    continue;
                }
                for i in 0..layer_sizes[l] {
                    rate_sum[l][i] += rates[l][i];
                }
            }
        }
        let inv_b = 1.0 / results.len() as f32;
        for row in &mut rate_sum {
            for v in row {
                *v *= inv_b;
            }
        }
        (total_loss, rate_sum)
    }

    /// Sprint 26: token-level rayon-parallel batch trainer. Each
    /// batch slot's `inputs[b]` / `targets[b]` are token id slices.
    ///
    /// # Panics
    /// Panics on length mismatch between `batch_inputs`,
    /// `batch_targets`, and `states`.
    pub fn forward_backward_batch_tokens_stateful(
        &mut self,
        batch_inputs: &[&[u32]],
        batch_targets: &[&[u32]],
        states: &mut [Vec<GatedState>],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        if batch_inputs.is_empty() {
            return 0.0;
        }
        let weights = &*self;
        let results: Vec<(f32, MultiLayerGatedGrads)> = batch_inputs
            .par_iter()
            .zip(batch_targets.par_iter())
            .zip(states.par_iter_mut())
            .map(|((inputs, targets), state)| {
                let mut grads = weights.alloc_grads();
                let cache = weights.forward_lm_tokens_stateful(inputs, state);
                let loss =
                    weights.backward_lm_tokens_into(&cache, inputs, targets, &mut grads);
                (loss, grads)
            })
            .collect();
        self.zero_grad();
        let mut total_loss = 0.0_f32;
        for (loss, grads) in &results {
            total_loss += loss;
            self.grads.add(grads);
        }
        total_loss
    }
}


fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
