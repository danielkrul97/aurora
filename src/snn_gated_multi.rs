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
/// Sprint 31: positive `b_h` init. With weights initialized small
/// (He-style), random one-hot/spike inputs land the membrane potential
/// near 0, far below θ=1.0 — the surrogate gradient is then ≈ 0 and
/// the network can't bootstrap firing. With β≈0.73 and `b_h = c`, the
/// equilibrium membrane is `c / (1−β) ≈ 3.7c`. Setting c=0.3 puts the
/// equilibrium near θ=1.0, so neurons fire at moderate rate (~20–30 %)
/// from step 0, surrogate grads flow, and recurrent dynamics don't
/// immediately blow up. b_h = 0.5 was tried but produced membrane
/// peaks > 40k from runaway recurrent integration.
const DEFAULT_B_H_INIT: f32 = 0.3;

impl GatedLayer {
    pub fn new<R: Rng>(n_in: usize, n_hidden: usize, rng: &mut R) -> Self {
        // He-style init. Variance-conserving init from Klos &
        // Memmesheimer 2024 was tried in Sprint 30 but doesn't apply
        // cleanly: layer 0 sees a one-hot input (effective rate 1/n_in,
        // not the canonical ν), so a single formula across layers
        // overestimates the input variance and silences the network.
        // Differentiating layer-0 vs deeper layers is open work.
        let scale_in = (2.0 / n_in as f32).sqrt();
        let scale_rec = 0.9 / (n_hidden as f32).sqrt();
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
            b_gate: vec![1.0; n_hidden], // forget-gate-bias-1 trick
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
        }
    }

    pub fn zero(&mut self) {
        for l in &mut self.layers {
            l.zero();
        }
        self.w_out.fill(0.0);
        self.b_o.fill(0.0);
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
    }
}

/// Per-layer cache (per-timestep) for one chunk's forward pass.
#[derive(Clone, Debug)]
pub struct LayerCache {
    pub h_v_pre: Vec<Vec<f32>>,
    pub h_v_after: Vec<Vec<f32>>,
    pub h_spike: Vec<Vec<f32>>,
    pub gate: Vec<Vec<f32>>,
    pub initial_h_v_after: Vec<f32>,
    pub initial_h_spike: Vec<f32>,
}

impl LayerCache {
    pub fn with_capacity(t_steps: usize) -> Self {
        Self {
            h_v_pre: Vec::with_capacity(t_steps),
            h_v_after: Vec::with_capacity(t_steps),
            h_spike: Vec::with_capacity(t_steps),
            gate: Vec::with_capacity(t_steps),
            initial_h_v_after: Vec::new(),
            initial_h_spike: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLayerCache {
    pub layers: Vec<LayerCache>,
    pub logits: Vec<Vec<f32>>,
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
        // target=0.1 is the firing rate that keeps the active-indices
        // compaction comfortably sparse. λ=0.01 is a gentle default —
        // doesn't kill firing on small/short tests, doesn't dominate
        // CE gradients. For overnight runs where membrane runaway
        // matters more, callers can crank λ on the network directly.
        // Adam's moment normalization mostly absorbs the constant
        // bias correction, so seeing a measurable rate shift requires
        // either much higher λ or many epochs.
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
            adam_t: 0,
            layer_adam,
            adam_w_out: AdamGroup::zeros(n_output * top_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
        }
    }

    pub fn fresh_states(&self) -> Vec<GatedState> {
        self.layers
            .iter()
            .map(|l| GatedState {
                h_v_after: vec![0.0_f32; l.n_hidden],
                h_spike: vec![0.0_f32; l.n_hidden],
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
        MultiLayerGatedGrads::zeros(&layer_dims, self.n_output)
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
        };
        for (l, st) in states.iter().enumerate() {
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
                    h_v_pre_t[hi] = state.h_v_after[hi] * gate[hi]
                        + h_in_val
                        + h_rec_val
                        + layer.b_h[hi];
                }
                let h_spike_t: Vec<f32> = h_v_pre_t
                    .iter()
                    .map(|&v| self.spike_fn.forward(v))
                    .collect();
                let h_v_after_t: Vec<f32> = h_v_pre_t
                    .iter()
                    .zip(h_spike_t.iter())
                    .map(|(&v, &s)| v - theta * s)
                    .collect();

                let active_t = active_indices(&h_spike_t);

                cache.layers[l].h_v_pre.push(h_v_pre_t);
                cache.layers[l].h_v_after.push(h_v_after_t.clone());
                cache.layers[l].h_spike.push(h_spike_t.clone());
                cache.layers[l].gate.push(gate);

                states[l].h_v_after = h_v_after_t;
                states[l].h_spike = h_spike_t;

                layer_input_active = active_t;
            }

            // Output projection from top layer's spike output.
            let top_active = layer_input_active; // == active indices of top spike
            let mut logits = self.b_o.clone();
            for o in 0..self.n_output {
                let row_offset = o * top_hidden;
                let mut acc = self.b_o[o];
                for &c in &top_active {
                    acc += self.w_out[row_offset + c];
                }
                logits[o] = acc;
            }
            cache.logits.push(logits);
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

        let mut total_loss = 0.0_f32;

        for t in (0..t_steps).rev() {
            let (loss_t, d_logits) =
                cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            // Top layer spike at t (used for W_out grad and back-prop
            // into top layer hidden).
            let top_layer = n_layers - 1;
            let active_top_t = active_indices(&cache.layers[top_layer].h_spike[t]);

            // grad_w_out and grad_b_o + d_h_top_spike via output.
            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                grads.b_o[o] += d;
                let row_offset = o * top_hidden;
                for &h in &active_top_t {
                    grads.w_out[row_offset + h] += d;
                }
            }
            let d_h_top_spike_from_out =
                matvec_transpose(&self.w_out, &d_logits, self.n_output, top_hidden);

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
                // (from above) + (recurrent feedback from t+1).
                let d_h_spike_t: Vec<f32> = (0..n_h)
                    .map(|i| d_h_spike_from_above[i] + d_h_spike_recurrent[l][i])
                    .collect();

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

                // Layer's own input active list (for grad_w_in /
                // grad_w_gate_in column writes). For layer 0: byte_t.
                // For layer L>0: active indices of layer L-1's spike
                // output at THIS timestep.
                let layer_input_active: Vec<usize> = if l == 0 {
                    vec![input_tokens[t] as usize]
                } else {
                    active_indices(&cache.layers[l - 1].h_spike[t])
                };
                // Recurrent: previous timestep's spike actives for
                // this layer.
                let active_prev = if t == 0 {
                    active_indices(&cache.layers[l].initial_h_spike)
                } else {
                    active_indices(&cache.layers[l].h_spike[t - 1])
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
                    for &c in &layer_input_active {
                        layer_grad.w_in[row_in + c] += d;
                    }
                    let row_rec = h * n_h;
                    for &j in &active_prev {
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
                    for &c in &layer_input_active {
                        layer_grad.w_gate_in[row_in + c] += d;
                    }
                    let row_rec = h * n_h;
                    for &j in &active_prev {
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

        // Sprint 32 spike-rate l2 regularization. Push b_h toward
        // settings that produce a target firing rate. Decoupled from
        // the surrogate path on purpose — prevents the runaway
        // membrane integration observed in Sprint 31 (peaks > 30 k)
        // even when the surrogate is saturated.
        let RateReg { target, lambda } = self.rate_reg;
        if lambda > 0.0 && t_steps > 0 {
            for (l, layer_cache) in cache.layers.iter().enumerate() {
                let n_h = self.layers[l].n_hidden;
                let total = (t_steps * n_h) as f32;
                if total == 0.0 {
                    continue;
                }
                let count: f32 = layer_cache.h_spike.iter().flatten().copied().sum();
                let rate = count / total;
                let correction = 2.0 * lambda * (rate - target);
                for bh in &mut grads.layers[l].b_h {
                    *bh += correction;
                }
            }
        }

        total_loss
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
                h_v_pre[hi] = states[l].h_v_after[hi] * gate[hi]
                    + h_in_val
                    + h_rec_val
                    + layer.b_h[hi];
            }
            let h_spike: Vec<f32> = h_v_pre
                .iter()
                .map(|&v| self.spike_fn.forward(v))
                .collect();
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

        let mut logits = self.b_o.clone();
        for o in 0..self.n_output {
            let row_offset = o * top_hidden;
            for &c in &layer_input_active {
                logits[o] += self.w_out[row_offset + c];
            }
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
