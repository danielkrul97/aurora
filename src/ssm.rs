#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

//! Sprint 46: diagonal state-space cell with spike output.
//!
//! Replacement for `GatedLayer`. Per-neuron diagonal recurrence
//! (no full `W_rec` matmul, no gate, no spike feedback) — pure linear
//! state evolution, heaviside read-out. Closer to S4 / Mamba than to
//! Aurora's gated LIF.
//!
//! Forward (per timestep):
//! ```text
//! s[t]     = a ⊙ s[t-1] + W_in · x[t] + b      (a is per-neuron decay in (0, 0.99])
//! spike[t] = heaviside(s[t] - θ)               (ANN→spike read-out)
//! ```
//!
//! Notably NOT here:
//! - **No `W_rec`** — recurrence is *only* via the diagonal `a` decay.
//! - **No subtract-on-spike** — `s` is not reset; bounded purely via
//!   `|a| < 1`. This makes the recurrence linear, future-proofing for
//!   parallel scan (Sprint 47+).
//! - **No gate** — `a` is fixed per neuron (learned, but input-independent).
//!
//! Trade-off vs `GatedLayer`: cheaper per timestep (no n_h² recurrent
//! matvec), longer effective memory (diagonal `a` close to 1.0 carries
//! signal across many steps), at the cost of expressivity from the gate.

use rand::Rng;
use rayon::prelude::*;

use crate::optimizer::{adam_apply, bias_correction, AdamConfig, AdamGroup};
use crate::snn_loss::cross_entropy_with_grad;
use crate::surrogate::SpikeFn;

const MEMBRANE_CLAMP: f32 = 10.0;
const A_MAX: f32 = 0.99;
const DEFAULT_ALPHA: f32 = 4.0;
/// Sprint 46: short memory init. `a = 0.5` matches Aurora's gated
/// effective β at `b_gate = 0`. First experiment with `a = 0.95`
/// (long memory) collapsed to winner-take-all — without
/// subtract-on-spike, state polarizes: a few neurons saturate at v=10
/// (clamp), most atrophy to v ≪ 0. Short memory limits the drift.
const DEFAULT_A_INIT: f32 = 0.5;

/// One SSM layer's parameters.
#[derive(Clone, Debug)]
pub struct SsmLayer {
    pub n_in: usize,
    pub n_hidden: usize,
    /// Per-neuron diagonal decay; clamped to `[0, A_MAX]` after each
    /// Adam step (keeps state bounded). Stored in raw form, no
    /// reparametrization — simpler gradients and visible param values.
    pub a: Vec<f32>,
    pub w_in: Vec<f32>, // [n_hidden, n_in]
    pub b: Vec<f32>,    // [n_hidden]
    pub alpha: f32,     // surrogate-gradient width (LSG)
}

impl SsmLayer {
    pub fn new<R: Rng>(n_in: usize, n_hidden: usize, rng: &mut R) -> Self {
        Self::new_with_a(n_in, n_hidden, DEFAULT_A_INIT, rng)
    }

    /// Sprint 47: configurable `a` init for the diagonal decay.
    /// Equilibrium membrane reaches `b / (1 - a)`; we set
    /// `b = (1 - a) · θ` so `v_eq = θ` regardless of `a`. Effective
    /// memory ≈ `1 / (1 - a)` timesteps.
    pub fn new_with_a<R: Rng>(n_in: usize, n_hidden: usize, a_init: f32, rng: &mut R) -> Self {
        Self::new_with_a_spread(n_in, n_hidden, a_init, 0.0, rng)
    }

    /// Sprint 48: HiPPO-style log-linear memory spectrum. Neurons
    /// span timescales from fast (`a ≈ a_lo`, ~1/(1-a_lo) steps memory)
    /// to slow (`a ≈ a_hi`, ~1/(1-a_hi) steps), with `log(1-a_i)`
    /// linearly interpolated between the two. Standard SSM init
    /// (Gu et al., HiPPO) — provides intrinsic multi-scale temporal
    /// representation without requiring time-varying gates.
    pub fn new_with_a_spectrum<R: Rng>(
        n_in: usize,
        n_hidden: usize,
        a_lo: f32,
        a_hi: f32,
        rng: &mut R,
    ) -> Self {
        let scale_in = (2.0 / n_in as f32).sqrt();
        let w_in = (0..n_hidden * n_in)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_in)
            .collect();
        let log_lo = (1.0 - a_lo).max(1e-3).ln();
        let log_hi = (1.0 - a_hi).max(1e-3).ln();
        let a: Vec<f32> = (0..n_hidden)
            .map(|i| {
                let frac = if n_hidden > 1 {
                    i as f32 / (n_hidden - 1) as f32
                } else {
                    0.5
                };
                let log_decay = log_lo + (log_hi - log_lo) * frac;
                (1.0 - log_decay.exp()).clamp(0.05, 0.99)
            })
            .collect();
        let b: Vec<f32> = a.iter().map(|&v| (1.0 - v).max(1e-3)).collect();
        Self {
            n_in,
            n_hidden,
            a,
            w_in,
            b,
            alpha: DEFAULT_ALPHA,
        }
    }

    /// Sprint 48: heterogenous diagonal `a`. Each neuron's `a` is
    /// drawn uniformly from `[a_init - spread, a_init + spread]`,
    /// clamped to `[0.05, 0.99]`. Per-neuron `b = (1 - a_i) · θ` keeps
    /// equilibrium at threshold for every neuron individually. Spread
    /// > 0 introduces a memory-spectrum: some neurons fast, some slow
    /// — diversity that may break the WTA failure mode of homogeneous
    /// long-memory init.
    pub fn new_with_a_spread<R: Rng>(
        n_in: usize,
        n_hidden: usize,
        a_init: f32,
        spread: f32,
        rng: &mut R,
    ) -> Self {
        let scale_in = (2.0 / n_in as f32).sqrt();
        let w_in = (0..n_hidden * n_in)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_in)
            .collect();
        let a: Vec<f32> = (0..n_hidden)
            .map(|_| {
                let r = if spread > 0.0 {
                    a_init + (rng.random::<f32>() - 0.5) * 2.0 * spread
                } else {
                    a_init
                };
                r.clamp(0.05, 0.99)
            })
            .collect();
        let b: Vec<f32> = a.iter().map(|&v| (1.0 - v).max(1e-3)).collect();
        Self {
            n_in,
            n_hidden,
            a,
            w_in,
            b,
            alpha: DEFAULT_ALPHA,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerGrads {
    pub a: Vec<f32>,
    pub w_in: Vec<f32>,
    pub b: Vec<f32>,
    pub d_alpha: f32,
}

impl LayerGrads {
    pub fn zeros(n_in: usize, n_hidden: usize) -> Self {
        Self {
            a: vec![0.0; n_hidden],
            w_in: vec![0.0; n_hidden * n_in],
            b: vec![0.0; n_hidden],
            d_alpha: 0.0,
        }
    }

    pub fn zero(&mut self) {
        self.a.fill(0.0);
        self.w_in.fill(0.0);
        self.b.fill(0.0);
        self.d_alpha = 0.0;
    }

    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.a.iter_mut().zip(&other.a) {
            *a += b;
        }
        for (a, b) in self.w_in.iter_mut().zip(&other.w_in) {
            *a += b;
        }
        for (a, b) in self.b.iter_mut().zip(&other.b) {
            *a += b;
        }
        self.d_alpha += other.d_alpha;
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &f32> {
        self.a
            .iter()
            .chain(&self.w_in)
            .chain(&self.b)
            .chain(std::iter::once(&self.d_alpha))
    }

    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.a
            .iter_mut()
            .chain(&mut self.w_in)
            .chain(&mut self.b)
            .chain(std::iter::once(&mut self.d_alpha))
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiLayerSsmGrads {
    pub layers: Vec<LayerGrads>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
}

impl MultiLayerSsmGrads {
    pub fn zeros(layer_dims: &[usize], n_output: usize) -> Self {
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

/// Per-layer cache for one chunk's forward pass.
#[derive(Clone, Debug)]
pub struct LayerCache {
    /// Pre-spike state at each timestep, [t][n_h]. Used in backward
    /// for surrogate-gradient evaluation and `d_a += d_s · s_prev`.
    pub state: Vec<Vec<f32>>,
    pub spike: Vec<Vec<f32>>,
    pub active: Vec<Vec<usize>>,
    /// State carried in from the previous chunk (or zeros if fresh).
    pub initial_state: Vec<f32>,
}

impl LayerCache {
    pub fn with_capacity(t_steps: usize) -> Self {
        Self {
            state: Vec::with_capacity(t_steps),
            spike: Vec::with_capacity(t_steps),
            active: Vec::with_capacity(t_steps),
            initial_state: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiLayerCache {
    pub layers: Vec<LayerCache>,
    pub logits: Vec<Vec<f32>>,
}

impl MultiLayerCache {
    #[must_use]
    pub fn firing_rates(&self) -> Vec<f32> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.spike.is_empty() {
                    return 0.0;
                }
                let n_h = layer.spike[0].len();
                let total = (layer.spike.len() * n_h) as f32;
                let count: f32 = layer.spike.iter().flatten().copied().sum();
                if total > 0.0 { count / total } else { 0.0 }
            })
            .collect()
    }

    #[must_use]
    pub fn membrane_stats(&self) -> Vec<(f32, f32)> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.state.is_empty() {
                    return (0.0, 0.0);
                }
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                let mut peak = f32::NEG_INFINITY;
                for row in &layer.state {
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

/// Per-layer state carried across chunks.
#[derive(Clone, Debug, Default)]
pub struct SsmState {
    pub state: Vec<f32>,
}

#[derive(Clone, Debug)]
struct LayerAdam {
    m_a: AdamGroup,
    m_w_in: AdamGroup,
    m_b: AdamGroup,
    m_alpha: AdamGroup,
}

impl LayerAdam {
    fn zeros(n_in: usize, n_hidden: usize) -> Self {
        Self {
            m_a: AdamGroup::zeros(n_hidden),
            m_w_in: AdamGroup::zeros(n_hidden * n_in),
            m_b: AdamGroup::zeros(n_hidden),
            m_alpha: AdamGroup::zeros(1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLayerSsmSnn {
    pub n_input: usize,
    pub n_output: usize,
    pub layers: Vec<SsmLayer>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
    pub spike_fn: SpikeFn,
    /// Sprint 49: when `true`, the cell skips the heaviside spike step
    /// — inter-layer signal is the real-valued state directly. Dense
    /// W_in / W_out matmul (no sparse active-set optimization) and no
    /// surrogate gradient (cleaner backprop). A/B test: does dropping
    /// the 1-bit quantization improve LM quality enough to justify
    /// losing Aurora's "biological" spike identity?
    pub continuous: bool,
    grads: MultiLayerSsmGrads,
    adam_t: usize,
    layer_adam: Vec<LayerAdam>,
    adam_w_out: AdamGroup,
    adam_b_o: AdamGroup,
}

impl MultiLayerSsmSnn {
    pub fn new<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        rng: &mut R,
    ) -> Self {
        Self::new_with_a(n_input, hidden_dims, n_output, spike_fn, DEFAULT_A_INIT, rng)
    }

    pub fn new_with_a<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        a_init: f32,
        rng: &mut R,
    ) -> Self {
        Self::new_with_a_spread(n_input, hidden_dims, n_output, spike_fn, a_init, 0.0, rng)
    }

    pub fn new_with_a_spread<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        a_init: f32,
        spread: f32,
        rng: &mut R,
    ) -> Self {
        assert!(!hidden_dims.is_empty(), "at least one layer required");
        let mut layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = n_input;
        for &h in hidden_dims {
            layers.push(SsmLayer::new_with_a_spread(prev_dim, h, a_init, spread, rng));
            prev_dim = h;
        }
        Self::finish_construction(layers, n_input, hidden_dims, n_output, spike_fn, rng)
    }

    /// Sprint 48: HiPPO-style construction.
    pub fn new_with_spectrum<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        a_lo: f32,
        a_hi: f32,
        rng: &mut R,
    ) -> Self {
        assert!(!hidden_dims.is_empty(), "at least one layer required");
        let mut layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = n_input;
        for &h in hidden_dims {
            layers.push(SsmLayer::new_with_a_spectrum(prev_dim, h, a_lo, a_hi, rng));
            prev_dim = h;
        }
        Self::finish_construction(layers, n_input, hidden_dims, n_output, spike_fn, rng)
    }

    fn finish_construction<R: Rng>(
        layers: Vec<SsmLayer>,
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        spike_fn: SpikeFn,
        rng: &mut R,
    ) -> Self {
        let top_hidden = *hidden_dims.last().unwrap();
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
            grads: MultiLayerSsmGrads::zeros(&layer_dims, n_output),
            layers,
            w_out,
            b_o: vec![0.0; n_output],
            spike_fn,
            continuous: false,
            adam_t: 0,
            layer_adam,
            adam_w_out: AdamGroup::zeros(n_output * top_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
        }
    }

    pub fn fresh_states(&self) -> Vec<SsmState> {
        self.layers
            .iter()
            .map(|l| SsmState {
                state: vec![0.0; l.n_hidden],
            })
            .collect()
    }

    pub const fn grads(&self) -> &MultiLayerSsmGrads {
        &self.grads
    }

    pub fn alloc_grads(&self) -> MultiLayerSsmGrads {
        let mut layer_dims = vec![self.n_input];
        for l in &self.layers {
            layer_dims.push(l.n_hidden);
        }
        MultiLayerSsmGrads::zeros(&layer_dims, self.n_output)
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

    pub fn add_grads(&mut self, other: &MultiLayerSsmGrads) {
        self.grads.add(other);
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let g = &self.grads.layers[l];
            let st = &mut self.layer_adam[l];
            adam_apply(&mut layer.a, &g.a, &mut st.m_a, config, bc1, bc2);
            // Clamp `a` to [0, A_MAX] to keep the state bounded.
            for v in &mut layer.a {
                *v = v.clamp(0.0, A_MAX);
            }
            adam_apply(&mut layer.w_in, &g.w_in, &mut st.m_w_in, config, bc1, bc2);
            adam_apply(&mut layer.b, &g.b, &mut st.m_b, config, bc1, bc2);
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
    /// Sprint 47: scan-based formulation. Within each layer, we
    /// pre-compute the input-driven term `u[t][i] = b[i] + Σ_{c}
    /// W_in[i,c]·x[t]` for the entire chunk, then run a per-neuron
    /// parallel scan on the diagonal recurrence to obtain all
    /// timesteps' states in one pass. Layers stay sequential (layer
    /// l's input is layer l-1's spike output), but within a layer the
    /// time dimension is now scan-able — directly maps to a parallel
    /// scan kernel on GPU when the port arrives.
    ///
    /// Membrane clamp is DROPPED in this formulation — the scan needs
    /// pure linear recurrence. With `a < 1` and grad-clipped weights,
    /// state stays bounded naturally.
    pub fn forward_lm_tokens_stateful(
        &self,
        input_tokens: &[u32],
        states: &mut [SsmState],
    ) -> MultiLayerCache {
        assert_eq!(states.len(), self.layers.len());
        let t_steps = input_tokens.len();
        let n_layers = self.layers.len();
        let mut cache = MultiLayerCache {
            layers: (0..n_layers).map(|_| LayerCache::with_capacity(t_steps)).collect(),
            logits: Vec::with_capacity(t_steps),
        };
        for (l, st) in states.iter().enumerate() {
            cache.layers[l].initial_state = st.state.clone();
        }
        let top_hidden = self.layers.last().unwrap().n_hidden;

        // Per-layer per-t input active set. Layer 0: each timestep's
        // single token. Layer l > 0: previous layer's spike actives.
        let mut layer_input_active_per_t: Vec<Vec<usize>> = (0..t_steps)
            .map(|t| vec![input_tokens[t] as usize])
            .collect();

        for l in 0..n_layers {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;

            // Step 1: precompute `u[t][i]` for the whole chunk.
            // Sparse path (spike mode or layer 0): sum W_in cols over
            // the active input set (signal=1 implicit).
            // Dense path (layer >0 in continuous mode): sum W_in over
            // all input dims, weighted by previous layer's real-valued
            // signal — full GEMV.
            let mut u_per_t: Vec<Vec<f32>> = (0..t_steps)
                .map(|_| layer.b.clone())
                .collect();
            let dense_signal: Option<&[Vec<f32>]> = if l > 0 && self.continuous {
                Some(&cache.layers[l - 1].spike)
            } else {
                None
            };
            for t in 0..t_steps {
                let actives = &layer_input_active_per_t[t];
                if let Some(sig_per_t) = dense_signal {
                    let signal = &sig_per_t[t];
                    for i in 0..n_h {
                        let row_in = i * n_in;
                        let row = &layer.w_in[row_in..row_in + n_in];
                        let mut delta = 0.0_f32;
                        for c in 0..n_in {
                            delta += row[c] * signal[c];
                        }
                        u_per_t[t][i] += delta;
                    }
                } else {
                    for i in 0..n_h {
                        let row_in = i * n_in;
                        let row = &layer.w_in[row_in..row_in + n_in];
                        let mut delta = 0.0_f32;
                        for &c in actives {
                            delta += row[c];
                        }
                        u_per_t[t][i] += delta;
                    }
                }
            }

            // Step 2: per-neuron parallel scan over time.
            // Reuse a single buffer for the per-neuron `u` slice to
            // keep allocation noise low.
            let s_init = states[l].state.clone();
            let mut state_per_t: Vec<Vec<f32>> = (0..t_steps)
                .map(|_| vec![0.0_f32; n_h])
                .collect();
            let mut u_neuron = vec![0.0_f32; t_steps];
            for i in 0..n_h {
                for t in 0..t_steps {
                    u_neuron[t] = u_per_t[t][i];
                }
                let s_seq = parallel_scan_diagonal(layer.a[i], &u_neuron, s_init[i]);
                for t in 0..t_steps {
                    state_per_t[t][i] = s_seq[t];
                }
            }

            // Step 3: spike + active-set scan per timestep. In
            // continuous mode (`self.continuous`), the layer's output
            // signal IS the state directly — no heaviside, no
            // quantization. Active-set degenerates to "all neurons"
            // and downstream matmuls run dense.
            let mut spike_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut active_per_t: Vec<Vec<usize>> = Vec::with_capacity(t_steps);
            for t in 0..t_steps {
                let mut spike = Vec::with_capacity(n_h);
                let mut active = Vec::new();
                for i in 0..n_h {
                    let s = if self.continuous {
                        state_per_t[t][i]
                    } else {
                        self.spike_fn.forward(state_per_t[t][i])
                    };
                    spike.push(s);
                    if self.continuous {
                        active.push(i);
                    } else if s != 0.0 {
                        active.push(i);
                    }
                }
                spike_per_t.push(spike);
                active_per_t.push(active);
            }

            // Step 4: carry final state into next chunk; cache.
            states[l]
                .state
                .clone_from(state_per_t.last().expect("t_steps ≥ 1"));
            cache.layers[l].state = state_per_t;
            cache.layers[l].spike = spike_per_t;

            // Set up input for next layer (clones to satisfy borrow on
            // active_per_t which we also stash in cache).
            layer_input_active_per_t = active_per_t.clone();
            cache.layers[l].active = active_per_t;
        }

        // Output projection from top layer. Sparse over `active` in
        // spike mode (binary 0/1 signal); dense GEMV over the full
        // state vector in continuous mode.
        let top = n_layers - 1;
        for t in 0..t_steps {
            let mut logits = Vec::with_capacity(self.n_output);
            if self.continuous {
                let top_signal = &cache.layers[top].spike[t];
                for o in 0..self.n_output {
                    let row_offset = o * top_hidden;
                    let mut acc = self.b_o[o];
                    for h in 0..top_hidden {
                        acc += self.w_out[row_offset + h] * top_signal[h];
                    }
                    logits.push(acc);
                }
            } else {
                let top_active = &cache.layers[top].active[t];
                for o in 0..self.n_output {
                    let row_offset = o * top_hidden;
                    let mut acc = self.b_o[o];
                    for &c in top_active {
                        acc += self.w_out[row_offset + c];
                    }
                    logits.push(acc);
                }
            }
            cache.logits.push(logits);
        }

        cache
    }

    pub fn forward_lm_tokens(&self, input_tokens: &[u32]) -> MultiLayerCache {
        let mut states = self.fresh_states();
        self.forward_lm_tokens_stateful(input_tokens, &mut states)
    }

    /// Backward over the chunk, accumulating into `grads`. Returns
    /// summed cross-entropy loss.
    /// Sprint 47: adjoint-scan formulation. Per layer we batch the
    /// surrogate-gradient evaluation across the entire chunk into a
    /// single per-neuron pass, then run a parallel adjoint scan on
    /// each neuron's reverse-time gradient sequence to fold in the
    /// recurrent contribution `d_state[t] += a · d_state[t+1]`. Layer
    /// loop stays top→bottom (cross-layer dependency); within a layer
    /// the time dimension is now scan-able.
    pub fn backward_lm_tokens_into(
        &self,
        cache: &MultiLayerCache,
        input_tokens: &[u32],
        targets: &[u32],
        grads: &mut MultiLayerSsmGrads,
    ) -> f32 {
        let t_steps = input_tokens.len();
        debug_assert_eq!(targets.len(), t_steps);
        let n_layers = self.layers.len();
        let top = n_layers - 1;
        let top_hidden = self.layers[top].n_hidden;
        let theta = self.spike_fn.threshold;

        // Step A: top-layer d_spike per timestep (and W_out / b_o grads).
        // Dense GEMV path in continuous mode; sparse over fired
        // indices in spike mode.
        let mut total_loss = 0.0_f32;
        let mut d_spike_from_above_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
        for t in 0..t_steps {
            let (loss_t, d_logits) =
                cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            if self.continuous {
                let top_signal = &cache.layers[top].spike[t];
                for o in 0..self.n_output {
                    let d = d_logits[o];
                    if d == 0.0 {
                        continue;
                    }
                    grads.b_o[o] += d;
                    let row = o * top_hidden;
                    for h in 0..top_hidden {
                        grads.w_out[row + h] += d * top_signal[h];
                    }
                }
            } else {
                let active_top = &cache.layers[top].active[t];
                for o in 0..self.n_output {
                    let d = d_logits[o];
                    if d == 0.0 {
                        continue;
                    }
                    grads.b_o[o] += d;
                    let row = o * top_hidden;
                    for &h in active_top {
                        grads.w_out[row + h] += d;
                    }
                }
            }
            let d_top_spike = matvec_transpose(&self.w_out, &d_logits, self.n_output, top_hidden);
            d_spike_from_above_per_t.push(d_top_spike);
        }

        // Step B: walk layers top → bottom. Within each layer, scan
        // produces the full d_state sequence in one shot.
        for l in (0..n_layers).rev() {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let alpha_l = layer.alpha;
            let layer_grad = &mut grads.layers[l];

            // B1. Per-step gradient from the spike → state path.
            // In spike mode, factor through surrogate
            // `1 / (1 + α|state-θ|)²` (and accumulate d_alpha).
            // In continuous mode, the spike → state map is identity,
            // gradient passes straight through (factor 1.0); α is
            // unused.
            let mut d_input_per_t: Vec<Vec<f32>> = (0..t_steps)
                .map(|_| vec![0.0_f32; n_h])
                .collect();
            for t in 0..t_steps {
                let d_above = &d_spike_from_above_per_t[t];
                if self.continuous {
                    for i in 0..n_h {
                        d_input_per_t[t][i] = d_above[i];
                    }
                } else {
                    let state_t = &cache.layers[l].state[t];
                    for i in 0..n_h {
                        let dv = state_t[i] - theta;
                        let abs_dv = dv.abs();
                        let denom = 1.0 + alpha_l * abs_dv;
                        let cubic = denom * denom * denom;
                        let s_grad = 1.0 / (denom * denom);
                        d_input_per_t[t][i] = d_above[i] * s_grad;
                        layer_grad.d_alpha += d_above[i] * (-2.0 * abs_dv / cubic);
                    }
                }
            }

            // B2. Per-neuron adjoint scan: full d_state[*][i].
            let mut d_state_per_t: Vec<Vec<f32>> = (0..t_steps)
                .map(|_| vec![0.0_f32; n_h])
                .collect();
            let mut d_in_neuron = vec![0.0_f32; t_steps];
            for i in 0..n_h {
                for t in 0..t_steps {
                    d_in_neuron[t] = d_input_per_t[t][i];
                }
                let d_state_seq = parallel_scan_diagonal_adjoint(layer.a[i], &d_in_neuron);
                for t in 0..t_steps {
                    d_state_per_t[t][i] = d_state_seq[t];
                }
            }

            // B3. Accumulate grads to `b`, `W_in`, `a`. Per-t loop
            // because input-active set varies with `t`.
            // Dense W_in grad path for layer >0 in continuous mode
            // (multiply by previous layer's real-valued signal).
            let dense_signal_back: Option<&[Vec<f32>]> =
                if l > 0 && self.continuous {
                    Some(&cache.layers[l - 1].spike)
                } else {
                    None
                };
            for t in 0..t_steps {
                let layer_input_one: [usize; 1];
                let layer_input_active: &[usize] = if l == 0 {
                    layer_input_one = [input_tokens[t] as usize];
                    &layer_input_one
                } else {
                    &cache.layers[l - 1].active[t]
                };
                let s_prev: &[f32] = if t == 0 {
                    &cache.layers[l].initial_state
                } else {
                    &cache.layers[l].state[t - 1]
                };
                if let Some(sig_per_t) = dense_signal_back {
                    let signal = &sig_per_t[t];
                    for i in 0..n_h {
                        let d = d_state_per_t[t][i];
                        if d == 0.0 {
                            continue;
                        }
                        layer_grad.b[i] += d;
                        layer_grad.a[i] += d * s_prev[i];
                        let row_in = i * n_in;
                        for c in 0..n_in {
                            layer_grad.w_in[row_in + c] += d * signal[c];
                        }
                    }
                } else {
                    for i in 0..n_h {
                        let d = d_state_per_t[t][i];
                        if d == 0.0 {
                            continue;
                        }
                        layer_grad.b[i] += d;
                        layer_grad.a[i] += d * s_prev[i];
                        let row_in = i * n_in;
                        for &c in layer_input_active {
                            layer_grad.w_in[row_in + c] += d;
                        }
                    }
                }
            }

            // B4. Cross-layer back-prop to layer l-1 (`d_spike` from
            // above for the next iteration). For each timestep:
            // `d_spike_below[t] = W_in_l^T · d_state[t]`.
            if l > 0 {
                let mut next_d_spike: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
                for t in 0..t_steps {
                    next_d_spike.push(matvec_transpose(
                        &layer.w_in,
                        &d_state_per_t[t],
                        n_h,
                        n_in,
                    ));
                }
                d_spike_from_above_per_t = next_d_spike;
            }
        }

        total_loss
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

    /// Streaming inference for a single token.
    pub fn step_lm_token(&self, token: u32, states: &mut [SsmState]) -> Vec<f32> {
        assert_eq!(states.len(), self.layers.len());
        let n_layers = self.layers.len();
        let top_hidden = self.layers.last().unwrap().n_hidden;
        let mut layer_input_active: Vec<usize> = vec![token as usize];
        for l in 0..n_layers {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let mut s_new = vec![0.0_f32; n_h];
            for hi in 0..n_h {
                let row_in = hi * n_in;
                let mut acc = layer.a[hi] * states[l].state[hi] + layer.b[hi];
                for &c in &layer_input_active {
                    acc += layer.w_in[row_in + c];
                }
                s_new[hi] = acc.clamp(-MEMBRANE_CLAMP, MEMBRANE_CLAMP);
            }
            let mut spike: Vec<f32> = Vec::with_capacity(n_h);
            let mut active_t: Vec<usize> = Vec::new();
            for (hi, &v) in s_new.iter().enumerate() {
                let s = self.spike_fn.forward(v);
                spike.push(s);
                if s != 0.0 {
                    active_t.push(hi);
                }
            }
            states[l].state = s_new;
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

    /// Rayon-parallel batched stateful trainer.
    pub fn forward_backward_batch_tokens_stateful(
        &mut self,
        batch_inputs: &[&[u32]],
        batch_targets: &[&[u32]],
        states: &mut [Vec<SsmState>],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        if batch_inputs.is_empty() {
            return 0.0;
        }
        let weights = &*self;
        let results: Vec<(f32, MultiLayerSsmGrads)> = batch_inputs
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

/// Sprint 47: Hillis-Steele inclusive prefix scan for the diagonal SSM
/// recurrence `s[t] = a · s[t-1] + u[t]` with **constant per-neuron**
/// `a` (the SSM cell does not learn time-varying `a`).
///
/// The scan is over the SSM compose operator on (A, U) pairs:
/// `(A1, U1) ⊕ (A2, U2) = (A1·A2, A2·U1 + U2)`
/// which is associative — Hillis-Steele applies. After the inclusive
/// scan, position `t` holds `(a^(t+1), Σ_{k=0}^t a^(t-k) · u[k])` and
/// the state is recovered as `s[t] = A_acc[t] · s_init + U_acc[t]`.
///
/// Cost: `O(T · log₂ T)` work, `O(log₂ T)` depth. Sequential CPU; the
/// shape is preserved so a future GPU port (parallel rows) is direct.
///
/// `a` is a scalar (per-neuron constant). `u` and the returned state
/// are length-T vectors. `s_init` is the carry-in (state from prior
/// chunk's last timestep, or zero on a fresh stream).
#[must_use]
pub fn parallel_scan_diagonal(a: f32, u: &[f32], s_init: f32) -> Vec<f32> {
    let t = u.len();
    if t == 0 {
        return Vec::new();
    }
    // Initialise the (A, U) buffers. Each timestep starts as the
    // single-step pair `(a, u[t])`.
    let mut acc_a = vec![a; t];
    let mut acc_u = u.to_vec();
    let mut buf_a = vec![0.0_f32; t];
    let mut buf_u = vec![0.0_f32; t];

    let mut stride = 1usize;
    while stride < t {
        // Positions [0, stride) are unchanged at this level.
        buf_a[..stride].copy_from_slice(&acc_a[..stride]);
        buf_u[..stride].copy_from_slice(&acc_u[..stride]);
        for i in stride..t {
            // Compose left = element at i-stride, right = element at i.
            // Result: (A_l · A_r, A_r · U_l + U_r).
            let a_l = acc_a[i - stride];
            let u_l = acc_u[i - stride];
            let a_r = acc_a[i];
            let u_r = acc_u[i];
            buf_a[i] = a_l * a_r;
            buf_u[i] = a_r * u_l + u_r;
        }
        std::mem::swap(&mut acc_a, &mut buf_a);
        std::mem::swap(&mut acc_u, &mut buf_u);
        stride *= 2;
    }

    let mut s = vec![0.0_f32; t];
    for i in 0..t {
        s[i] = acc_a[i] * s_init + acc_u[i];
    }
    s
}

/// Sprint 47: adjoint scan for the SSM backward pass.
///
/// Given upstream gradients `d_above[t] = ∂L/∂s[t]` (contributed by
/// the spike-output path at timestep `t`), the recurrence
/// `d_total[t] = d_above[t] + a · d_total[t+1]` (with `d_total[T] = 0`)
/// computes the full gradient on each timestep's state including
/// recurrent feedback from later timesteps.
///
/// This is the same linear recurrence as the forward, just over the
/// reverse-time sequence. Reuses `parallel_scan_diagonal` after
/// reversing inputs and reversing the output.
#[must_use]
pub fn parallel_scan_diagonal_adjoint(a: f32, d_above: &[f32]) -> Vec<f32> {
    let t = d_above.len();
    if t == 0 {
        return Vec::new();
    }
    // Reverse-time view: define τ = T-1-t. The recurrence on d_total
    // reads, in reverse: d̃[τ=0] = d_above[T-1],
    // d̃[τ=k] = d_above[T-1-k] + a · d̃[τ=k-1].
    // That's exactly the forward SSM recurrence with `u_τ` =
    // d_above[T-1-τ] and zero initial carry.
    let u_rev: Vec<f32> = d_above.iter().rev().copied().collect();
    let s_rev = parallel_scan_diagonal(a, &u_rev, 0.0);
    s_rev.into_iter().rev().collect()
}

fn matvec_transpose(w: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // out[c] = Σ_r w[r, c] · v[r]
    let mut out = vec![0.0_f32; cols];
    for r in 0..rows {
        let row_off = r * cols;
        let vr = v[r];
        if vr == 0.0 {
            continue;
        }
        for c in 0..cols {
            out[c] += w[row_off + c] * vr;
        }
    }
    out
}
