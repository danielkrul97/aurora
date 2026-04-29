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
/// Equilibrium membrane = `b / (1 - a)`. With `a = 0.5`, `b = 0.5`
/// puts equilibrium at θ = 1.0 — Aurora's gated default ported.
const DEFAULT_B_INIT: f32 = 0.5;

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
        // He-style for `W_in`. No `W_rec`, no gate weights.
        let scale_in = (2.0 / n_in as f32).sqrt();
        let w_in = (0..n_hidden * n_in)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_in)
            .collect();
        Self {
            n_in,
            n_hidden,
            a: vec![DEFAULT_A_INIT; n_hidden],
            w_in,
            b: vec![DEFAULT_B_INIT; n_hidden],
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
        assert!(!hidden_dims.is_empty(), "at least one layer required");
        let mut layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = n_input;
        for &h in hidden_dims {
            layers.push(SsmLayer::new(prev_dim, h, rng));
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
            grads: MultiLayerSsmGrads::zeros(&layer_dims, n_output),
            layers,
            w_out,
            b_o: vec![0.0; n_output],
            spike_fn,
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

        for &tok in input_tokens {
            let b = tok as usize;
            debug_assert!(b < self.n_input);
            // Active inputs into layer l. For layer 0: [byte/token].
            // For layer l > 0: layer l-1's spike actives at this t.
            let mut layer_input_active: Vec<usize> = vec![b];

            for l in 0..n_layers {
                let layer = &self.layers[l];
                let n_h = layer.n_hidden;
                let n_in = layer.n_in;

                // s[t] = a ⊙ s[t-1] + W_in · x[t] + b
                let mut s_new = vec![0.0_f32; n_h];
                for hi in 0..n_h {
                    let row_in = hi * n_in;
                    let mut acc = layer.a[hi] * states[l].state[hi] + layer.b[hi];
                    for &c in &layer_input_active {
                        acc += layer.w_in[row_in + c];
                    }
                    s_new[hi] = acc.clamp(-MEMBRANE_CLAMP, MEMBRANE_CLAMP);
                }

                // Spike + active-set scan.
                let mut spike_t: Vec<f32> = Vec::with_capacity(n_h);
                let mut active_t: Vec<usize> = Vec::new();
                for (hi, &v) in s_new.iter().enumerate() {
                    let s = self.spike_fn.forward(v);
                    spike_t.push(s);
                    if s != 0.0 {
                        active_t.push(hi);
                    }
                }

                // Update carry-state for next timestep (no reset on spike).
                states[l].state.clone_from(&s_new);

                cache.layers[l].state.push(s_new);
                cache.layers[l].spike.push(spike_t);
                cache.layers[l].active.push(active_t.clone());

                layer_input_active = active_t;
            }

            // Output projection from top spike (sparse).
            let top_active = layer_input_active;
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

        cache
    }

    pub fn forward_lm_tokens(&self, input_tokens: &[u32]) -> MultiLayerCache {
        let mut states = self.fresh_states();
        self.forward_lm_tokens_stateful(input_tokens, &mut states)
    }

    /// Backward over the chunk, accumulating into `grads`. Returns
    /// summed cross-entropy loss.
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
        let top_hidden = self.layers.last().unwrap().n_hidden;
        let theta = self.spike_fn.threshold;

        // Per-layer carry: gradient on state at t+1 propagates to t via
        // d_s_carry[t] = a ⊙ d_s[t+1].
        let mut d_s_carry: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| vec![0.0_f32; l.n_hidden])
            .collect();

        let mut total_loss = 0.0_f32;

        for t in (0..t_steps).rev() {
            let (loss_t, d_logits) =
                cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            // grad on top-layer spike (and W_out / b_o updates).
            let top = n_layers - 1;
            let active_top_t = &cache.layers[top].active[t];
            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                grads.b_o[o] += d;
                let row = o * top_hidden;
                for &h in active_top_t {
                    grads.w_out[row + h] += d;
                }
            }
            // d_h_spike from W_out^T · d_logits (dense, length top_hidden).
            let d_top_spike = matvec_transpose(&self.w_out, &d_logits, self.n_output, top_hidden);

            // Walk layers top → bottom; d_spike for layer l-1 is built
            // from layer l's d_input.
            let mut d_spike_from_above: Vec<f32> = d_top_spike;

            for l in (0..n_layers).rev() {
                let layer = &self.layers[l];
                let n_h = layer.n_hidden;
                let n_in = layer.n_in;
                let layer_grad = &mut grads.layers[l];

                // d_state[t] = d_spike[t] · surrogate(state[t] − θ)  +  carry[t+1]
                // Surrogate: s_grad = 1 / (1 + α|state - θ|)²,
                //            ∂s_grad/∂α via cubic denom (Sprint 30 LSG).
                let alpha_l = layer.alpha;
                let mut d_state_t = vec![0.0_f32; n_h];
                for i in 0..n_h {
                    let dv = cache.layers[l].state[t][i] - theta;
                    let abs_dv = dv.abs();
                    let denom = 1.0 + alpha_l * abs_dv;
                    let cubic = denom * denom * denom;
                    let s_grad = 1.0 / (denom * denom);
                    d_state_t[i] = d_spike_from_above[i] * s_grad + d_s_carry[l][i];
                    // ∂L/∂α += d_spike × ∂surrogate/∂α
                    layer_grad.d_alpha += d_spike_from_above[i] * (-2.0 * abs_dv / cubic);
                }

                // Layer's input active set at this t.
                let layer_input_one: [usize; 1];
                let layer_input_active: &[usize] = if l == 0 {
                    layer_input_one = [input_tokens[t] as usize];
                    &layer_input_one
                } else {
                    &cache.layers[l - 1].active[t]
                };

                // grad b: d_state_t[i] adds to b[i]
                // grad W_in: d_state_t[i] adds to W_in[i, c] for each active input c
                // grad a: d_state_t[i] · s_prev[i] (where s_prev = state[t-1] or initial_state)
                let s_prev: &[f32] = if t == 0 {
                    &cache.layers[l].initial_state
                } else {
                    &cache.layers[l].state[t - 1]
                };
                for i in 0..n_h {
                    let d = d_state_t[i];
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

                // Carry into t-1: d_s_carry[l][i] = a[i] · d_state_t[i]
                for i in 0..n_h {
                    d_s_carry[l][i] = layer.a[i] * d_state_t[i];
                }

                // Cross-layer back-prop: layer l's d_state through W_in
                // becomes layer l-1's d_spike_from_above.
                if l > 0 {
                    d_spike_from_above = matvec_transpose(&layer.w_in, &d_state_t, n_h, n_in);
                }
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
