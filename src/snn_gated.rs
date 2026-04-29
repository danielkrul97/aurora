#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::if_not_else,
    clippy::too_many_lines
)]

//! Gated recurrent surrogate-gradient SNN.
//!
//! Like `RecurrentSurrogateSnn` but the membrane carryover is
//! multiplied by a learnable per-neuron forget gate `β_t ∈ [0,1]`
//! computed as `sigmoid(W_β_in @ x_t + W_β_rec @ h_spike_{t-1} + b_β)`.
//! When the gate is close to 1 the neuron remembers; close to 0 it
//! resets. This is the SNN analogue of LSTM/GRU forget gating and
//! breaks the fixed `exp(-dt/τ)` decay that limits vanilla recurrent
//! SNNs on long-range dependencies.
//!
//! The cell:
//! ```text
//! β_t       = sigmoid(W_β_in @ x_t + W_β_rec @ h_spike_{t-1} + b_β)
//! h_v_pre_t = h_v_after_{t-1} ⊙ β_t
//!           + W_in @ x_t + W_rec @ h_spike_{t-1} + b_h
//! h_spike_t = heaviside(h_v_pre_t - θ)
//! h_v_after_t = h_v_pre_t - θ ⊙ h_spike_t
//! logits_t  = W_out @ h_spike_t + b_o
//! ```

use rand::Rng;
use rayon::prelude::*;

use crate::linalg::{active_indices, matvec_transpose};
use crate::optimizer::{adam_apply, bias_correction, AdamConfig, AdamGroup};
use crate::snn_loss::cross_entropy_with_grad;
use crate::surrogate::SpikeFn;

/// Eight gradient buffers — one per parameter group of `GatedRecurrentSnn`.
/// Externalized so the parallel batch trainer can spawn per-thread
/// accumulators without cloning the whole network.
#[derive(Clone, Debug, Default)]
pub struct GatedGrads {
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_out: Vec<f32>,
    pub w_gate_in: Vec<f32>,
    pub w_gate_rec: Vec<f32>,
    pub b_h: Vec<f32>,
    pub b_o: Vec<f32>,
    pub b_gate: Vec<f32>,
}

impl GatedGrads {
    #[must_use]
    pub fn zeros(n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        Self {
            w_in: vec![0.0; n_hidden * n_input],
            w_rec: vec![0.0; n_hidden * n_hidden],
            w_out: vec![0.0; n_output * n_hidden],
            w_gate_in: vec![0.0; n_hidden * n_input],
            w_gate_rec: vec![0.0; n_hidden * n_hidden],
            b_h: vec![0.0; n_hidden],
            b_o: vec![0.0; n_output],
            b_gate: vec![0.0; n_hidden],
        }
    }

    pub fn zero(&mut self) {
        self.w_in.fill(0.0);
        self.w_rec.fill(0.0);
        self.w_out.fill(0.0);
        self.w_gate_in.fill(0.0);
        self.w_gate_rec.fill(0.0);
        self.b_h.fill(0.0);
        self.b_o.fill(0.0);
        self.b_gate.fill(0.0);
    }

    pub fn add(&mut self, other: &Self) {
        debug_assert_eq!(self.w_in.len(), other.w_in.len());
        for (a, b) in self.w_in.iter_mut().zip(&other.w_in) {
            *a += b;
        }
        for (a, b) in self.w_rec.iter_mut().zip(&other.w_rec) {
            *a += b;
        }
        for (a, b) in self.w_out.iter_mut().zip(&other.w_out) {
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
        for (a, b) in self.b_o.iter_mut().zip(&other.b_o) {
            *a += b;
        }
        for (a, b) in self.b_gate.iter_mut().zip(&other.b_gate) {
            *a += b;
        }
    }

    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        let mut sq = 0.0_f32;
        for v in self
            .w_in
            .iter()
            .chain(&self.w_rec)
            .chain(&self.w_out)
            .chain(&self.w_gate_in)
            .chain(&self.w_gate_rec)
            .chain(&self.b_h)
            .chain(&self.b_o)
            .chain(&self.b_gate)
        {
            sq += v * v;
        }
        sq.sqrt()
    }

    pub fn scale(&mut self, factor: f32) {
        for v in self
            .w_in
            .iter_mut()
            .chain(&mut self.w_rec)
            .chain(&mut self.w_out)
            .chain(&mut self.w_gate_in)
            .chain(&mut self.w_gate_rec)
            .chain(&mut self.b_h)
            .chain(&mut self.b_o)
            .chain(&mut self.b_gate)
        {
            *v *= factor;
        }
    }
}

#[derive(Clone, Debug)]
pub struct GatedLmCache {
    pub h_v_pre: Vec<Vec<f32>>,
    pub h_v_after: Vec<Vec<f32>>,
    pub h_spike: Vec<Vec<f32>>,
    pub gate: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
    /// Hidden state at t = -1 (before the first timestep of this chunk).
    /// Zeros for stateless calls, the previous chunk's final values for
    /// stateful BPTT. Backward uses it to compute correct gate gradient
    /// and `W_rec`/`W_gate_rec` gradient at t = 0; gradients across the chunk
    /// boundary are NOT propagated (DETACH).
    pub initial_h_v_after: Vec<f32>,
    pub initial_h_spike: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GatedState {
    pub h_v_after: Vec<f32>,
    pub h_spike: Vec<f32>,
    /// Sprint 38: SDSA running attention accumulator. Empty `Vec` when
    /// SDSA is not in use (per-layer state for non-top layers, or any
    /// layer when the network has no `sdsa: Option<SdsaCell>`). For
    /// the top layer with SDSA enabled, sized to that layer's hidden.
    pub attn_state: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GatedRecurrentSnn {
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_out: Vec<f32>,
    pub w_gate_in: Vec<f32>,
    pub w_gate_rec: Vec<f32>,
    pub b_h: Vec<f32>,
    pub b_o: Vec<f32>,
    pub b_gate: Vec<f32>,
    grads: GatedGrads,
    pub spike_fn: SpikeFn,
    adam_t: usize,
    adam_w_in: AdamGroup,
    adam_w_rec: AdamGroup,
    adam_w_out: AdamGroup,
    adam_w_gate_in: AdamGroup,
    adam_w_gate_rec: AdamGroup,
    adam_b_h: AdamGroup,
    adam_b_o: AdamGroup,
    adam_b_gate: AdamGroup,
}

impl GatedRecurrentSnn {
    pub fn new<R: Rng>(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        spike_fn: SpikeFn,
        rng: &mut R,
    ) -> Self {
        let scale_in = (2.0 / n_input as f32).sqrt();
        let scale_out = (2.0 / n_hidden as f32).sqrt();
        let scale_rec = 0.9 / (n_hidden as f32).sqrt();

        let w_in = (0..n_hidden * n_input)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_in)
            .collect();
        let w_rec = (0..n_hidden * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_rec)
            .collect();
        let w_out = (0..n_output * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_out)
            .collect();
        // Gate weights start small so initial sigmoid is near 0.5; bias 1.0
        // pushes initial decay toward sigmoid(1.0) ≈ 0.73 — "remember by
        // default", which is the LSTM-forget-gate-bias-1 trick (Jozefowicz
        // et al. 2015).
        let scale_gate_in = 0.1 * scale_in;
        let scale_gate_rec = 0.1 * scale_rec;
        let w_gate_in = (0..n_hidden * n_input)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_gate_in)
            .collect();
        let w_gate_rec = (0..n_hidden * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale_gate_rec)
            .collect();

        Self {
            n_input,
            n_hidden,
            n_output,
            grads: GatedGrads::zeros(n_input, n_hidden, n_output),
            w_in,
            w_rec,
            w_out,
            w_gate_in,
            w_gate_rec,
            b_h: vec![0.0; n_hidden],
            b_o: vec![0.0; n_output],
            b_gate: vec![1.0; n_hidden],
            spike_fn,
            adam_t: 0,
            adam_w_in: AdamGroup::zeros(n_hidden * n_input),
            adam_w_rec: AdamGroup::zeros(n_hidden * n_hidden),
            adam_w_out: AdamGroup::zeros(n_output * n_hidden),
            adam_w_gate_in: AdamGroup::zeros(n_hidden * n_input),
            adam_w_gate_rec: AdamGroup::zeros(n_hidden * n_hidden),
            adam_b_h: AdamGroup::zeros(n_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
            adam_b_gate: AdamGroup::zeros(n_hidden),
        }
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        let groups: [(&mut Vec<f32>, &Vec<f32>, &mut AdamGroup); 8] = [
            (&mut self.w_in, &self.grads.w_in, &mut self.adam_w_in),
            (&mut self.w_rec, &self.grads.w_rec, &mut self.adam_w_rec),
            (&mut self.w_out, &self.grads.w_out, &mut self.adam_w_out),
            (
                &mut self.w_gate_in,
                &self.grads.w_gate_in,
                &mut self.adam_w_gate_in,
            ),
            (
                &mut self.w_gate_rec,
                &self.grads.w_gate_rec,
                &mut self.adam_w_gate_rec,
            ),
            (&mut self.b_h, &self.grads.b_h, &mut self.adam_b_h),
            (&mut self.b_o, &self.grads.b_o, &mut self.adam_b_o),
            (&mut self.b_gate, &self.grads.b_gate, &mut self.adam_b_gate),
        ];
        for (params, grads, group) in groups {
            adam_apply(params, grads, group, config, bc1, bc2);
        }
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

    #[must_use]
    pub fn alloc_grads(&self) -> GatedGrads {
        GatedGrads::zeros(self.n_input, self.n_hidden, self.n_output)
    }

    /// Add an externally-accumulated gradient set (e.g., from a parallel
    /// batch worker) into `self.grads`.
    pub fn add_grads(&mut self, other: &GatedGrads) {
        self.grads.add(other);
    }

    #[must_use]
    pub fn fresh_state(&self) -> GatedState {
        GatedState {
            h_v_after: vec![0.0_f32; self.n_hidden],
            h_spike: vec![0.0_f32; self.n_hidden],
            attn_state: Vec::new(),
        }
    }

    /// Sparse byte-level LM forward. Logits are non-spiking linear
    /// projection of hidden spikes. Stateless: starts hidden state at
    /// zero. For TBPTT-with-detach across chunk boundaries see
    /// `forward_lm_bytes_stateful`.
    pub fn forward_lm_bytes(&self, input_bytes: &[u8]) -> GatedLmCache {
        let mut state = self.fresh_state();
        self.forward_lm_bytes_stateful(input_bytes, &mut state)
    }

    /// Stateful sparse byte-level LM forward (truncated BPTT). Hidden
    /// state at t = -1 comes from `state`; at end of forward, `state`
    /// is overwritten with the chunk's final hidden values for use as
    /// t = -1 of the next chunk. Gradients across the chunk boundary
    /// are detached automatically by `backward_lm_bytes_into`: weights
    /// receive correct local-time updates but no signal propagates back
    /// into the previous chunk.
    pub fn forward_lm_bytes_stateful(
        &self,
        input_bytes: &[u8],
        state: &mut GatedState,
    ) -> GatedLmCache {
        let t_steps = input_bytes.len();
        let h_dim = self.n_hidden;
        let i_dim = self.n_input;
        let o_dim = self.n_output;
        let theta = self.spike_fn.threshold;

        let initial_h_v_after = state.h_v_after.clone();
        let initial_h_spike = state.h_spike.clone();

        let mut h_v_after_prev = state.h_v_after.clone();
        let mut active_prev = active_indices(&state.h_spike);

        let mut h_v_pre_seq = Vec::with_capacity(t_steps);
        let mut h_v_after_seq = Vec::with_capacity(t_steps);
        let mut h_spike_seq = Vec::with_capacity(t_steps);
        let mut gate_seq = Vec::with_capacity(t_steps);
        let mut logits_seq = Vec::with_capacity(t_steps);

        for &byte in input_bytes {
            let b = byte as usize;
            debug_assert!(b < i_dim);

            // Gate pre-activation (sigmoid input).
            let mut gate_pre = vec![0.0_f32; h_dim];
            for hi in 0..h_dim {
                let row_offset = hi * h_dim;
                let mut acc = self.b_gate[hi] + self.w_gate_in[hi * i_dim + b];
                for &c in &active_prev {
                    acc += self.w_gate_rec[row_offset + c];
                }
                gate_pre[hi] = acc;
            }
            let gate: Vec<f32> = gate_pre.iter().map(|&v| sigmoid(v)).collect();

            // Membrane integration with adaptive decay.
            let mut h_v_pre_t = vec![0.0_f32; h_dim];
            for hi in 0..h_dim {
                let row_offset = hi * h_dim;
                let mut h_rec_val = 0.0_f32;
                for &c in &active_prev {
                    h_rec_val += self.w_rec[row_offset + c];
                }
                let h_in_val = self.w_in[hi * i_dim + b];
                h_v_pre_t[hi] = h_v_after_prev[hi] * gate[hi] + h_in_val + h_rec_val + self.b_h[hi];
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

            let mut logits_t = vec![0.0_f32; o_dim];
            for o in 0..o_dim {
                let row_offset = o * h_dim;
                let mut acc = self.b_o[o];
                for &c in &active_t {
                    acc += self.w_out[row_offset + c];
                }
                logits_t[o] = acc;
            }

            h_v_after_prev.clone_from(&h_v_after_t);
            active_prev = active_t;
            h_v_pre_seq.push(h_v_pre_t);
            h_v_after_seq.push(h_v_after_t);
            h_spike_seq.push(h_spike_t);
            gate_seq.push(gate);
            logits_seq.push(logits_t);
        }

        if let Some(last_h_v_after) = h_v_after_seq.last() {
            state.h_v_after.clone_from(last_h_v_after);
        }
        if let Some(last_h_spike) = h_spike_seq.last() {
            state.h_spike.clone_from(last_h_spike);
        }

        GatedLmCache {
            h_v_pre: h_v_pre_seq,
            h_v_after: h_v_after_seq,
            h_spike: h_spike_seq,
            gate: gate_seq,
            logits: logits_seq,
            initial_h_v_after,
            initial_h_spike,
        }
    }

    /// In-place backward writing into `self.grads`.
    pub fn backward_lm_bytes(
        &mut self,
        cache: &GatedLmCache,
        input_bytes: &[u8],
        targets: &[u8],
    ) -> f32 {
        let mut grads = std::mem::take(&mut self.grads);
        let loss = self.backward_lm_bytes_into(cache, input_bytes, targets, &mut grads);
        self.grads = grads;
        loss
    }

    /// Backward into a caller-provided gradient buffer (read-only on
    /// `self`). Required for parallel batch training where each worker
    /// owns its own gradient accumulator.
    pub fn backward_lm_bytes_into(
        &self,
        cache: &GatedLmCache,
        input_bytes: &[u8],
        targets: &[u8],
        grads: &mut GatedGrads,
    ) -> f32 {
        let t_steps = input_bytes.len();
        debug_assert_eq!(targets.len(), t_steps);
        let h_dim = self.n_hidden;
        let i_dim = self.n_input;

        // d_loss / d_h_v_after_prev (carries to t-1)
        let mut d_h_v_after_carry = vec![0.0_f32; h_dim];
        // d_loss / d_h_spike_{t-1} from this step's recurrent and gate-recurrent paths
        let mut d_h_spike_recurrent = vec![0.0_f32; h_dim];
        let mut total_loss = 0.0_f32;

        for t in (0..t_steps).rev() {
            let (loss_t, d_logits) = cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            let active_t = active_indices(&cache.h_spike[t]);

            // grad_w_out and d_h_spike via output.
            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                grads.b_o[o] += d;
                let row_offset = o * h_dim;
                for &h in &active_t {
                    grads.w_out[row_offset + h] += d;
                }
            }
            let d_h_spike_from_out = matvec_transpose(&self.w_out, &d_logits, self.n_output, h_dim);

            // Total d_h_spike at step t = output-side + recurrent-feedback-from-step-(t+1).
            let d_h_spike_t: Vec<f32> = (0..h_dim)
                .map(|i| d_h_spike_from_out[i] + d_h_spike_recurrent[i])
                .collect();

            // d_h_v_pre[t] = d_h_spike_t * surrogate(h_v_pre[t]) + d_h_v_after_carry from t+1
            // (h_v_after at step t flows into membrane at step t+1 multiplied by gate at t+1).
            let mut d_h_v_pre_t = vec![0.0_f32; h_dim];
            for i in 0..h_dim {
                d_h_v_pre_t[i] = d_h_spike_t[i] * self.spike_fn.backward(cache.h_v_pre[t][i])
                    + d_h_v_after_carry[i];
            }

            // grad_b_h, grad_w_in (column update for byte_t), grad_w_rec (sparse over active_prev).
            // For TBPTT-with-detach: at t=0 we still update grad_w_rec /
            // grad_w_gate_rec via cache.initial_h_spike (local-time gradient
            // is valid); only the carry-over into step t-1 is dropped (which
            // we already handle by simply exiting the loop after t=0).
            let active_prev = if t == 0 {
                active_indices(&cache.initial_h_spike)
            } else {
                active_indices(&cache.h_spike[t - 1])
            };
            let byte_t = input_bytes[t] as usize;
            for h in 0..h_dim {
                let d = d_h_v_pre_t[h];
                if d != 0.0 {
                    grads.b_h[h] += d;
                    grads.w_in[h * i_dim + byte_t] += d;
                    let row_offset = h * h_dim;
                    for &j in &active_prev {
                        grads.w_rec[row_offset + j] += d;
                    }
                }
            }

            // Gate gradient. d_loss / d_gate[h] = d_h_v_pre[t][h] * h_v_after_prev[h]
            let h_v_after_prev: &[f32] = if t == 0 {
                &cache.initial_h_v_after
            } else {
                &cache.h_v_after[t - 1]
            };
            let mut d_gate_pre = vec![0.0_f32; h_dim];
            for h in 0..h_dim {
                let d_gate_h = d_h_v_pre_t[h] * h_v_after_prev[h];
                let g = cache.gate[t][h];
                d_gate_pre[h] = d_gate_h * g * (1.0 - g);
            }

            // grad_b_gate, grad_w_gate_in (column for byte_t), grad_w_gate_rec (sparse).
            for h in 0..h_dim {
                let d = d_gate_pre[h];
                if d == 0.0 {
                    continue;
                }
                grads.b_gate[h] += d;
                grads.w_gate_in[h * i_dim + byte_t] += d;
                let row_offset = h * h_dim;
                for &j in &active_prev {
                    grads.w_gate_rec[row_offset + j] += d;
                }
            }

            // Carryover into step t-1:
            //   d_h_v_after[t-1] = d_h_v_pre[t] * gate[t]
            //   d_h_spike_recurrent[t-1] = W_rec.T @ d_h_v_pre[t]
            //                            + W_gate_rec.T @ d_gate_pre
            for h in 0..h_dim {
                d_h_v_after_carry[h] = d_h_v_pre_t[h] * cache.gate[t][h];
            }
            let d_from_rec = matvec_transpose(&self.w_rec, &d_h_v_pre_t, h_dim, h_dim);
            let d_from_gate_rec = matvec_transpose(&self.w_gate_rec, &d_gate_pre, h_dim, h_dim);
            for h in 0..h_dim {
                d_h_spike_recurrent[h] = d_from_rec[h] + d_from_gate_rec[h];
            }
        }

        total_loss
    }

    /// Streaming inference for a single byte. State persists across calls.
    pub fn step_lm_byte(&self, byte: u8, state: &mut GatedState) -> Vec<f32> {
        let h_dim = self.n_hidden;
        let i_dim = self.n_input;
        let theta = self.spike_fn.threshold;
        let b = byte as usize;

        let active_prev = active_indices(&state.h_spike);

        let mut gate_pre = vec![0.0_f32; h_dim];
        for hi in 0..h_dim {
            let row_offset = hi * h_dim;
            let mut acc = self.b_gate[hi] + self.w_gate_in[hi * i_dim + b];
            for &c in &active_prev {
                acc += self.w_gate_rec[row_offset + c];
            }
            gate_pre[hi] = acc;
        }
        let gate: Vec<f32> = gate_pre.iter().map(|&v| sigmoid(v)).collect();

        let mut h_v_pre = vec![0.0_f32; h_dim];
        for hi in 0..h_dim {
            let row_offset = hi * h_dim;
            let mut h_rec_val = 0.0_f32;
            for &c in &active_prev {
                h_rec_val += self.w_rec[row_offset + c];
            }
            let h_in_val = self.w_in[hi * i_dim + b];
            h_v_pre[hi] = state.h_v_after[hi] * gate[hi] + h_in_val + h_rec_val + self.b_h[hi];
        }
        let h_spike: Vec<f32> = h_v_pre.iter().map(|&v| self.spike_fn.forward(v)).collect();
        let h_v_after: Vec<f32> = h_v_pre
            .iter()
            .zip(h_spike.iter())
            .map(|(&v, &s)| v - theta * s)
            .collect();

        let mut logits = self.b_o.clone();
        for o in 0..self.n_output {
            let row_offset = o * h_dim;
            for (h, &s) in h_spike.iter().enumerate() {
                if s != 0.0 {
                    logits[o] += self.w_out[row_offset + h];
                }
            }
        }

        state.h_v_after = h_v_after;
        state.h_spike = h_spike;
        logits
    }

    #[must_use]
    pub const fn grads(&self) -> &GatedGrads {
        &self.grads
    }

    /// Train one mini-batch in parallel: forward + backward each
    /// `(input, target)` pair on its own thread (rayon over batch dim),
    /// then sum the per-thread gradients into `self.grads`. Caller is
    /// responsible for `clip_grads` + `step_adam` after this returns.
    /// Returns total loss summed over all batch samples.
    ///
    /// # Panics
    /// Panics if `batch_inputs.len() != batch_targets.len()`.
    pub fn forward_backward_batch(
        &mut self,
        batch_inputs: &[&[u8]],
        batch_targets: &[&[u8]],
    ) -> f32 {
        let mut states: Vec<GatedState> = (0..batch_inputs.len())
            .map(|_| self.fresh_state())
            .collect();
        self.forward_backward_batch_stateful(batch_inputs, batch_targets, &mut states)
    }

    /// Stateful batched training: each batch slot has its own
    /// `GatedState` carried across calls. Use this for TBPTT-with-detach,
    /// where each batch position streams contiguous chunks of a long
    /// sequence and hidden state persists across chunk boundaries.
    /// Caller manages the `Vec<GatedState>` (one per batch slot) — this
    /// method reads the incoming state, runs forward+backward+grad
    /// accumulation in parallel, and writes the chunk's final state back.
    ///
    /// # Panics
    /// Panics if `batch_inputs.len() != batch_targets.len()` or
    /// `batch_inputs.len() != states.len()`.
    pub fn forward_backward_batch_stateful(
        &mut self,
        batch_inputs: &[&[u8]],
        batch_targets: &[&[u8]],
        states: &mut [GatedState],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        if batch_inputs.is_empty() {
            return 0.0;
        }
        let weights = &*self;
        let results: Vec<(f32, GatedGrads)> = batch_inputs
            .par_iter()
            .zip(batch_targets.par_iter())
            .zip(states.par_iter_mut())
            .map(|((inputs, targets), state)| {
                let mut grads = weights.alloc_grads();
                let cache = weights.forward_lm_bytes_stateful(inputs, state);
                let loss = weights.backward_lm_bytes_into(&cache, inputs, targets, &mut grads);
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
