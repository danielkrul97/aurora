#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::if_not_else
)]

//! Recurrent surrogate-gradient SNN with hidden→hidden weights and biases.
//!
//! BPTT routes gradient into `h_v_pre[t-1]` via three paths: decay carryover,
//! recurrent feedback through `W_rec`, and output-side via `W_out.T`.

use rand::Rng;

use crate::linalg::{active_indices, matvec, matvec_transpose};
use crate::optimizer::{adam_apply, bias_correction, sgd_apply, AdamConfig, AdamGroup};
use crate::snn_loss::{argmax, cross_entropy_with_grad};
use crate::surrogate::SpikeFn;

/// Persistent activation state for streaming inference (char-LM
/// generation). Created via `RecurrentSurrogateSnn::fresh_state`.
#[derive(Clone, Debug)]
pub struct SnnState {
    pub h_v_after: Vec<f32>,
    pub o_v_after: Vec<f32>,
    pub h_spike: Vec<f32>,
}

/// LM-mode forward cache.
///
/// Hidden dynamics same as classification, but output is a non-spiking
/// linear projection — `logits[t] = W_out @ h_spike[t] + b_o`. The
/// classification path uses `forward` and emits `ForwardCache`; do not
/// mix.
#[derive(Clone, Debug)]
pub struct LmCache {
    pub h_v_pre: Vec<Vec<f32>>,
    pub h_v_after: Vec<Vec<f32>>,
    pub h_spike: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub h_v_pre: Vec<Vec<f32>>,
    pub h_v_after: Vec<Vec<f32>>,
    pub h_spike: Vec<Vec<f32>>,
    pub o_v_pre: Vec<Vec<f32>>,
    pub o_v_after: Vec<Vec<f32>>,
    pub o_spike: Vec<Vec<f32>>,
    pub spike_counts: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct RecurrentSurrogateSnn {
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_out: Vec<f32>,
    pub b_h: Vec<f32>,
    pub b_o: Vec<f32>,
    grad_w_in: Vec<f32>,
    grad_w_rec: Vec<f32>,
    grad_w_out: Vec<f32>,
    grad_b_h: Vec<f32>,
    grad_b_o: Vec<f32>,
    pub spike_fn: SpikeFn,
    pub tau: f32,
    pub dt: f32,
    adam_t: usize,
    adam_w_in: AdamGroup,
    adam_w_rec: AdamGroup,
    adam_w_out: AdamGroup,
    adam_b_h: AdamGroup,
    adam_b_o: AdamGroup,
}

impl RecurrentSurrogateSnn {
    pub fn new<R: Rng>(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        spike_fn: SpikeFn,
        tau: f32,
        dt: f32,
        rng: &mut R,
    ) -> Self {
        let scale_in = (2.0 / n_input as f32).sqrt();
        let scale_out = (2.0 / n_hidden as f32).sqrt();
        // 0.9/√N keeps spectral radius near 1 — fast info propagation across
        // timesteps while staying just under the explosion edge.
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
        Self {
            n_input,
            n_hidden,
            n_output,
            grad_w_in: vec![0.0; n_hidden * n_input],
            grad_w_rec: vec![0.0; n_hidden * n_hidden],
            grad_w_out: vec![0.0; n_output * n_hidden],
            grad_b_h: vec![0.0; n_hidden],
            grad_b_o: vec![0.0; n_output],
            w_in,
            w_rec,
            w_out,
            b_h: vec![0.0; n_hidden],
            b_o: vec![0.0; n_output],
            spike_fn,
            tau,
            dt,
            adam_t: 0,
            adam_w_in: AdamGroup::zeros(n_hidden * n_input),
            adam_w_rec: AdamGroup::zeros(n_hidden * n_hidden),
            adam_w_out: AdamGroup::zeros(n_output * n_hidden),
            adam_b_h: AdamGroup::zeros(n_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
        }
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        let groups: [(&mut Vec<f32>, &Vec<f32>, &mut AdamGroup); 5] = [
            (&mut self.w_in, &self.grad_w_in, &mut self.adam_w_in),
            (&mut self.w_rec, &self.grad_w_rec, &mut self.adam_w_rec),
            (&mut self.w_out, &self.grad_w_out, &mut self.adam_w_out),
            (&mut self.b_h, &self.grad_b_h, &mut self.adam_b_h),
            (&mut self.b_o, &self.grad_b_o, &mut self.adam_b_o),
        ];
        for (params, grads, group) in groups {
            adam_apply(params, grads, group, config, bc1, bc2);
        }
    }

    /// Read-only access to accumulated gradients (set by `backward_lm` /
    /// `backward_lm_bytes` / `backward`, cleared by `zero_grad`). Useful
    /// for parity tests and gradient-norm diagnostics.
    #[must_use]
    pub fn grad_w_in(&self) -> &[f32] {
        &self.grad_w_in
    }

    #[must_use]
    pub fn grad_w_rec(&self) -> &[f32] {
        &self.grad_w_rec
    }

    #[must_use]
    pub fn grad_w_out(&self) -> &[f32] {
        &self.grad_w_out
    }

    #[must_use]
    pub fn grad_b_h(&self) -> &[f32] {
        &self.grad_b_h
    }

    #[must_use]
    pub fn grad_b_o(&self) -> &[f32] {
        &self.grad_b_o
    }

    pub fn zero_grad(&mut self) {
        self.grad_w_in.fill(0.0);
        self.grad_w_rec.fill(0.0);
        self.grad_w_out.fill(0.0);
        self.grad_b_h.fill(0.0);
        self.grad_b_o.fill(0.0);
    }

    /// Persistent hidden+output state across `step` calls. Used by the
    /// generation loop in char-LM where we feed bytes one at a time and
    /// want soma membranes to carry over.
    #[must_use]
    pub fn fresh_state(&self) -> SnnState {
        SnnState {
            h_v_after: vec![0.0_f32; self.n_hidden],
            o_v_after: vec![0.0_f32; self.n_output],
            h_spike: vec![0.0_f32; self.n_hidden],
        }
    }

    /// Single-timestep forward that mutates `state` in place. Returns
    /// `o_v_pre` (logits-like) and `o_spike` (binary fires) for this step.
    /// Useful for autoregressive sampling — the training loop should keep
    /// using `forward` for full BPTT.
    pub fn step(&self, input: &[f32], state: &mut SnnState) -> (Vec<f32>, Vec<f32>) {
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;

        let h_in = matvec(&self.w_in, input, self.n_hidden, self.n_input);
        let h_rec = matvec(&self.w_rec, &state.h_spike, self.n_hidden, self.n_hidden);
        let h_v_pre: Vec<f32> = (0..self.n_hidden)
            .map(|i| state.h_v_after[i] * decay + h_in[i] + h_rec[i] + self.b_h[i])
            .collect();
        let h_spike: Vec<f32> = h_v_pre.iter().map(|&v| self.spike_fn.forward(v)).collect();
        let h_v_after: Vec<f32> = h_v_pre
            .iter()
            .zip(h_spike.iter())
            .map(|(&v, &s)| v - theta * s)
            .collect();

        let o_lin = matvec(&self.w_out, &h_spike, self.n_output, self.n_hidden);
        let o_v_pre: Vec<f32> = (0..self.n_output)
            .map(|i| state.o_v_after[i] * decay + o_lin[i] + self.b_o[i])
            .collect();
        let o_spike: Vec<f32> = o_v_pre.iter().map(|&v| self.spike_fn.forward(v)).collect();
        let o_v_after: Vec<f32> = o_v_pre
            .iter()
            .zip(o_spike.iter())
            .map(|(&v, &s)| v - theta * s)
            .collect();

        state.h_v_after = h_v_after;
        state.o_v_after = o_v_after;
        state.h_spike = h_spike;
        (o_v_pre, o_spike)
    }

    /// LM forward: hidden dynamics identical to classification mode, but
    /// output is non-spiking linear projection (treated as logits per
    /// timestep). Standard SNN LM design — output spikes are noisy
    /// targets for next-token prediction.
    pub fn forward_lm(&self, input_seq: &[Vec<f32>]) -> LmCache {
        let t_steps = input_seq.len();
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;

        let mut h_v_after_prev = vec![0.0_f32; self.n_hidden];
        let mut h_spike_prev = vec![0.0_f32; self.n_hidden];

        let mut h_v_pre = Vec::with_capacity(t_steps);
        let mut h_v_after = Vec::with_capacity(t_steps);
        let mut h_spike = Vec::with_capacity(t_steps);
        let mut logits = Vec::with_capacity(t_steps);

        for input_t in input_seq {
            let h_in = matvec(&self.w_in, input_t, self.n_hidden, self.n_input);
            let h_rec = matvec(&self.w_rec, &h_spike_prev, self.n_hidden, self.n_hidden);
            let h_v_pre_t: Vec<f32> = (0..self.n_hidden)
                .map(|i| h_v_after_prev[i] * decay + h_in[i] + h_rec[i] + self.b_h[i])
                .collect();
            let h_spike_t: Vec<f32> = h_v_pre_t
                .iter()
                .map(|&v| self.spike_fn.forward(v))
                .collect();
            let h_v_after_t: Vec<f32> = h_v_pre_t
                .iter()
                .zip(h_spike_t.iter())
                .map(|(&v, &s)| v - theta * s)
                .collect();

            let o_lin = matvec(&self.w_out, &h_spike_t, self.n_output, self.n_hidden);
            let logits_t: Vec<f32> = (0..self.n_output).map(|i| o_lin[i] + self.b_o[i]).collect();

            h_v_after_prev.clone_from(&h_v_after_t);
            h_spike_prev.clone_from(&h_spike_t);
            h_v_pre.push(h_v_pre_t);
            h_v_after.push(h_v_after_t);
            h_spike.push(h_spike_t);
            logits.push(logits_t);
        }

        LmCache {
            h_v_pre,
            h_v_after,
            h_spike,
            logits,
        }
    }

    /// LM backward: per-step cross-entropy on `targets[t]` against
    /// `cache.logits[t]`. Accumulates gradients; caller is responsible
    /// for `zero_grad`. Returns total loss in **nats** so the caller can
    /// convert to bits-per-byte.
    pub fn backward_lm(&mut self, cache: &LmCache, input_seq: &[Vec<f32>], targets: &[u8]) -> f32 {
        let t_steps = input_seq.len();
        debug_assert_eq!(targets.len(), t_steps);
        let decay = (-self.dt / self.tau).exp();

        let mut d_h_v_pre_next = vec![0.0_f32; self.n_hidden];
        let mut d_h_spike_recurrent = vec![0.0_f32; self.n_hidden];
        let mut total_loss = 0.0_f32;

        for t in (0..t_steps).rev() {
            let (loss_t, d_logits) = cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_o[o] += d;
                for h in 0..self.n_hidden {
                    self.grad_w_out[o * self.n_hidden + h] += d * cache.h_spike[t][h];
                }
            }
            let d_h_spike_from_out =
                matvec_transpose(&self.w_out, &d_logits, self.n_output, self.n_hidden);

            let d_h_spike_t: Vec<f32> = (0..self.n_hidden)
                .map(|i| d_h_spike_from_out[i] + d_h_spike_recurrent[i])
                .collect();

            let mut d_h_v_pre_t = d_h_v_pre_next.clone();
            for i in 0..self.n_hidden {
                d_h_v_pre_t[i] += d_h_spike_t[i] * self.spike_fn.backward(cache.h_v_pre[t][i]);
            }
            for h in 0..self.n_hidden {
                let d = d_h_v_pre_t[h];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_h[h] += d;
                for i in 0..self.n_input {
                    self.grad_w_in[h * self.n_input + i] += d * input_seq[t][i];
                }
                if t > 0 {
                    for j in 0..self.n_hidden {
                        self.grad_w_rec[h * self.n_hidden + j] += d * cache.h_spike[t - 1][j];
                    }
                }
            }

            for i in 0..self.n_hidden {
                d_h_v_pre_next[i] = d_h_v_pre_t[i] * decay;
            }
            d_h_spike_recurrent =
                matvec_transpose(&self.w_rec, &d_h_v_pre_t, self.n_hidden, self.n_hidden);
        }

        let _ = decay;
        total_loss
    }

    /// Sparse LM forward for byte-level input. Identical numerics to
    /// `forward_lm` (one-hot inputs and binary spikes only contribute
    /// non-zero terms in IEEE 754, so skipping zero summands is exact),
    /// but ~3-4× faster: `W_in @ onehot(b)` collapses to a column read,
    /// `W_rec @ h_spike` and `W_out @ h_spike` skip non-firing columns.
    pub fn forward_lm_bytes(&self, input_bytes: &[u8]) -> LmCache {
        let t_steps = input_bytes.len();
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;
        let h_dim = self.n_hidden;
        let i_dim = self.n_input;
        let o_dim = self.n_output;

        let mut h_v_after_prev = vec![0.0_f32; h_dim];
        let mut active_prev: Vec<usize> = Vec::with_capacity(h_dim);

        let mut h_v_pre_seq = Vec::with_capacity(t_steps);
        let mut h_v_after_seq = Vec::with_capacity(t_steps);
        let mut h_spike_seq = Vec::with_capacity(t_steps);
        let mut logits_seq = Vec::with_capacity(t_steps);

        for &byte in input_bytes {
            let b = byte as usize;
            debug_assert!(b < i_dim);
            let mut h_v_pre_t = vec![0.0_f32; h_dim];
            for hi in 0..h_dim {
                let row_offset = hi * h_dim;
                let mut h_rec_val = 0.0_f32;
                for &c in &active_prev {
                    h_rec_val += self.w_rec[row_offset + c];
                }
                let h_in_val = self.w_in[hi * i_dim + b];
                h_v_pre_t[hi] = h_v_after_prev[hi] * decay + h_in_val + h_rec_val + self.b_h[hi];
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
            logits_seq.push(logits_t);
        }

        LmCache {
            h_v_pre: h_v_pre_seq,
            h_v_after: h_v_after_seq,
            h_spike: h_spike_seq,
            logits: logits_seq,
        }
    }

    /// Sparse LM backward. Numerically identical to `backward_lm` for
    /// one-hot inputs and binary spikes; skips zero-valued multiplications
    /// in `grad_w_in` (only the byte's column updates), `grad_w_rec`, and
    /// `grad_w_out` (only firing-hidden columns update).
    pub fn backward_lm_bytes(
        &mut self,
        cache: &LmCache,
        input_bytes: &[u8],
        targets: &[u8],
    ) -> f32 {
        let t_steps = input_bytes.len();
        debug_assert_eq!(targets.len(), t_steps);
        let decay = (-self.dt / self.tau).exp();
        let h_dim = self.n_hidden;
        let i_dim = self.n_input;

        let mut d_h_v_pre_next = vec![0.0_f32; h_dim];
        let mut d_h_spike_recurrent = vec![0.0_f32; h_dim];
        let mut total_loss = 0.0_f32;

        for t in (0..t_steps).rev() {
            let (loss_t, d_logits) = cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            let active_t = active_indices(&cache.h_spike[t]);

            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_o[o] += d;
                let row_offset = o * h_dim;
                for &h in &active_t {
                    self.grad_w_out[row_offset + h] += d;
                }
            }
            let d_h_spike_from_out = matvec_transpose(&self.w_out, &d_logits, self.n_output, h_dim);

            let d_h_spike_t: Vec<f32> = (0..h_dim)
                .map(|i| d_h_spike_from_out[i] + d_h_spike_recurrent[i])
                .collect();

            let mut d_h_v_pre_t = d_h_v_pre_next.clone();
            for i in 0..h_dim {
                d_h_v_pre_t[i] += d_h_spike_t[i] * self.spike_fn.backward(cache.h_v_pre[t][i]);
            }
            let active_prev: Vec<usize> = if t == 0 {
                Vec::new()
            } else {
                active_indices(&cache.h_spike[t - 1])
            };
            let byte_t = input_bytes[t] as usize;
            for h in 0..h_dim {
                let d = d_h_v_pre_t[h];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_h[h] += d;
                self.grad_w_in[h * i_dim + byte_t] += d;
                if t > 0 {
                    let row_offset = h * h_dim;
                    for &j in &active_prev {
                        self.grad_w_rec[row_offset + j] += d;
                    }
                }
            }

            for i in 0..h_dim {
                d_h_v_pre_next[i] = d_h_v_pre_t[i] * decay;
            }
            d_h_spike_recurrent = matvec_transpose(&self.w_rec, &d_h_v_pre_t, h_dim, h_dim);
        }

        let _ = decay;
        total_loss
    }

    /// LM streaming step: feed one input vector, return logits for next-
    /// token prediction. State persists across calls. Use this for
    /// generation only — training should batch with `forward_lm`.
    pub fn step_lm(&self, input: &[f32], state: &mut SnnState) -> Vec<f32> {
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;

        let h_in = matvec(&self.w_in, input, self.n_hidden, self.n_input);
        let h_rec = matvec(&self.w_rec, &state.h_spike, self.n_hidden, self.n_hidden);
        let h_v_pre: Vec<f32> = (0..self.n_hidden)
            .map(|i| state.h_v_after[i] * decay + h_in[i] + h_rec[i] + self.b_h[i])
            .collect();
        let h_spike: Vec<f32> = h_v_pre.iter().map(|&v| self.spike_fn.forward(v)).collect();
        let h_v_after: Vec<f32> = h_v_pre
            .iter()
            .zip(h_spike.iter())
            .map(|(&v, &s)| v - theta * s)
            .collect();

        let o_lin = matvec(&self.w_out, &h_spike, self.n_output, self.n_hidden);
        let logits: Vec<f32> = (0..self.n_output).map(|i| o_lin[i] + self.b_o[i]).collect();

        state.h_v_after = h_v_after;
        state.h_spike = h_spike;
        logits
    }

    pub fn forward(&self, input_spikes: &[Vec<f32>]) -> ForwardCache {
        let t_steps = input_spikes.len();
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;

        let mut h_v_after_prev = vec![0.0_f32; self.n_hidden];
        let mut o_v_after_prev = vec![0.0_f32; self.n_output];
        let mut h_spike_prev = vec![0.0_f32; self.n_hidden];

        let mut h_v_pre = Vec::with_capacity(t_steps);
        let mut h_v_after = Vec::with_capacity(t_steps);
        let mut h_spike = Vec::with_capacity(t_steps);
        let mut o_v_pre = Vec::with_capacity(t_steps);
        let mut o_v_after = Vec::with_capacity(t_steps);
        let mut o_spike = Vec::with_capacity(t_steps);
        let mut spike_counts = vec![0.0_f32; self.n_output];

        for input_t in input_spikes {
            let h_in = matvec(&self.w_in, input_t, self.n_hidden, self.n_input);
            let h_rec = matvec(&self.w_rec, &h_spike_prev, self.n_hidden, self.n_hidden);
            let h_v_pre_t: Vec<f32> = (0..self.n_hidden)
                .map(|i| h_v_after_prev[i] * decay + h_in[i] + h_rec[i] + self.b_h[i])
                .collect();
            let h_spike_t: Vec<f32> = h_v_pre_t
                .iter()
                .map(|&v| self.spike_fn.forward(v))
                .collect();
            let h_v_after_t: Vec<f32> = h_v_pre_t
                .iter()
                .zip(h_spike_t.iter())
                .map(|(&v, &s)| v - theta * s)
                .collect();

            let o_lin = matvec(&self.w_out, &h_spike_t, self.n_output, self.n_hidden);
            let o_v_pre_t: Vec<f32> = (0..self.n_output)
                .map(|i| o_v_after_prev[i] * decay + o_lin[i] + self.b_o[i])
                .collect();
            let o_spike_t: Vec<f32> = o_v_pre_t
                .iter()
                .map(|&v| self.spike_fn.forward(v))
                .collect();
            let o_v_after_t: Vec<f32> = o_v_pre_t
                .iter()
                .zip(o_spike_t.iter())
                .map(|(&v, &s)| v - theta * s)
                .collect();

            for (acc, &s) in spike_counts.iter_mut().zip(o_spike_t.iter()) {
                *acc += s;
            }

            h_v_after_prev.clone_from(&h_v_after_t);
            o_v_after_prev.clone_from(&o_v_after_t);
            h_spike_prev.clone_from(&h_spike_t);
            h_v_pre.push(h_v_pre_t);
            h_v_after.push(h_v_after_t);
            h_spike.push(h_spike_t);
            o_v_pre.push(o_v_pre_t);
            o_v_after.push(o_v_after_t);
            o_spike.push(o_spike_t);
        }

        ForwardCache {
            h_v_pre,
            h_v_after,
            h_spike,
            o_v_pre,
            o_v_after,
            o_spike,
            spike_counts,
        }
    }

    pub fn backward(
        &mut self,
        cache: &ForwardCache,
        input_spikes: &[Vec<f32>],
        target: usize,
    ) -> (f32, usize) {
        let t_steps = input_spikes.len();
        let decay = (-self.dt / self.tau).exp();

        let (loss, d_spike_counts) = cross_entropy_with_grad(&cache.spike_counts, target);
        let predicted = argmax(&cache.spike_counts);

        let mut d_h_v_pre_next = vec![0.0_f32; self.n_hidden];
        let mut d_o_v_pre_next = vec![0.0_f32; self.n_output];
        let mut d_h_spike_recurrent = vec![0.0_f32; self.n_hidden];

        for t in (0..t_steps).rev() {
            let mut d_o_v_pre_t = d_o_v_pre_next.clone();
            for i in 0..self.n_output {
                d_o_v_pre_t[i] += d_spike_counts[i] * self.spike_fn.backward(cache.o_v_pre[t][i]);
            }
            for o in 0..self.n_output {
                let d = d_o_v_pre_t[o];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_o[o] += d;
                for h in 0..self.n_hidden {
                    self.grad_w_out[o * self.n_hidden + h] += d * cache.h_spike[t][h];
                }
            }
            let d_h_spike_from_out =
                matvec_transpose(&self.w_out, &d_o_v_pre_t, self.n_output, self.n_hidden);

            let d_h_spike_t: Vec<f32> = (0..self.n_hidden)
                .map(|i| d_h_spike_from_out[i] + d_h_spike_recurrent[i])
                .collect();

            let mut d_h_v_pre_t = d_h_v_pre_next.clone();
            for i in 0..self.n_hidden {
                d_h_v_pre_t[i] += d_h_spike_t[i] * self.spike_fn.backward(cache.h_v_pre[t][i]);
            }
            for h in 0..self.n_hidden {
                let d = d_h_v_pre_t[h];
                if d == 0.0 {
                    continue;
                }
                self.grad_b_h[h] += d;
                for i in 0..self.n_input {
                    self.grad_w_in[h * self.n_input + i] += d * input_spikes[t][i];
                }
                if t > 0 {
                    for j in 0..self.n_hidden {
                        self.grad_w_rec[h * self.n_hidden + j] += d * cache.h_spike[t - 1][j];
                    }
                }
            }

            for i in 0..self.n_output {
                d_o_v_pre_next[i] = d_o_v_pre_t[i] * decay;
            }
            for i in 0..self.n_hidden {
                d_h_v_pre_next[i] = d_h_v_pre_t[i] * decay;
            }
            d_h_spike_recurrent =
                matvec_transpose(&self.w_rec, &d_h_v_pre_t, self.n_hidden, self.n_hidden);
        }

        (loss, predicted)
    }

    pub fn clip_grads(&mut self, max_norm: f32) {
        let groups: [&[f32]; 5] = [
            &self.grad_w_in,
            &self.grad_w_rec,
            &self.grad_w_out,
            &self.grad_b_h,
            &self.grad_b_o,
        ];
        let sq: f32 = groups.iter().flat_map(|g| g.iter()).map(|x| x * x).sum();
        let norm = sq.sqrt();
        if norm <= max_norm || norm == 0.0 {
            return;
        }
        let scale = max_norm / norm;
        let groups_mut: [&mut [f32]; 5] = [
            &mut self.grad_w_in,
            &mut self.grad_w_rec,
            &mut self.grad_w_out,
            &mut self.grad_b_h,
            &mut self.grad_b_o,
        ];
        for grads in groups_mut {
            for g in grads {
                *g *= scale;
            }
        }
    }

    pub fn step_sgd(&mut self, lr: f32) {
        sgd_apply(&mut self.w_in, &self.grad_w_in, lr);
        sgd_apply(&mut self.w_rec, &self.grad_w_rec, lr);
        sgd_apply(&mut self.w_out, &self.grad_w_out, lr);
        sgd_apply(&mut self.b_h, &self.grad_b_h, lr);
        sgd_apply(&mut self.b_o, &self.grad_b_o, lr);
    }
}
