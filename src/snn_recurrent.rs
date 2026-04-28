#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::too_many_arguments
)]

//! Recurrent surrogate-gradient SNN with hidden→hidden weights and biases.
//!
//! BPTT routes gradient into `h_v_pre[t-1]` via three paths: decay carryover,
//! recurrent feedback through `W_rec`, and output-side via `W_out.T`.

use rand::Rng;

use crate::linalg::{matvec, matvec_transpose};
use crate::optimizer::{adam_apply, bias_correction, sgd_apply, AdamConfig, AdamGroup};
use crate::snn_loss::{argmax, cross_entropy_with_grad};
use crate::surrogate::SpikeFn;

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

    pub fn zero_grad(&mut self) {
        self.grad_w_in.fill(0.0);
        self.grad_w_rec.fill(0.0);
        self.grad_w_out.fill(0.0);
        self.grad_b_h.fill(0.0);
        self.grad_b_o.fill(0.0);
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
