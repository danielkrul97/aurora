#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate
)]

//! Two-layer dense SNN with surrogate-gradient BPTT.
//! Subtraction reset; loss = `CE(softmax(spike_counts), label)`.

use rand::Rng;

use crate::linalg::{matvec, matvec_transpose};
use crate::optimizer::sgd_apply;
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
pub struct SurrogateSnn {
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    grad_w1: Vec<f32>,
    grad_w2: Vec<f32>,
    pub spike_fn: SpikeFn,
    pub tau: f32,
    pub dt: f32,
}

impl SurrogateSnn {
    pub fn new<R: Rng>(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        spike_fn: SpikeFn,
        tau: f32,
        dt: f32,
        rng: &mut R,
    ) -> Self {
        let scale1 = (2.0 / n_input as f32).sqrt();
        let scale2 = (2.0 / n_hidden as f32).sqrt();
        let w1 = (0..n_hidden * n_input)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale1)
            .collect();
        let w2 = (0..n_output * n_hidden)
            .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale2)
            .collect();
        Self {
            n_input,
            n_hidden,
            n_output,
            grad_w1: vec![0.0; n_hidden * n_input],
            grad_w2: vec![0.0; n_output * n_hidden],
            w1,
            w2,
            spike_fn,
            tau,
            dt,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_w1.fill(0.0);
        self.grad_w2.fill(0.0);
    }

    pub fn forward(&self, input_spikes: &[Vec<f32>]) -> ForwardCache {
        let t_steps = input_spikes.len();
        let decay = (-self.dt / self.tau).exp();
        let theta = self.spike_fn.threshold;

        let mut h_v_after_prev = vec![0.0_f32; self.n_hidden];
        let mut o_v_after_prev = vec![0.0_f32; self.n_output];

        let mut h_v_pre = Vec::with_capacity(t_steps);
        let mut h_v_after = Vec::with_capacity(t_steps);
        let mut h_spike = Vec::with_capacity(t_steps);
        let mut o_v_pre = Vec::with_capacity(t_steps);
        let mut o_v_after = Vec::with_capacity(t_steps);
        let mut o_spike = Vec::with_capacity(t_steps);
        let mut spike_counts = vec![0.0_f32; self.n_output];

        for input_t in input_spikes {
            let h_lin = matvec(&self.w1, input_t, self.n_hidden, self.n_input);
            let h_v_pre_t: Vec<f32> = h_v_after_prev
                .iter()
                .zip(h_lin.iter())
                .map(|(&prev, &lin)| prev * decay + lin)
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

            let o_lin = matvec(&self.w2, &h_spike_t, self.n_output, self.n_hidden);
            let o_v_pre_t: Vec<f32> = o_v_after_prev
                .iter()
                .zip(o_lin.iter())
                .map(|(&prev, &lin)| prev * decay + lin)
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

        for t in (0..t_steps).rev() {
            let mut d_o_v_pre_t = d_o_v_pre_next.clone();
            for (i, d) in d_o_v_pre_t.iter_mut().enumerate() {
                *d += d_spike_counts[i] * self.spike_fn.backward(cache.o_v_pre[t][i]);
            }
            for o in 0..self.n_output {
                let d = d_o_v_pre_t[o];
                if d == 0.0 {
                    continue;
                }
                for h in 0..self.n_hidden {
                    self.grad_w2[o * self.n_hidden + h] += d * cache.h_spike[t][h];
                }
            }
            let d_h_spike_t =
                matvec_transpose(&self.w2, &d_o_v_pre_t, self.n_output, self.n_hidden);

            for (next, &cur) in d_o_v_pre_next.iter_mut().zip(d_o_v_pre_t.iter()) {
                *next = cur * decay;
            }

            let mut d_h_v_pre_t = d_h_v_pre_next.clone();
            for (i, d) in d_h_v_pre_t.iter_mut().enumerate() {
                *d += d_h_spike_t[i] * self.spike_fn.backward(cache.h_v_pre[t][i]);
            }
            for h in 0..self.n_hidden {
                let d = d_h_v_pre_t[h];
                if d == 0.0 {
                    continue;
                }
                for i in 0..self.n_input {
                    self.grad_w1[h * self.n_input + i] += d * input_spikes[t][i];
                }
            }
            for (next, &cur) in d_h_v_pre_next.iter_mut().zip(d_h_v_pre_t.iter()) {
                *next = cur * decay;
            }
        }

        (loss, predicted)
    }

    pub fn step_sgd(&mut self, lr: f32) {
        sgd_apply(&mut self.w1, &self.grad_w1, lr);
        sgd_apply(&mut self.w2, &self.grad_w2, lr);
    }
}
