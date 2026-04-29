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

//! Sprint 50: closed-form continuous-time (CfC) Liquid Neural Network
//! cell.
//!
//! Successor to the Sprint 46+ diagonal SSM cell. Where SSM has a
//! per-neuron decay `a_i` that is **constant in time**, CfC's effective
//! decay is **input-dependent** at every timestep — Hasani et al.
//! (Nature MI 2022, arXiv:2106.13898). Same architectural lineage as
//! Mamba's selective SSM but with biological motivation (originally
//! inspired by C. elegans nervous system, Hasani LTC 2020).
//!
//! Forward (per timestep, per neuron `i`):
//! ```text
//! z_t          = [I_t, x_{t-1}]                       (concat)
//! f_pre[i]     = b_f[i] + W_f[i,:] · z_t
//! g_pre[i]     = b_g[i] + W_g[i,:] · z_t
//! h_pre[i]     = b_h[i] + W_h[i,:] · z_t
//! f[i]         = sigmoid(f_pre[i])                     (gating logit)
//! g[i]         = tanh(g_pre[i])                        (fast sub-state)
//! h[i]         = tanh(h_pre[i])                        (slow sub-state)
//! tau[i]       = exp(log_tau[i])                       (per-neuron timescale, > 0)
//! alpha[i]     = sigmoid(-f[i] · tau[i])               (interpolation weight ∈ (0,1))
//! x[t][i]      = alpha[i] · g[i] + (1 - alpha[i]) · h[i]
//! ```
//!
//! Read-out is **continuous** (real-valued state directly, no
//! heaviside). Output projection is a dense linear layer over `x[t]`.
//! Cell state is bounded by construction: `tanh` saturates `g, h` to
//! `[-1, 1]`, and `x` is a convex combination → `x ∈ [-1, 1]^n_h`.
//!
//! No parallel scan: input-dependent decay breaks Hillis-Steele
//! associativity. Sequential per-timestep forward and reverse-time
//! BPTT backward.
//!
//! API mirrors `crate::ssm::MultiLayerSsmSnn` so `aurora_lnn_train` can
//! reuse the same chunk/batch/Adam loop shape.

use rand::Rng;
use rayon::prelude::*;

use crate::optimizer::{adam_apply, bias_correction, AdamConfig, AdamGroup};
use crate::loss::cross_entropy_with_grad;

const LOG_TAU_CLAMP: f32 = 3.0;
const DEFAULT_LOG_TAU_INIT: f32 = 0.0;
/// Memory neurons (M-LTC, Srinivas et al. 2025) get a higher floor on
/// log_tau to keep their timescale slow. With min = 1.0, τ ≥ 2.7.
const LOG_TAU_MEM_MIN: f32 = 1.0;
/// Memory neurons start with τ ≈ 4.5.
const DEFAULT_LOG_TAU_MEM_INIT: f32 = 1.5;

/// One CfC layer's parameters. Three weight matrices feed three
/// sub-states (`f`, `g`, `h`); a per-neuron `log_tau` scales the
/// gating sigmoid argument.
///
/// The first `n_memory` neurons are M-LTC memory neurons: periodic
/// activation (`sin/cos`) on `g`/`h`, and `log_tau` floored at
/// `LOG_TAU_MEM_MIN`. The remaining `n_hidden - n_memory` neurons are
/// classic CfC (tanh activation, `log_tau` floored at -LOG_TAU_CLAMP).
#[derive(Clone, Debug)]
pub struct LnnLayer {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_z: usize,
    /// Number of leading neurons running the M-LTC memory variant.
    pub n_memory: usize,
    /// `[n_hidden, n_z]` row-major. `f_pre = b_f + W_f · z_t`.
    pub w_f: Vec<f32>,
    pub w_g: Vec<f32>,
    pub w_h: Vec<f32>,
    pub b_f: Vec<f32>,
    pub b_g: Vec<f32>,
    pub b_h: Vec<f32>,
    /// Per-neuron log-timescale; clamped to `[-LOG_TAU_CLAMP, +LOG_TAU_CLAMP]`
    /// after each Adam step so `tau ∈ [exp(-3), exp(3)] ≈ [0.05, 20]`.
    /// Memory neurons additionally floor at `LOG_TAU_MEM_MIN`.
    pub log_tau: Vec<f32>,
}

impl LnnLayer {
    pub fn new<R: Rng>(n_in: usize, n_hidden: usize, rng: &mut R) -> Self {
        Self::new_with_log_tau(n_in, n_hidden, DEFAULT_LOG_TAU_INIT, rng)
    }

    pub fn new_with_log_tau<R: Rng>(
        n_in: usize,
        n_hidden: usize,
        log_tau_init: f32,
        rng: &mut R,
    ) -> Self {
        Self::new_with_memory(n_in, n_hidden, 0, log_tau_init, DEFAULT_LOG_TAU_MEM_INIT, rng)
    }

    pub fn new_with_memory<R: Rng>(
        n_in: usize,
        n_hidden: usize,
        n_memory: usize,
        log_tau_init: f32,
        log_tau_mem_init: f32,
        rng: &mut R,
    ) -> Self {
        assert!(n_memory <= n_hidden, "n_memory must not exceed n_hidden");
        let n_z = n_in + n_hidden;
        let scale = (2.0 / n_z as f32).sqrt();
        let w_init = |rng: &mut R| (rng.random::<f32>() - 0.5) * 2.0 * scale;
        let w_f: Vec<f32> = (0..n_hidden * n_z).map(|_| w_init(rng)).collect();
        let w_g: Vec<f32> = (0..n_hidden * n_z).map(|_| w_init(rng)).collect();
        let w_h: Vec<f32> = (0..n_hidden * n_z).map(|_| w_init(rng)).collect();
        let log_tau_learn = log_tau_init.clamp(-LOG_TAU_CLAMP, LOG_TAU_CLAMP);
        let log_tau_mem = log_tau_mem_init.clamp(LOG_TAU_MEM_MIN, LOG_TAU_CLAMP);
        let log_tau: Vec<f32> = (0..n_hidden)
            .map(|i| if i < n_memory { log_tau_mem } else { log_tau_learn })
            .collect();
        Self {
            n_in,
            n_hidden,
            n_z,
            n_memory,
            w_f,
            w_g,
            w_h,
            b_f: vec![0.0; n_hidden],
            b_g: vec![0.0; n_hidden],
            b_h: vec![0.0; n_hidden],
            log_tau,
        }
    }

    #[inline]
    pub const fn is_memory(&self, i: usize) -> bool {
        i < self.n_memory
    }
}

#[derive(Clone, Debug)]
pub struct LayerGrads {
    pub w_f: Vec<f32>,
    pub w_g: Vec<f32>,
    pub w_h: Vec<f32>,
    pub b_f: Vec<f32>,
    pub b_g: Vec<f32>,
    pub b_h: Vec<f32>,
    pub log_tau: Vec<f32>,
}

impl LayerGrads {
    pub fn zeros(n_in: usize, n_hidden: usize) -> Self {
        let n_z = n_in + n_hidden;
        Self {
            w_f: vec![0.0; n_hidden * n_z],
            w_g: vec![0.0; n_hidden * n_z],
            w_h: vec![0.0; n_hidden * n_z],
            b_f: vec![0.0; n_hidden],
            b_g: vec![0.0; n_hidden],
            b_h: vec![0.0; n_hidden],
            log_tau: vec![0.0; n_hidden],
        }
    }

    pub fn zero(&mut self) {
        self.w_f.fill(0.0);
        self.w_g.fill(0.0);
        self.w_h.fill(0.0);
        self.b_f.fill(0.0);
        self.b_g.fill(0.0);
        self.b_h.fill(0.0);
        self.log_tau.fill(0.0);
    }

    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.w_f.iter_mut().zip(&other.w_f) {
            *a += b;
        }
        for (a, b) in self.w_g.iter_mut().zip(&other.w_g) {
            *a += b;
        }
        for (a, b) in self.w_h.iter_mut().zip(&other.w_h) {
            *a += b;
        }
        for (a, b) in self.b_f.iter_mut().zip(&other.b_f) {
            *a += b;
        }
        for (a, b) in self.b_g.iter_mut().zip(&other.b_g) {
            *a += b;
        }
        for (a, b) in self.b_h.iter_mut().zip(&other.b_h) {
            *a += b;
        }
        for (a, b) in self.log_tau.iter_mut().zip(&other.log_tau) {
            *a += b;
        }
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &f32> {
        self.w_f
            .iter()
            .chain(&self.w_g)
            .chain(&self.w_h)
            .chain(&self.b_f)
            .chain(&self.b_g)
            .chain(&self.b_h)
            .chain(&self.log_tau)
    }

    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.w_f
            .iter_mut()
            .chain(&mut self.w_g)
            .chain(&mut self.w_h)
            .chain(&mut self.b_f)
            .chain(&mut self.b_g)
            .chain(&mut self.b_h)
            .chain(&mut self.log_tau)
    }
}

#[derive(Clone, Debug, Default)]
pub struct MultiLayerLnnGrads {
    pub layers: Vec<LayerGrads>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
}

impl MultiLayerLnnGrads {
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

/// Per-layer cache for one chunk's forward pass. Stores everything the
/// backward pass needs without recomputing forward activations.
#[derive(Clone, Debug)]
pub struct LayerCache {
    /// Sigmoid output `f[t][i]`, length `t_steps`.
    pub f: Vec<Vec<f32>>,
    /// Activation output `g[t][i]` — `tanh(g_pre)` for learning neurons,
    /// `sin(g_pre)` for memory neurons.
    pub g: Vec<Vec<f32>>,
    pub h_act: Vec<Vec<f32>>,
    /// `dg/d(g_pre)` — `1 - g²` for tanh neurons, `cos(g_pre)` for sin
    /// neurons. Cached at forward time so backward can stay uniform.
    pub g_deriv: Vec<Vec<f32>>,
    pub h_deriv: Vec<Vec<f32>>,
    /// Interpolation weight `alpha[t][i]`.
    pub alpha: Vec<Vec<f32>>,
    /// State `x[t][i]` after the convex combination.
    pub x: Vec<Vec<f32>>,
    /// State carried in from the previous chunk (or zeros if fresh).
    pub initial_state: Vec<f32>,
}

impl LayerCache {
    pub fn with_capacity(t_steps: usize) -> Self {
        Self {
            f: Vec::with_capacity(t_steps),
            g: Vec::with_capacity(t_steps),
            h_act: Vec::with_capacity(t_steps),
            g_deriv: Vec::with_capacity(t_steps),
            h_deriv: Vec::with_capacity(t_steps),
            alpha: Vec::with_capacity(t_steps),
            x: Vec::with_capacity(t_steps),
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
    /// Mean absolute state magnitude per layer — the closest analog to
    /// the SSM/gated-cell "firing rate" diagnostic. State is bounded
    /// to `[-1, 1]` by construction, so values near 1.0 indicate
    /// saturation.
    #[must_use]
    pub fn activity(&self) -> Vec<f32> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.x.is_empty() {
                    return 0.0;
                }
                let n_h = layer.x[0].len();
                let total = (layer.x.len() * n_h) as f32;
                let sum: f32 = layer
                    .x
                    .iter()
                    .flat_map(|row| row.iter().map(|v| v.abs()))
                    .sum();
                if total > 0.0 { sum / total } else { 0.0 }
            })
            .collect()
    }

    /// Mean alpha (gating interpolation weight) per layer. Near 0.5 =
    /// balanced fast/slow mix; far from 0.5 = one path dominates.
    #[must_use]
    pub fn alpha_stats(&self) -> Vec<(f32, f32)> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.alpha.is_empty() {
                    return (0.0, 0.0);
                }
                let mut sum = 0.0_f32;
                let mut peak = 0.0_f32;
                let mut count = 0_usize;
                for row in &layer.alpha {
                    for &a in row {
                        sum += a;
                        let dev = (a - 0.5).abs();
                        if dev > peak {
                            peak = dev;
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
pub struct LnnState {
    pub x: Vec<f32>,
}

#[derive(Clone, Debug)]
struct LayerAdam {
    m_w_f: AdamGroup,
    m_w_g: AdamGroup,
    m_w_h: AdamGroup,
    m_b_f: AdamGroup,
    m_b_g: AdamGroup,
    m_b_h: AdamGroup,
    m_log_tau: AdamGroup,
}

impl LayerAdam {
    fn zeros(n_in: usize, n_hidden: usize) -> Self {
        let n_z = n_in + n_hidden;
        Self {
            m_w_f: AdamGroup::zeros(n_hidden * n_z),
            m_w_g: AdamGroup::zeros(n_hidden * n_z),
            m_w_h: AdamGroup::zeros(n_hidden * n_z),
            m_b_f: AdamGroup::zeros(n_hidden),
            m_b_g: AdamGroup::zeros(n_hidden),
            m_b_h: AdamGroup::zeros(n_hidden),
            m_log_tau: AdamGroup::zeros(n_hidden),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiLayerLnnSnn {
    pub n_input: usize,
    pub n_output: usize,
    pub layers: Vec<LnnLayer>,
    pub w_out: Vec<f32>,
    pub b_o: Vec<f32>,
    grads: MultiLayerLnnGrads,
    adam_t: usize,
    layer_adam: Vec<LayerAdam>,
    adam_w_out: AdamGroup,
    adam_b_o: AdamGroup,
}

impl MultiLayerLnnSnn {
    pub fn new<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        rng: &mut R,
    ) -> Self {
        Self::new_with_log_tau(n_input, hidden_dims, n_output, DEFAULT_LOG_TAU_INIT, rng)
    }

    pub fn new_with_log_tau<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_output: usize,
        log_tau_init: f32,
        rng: &mut R,
    ) -> Self {
        Self::new_with_memory(
            n_input,
            hidden_dims,
            &vec![0; hidden_dims.len()],
            n_output,
            log_tau_init,
            DEFAULT_LOG_TAU_MEM_INIT,
            rng,
        )
    }

    /// `n_memory_per_layer[l]` controls how many of layer `l`'s neurons
    /// are M-LTC memory neurons. Pass an array of zeros to recover the
    /// classic CfC behavior.
    pub fn new_with_memory<R: Rng>(
        n_input: usize,
        hidden_dims: &[usize],
        n_memory_per_layer: &[usize],
        n_output: usize,
        log_tau_init: f32,
        log_tau_mem_init: f32,
        rng: &mut R,
    ) -> Self {
        assert!(!hidden_dims.is_empty(), "at least one layer required");
        assert_eq!(
            hidden_dims.len(),
            n_memory_per_layer.len(),
            "n_memory_per_layer must match hidden_dims length"
        );
        let mut layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = n_input;
        for (l, &h) in hidden_dims.iter().enumerate() {
            layers.push(LnnLayer::new_with_memory(
                prev_dim,
                h,
                n_memory_per_layer[l],
                log_tau_init,
                log_tau_mem_init,
                rng,
            ));
            prev_dim = h;
        }
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
            grads: MultiLayerLnnGrads::zeros(&layer_dims, n_output),
            layers,
            w_out,
            b_o: vec![0.0; n_output],
            adam_t: 0,
            layer_adam,
            adam_w_out: AdamGroup::zeros(n_output * top_hidden),
            adam_b_o: AdamGroup::zeros(n_output),
        }
    }

    pub fn fresh_states(&self) -> Vec<LnnState> {
        self.layers
            .iter()
            .map(|l| LnnState {
                x: vec![0.0; l.n_hidden],
            })
            .collect()
    }

    pub const fn grads(&self) -> &MultiLayerLnnGrads {
        &self.grads
    }

    pub fn alloc_grads(&self) -> MultiLayerLnnGrads {
        let mut layer_dims = vec![self.n_input];
        for l in &self.layers {
            layer_dims.push(l.n_hidden);
        }
        MultiLayerLnnGrads::zeros(&layer_dims, self.n_output)
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

    pub fn add_grads(&mut self, other: &MultiLayerLnnGrads) {
        self.grads.add(other);
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let g = &self.grads.layers[l];
            let st = &mut self.layer_adam[l];
            adam_apply(&mut layer.w_f, &g.w_f, &mut st.m_w_f, config, bc1, bc2);
            adam_apply(&mut layer.w_g, &g.w_g, &mut st.m_w_g, config, bc1, bc2);
            adam_apply(&mut layer.w_h, &g.w_h, &mut st.m_w_h, config, bc1, bc2);
            adam_apply(&mut layer.b_f, &g.b_f, &mut st.m_b_f, config, bc1, bc2);
            adam_apply(&mut layer.b_g, &g.b_g, &mut st.m_b_g, config, bc1, bc2);
            adam_apply(&mut layer.b_h, &g.b_h, &mut st.m_b_h, config, bc1, bc2);
            adam_apply(
                &mut layer.log_tau,
                &g.log_tau,
                &mut st.m_log_tau,
                config,
                bc1,
                bc2,
            );
            // Memory neurons get a higher floor so the slow timescale
            // can't collapse; learning neurons use the standard symmetric
            // clamp around the log_tau init.
            let n_mem = layer.n_memory;
            for (i, v) in layer.log_tau.iter_mut().enumerate() {
                let lo = if i < n_mem { LOG_TAU_MEM_MIN } else { -LOG_TAU_CLAMP };
                *v = v.clamp(lo, LOG_TAU_CLAMP);
            }
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

    /// Stateful forward over a chunk of token IDs. Layer 0 input is the
    /// one-hot token vector (sparse, single column per timestep). Deep
    /// layers receive the previous layer's continuous state vector
    /// (dense GEMV). Output projection is dense over the top layer's
    /// state.
    pub fn forward_lm_tokens_stateful(
        &self,
        input_tokens: &[u32],
        states: &mut [LnnState],
    ) -> MultiLayerCache {
        assert_eq!(states.len(), self.layers.len());
        let t_steps = input_tokens.len();
        let n_layers = self.layers.len();
        let mut cache = MultiLayerCache {
            layers: (0..n_layers).map(|_| LayerCache::with_capacity(t_steps)).collect(),
            logits: Vec::with_capacity(t_steps),
        };
        for (l, st) in states.iter().enumerate() {
            cache.layers[l].initial_state = st.x.clone();
        }
        let top_hidden = self.layers.last().unwrap().n_hidden;

        for l in 0..n_layers {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let n_z = layer.n_z;

            let mut f_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut g_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut h_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut g_deriv_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut h_deriv_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut alpha_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
            let mut x_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);

            for t in 0..t_steps {
                let x_prev: &[f32] = if t == 0 {
                    &cache.layers[l].initial_state
                } else {
                    &x_per_t[t - 1]
                };

                let mut f_pre = layer.b_f.clone();
                let mut g_pre = layer.b_g.clone();
                let mut h_pre = layer.b_h.clone();

                if l == 0 {
                    let token = input_tokens[t] as usize;
                    for i in 0..n_h {
                        let row = i * n_z;
                        f_pre[i] += layer.w_f[row + token];
                        g_pre[i] += layer.w_g[row + token];
                        h_pre[i] += layer.w_h[row + token];
                    }
                } else {
                    let inp = &cache.layers[l - 1].x[t];
                    debug_assert_eq!(inp.len(), n_in);
                    for i in 0..n_h {
                        let row = i * n_z;
                        let mut acc_f = 0.0_f32;
                        let mut acc_g = 0.0_f32;
                        let mut acc_h = 0.0_f32;
                        for j in 0..n_in {
                            let v = inp[j];
                            acc_f += layer.w_f[row + j] * v;
                            acc_g += layer.w_g[row + j] * v;
                            acc_h += layer.w_h[row + j] * v;
                        }
                        f_pre[i] += acc_f;
                        g_pre[i] += acc_g;
                        h_pre[i] += acc_h;
                    }
                }

                for i in 0..n_h {
                    let row = i * n_z + n_in;
                    let mut acc_f = 0.0_f32;
                    let mut acc_g = 0.0_f32;
                    let mut acc_h = 0.0_f32;
                    for j in 0..n_h {
                        let v = x_prev[j];
                        acc_f += layer.w_f[row + j] * v;
                        acc_g += layer.w_g[row + j] * v;
                        acc_h += layer.w_h[row + j] * v;
                    }
                    f_pre[i] += acc_f;
                    g_pre[i] += acc_g;
                    h_pre[i] += acc_h;
                }

                let mut f = vec![0.0_f32; n_h];
                let mut g = vec![0.0_f32; n_h];
                let mut h_a = vec![0.0_f32; n_h];
                let mut g_deriv = vec![0.0_f32; n_h];
                let mut h_deriv = vec![0.0_f32; n_h];
                let mut alpha = vec![0.0_f32; n_h];
                let mut x = vec![0.0_f32; n_h];
                for i in 0..n_h {
                    let f_i = sigmoid(f_pre[i]);
                    let (g_i, g_d, h_i, h_d) = if layer.is_memory(i) {
                        let gp = g_pre[i];
                        let hp = h_pre[i];
                        (gp.sin(), gp.cos(), hp.sin(), hp.cos())
                    } else {
                        let g_t = g_pre[i].tanh();
                        let h_t = h_pre[i].tanh();
                        (g_t, 1.0 - g_t * g_t, h_t, 1.0 - h_t * h_t)
                    };
                    let tau_i = layer.log_tau[i].exp();
                    let alpha_i = sigmoid(-f_i * tau_i);
                    let x_i = alpha_i * g_i + (1.0 - alpha_i) * h_i;
                    f[i] = f_i;
                    g[i] = g_i;
                    h_a[i] = h_i;
                    g_deriv[i] = g_d;
                    h_deriv[i] = h_d;
                    alpha[i] = alpha_i;
                    x[i] = x_i;
                }

                f_per_t.push(f);
                g_per_t.push(g);
                h_per_t.push(h_a);
                g_deriv_per_t.push(g_deriv);
                h_deriv_per_t.push(h_deriv);
                alpha_per_t.push(alpha);
                x_per_t.push(x);
            }

            states[l].x.clone_from(x_per_t.last().expect("t_steps ≥ 1"));
            cache.layers[l].f = f_per_t;
            cache.layers[l].g = g_per_t;
            cache.layers[l].h_act = h_per_t;
            cache.layers[l].g_deriv = g_deriv_per_t;
            cache.layers[l].h_deriv = h_deriv_per_t;
            cache.layers[l].alpha = alpha_per_t;
            cache.layers[l].x = x_per_t;
        }

        let top = n_layers - 1;
        for t in 0..t_steps {
            let signal = &cache.layers[top].x[t];
            let mut logits = Vec::with_capacity(self.n_output);
            for o in 0..self.n_output {
                let row_off = o * top_hidden;
                let mut acc = self.b_o[o];
                for k in 0..top_hidden {
                    acc += self.w_out[row_off + k] * signal[k];
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
    /// summed cross-entropy loss in nats.
    ///
    /// Walks layers top→bottom; within each layer, walks `t` from
    /// `T-1` down to `0` carrying `d_x_carry` for the recurrent path
    /// `z_t = [I_t, x_{t-1}]`. The CfC backward differentiates seven
    /// gradient terms per layer per timestep:
    ///
    /// 1. `x = alpha·g + (1-alpha)·h`           → `d_alpha, d_g, d_h`
    /// 2. `g = tanh(g_pre), h = tanh(h_pre)`    → `d_g_pre, d_h_pre`
    /// 3. `alpha = sigmoid(-f·tau)`             → `d_f, d_tau`
    /// 4. `f = sigmoid(f_pre)`                  → `d_f_pre`
    /// 5. `tau = exp(log_tau)`                  → `d_log_tau`
    /// 6. `f_pre, g_pre, h_pre = b_* + W_* · z` → param grads + `d_z`
    /// 7. split `d_z = [d_I_t, d_x_carry]`      → cross-layer + recurrent
    pub fn backward_lm_tokens_into(
        &self,
        cache: &MultiLayerCache,
        input_tokens: &[u32],
        targets: &[u32],
        grads: &mut MultiLayerLnnGrads,
    ) -> f32 {
        let t_steps = input_tokens.len();
        debug_assert_eq!(targets.len(), t_steps);
        let n_layers = self.layers.len();
        let top = n_layers - 1;
        let top_hidden = self.layers[top].n_hidden;

        let mut total_loss = 0.0_f32;
        let mut d_x_above_per_t: Vec<Vec<f32>> = Vec::with_capacity(t_steps);
        for t in 0..t_steps {
            let (loss_t, d_logits) =
                cross_entropy_with_grad(&cache.logits[t], targets[t] as usize);
            total_loss += loss_t;

            let signal = &cache.layers[top].x[t];
            for o in 0..self.n_output {
                let d = d_logits[o];
                grads.b_o[o] += d;
                let row = o * top_hidden;
                for k in 0..top_hidden {
                    grads.w_out[row + k] += d * signal[k];
                }
            }
            let mut d_top = vec![0.0_f32; top_hidden];
            for o in 0..self.n_output {
                let d = d_logits[o];
                if d == 0.0 {
                    continue;
                }
                let row = o * top_hidden;
                for k in 0..top_hidden {
                    d_top[k] += self.w_out[row + k] * d;
                }
            }
            d_x_above_per_t.push(d_top);
        }

        for l in (0..n_layers).rev() {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let n_z = layer.n_z;
            let layer_grad = &mut grads.layers[l];
            let layer_cache = &cache.layers[l];

            let mut d_x_carry = vec![0.0_f32; n_h];
            let mut d_input_per_t: Vec<Vec<f32>> = vec![Vec::new(); t_steps];

            for t in (0..t_steps).rev() {
                let d_above = &d_x_above_per_t[t];
                let mut d_x = vec![0.0_f32; n_h];
                for i in 0..n_h {
                    d_x[i] = d_above[i] + d_x_carry[i];
                }

                let f_t = &layer_cache.f[t];
                let g_t = &layer_cache.g[t];
                let h_t = &layer_cache.h_act[t];
                let g_deriv_t = &layer_cache.g_deriv[t];
                let h_deriv_t = &layer_cache.h_deriv[t];
                let alpha_t = &layer_cache.alpha[t];
                let x_prev: &[f32] = if t == 0 {
                    &layer_cache.initial_state
                } else {
                    &layer_cache.x[t - 1]
                };

                let mut d_f_pre = vec![0.0_f32; n_h];
                let mut d_g_pre = vec![0.0_f32; n_h];
                let mut d_h_pre = vec![0.0_f32; n_h];

                for i in 0..n_h {
                    let g_i = g_t[i];
                    let h_i = h_t[i];
                    let alpha_i = alpha_t[i];
                    let f_i = f_t[i];
                    let tau_i = layer.log_tau[i].exp();
                    let dx_i = d_x[i];

                    // x = α·g + (1-α)·h
                    let d_alpha = dx_i * (g_i - h_i);
                    let d_g = dx_i * alpha_i;
                    let d_h = dx_i * (1.0 - alpha_i);

                    // Activation deriv: cached at forward time. Tanh =>
                    // 1 - g², sin => cos(g_pre).
                    d_g_pre[i] = d_g * g_deriv_t[i];
                    d_h_pre[i] = d_h * h_deriv_t[i];

                    // α = σ(-f·τ): dα/d(arg) = α(1-α);
                    // arg = -f·τ → d(arg)/df = -τ, d(arg)/dτ = -f.
                    let d_arg = d_alpha * alpha_i * (1.0 - alpha_i);
                    let d_f = d_arg * (-tau_i);
                    let d_tau = d_arg * (-f_i);

                    // τ = exp(log_τ): d_log_τ = d_τ · τ
                    layer_grad.log_tau[i] += d_tau * tau_i;

                    // f = σ(f_pre): df/df_pre = f(1-f)
                    d_f_pre[i] = d_f * f_i * (1.0 - f_i);
                }

                // Bias grads: d_b_* += d_*_pre
                for i in 0..n_h {
                    layer_grad.b_f[i] += d_f_pre[i];
                    layer_grad.b_g[i] += d_g_pre[i];
                    layer_grad.b_h[i] += d_h_pre[i];
                }

                // Weight grads + d_z accumulation. z = [I_t, x_prev].
                // For layer 0, I_t is one-hot (single column); for
                // deeper layers, I_t is the previous layer's x[t].
                let mut d_z = vec![0.0_f32; n_z];

                if l == 0 {
                    let token = input_tokens[t] as usize;
                    for i in 0..n_h {
                        let row = i * n_z;
                        let dfp = d_f_pre[i];
                        let dgp = d_g_pre[i];
                        let dhp = d_h_pre[i];
                        layer_grad.w_f[row + token] += dfp;
                        layer_grad.w_g[row + token] += dgp;
                        layer_grad.w_h[row + token] += dhp;
                    }
                } else {
                    let inp = &cache.layers[l - 1].x[t];
                    for i in 0..n_h {
                        let row = i * n_z;
                        let dfp = d_f_pre[i];
                        let dgp = d_g_pre[i];
                        let dhp = d_h_pre[i];
                        for j in 0..n_in {
                            let v = inp[j];
                            layer_grad.w_f[row + j] += dfp * v;
                            layer_grad.w_g[row + j] += dgp * v;
                            layer_grad.w_h[row + j] += dhp * v;
                            d_z[j] += layer.w_f[row + j] * dfp
                                + layer.w_g[row + j] * dgp
                                + layer.w_h[row + j] * dhp;
                        }
                    }
                }

                for i in 0..n_h {
                    let row = i * n_z;
                    let dfp = d_f_pre[i];
                    let dgp = d_g_pre[i];
                    let dhp = d_h_pre[i];
                    for j in 0..n_h {
                        let v = x_prev[j];
                        layer_grad.w_f[row + n_in + j] += dfp * v;
                        layer_grad.w_g[row + n_in + j] += dgp * v;
                        layer_grad.w_h[row + n_in + j] += dhp * v;
                        d_z[n_in + j] += layer.w_f[row + n_in + j] * dfp
                            + layer.w_g[row + n_in + j] * dgp
                            + layer.w_h[row + n_in + j] * dhp;
                    }
                }

                if l > 0 {
                    d_input_per_t[t] = d_z[..n_in].to_vec();
                }
                for j in 0..n_h {
                    d_x_carry[j] = d_z[n_in + j];
                }
            }

            if l > 0 {
                d_x_above_per_t = d_input_per_t;
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

    /// Streaming inference for a single token. State is updated in
    /// place. Returns logits for the next-token distribution.
    pub fn step_lm_token(&self, token: u32, states: &mut [LnnState]) -> Vec<f32> {
        assert_eq!(states.len(), self.layers.len());
        let n_layers = self.layers.len();
        let top_hidden = self.layers.last().unwrap().n_hidden;
        let mut layer_input: Vec<f32> = Vec::new();
        for l in 0..n_layers {
            let layer = &self.layers[l];
            let n_h = layer.n_hidden;
            let n_in = layer.n_in;
            let n_z = layer.n_z;
            let x_prev = states[l].x.clone();
            let mut x_new = vec![0.0_f32; n_h];
            for i in 0..n_h {
                let row = i * n_z;
                let mut f_pre = layer.b_f[i];
                let mut g_pre = layer.b_g[i];
                let mut h_pre = layer.b_h[i];
                if l == 0 {
                    let t = token as usize;
                    f_pre += layer.w_f[row + t];
                    g_pre += layer.w_g[row + t];
                    h_pre += layer.w_h[row + t];
                } else {
                    for j in 0..n_in {
                        let v = layer_input[j];
                        f_pre += layer.w_f[row + j] * v;
                        g_pre += layer.w_g[row + j] * v;
                        h_pre += layer.w_h[row + j] * v;
                    }
                }
                for j in 0..n_h {
                    let v = x_prev[j];
                    f_pre += layer.w_f[row + n_in + j] * v;
                    g_pre += layer.w_g[row + n_in + j] * v;
                    h_pre += layer.w_h[row + n_in + j] * v;
                }
                let f_i = sigmoid(f_pre);
                let (g_i, h_i) = if layer.is_memory(i) {
                    (g_pre.sin(), h_pre.sin())
                } else {
                    (g_pre.tanh(), h_pre.tanh())
                };
                let tau_i = layer.log_tau[i].exp();
                let alpha_i = sigmoid(-f_i * tau_i);
                x_new[i] = alpha_i * g_i + (1.0 - alpha_i) * h_i;
            }
            states[l].x.clone_from(&x_new);
            layer_input = x_new;
        }
        let mut logits = Vec::with_capacity(self.n_output);
        for o in 0..self.n_output {
            let row_off = o * top_hidden;
            let mut acc = self.b_o[o];
            for k in 0..top_hidden {
                acc += self.w_out[row_off + k] * layer_input[k];
            }
            logits.push(acc);
        }
        logits
    }

    /// Rayon-parallel batched stateful trainer. Each batch element
    /// runs forward+backward independently into its own grad buffer;
    /// the main thread then sums grads into `self.grads` for the
    /// subsequent Adam step.
    pub fn forward_backward_batch_tokens_stateful(
        &mut self,
        batch_inputs: &[&[u32]],
        batch_targets: &[&[u32]],
        states: &mut [Vec<LnnState>],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        assert_eq!(batch_inputs.len(), states.len());
        if batch_inputs.is_empty() {
            return 0.0;
        }
        let weights = &*self;
        let results: Vec<(f32, MultiLayerLnnGrads)> = batch_inputs
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

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
