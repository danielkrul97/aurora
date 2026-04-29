#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions
)]

//! Sprint 35: Spike-Driven Self-Attention (Yao et al., NeurIPS 2023,
//! arXiv:2307.01694) integrated into Aurora's recurrent SNN.
//!
//! Replaces softmax(Q·Kᵀ/√d)·V with a *linear* attention over binary
//! spike Q/K/V:
//!
//! ```text
//! h_norm     = LayerNorm(h_spike)
//! Q          = step(W_q · h_norm − θ)
//! K          = step(W_k · h_norm − θ)
//! V          = step(W_v · h_norm − θ)
//! attn_t     = attn_{t−1} + K_t ⊙ V_t           (running accumulator)
//! sdsa_out_t = Q_t ⊙ attn_t                     (Hadamard, not matmul)
//! y_t        = h_spike_t + sdsa_out_t           (residual add)
//! ```
//!
//! Linear in T (not quadratic), no softmax (poison for fp32-only WGSL),
//! and the accumulator naturally carries between Aurora's stateful
//! BPTT chunks the same way `h_v_after` does. Dimensions are kept
//! identical at `d_attn = n_hidden` to skip a separate output projection.
//!
//! CPU-only in this sprint; GPU port is Sprint 36.

use rand::Rng;

use crate::attention::{
    layer_norm_backward, layer_norm_forward, linear_backward, linear_forward, LayerNorm,
    LayerNormCache, LayerNormGrads, LinearGrads, LinearProj,
};
use crate::optimizer::{adam_apply, AdamConfig, AdamGroup};
use crate::surrogate::SpikeFn;

#[derive(Clone, Debug)]
pub struct SdsaCell {
    pub n_hidden: usize,
    pub layer_norm: LayerNorm,
    pub w_q: LinearProj,
    pub w_k: LinearProj,
    pub w_v: LinearProj,
    pub spike_fn: SpikeFn,
}

impl SdsaCell {
    pub fn new<R: Rng>(n_hidden: usize, rng: &mut R) -> Self {
        Self {
            n_hidden,
            layer_norm: LayerNorm::new(n_hidden),
            w_q: LinearProj::new(n_hidden, n_hidden, rng),
            w_k: LinearProj::new(n_hidden, n_hidden, rng),
            w_v: LinearProj::new(n_hidden, n_hidden, rng),
            spike_fn: SpikeFn::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SdsaCache {
    pub h_norm: Vec<f32>,
    pub layer_norm_cache: LayerNormCache,
    pub q_pre: Vec<f32>,
    pub k_pre: Vec<f32>,
    pub v_pre: Vec<f32>,
    pub q_spike: Vec<f32>,
    pub k_spike: Vec<f32>,
    pub v_spike: Vec<f32>,
    /// Flat row-major `[T * n_hidden]`. `attn_states[t*n .. (t+1)*n]`
    /// = state AFTER adding `K_t ⊙ V_t`. The very first slice is
    /// `initial_attn_state + K_0 ⊙ V_0`. Stored flat to avoid per-step
    /// `Vec<f32>` allocations on the forward path.
    pub attn_states: Vec<f32>,
    pub initial_attn_state: Vec<f32>,
    pub final_attn_state: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct SdsaGrads {
    pub layer_norm: LayerNormGrads,
    pub w_q: LinearGrads,
    pub w_k: LinearGrads,
    pub w_v: LinearGrads,
}

impl SdsaGrads {
    pub fn zeros(n_hidden: usize) -> Self {
        Self {
            layer_norm: LayerNormGrads {
                d_gamma: vec![0.0; n_hidden],
                d_beta: vec![0.0; n_hidden],
            },
            w_q: LinearGrads {
                d_w: vec![0.0; n_hidden * n_hidden],
            },
            w_k: LinearGrads {
                d_w: vec![0.0; n_hidden * n_hidden],
            },
            w_v: LinearGrads {
                d_w: vec![0.0; n_hidden * n_hidden],
            },
        }
    }

    pub fn zero(&mut self) {
        self.layer_norm.d_gamma.fill(0.0);
        self.layer_norm.d_beta.fill(0.0);
        self.w_q.d_w.fill(0.0);
        self.w_k.d_w.fill(0.0);
        self.w_v.d_w.fill(0.0);
    }

    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.layer_norm.d_gamma.iter_mut().zip(&other.layer_norm.d_gamma) {
            *a += b;
        }
        for (a, b) in self.layer_norm.d_beta.iter_mut().zip(&other.layer_norm.d_beta) {
            *a += b;
        }
        for (a, b) in self.w_q.d_w.iter_mut().zip(&other.w_q.d_w) {
            *a += b;
        }
        for (a, b) in self.w_k.d_w.iter_mut().zip(&other.w_k.d_w) {
            *a += b;
        }
        for (a, b) in self.w_v.d_w.iter_mut().zip(&other.w_v.d_w) {
            *a += b;
        }
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &f32> {
        self.layer_norm
            .d_gamma
            .iter()
            .chain(&self.layer_norm.d_beta)
            .chain(&self.w_q.d_w)
            .chain(&self.w_k.d_w)
            .chain(&self.w_v.d_w)
    }

    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.layer_norm
            .d_gamma
            .iter_mut()
            .chain(&mut self.layer_norm.d_beta)
            .chain(&mut self.w_q.d_w)
            .chain(&mut self.w_k.d_w)
            .chain(&mut self.w_v.d_w)
    }
}

#[derive(Clone, Debug)]
pub struct SdsaAdam {
    pub layer_norm: (AdamGroup, AdamGroup), // (γ, β)
    pub w_q: AdamGroup,
    pub w_k: AdamGroup,
    pub w_v: AdamGroup,
}

impl SdsaAdam {
    pub fn zeros(n_hidden: usize) -> Self {
        Self {
            layer_norm: (AdamGroup::zeros(n_hidden), AdamGroup::zeros(n_hidden)),
            w_q: AdamGroup::zeros(n_hidden * n_hidden),
            w_k: AdamGroup::zeros(n_hidden * n_hidden),
            w_v: AdamGroup::zeros(n_hidden * n_hidden),
        }
    }
}

/// Forward through the SDSA block. Returns the residual output
/// `y_t = h_spike_t + sdsa_out_t` (shape `[T × n_hidden]`) and a
/// cache for backward.
pub fn sdsa_forward(
    cell: &SdsaCell,
    h_spike: &[f32],
    initial_attn_state: &[f32],
) -> (Vec<f32>, SdsaCache) {
    let n = cell.n_hidden;
    debug_assert_eq!(h_spike.len() % n, 0);
    debug_assert_eq!(initial_attn_state.len(), n);
    let t_steps = h_spike.len() / n;

    let (h_norm, layer_norm_cache) = layer_norm_forward(&cell.layer_norm, h_spike);
    let q_pre = linear_forward(&cell.w_q, &h_norm);
    let k_pre = linear_forward(&cell.w_k, &h_norm);
    let v_pre = linear_forward(&cell.w_v, &h_norm);

    let theta = cell.spike_fn.threshold;
    let q_spike: Vec<f32> = q_pre.iter().map(|&v| if v >= theta { 1.0 } else { 0.0 }).collect();
    let k_spike: Vec<f32> = k_pre.iter().map(|&v| if v >= theta { 1.0 } else { 0.0 }).collect();
    let v_spike: Vec<f32> = v_pre.iter().map(|&v| if v >= theta { 1.0 } else { 0.0 }).collect();

    let mut attn_states = vec![0.0_f32; t_steps * n];
    let mut sdsa_out = vec![0.0_f32; t_steps * n];
    let mut state = initial_attn_state.to_vec();

    for t in 0..t_steps {
        let off = t * n;
        // attn_state += K_t ⊙ V_t
        for i in 0..n {
            state[i] += k_spike[off + i] * v_spike[off + i];
        }
        attn_states[off..off + n].copy_from_slice(&state);
        // out_t = Q_t ⊙ attn_state
        for i in 0..n {
            sdsa_out[off + i] = q_spike[off + i] * state[i];
        }
    }

    // Residual add: y_t = h_spike_t + sdsa_out_t
    let mut y = h_spike.to_vec();
    for (yi, oi) in y.iter_mut().zip(sdsa_out.iter()) {
        *yi += oi;
    }

    (
        y,
        SdsaCache {
            h_norm,
            layer_norm_cache,
            q_pre,
            k_pre,
            v_pre,
            q_spike,
            k_spike,
            v_spike,
            attn_states,
            initial_attn_state: initial_attn_state.to_vec(),
            final_attn_state: state,
        },
    )
}

/// Backward through the SDSA block. `d_y` is the upstream gradient
/// from the residual sum (shape `[T × n_hidden]`). Returns
/// `d_h_spike` (input gradient to feed into the SNN's top layer),
/// and accumulates parameter gradients into `grads`.
pub fn sdsa_backward(
    cell: &SdsaCell,
    cache: &SdsaCache,
    h_spike: &[f32],
    d_y: &[f32],
    grads: &mut SdsaGrads,
) -> Vec<f32> {
    let n = cell.n_hidden;
    let t_steps = d_y.len() / n;

    // Residual: d_h_spike (direct path) = d_y; d_sdsa_out = d_y.
    let d_sdsa_out = d_y;
    let mut d_h_spike_direct = d_y.to_vec();

    // Reverse-time sweep through the SDSA core.
    // out_t = Q_t ⊙ attn_t  where attn_t = attn_{t-1} + K_t ⊙ V_t.
    // The "reverse cumulative" gradient on attn_state at time t is
    // d_attn_cum[t] = Σ_{s≥t} (Q_s ⊙ d_sdsa_out_s).
    // Then ∂L/∂(K_s ⊙ V_s) = d_attn_cum[s].
    let mut d_q_spike = vec![0.0_f32; t_steps * n];
    let mut d_k_spike = vec![0.0_f32; t_steps * n];
    let mut d_v_spike = vec![0.0_f32; t_steps * n];
    let mut d_attn_cum = vec![0.0_f32; n];
    for t in (0..t_steps).rev() {
        let off = t * n;
        let attn_t = &cache.attn_states[off..off + n];
        // d_Q_t = d_sdsa_out_t ⊙ attn_t
        // d_attn_t = d_sdsa_out_t ⊙ Q_t  → accumulate into d_attn_cum
        for i in 0..n {
            d_q_spike[off + i] = d_sdsa_out[off + i] * attn_t[i];
            d_attn_cum[i] += d_sdsa_out[off + i] * cache.q_spike[off + i];
        }
        // d_K_t = d_attn_cum ⊙ V_t  ;  d_V_t = d_attn_cum ⊙ K_t
        for i in 0..n {
            d_k_spike[off + i] = d_attn_cum[i] * cache.v_spike[off + i];
            d_v_spike[off + i] = d_attn_cum[i] * cache.k_spike[off + i];
        }
    }

    // Through spike (surrogate gradient).
    let theta = cell.spike_fn.threshold;
    let mut d_q_pre = vec![0.0_f32; t_steps * n];
    let mut d_k_pre = vec![0.0_f32; t_steps * n];
    let mut d_v_pre = vec![0.0_f32; t_steps * n];
    for i in 0..d_q_pre.len() {
        d_q_pre[i] = d_q_spike[i] * cell.spike_fn.backward(cache.q_pre[i] - theta);
        d_k_pre[i] = d_k_spike[i] * cell.spike_fn.backward(cache.k_pre[i] - theta);
        d_v_pre[i] = d_v_spike[i] * cell.spike_fn.backward(cache.v_pre[i] - theta);
    }

    // Through linear projections — accumulate weight grads, get d_h_norm.
    let dh_norm_q =
        linear_backward(&cell.w_q, &cache.h_norm, &d_q_pre, &mut grads.w_q);
    let dh_norm_k =
        linear_backward(&cell.w_k, &cache.h_norm, &d_k_pre, &mut grads.w_k);
    let dh_norm_v =
        linear_backward(&cell.w_v, &cache.h_norm, &d_v_pre, &mut grads.w_v);

    let mut d_h_norm = vec![0.0_f32; t_steps * n];
    for i in 0..d_h_norm.len() {
        d_h_norm[i] = dh_norm_q[i] + dh_norm_k[i] + dh_norm_v[i];
    }

    // Through LayerNorm.
    let d_h_spike_norm = layer_norm_backward(
        &cell.layer_norm,
        &cache.layer_norm_cache,
        &d_h_norm,
        &mut grads.layer_norm,
    );

    // Sum residual + norm-path d_h_spike.
    for (a, b) in d_h_spike_direct.iter_mut().zip(d_h_spike_norm.iter()) {
        *a += b;
    }
    let _ = h_spike;
    d_h_spike_direct
}

pub fn sdsa_step_adam(
    cell: &mut SdsaCell,
    grads: &SdsaGrads,
    adam: &mut SdsaAdam,
    config: &AdamConfig,
    bc1: f32,
    bc2: f32,
) {
    adam_apply(
        &mut cell.layer_norm.gamma,
        &grads.layer_norm.d_gamma,
        &mut adam.layer_norm.0,
        config,
        bc1,
        bc2,
    );
    adam_apply(
        &mut cell.layer_norm.beta,
        &grads.layer_norm.d_beta,
        &mut adam.layer_norm.1,
        config,
        bc1,
        bc2,
    );
    adam_apply(
        &mut cell.w_q.w,
        &grads.w_q.d_w,
        &mut adam.w_q,
        config,
        bc1,
        bc2,
    );
    adam_apply(
        &mut cell.w_k.w,
        &grads.w_k.d_w,
        &mut adam.w_k,
        config,
        bc1,
        bc2,
    );
    adam_apply(
        &mut cell.w_v.w,
        &grads.w_v.d_w,
        &mut adam.w_v,
        config,
        bc1,
        bc2,
    );
}
