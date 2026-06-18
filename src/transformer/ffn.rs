#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::doc_markdown
)]

//! Feed-forward block: `Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)`.
//!
//! Output projection uses nanoGPT residual scaling (init divided by
//! `sqrt(2 * n_layer)`) to keep activations stable in pre-norm stacks.

use rand::Rng;

use crate::attention::{linear_backward, linear_forward, LinearGrads, LinearProj};
use crate::linalg::{gelu, gelu_backward, relu2, relu2_backward, silu, silu_backward};

#[derive(Clone, Debug)]
pub struct FfnWeights {
    pub w_fc: LinearProj,
    pub w_proj: LinearProj,
    /// SwiGLU gate projection `[d_model → d_ff]`; `None` for plain FFN.
    /// When present, `w_fc` is the "up" projection and the hidden is
    /// `silu(w_gate·x) ⊙ (w_fc·x)`. Pass a reduced `d_ff` (≈⅔) to stay
    /// iso-parameter with the 2-matrix FFN.
    pub w_gate: Option<LinearProj>,
    pub d_model: usize,
    pub d_ff: usize,
    /// Squared-ReLU activation instead of GELU.
    pub relu2: bool,
    pub swiglu: bool,
}

impl FfnWeights {
    pub fn new<R: Rng>(
        d_model: usize,
        d_ff: usize,
        n_layer: usize,
        swiglu: bool,
        rng: &mut R,
    ) -> Self {
        let resid_scale = (2.0_f32 * n_layer as f32).sqrt().recip();
        Self {
            w_fc: LinearProj::new(d_model, d_ff, rng),
            w_proj: LinearProj::new_scaled(d_ff, d_model, resid_scale, rng),
            w_gate: swiglu.then(|| LinearProj::new(d_model, d_ff, rng)),
            d_model,
            d_ff,
            relu2: false,
            swiglu,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FfnGrads {
    pub d_w_fc: LinearGrads,
    pub d_w_proj: LinearGrads,
    /// SwiGLU gate grad; empty `d_w` when not using SwiGLU.
    pub d_w_gate: LinearGrads,
}

impl FfnGrads {
    #[must_use]
    pub fn zeros(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_w_fc: LinearGrads::zeros(d_model, d_ff),
            d_w_proj: LinearGrads::zeros(d_ff, d_model),
            d_w_gate: LinearGrads::default(),
        }
    }

    pub fn zero(&mut self) {
        self.d_w_fc.zero();
        self.d_w_proj.zero();
        self.d_w_gate.zero();
    }
}

#[derive(Clone, Debug, Default)]
pub struct FfnCache {
    pub x: Vec<f32>,
    /// Pre-activation `[T * d_ff]` (the "up" projection under SwiGLU).
    pub h_pre: Vec<f32>,
    /// Post-activation `[T * d_ff]` (input to the down projection).
    pub h_post: Vec<f32>,
    /// SwiGLU gate pre-activation `[T * d_ff]`; empty for plain FFN.
    pub gate_pre: Vec<f32>,
}

pub fn ffn_forward(w: &FfnWeights, x: &[f32]) -> (Vec<f32>, FfnCache) {
    let h_pre = linear_forward(&w.w_fc, x);
    let (h_post, gate_pre): (Vec<f32>, Vec<f32>) = if let Some(w_gate) = w.w_gate.as_ref() {
        // SwiGLU: hidden = silu(gate) ⊙ up, with up = w_fc·x.
        let gate_pre = linear_forward(w_gate, x);
        let h_post = gate_pre
            .iter()
            .zip(&h_pre)
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        (h_post, gate_pre)
    } else if w.relu2 {
        (h_pre.iter().map(|&v| relu2(v)).collect(), Vec::new())
    } else {
        (h_pre.iter().map(|&v| gelu(v)).collect(), Vec::new())
    };
    let y = linear_forward(&w.w_proj, &h_post);
    (
        y,
        FfnCache {
            x: x.to_vec(),
            h_pre,
            h_post,
            gate_pre,
        },
    )
}

pub fn ffn_backward(w: &FfnWeights, cache: &FfnCache, dy: &[f32], grads: &mut FfnGrads) -> Vec<f32> {
    let d_h_post = linear_backward(&w.w_proj, &cache.h_post, dy, &mut grads.d_w_proj);
    if let Some(w_gate) = w.w_gate.as_ref() {
        // h_post = silu(gate_pre) ⊙ up;  up = h_pre = w_fc·x.
        let mut d_up = vec![0.0_f32; cache.h_pre.len()];
        let mut d_gate_pre = vec![0.0_f32; cache.gate_pre.len()];
        for i in 0..cache.h_pre.len() {
            let g = cache.gate_pre[i];
            d_up[i] = d_h_post[i] * silu(g);
            d_gate_pre[i] = d_h_post[i] * cache.h_pre[i] * silu_backward(g);
        }
        let d_x_up = linear_backward(&w.w_fc, &cache.x, &d_up, &mut grads.d_w_fc);
        let d_x_gate = linear_backward(w_gate, &cache.x, &d_gate_pre, &mut grads.d_w_gate);
        let mut d_x = d_x_up;
        for i in 0..d_x.len() {
            d_x[i] += d_x_gate[i];
        }
        return d_x;
    }
    let mut d_h_pre = vec![0.0_f32; cache.h_pre.len()];
    if w.relu2 {
        for i in 0..cache.h_pre.len() {
            d_h_pre[i] = d_h_post[i] * relu2_backward(cache.h_pre[i]);
        }
    } else {
        for i in 0..cache.h_pre.len() {
            d_h_pre[i] = d_h_post[i] * gelu_backward(cache.h_pre[i]);
        }
    }
    linear_backward(&w.w_fc, &cache.x, &d_h_pre, &mut grads.d_w_fc)
}
