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
use crate::linalg::{gelu, gelu_backward};

#[derive(Clone, Debug)]
pub struct FfnWeights {
    pub w_fc: LinearProj,
    pub w_proj: LinearProj,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FfnWeights {
    pub fn new<R: Rng>(d_model: usize, d_ff: usize, n_layer: usize, rng: &mut R) -> Self {
        let resid_scale = (2.0_f32 * n_layer as f32).sqrt().recip();
        Self {
            w_fc: LinearProj::new(d_model, d_ff, rng),
            w_proj: LinearProj::new_scaled(d_ff, d_model, resid_scale, rng),
            d_model,
            d_ff,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FfnGrads {
    pub d_w_fc: LinearGrads,
    pub d_w_proj: LinearGrads,
}

impl FfnGrads {
    #[must_use]
    pub fn zeros(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_w_fc: LinearGrads::zeros(d_model, d_ff),
            d_w_proj: LinearGrads::zeros(d_ff, d_model),
        }
    }

    pub fn zero(&mut self) {
        self.d_w_fc.zero();
        self.d_w_proj.zero();
    }
}

#[derive(Clone, Debug, Default)]
pub struct FfnCache {
    pub x: Vec<f32>,
    /// Pre-GELU activations `[T * d_ff]`, kept for backward through GELU.
    pub h_pre: Vec<f32>,
    /// Post-GELU activations `[T * d_ff]`.
    pub h_post: Vec<f32>,
}

pub fn ffn_forward(w: &FfnWeights, x: &[f32]) -> (Vec<f32>, FfnCache) {
    let h_pre = linear_forward(&w.w_fc, x);
    let h_post: Vec<f32> = h_pre.iter().map(|&v| gelu(v)).collect();
    let y = linear_forward(&w.w_proj, &h_post);
    (
        y,
        FfnCache {
            x: x.to_vec(),
            h_pre,
            h_post,
        },
    )
}

pub fn ffn_backward(w: &FfnWeights, cache: &FfnCache, dy: &[f32], grads: &mut FfnGrads) -> Vec<f32> {
    let d_h_post = linear_backward(&w.w_proj, &cache.h_post, dy, &mut grads.d_w_proj);
    let mut d_h_pre = vec![0.0_f32; cache.h_pre.len()];
    for i in 0..cache.h_pre.len() {
        d_h_pre[i] = d_h_post[i] * gelu_backward(cache.h_pre[i]);
    }
    linear_backward(&w.w_fc, &cache.x, &d_h_pre, &mut grads.d_w_fc)
}
