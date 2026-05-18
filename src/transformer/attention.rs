#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]

//! Causal multi-head attention.
//!
//! Wraps single-head `scaled_dot_product_attention` from `crate::attention`
//! with per-head split/concat and causal masking. Forward + backward
//! return caches sized for the full batch × time × dim grid.

use rand::Rng;

use crate::attention::{
    linear_backward, linear_forward, scaled_dot_product_attention,
    scaled_dot_product_attention_backward, AttentionCache, LinearGrads, LinearProj,
};

#[derive(Clone, Debug)]
pub struct MhaWeights {
    pub w_q: LinearProj,
    pub w_k: LinearProj,
    pub w_v: LinearProj,
    pub w_o: LinearProj,
    pub n_head: usize,
    pub d_head: usize,
    pub d_model: usize,
}

impl MhaWeights {
    /// Initialize Q/K/V with plain Xavier; O with nanoGPT residual
    /// scaling `1 / sqrt(2 * n_layer)` for stable pre-norm stacks.
    pub fn new<R: Rng>(d_model: usize, n_head: usize, n_layer: usize, rng: &mut R) -> Self {
        assert!(d_model % n_head == 0, "d_model must be divisible by n_head");
        let d_head = d_model / n_head;
        let resid_scale = (2.0_f32 * n_layer as f32).sqrt().recip();
        Self {
            w_q: LinearProj::new(d_model, d_model, rng),
            w_k: LinearProj::new(d_model, d_model, rng),
            w_v: LinearProj::new(d_model, d_model, rng),
            w_o: LinearProj::new_scaled(d_model, d_model, resid_scale, rng),
            n_head,
            d_head,
            d_model,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MhaGrads {
    pub d_w_q: LinearGrads,
    pub d_w_k: LinearGrads,
    pub d_w_v: LinearGrads,
    pub d_w_o: LinearGrads,
}

impl MhaGrads {
    #[must_use]
    pub fn zeros(d_model: usize) -> Self {
        Self {
            d_w_q: LinearGrads::zeros(d_model, d_model),
            d_w_k: LinearGrads::zeros(d_model, d_model),
            d_w_v: LinearGrads::zeros(d_model, d_model),
            d_w_o: LinearGrads::zeros(d_model, d_model),
        }
    }

    pub fn zero(&mut self) {
        self.d_w_q.zero();
        self.d_w_k.zero();
        self.d_w_v.zero();
        self.d_w_o.zero();
    }
}

#[derive(Clone, Debug, Default)]
pub struct MhaCache {
    pub x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub heads_out: Vec<f32>,
    pub attn_caches: Vec<AttentionCache>,
}

/// Causal MHA forward. `x: [T * D]` → `y: [T * D]` with `D = n_head * d_head`.
pub fn causal_mha_forward(w: &MhaWeights, x: &[f32]) -> (Vec<f32>, MhaCache) {
    let d = w.d_model;
    let n_head = w.n_head;
    let d_head = w.d_head;
    debug_assert_eq!(x.len() % d, 0);
    let t = x.len() / d;

    let q = linear_forward(&w.w_q, x);
    let k = linear_forward(&w.w_k, x);
    let v = linear_forward(&w.w_v, x);

    let mut heads_out = vec![0.0_f32; t * d];
    let mut attn_caches = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let mut q_h = vec![0.0_f32; t * d_head];
        let mut k_h = vec![0.0_f32; t * d_head];
        let mut v_h = vec![0.0_f32; t * d_head];
        for ti in 0..t {
            let src_off = ti * d + h * d_head;
            let dst_off = ti * d_head;
            q_h[dst_off..dst_off + d_head].copy_from_slice(&q[src_off..src_off + d_head]);
            k_h[dst_off..dst_off + d_head].copy_from_slice(&k[src_off..src_off + d_head]);
            v_h[dst_off..dst_off + d_head].copy_from_slice(&v[src_off..src_off + d_head]);
        }
        let (out_h, cache_h) =
            scaled_dot_product_attention(&q_h, &k_h, &v_h, d_head, d_head, true);
        for ti in 0..t {
            let dst_off = ti * d + h * d_head;
            let src_off = ti * d_head;
            heads_out[dst_off..dst_off + d_head]
                .copy_from_slice(&out_h[src_off..src_off + d_head]);
        }
        attn_caches.push(cache_h);
    }

    let y = linear_forward(&w.w_o, &heads_out);
    (
        y,
        MhaCache {
            x: x.to_vec(),
            q,
            k,
            v,
            heads_out,
            attn_caches,
        },
    )
}

/// Causal MHA backward. Returns `dx` and accumulates per-projection grads.
pub fn causal_mha_backward(
    w: &MhaWeights,
    cache: &MhaCache,
    dy: &[f32],
    grads: &mut MhaGrads,
) -> Vec<f32> {
    let d = w.d_model;
    let n_head = w.n_head;
    let d_head = w.d_head;
    let t = cache.x.len() / d;

    let d_heads_out = linear_backward(&w.w_o, &cache.heads_out, dy, &mut grads.d_w_o);

    let mut d_q = vec![0.0_f32; t * d];
    let mut d_k = vec![0.0_f32; t * d];
    let mut d_v = vec![0.0_f32; t * d];
    for h in 0..n_head {
        let mut q_h = vec![0.0_f32; t * d_head];
        let mut k_h = vec![0.0_f32; t * d_head];
        let mut v_h = vec![0.0_f32; t * d_head];
        let mut d_out_h = vec![0.0_f32; t * d_head];
        for ti in 0..t {
            let src_off = ti * d + h * d_head;
            let dst_off = ti * d_head;
            q_h[dst_off..dst_off + d_head].copy_from_slice(&cache.q[src_off..src_off + d_head]);
            k_h[dst_off..dst_off + d_head].copy_from_slice(&cache.k[src_off..src_off + d_head]);
            v_h[dst_off..dst_off + d_head].copy_from_slice(&cache.v[src_off..src_off + d_head]);
            d_out_h[dst_off..dst_off + d_head]
                .copy_from_slice(&d_heads_out[src_off..src_off + d_head]);
        }
        let (dq_h, dk_h, dv_h) = scaled_dot_product_attention_backward(
            &q_h,
            &k_h,
            &v_h,
            &cache.attn_caches[h],
            &d_out_h,
        );
        for ti in 0..t {
            let dst_off = ti * d + h * d_head;
            let src_off = ti * d_head;
            d_q[dst_off..dst_off + d_head]
                .copy_from_slice(&dq_h[src_off..src_off + d_head]);
            d_k[dst_off..dst_off + d_head]
                .copy_from_slice(&dk_h[src_off..src_off + d_head]);
            d_v[dst_off..dst_off + d_head]
                .copy_from_slice(&dv_h[src_off..src_off + d_head]);
        }
    }

    let dx_q = linear_backward(&w.w_q, &cache.x, &d_q, &mut grads.d_w_q);
    let dx_k = linear_backward(&w.w_k, &cache.x, &d_k, &mut grads.d_w_k);
    let dx_v = linear_backward(&w.w_v, &cache.x, &d_v, &mut grads.d_w_v);
    let mut dx = vec![0.0_f32; cache.x.len()];
    for i in 0..dx.len() {
        dx[i] = dx_q[i] + dx_k[i] + dx_v[i];
    }
    dx
}

