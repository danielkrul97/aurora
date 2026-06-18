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
    linear_backward, linear_forward, qk_norm_backward, qk_norm_forward, rope_rotate,
    scaled_dot_product_attention, scaled_dot_product_attention_backward, AttentionCache,
    LinearGrads, LinearProj,
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
    /// Apply rotary position embedding to Q/K after projection. The
    /// cached q/k (and any KV history built from them) hold the
    /// already-rotated values.
    pub rope: bool,
    /// Parameter-free RMS norm of each per-head Q/K row after
    /// projection, before RoPE.
    pub qk_norm: bool,
    /// Differential attention (DIFF Transformer): two softmax maps from
    /// the two halves of Q/K, combined as `A1 − λ·A2` applied to V.
    /// `lambda` is one learned scalar per head; empty when off.
    pub diff_attn: bool,
    pub lambda: Vec<f32>,
}

impl MhaWeights {
    /// Q/K/V Xavier; w_o is zero-initialized so the MHA sublayer is an
    /// identity at init (output = h_in). Residual stream dominates
    /// early; attention learns its own scale from gradient signal.
    pub fn new<R: Rng>(d_model: usize, n_head: usize, rng: &mut R) -> Self {
        assert!(d_model % n_head == 0, "d_model must be divisible by n_head");
        let d_head = d_model / n_head;
        Self {
            w_q: LinearProj::new(d_model, d_model, rng),
            w_k: LinearProj::new(d_model, d_model, rng),
            w_v: LinearProj::new(d_model, d_model, rng),
            w_o: LinearProj::new_scaled(d_model, d_model, 0.0, rng),
            n_head,
            d_head,
            d_model,
            rope: false,
            qk_norm: false,
            diff_attn: false,
            lambda: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MhaGrads {
    pub d_w_q: LinearGrads,
    pub d_w_k: LinearGrads,
    pub d_w_v: LinearGrads,
    pub d_w_o: LinearGrads,
    /// Per-head diff-attention λ grad; empty when diff_attn is off.
    pub d_lambda: Vec<f32>,
}

impl MhaGrads {
    #[must_use]
    pub fn zeros(d_model: usize) -> Self {
        Self {
            d_w_q: LinearGrads::zeros(d_model, d_model),
            d_w_k: LinearGrads::zeros(d_model, d_model),
            d_w_v: LinearGrads::zeros(d_model, d_model),
            d_w_o: LinearGrads::zeros(d_model, d_model),
            d_lambda: Vec::new(),
        }
    }

    pub fn zero(&mut self) {
        self.d_w_q.zero();
        self.d_w_k.zero();
        self.d_w_v.zero();
        self.d_w_o.zero();
        self.d_lambda.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct MhaCache {
    pub x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Pre-norm Q/K (empty unless `qk_norm`); needed by its backward.
    pub q_raw: Vec<f32>,
    pub k_raw: Vec<f32>,
    pub heads_out: Vec<f32>,
    /// Second-map attention output `[T * D]` (diff_attn only), kept for
    /// the λ gradient. Empty otherwise. With diff_attn, `attn_caches`
    /// holds two caches per head (map 1 then map 2).
    pub out2: Vec<f32>,
    pub attn_caches: Vec<AttentionCache>,
}

/// Causal MHA forward. `x: [T * D]` → `y: [T * D]` with `D = n_head * d_head`.
pub fn causal_mha_forward(w: &MhaWeights, x: &[f32]) -> (Vec<f32>, MhaCache) {
    let d = w.d_model;
    let n_head = w.n_head;
    let d_head = w.d_head;
    debug_assert_eq!(x.len() % d, 0);
    let t = x.len() / d;

    let mut q = linear_forward(&w.w_q, x);
    let mut k = linear_forward(&w.w_k, x);
    let v = linear_forward(&w.w_v, x);
    let (q_raw, k_raw) = if w.qk_norm {
        let q_raw = q.clone();
        let k_raw = k.clone();
        qk_norm_forward(&mut q, n_head, d_head);
        qk_norm_forward(&mut k, n_head, d_head);
        (q_raw, k_raw)
    } else {
        (Vec::new(), Vec::new())
    };
    if w.rope {
        rope_rotate(&mut q, n_head, d_head, 0, false);
        rope_rotate(&mut k, n_head, d_head, 0, false);
    }

    let mut heads_out = vec![0.0_f32; t * d];
    let mut out2_full = if w.diff_attn { vec![0.0_f32; t * d] } else { Vec::new() };
    let caches_per_head = if w.diff_attn { 2 } else { 1 };
    let mut attn_caches = Vec::with_capacity(n_head * caches_per_head);
    // Per-head scratch reused across heads (every element is overwritten
    // by the gather loop below).
    let mut q_h = vec![0.0_f32; t * d_head];
    let mut k_h = vec![0.0_f32; t * d_head];
    let mut v_h = vec![0.0_f32; t * d_head];
    for h in 0..n_head {
        for ti in 0..t {
            let src_off = ti * d + h * d_head;
            let dst_off = ti * d_head;
            q_h[dst_off..dst_off + d_head].copy_from_slice(&q[src_off..src_off + d_head]);
            k_h[dst_off..dst_off + d_head].copy_from_slice(&k[src_off..src_off + d_head]);
            v_h[dst_off..dst_off + d_head].copy_from_slice(&v[src_off..src_off + d_head]);
        }
        if w.diff_attn {
            // Two softmax maps from the two halves of Q/K (shared V), then
            // out = A1·V − λ·A2·V = SDPA1 − λ·SDPA2 (distributes over V).
            let half = d_head / 2;
            let mut q1 = vec![0.0_f32; t * half];
            let mut q2 = vec![0.0_f32; t * half];
            let mut k1 = vec![0.0_f32; t * half];
            let mut k2 = vec![0.0_f32; t * half];
            for ti in 0..t {
                for c in 0..half {
                    q1[ti * half + c] = q_h[ti * d_head + c];
                    q2[ti * half + c] = q_h[ti * d_head + half + c];
                    k1[ti * half + c] = k_h[ti * d_head + c];
                    k2[ti * half + c] = k_h[ti * d_head + half + c];
                }
            }
            let (out1, cache1) = scaled_dot_product_attention(&q1, &k1, &v_h, half, d_head, true);
            let (out2, cache2) = scaled_dot_product_attention(&q2, &k2, &v_h, half, d_head, true);
            let lam = w.lambda[h];
            for ti in 0..t {
                let dst_off = ti * d + h * d_head;
                let src_off = ti * d_head;
                for c in 0..d_head {
                    heads_out[dst_off + c] = out1[src_off + c] - lam * out2[src_off + c];
                    out2_full[dst_off + c] = out2[src_off + c];
                }
            }
            attn_caches.push(cache1);
            attn_caches.push(cache2);
        } else {
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
    }

    let y = linear_forward(&w.w_o, &heads_out);
    (
        y,
        MhaCache {
            x: x.to_vec(),
            q,
            k,
            v,
            q_raw,
            k_raw,
            heads_out,
            out2: out2_full,
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
    let mut q_h = vec![0.0_f32; t * d_head];
    let mut k_h = vec![0.0_f32; t * d_head];
    let mut v_h = vec![0.0_f32; t * d_head];
    let mut d_out_h = vec![0.0_f32; t * d_head];
    for h in 0..n_head {
        for ti in 0..t {
            let src_off = ti * d + h * d_head;
            let dst_off = ti * d_head;
            q_h[dst_off..dst_off + d_head].copy_from_slice(&cache.q[src_off..src_off + d_head]);
            k_h[dst_off..dst_off + d_head].copy_from_slice(&cache.k[src_off..src_off + d_head]);
            v_h[dst_off..dst_off + d_head].copy_from_slice(&cache.v[src_off..src_off + d_head]);
            d_out_h[dst_off..dst_off + d_head]
                .copy_from_slice(&d_heads_out[src_off..src_off + d_head]);
        }
        if w.diff_attn {
            let half = d_head / 2;
            let lam = w.lambda[h];
            let mut q1 = vec![0.0_f32; t * half];
            let mut q2 = vec![0.0_f32; t * half];
            let mut k1 = vec![0.0_f32; t * half];
            let mut k2 = vec![0.0_f32; t * half];
            for ti in 0..t {
                for c in 0..half {
                    q1[ti * half + c] = q_h[ti * d_head + c];
                    q2[ti * half + c] = q_h[ti * d_head + half + c];
                    k1[ti * half + c] = k_h[ti * d_head + c];
                    k2[ti * half + c] = k_h[ti * d_head + half + c];
                }
            }
            // out = out1 − λ·out2 ⇒ d_out1 = d_out_h, d_out2 = −λ·d_out_h,
            // dλ = −Σ d_out_h ⊙ out2.
            let mut d_out2 = vec![0.0_f32; t * d_head];
            for ti in 0..t {
                for c in 0..d_head {
                    let g = d_out_h[ti * d_head + c];
                    grads.d_lambda[h] -= g * cache.out2[ti * d + h * d_head + c];
                    d_out2[ti * d_head + c] = -lam * g;
                }
            }
            let (dq1, dk1, dv1) = scaled_dot_product_attention_backward(
                &q1, &k1, &v_h, &cache.attn_caches[2 * h], &d_out_h,
            );
            let (dq2, dk2, dv2) = scaled_dot_product_attention_backward(
                &q2, &k2, &v_h, &cache.attn_caches[2 * h + 1], &d_out2,
            );
            for ti in 0..t {
                let dst = ti * d + h * d_head;
                for c in 0..half {
                    d_q[dst + c] = dq1[ti * half + c];
                    d_q[dst + half + c] = dq2[ti * half + c];
                    d_k[dst + c] = dk1[ti * half + c];
                    d_k[dst + half + c] = dk2[ti * half + c];
                }
                for c in 0..d_head {
                    d_v[dst + c] = dv1[ti * d_head + c] + dv2[ti * d_head + c];
                }
            }
        } else {
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
    }

    // d_q/d_k are gradients w.r.t. the rotated q/k; map them back
    // through the (orthogonal) rotation before the projection backward.
    if w.rope {
        rope_rotate(&mut d_q, n_head, d_head, 0, true);
        rope_rotate(&mut d_k, n_head, d_head, 0, true);
    }
    if w.qk_norm {
        qk_norm_backward(&cache.q_raw, &mut d_q, n_head, d_head);
        qk_norm_backward(&cache.k_raw, &mut d_k, n_head, d_head);
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

