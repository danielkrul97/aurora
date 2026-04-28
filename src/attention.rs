#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::doc_overindented_list_items,
    clippy::many_single_char_names,
    clippy::too_long_first_doc_paragraph
)]

//! Sprint 27: spiking attention building blocks.
//!
//! Stand-alone forward + backward primitives. Nothing here is integrated
//! into the gated-SNN training pipeline yet — these are unit-tested
//! components that future sprints will assemble into a hybrid
//! SNN/Transformer cell.
//!
//! Conventions:
//! * sequences are flat row-major `[T * D]` tensors
//! * `T` = sequence length, `D` = feature dim, `H` = head dim
//! * all gradients are returned alongside saved-for-backward caches so
//!   callers can chain modules without any global state
//!
//! Components:
//! * [`LayerNorm`] — per-token affine standardization
//! * [`LinearProj`] — bias-less linear (used for Q/K/V/output)
//! * [`scaled_dot_product_attention`] — softmax(QKᵀ/√d_k) V
//! * [`spike_gated_attention`] — same, but query/key/value rows masked
//!   by a binary spike vector (positions with spike=0 don't contribute)

use rand::Rng;

const NEG_INF: f32 = -1.0e30;

// --- LayerNorm -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
    pub d: usize,
}

impl LayerNorm {
    #[must_use]
    pub fn new(d: usize) -> Self {
        Self {
            gamma: vec![1.0; d],
            beta: vec![0.0; d],
            eps: 1e-5,
            d,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerNormCache {
    pub mean: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub x_hat: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct LayerNormGrads {
    pub d_gamma: Vec<f32>,
    pub d_beta: Vec<f32>,
}

/// Forward LayerNorm over the last axis.
/// `x` shape: `[T * D]`. Output `y` has the same shape.
pub fn layer_norm_forward(ln: &LayerNorm, x: &[f32]) -> (Vec<f32>, LayerNormCache) {
    let d = ln.d;
    debug_assert_eq!(x.len() % d, 0);
    let t = x.len() / d;
    let mut y = vec![0.0_f32; x.len()];
    let mut mean = vec![0.0_f32; t];
    let mut inv_std = vec![0.0_f32; t];
    let mut x_hat = vec![0.0_f32; x.len()];
    let inv_d = 1.0 / d as f32;
    for i in 0..t {
        let row = &x[i * d..(i + 1) * d];
        let mu: f32 = row.iter().sum::<f32>() * inv_d;
        let var: f32 = row.iter().map(|v| (v - mu).powi(2)).sum::<f32>() * inv_d;
        let s = (var + ln.eps).sqrt().recip();
        mean[i] = mu;
        inv_std[i] = s;
        for j in 0..d {
            let h = (row[j] - mu) * s;
            x_hat[i * d + j] = h;
            y[i * d + j] = h * ln.gamma[j] + ln.beta[j];
        }
    }
    (y, LayerNormCache { mean, inv_std, x_hat })
}

/// Backward LayerNorm. Returns `dx` and accumulates `dγ`, `dβ` into `grads`.
pub fn layer_norm_backward(
    ln: &LayerNorm,
    cache: &LayerNormCache,
    dy: &[f32],
    grads: &mut LayerNormGrads,
) -> Vec<f32> {
    let d = ln.d;
    let t = dy.len() / d;
    if grads.d_gamma.len() != d {
        grads.d_gamma = vec![0.0; d];
        grads.d_beta = vec![0.0; d];
    }
    let mut dx = vec![0.0_f32; dy.len()];
    let inv_d = 1.0 / d as f32;
    for i in 0..t {
        let off = i * d;
        let s = cache.inv_std[i];
        let xh = &cache.x_hat[off..off + d];
        let dy_row = &dy[off..off + d];

        let mut sum_dy_xh_gamma = 0.0_f32;
        let mut sum_dy_gamma = 0.0_f32;
        for j in 0..d {
            grads.d_gamma[j] += dy_row[j] * xh[j];
            grads.d_beta[j] += dy_row[j];
            let g = dy_row[j] * ln.gamma[j];
            sum_dy_gamma += g;
            sum_dy_xh_gamma += g * xh[j];
        }
        for j in 0..d {
            let g = dy_row[j] * ln.gamma[j];
            dx[off + j] = s * (g - inv_d * (sum_dy_gamma + xh[j] * sum_dy_xh_gamma));
        }
    }
    dx
}

// --- Linear projection ---------------------------------------------------

#[derive(Clone, Debug)]
pub struct LinearProj {
    /// row-major `[d_out * d_in]`
    pub w: Vec<f32>,
    pub d_in: usize,
    pub d_out: usize,
}

impl LinearProj {
    /// Xavier-uniform init.
    pub fn new(d_in: usize, d_out: usize, rng: &mut impl Rng) -> Self {
        let bound = (6.0_f32 / (d_in + d_out) as f32).sqrt();
        let w = (0..d_in * d_out)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * bound)
            .collect();
        Self { w, d_in, d_out }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinearGrads {
    pub d_w: Vec<f32>,
}

/// Forward `y = x @ Wᵀ`. `x: [T * d_in]`, output `y: [T * d_out]`.
pub fn linear_forward(proj: &LinearProj, x: &[f32]) -> Vec<f32> {
    let t = x.len() / proj.d_in;
    let mut y = vec![0.0_f32; t * proj.d_out];
    for i in 0..t {
        let xr = &x[i * proj.d_in..(i + 1) * proj.d_in];
        let yr = &mut y[i * proj.d_out..(i + 1) * proj.d_out];
        for o in 0..proj.d_out {
            let row = &proj.w[o * proj.d_in..(o + 1) * proj.d_in];
            let mut acc = 0.0_f32;
            for k in 0..proj.d_in {
                acc += row[k] * xr[k];
            }
            yr[o] = acc;
        }
    }
    y
}

/// Backward linear. Returns `dx`, accumulates `d_w` into `grads`.
pub fn linear_backward(
    proj: &LinearProj,
    x: &[f32],
    dy: &[f32],
    grads: &mut LinearGrads,
) -> Vec<f32> {
    if grads.d_w.len() != proj.w.len() {
        grads.d_w = vec![0.0; proj.w.len()];
    }
    let t = x.len() / proj.d_in;
    let mut dx = vec![0.0_f32; x.len()];
    for i in 0..t {
        let xr = &x[i * proj.d_in..(i + 1) * proj.d_in];
        let dyr = &dy[i * proj.d_out..(i + 1) * proj.d_out];
        let dxr = &mut dx[i * proj.d_in..(i + 1) * proj.d_in];
        for o in 0..proj.d_out {
            let g = dyr[o];
            let row = &proj.w[o * proj.d_in..(o + 1) * proj.d_in];
            let drow = &mut grads.d_w[o * proj.d_in..(o + 1) * proj.d_in];
            for k in 0..proj.d_in {
                drow[k] += g * xr[k];
                dxr[k] += g * row[k];
            }
        }
    }
    dx
}

// --- Scaled dot-product attention ---------------------------------------

#[derive(Clone, Debug, Default)]
pub struct AttentionCache {
    /// Softmax-normalized scores, `[T_q * T_k]`.
    pub probs: Vec<f32>,
    pub t_q: usize,
    pub t_k: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub scale: f32,
}

/// Single-head scaled dot-product attention.
///
/// * `q`: `[T_q * d_k]`
/// * `k`: `[T_k * d_k]`
/// * `v`: `[T_k * d_v]`
/// * `key_mask`: optional `[T_k]`; positions with `mask[i] == 0.0` are
///   set to `-∞` before the softmax (used for spike gating).
///
/// Returns `(out: [T_q * d_v], cache)` where the cache holds the softmax
/// output for the backward pass.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_k: usize,
    d_v: usize,
    key_mask: Option<&[f32]>,
) -> (Vec<f32>, AttentionCache) {
    let t_q = q.len() / d_k;
    let t_k = k.len() / d_k;
    debug_assert_eq!(v.len(), t_k * d_v);
    let scale = (d_k as f32).sqrt().recip();
    let mut probs = vec![0.0_f32; t_q * t_k];

    // scores S = Q Kᵀ / √d_k, then row-wise softmax (with optional mask).
    for i in 0..t_q {
        let qr = &q[i * d_k..(i + 1) * d_k];
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..t_k {
            let kr = &k[j * d_k..(j + 1) * d_k];
            let mut s = 0.0_f32;
            for c in 0..d_k {
                s += qr[c] * kr[c];
            }
            s *= scale;
            if let Some(m) = key_mask {
                if m[j] == 0.0 {
                    s = NEG_INF;
                }
            }
            probs[i * t_k + j] = s;
            if s > row_max {
                row_max = s;
            }
        }
        let mut sum = 0.0_f32;
        for j in 0..t_k {
            let e = (probs[i * t_k + j] - row_max).exp();
            probs[i * t_k + j] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for j in 0..t_k {
            probs[i * t_k + j] *= inv;
        }
    }

    // out = probs @ V
    let mut out = vec![0.0_f32; t_q * d_v];
    for i in 0..t_q {
        let pr = &probs[i * t_k..(i + 1) * t_k];
        let or = &mut out[i * d_v..(i + 1) * d_v];
        for j in 0..t_k {
            let p = pr[j];
            if p == 0.0 {
                continue;
            }
            let vr = &v[j * d_v..(j + 1) * d_v];
            for c in 0..d_v {
                or[c] += p * vr[c];
            }
        }
    }

    (
        out,
        AttentionCache {
            probs,
            t_q,
            t_k,
            d_k,
            d_v,
            scale,
        },
    )
}

/// Backward scaled dot-product attention.
///
/// Returns `(dq, dk, dv)`. The mask used in forward (if any) is implicit
/// in `cache.probs` — masked-out positions already have `prob == 0`, so
/// their gradients are naturally zero.
pub fn scaled_dot_product_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cache: &AttentionCache,
    d_out: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let t_q = cache.t_q;
    let t_k = cache.t_k;
    let d_k = cache.d_k;
    let d_v = cache.d_v;

    // dV[j] = Σ_i probs[i,j] * d_out[i]
    let mut dv = vec![0.0_f32; t_k * d_v];
    for i in 0..t_q {
        let pr = &cache.probs[i * t_k..(i + 1) * t_k];
        let dor = &d_out[i * d_v..(i + 1) * d_v];
        for j in 0..t_k {
            let p = pr[j];
            if p == 0.0 {
                continue;
            }
            let dvr = &mut dv[j * d_v..(j + 1) * d_v];
            for c in 0..d_v {
                dvr[c] += p * dor[c];
            }
        }
    }

    // dProbs[i,j] = d_out[i] · V[j]
    let mut d_probs = vec![0.0_f32; t_q * t_k];
    for i in 0..t_q {
        let dor = &d_out[i * d_v..(i + 1) * d_v];
        for j in 0..t_k {
            let vr = &v[j * d_v..(j + 1) * d_v];
            let mut s = 0.0_f32;
            for c in 0..d_v {
                s += dor[c] * vr[c];
            }
            d_probs[i * t_k + j] = s;
        }
    }

    // dScores[i,j] = probs[i,j] * (dProbs[i,j] - Σ_k probs[i,k] dProbs[i,k])
    // (standard softmax-Jacobian-vector product)
    let mut d_scores = vec![0.0_f32; t_q * t_k];
    for i in 0..t_q {
        let pr = &cache.probs[i * t_k..(i + 1) * t_k];
        let dpr = &d_probs[i * t_k..(i + 1) * t_k];
        let mut row_dot = 0.0_f32;
        for j in 0..t_k {
            row_dot += pr[j] * dpr[j];
        }
        for j in 0..t_k {
            d_scores[i * t_k + j] = pr[j] * (dpr[j] - row_dot);
        }
    }

    // dQ = (dScores * scale) @ K
    // dK = (dScoresᵀ * scale) @ Q
    let mut dq = vec![0.0_f32; t_q * d_k];
    let mut dk_buf = vec![0.0_f32; t_k * d_k];
    let scale = cache.scale;
    for i in 0..t_q {
        let qr = &q[i * d_k..(i + 1) * d_k];
        let dqr = &mut dq[i * d_k..(i + 1) * d_k];
        let dsr = &d_scores[i * t_k..(i + 1) * t_k];
        for j in 0..t_k {
            let g = dsr[j] * scale;
            if g == 0.0 {
                continue;
            }
            let kr = &k[j * d_k..(j + 1) * d_k];
            let dkr = &mut dk_buf[j * d_k..(j + 1) * d_k];
            for c in 0..d_k {
                dqr[c] += g * kr[c];
                dkr[c] += g * qr[c];
            }
        }
    }

    (dq, dk_buf, dv)
}

/// Convenience wrapper: spike-gated attention. The `key_spikes` vector
/// must have the same length as the number of key/value tokens; positions
/// where `key_spikes[i] == 0.0` are completely masked from the softmax.
///
/// This is the key biologically-flavored variant: only currently-spiking
/// tokens contribute to attention. With all-ones it degenerates to plain
/// SDPA.
pub fn spike_gated_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_k: usize,
    d_v: usize,
    key_spikes: &[f32],
) -> (Vec<f32>, AttentionCache) {
    scaled_dot_product_attention(q, k, v, d_k, d_v, Some(key_spikes))
}
