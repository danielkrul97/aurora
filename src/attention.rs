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

//! Building blocks for transformer attention.
//!
//! Forward + backward primitives, shared by CPU and (eventually) GPU.
//! All gradients are returned alongside saved-for-backward caches so
//! callers can chain modules without any global state.
//!
//! Layout conventions:
//! * sequences are flat row-major `[T * D]` tensors
//! * `T` = sequence length, `D` = feature dim, `H` = head dim

use rand::Rng;

const NEG_INF: f32 = -1.0e30;

// --- LayerNorm -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
    pub d: usize,
    /// RMSNorm mode: no mean subtraction, no beta (stays allocated as
    /// zeros so the checkpoint layout is unchanged, receives no grad).
    pub rms: bool,
}

impl LayerNorm {
    #[must_use]
    pub fn new(d: usize) -> Self {
        Self {
            gamma: vec![1.0; d],
            beta: vec![0.0; d],
            eps: 1e-5,
            d,
            rms: false,
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

impl LayerNormGrads {
    #[must_use]
    pub fn zeros(d: usize) -> Self {
        Self {
            d_gamma: vec![0.0; d],
            d_beta: vec![0.0; d],
        }
    }

    pub fn zero(&mut self) {
        self.d_gamma.fill(0.0);
        self.d_beta.fill(0.0);
    }
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
        let mu: f32 = if ln.rms {
            0.0
        } else {
            row.iter().sum::<f32>() * inv_d
        };
        let var: f32 = row.iter().map(|v| (v - mu).powi(2)).sum::<f32>() * inv_d;
        let s = (var + ln.eps).sqrt().recip();
        mean[i] = mu;
        inv_std[i] = s;
        for j in 0..d {
            let h = (row[j] - mu) * s;
            x_hat[i * d + j] = h;
            y[i * d + j] = if ln.rms {
                h * ln.gamma[j]
            } else {
                h * ln.gamma[j] + ln.beta[j]
            };
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
            if !ln.rms {
                grads.d_beta[j] += dy_row[j];
            }
            let g = dy_row[j] * ln.gamma[j];
            sum_dy_gamma += g;
            sum_dy_xh_gamma += g * xh[j];
        }
        // RMS mode: identical to LN minus the mean-coupling term.
        let mean_term = if ln.rms { 0.0 } else { 1.0 };
        for j in 0..d {
            let g = dy_row[j] * ln.gamma[j];
            dx[off + j] = s * (g - inv_d * (mean_term * sum_dy_gamma + xh[j] * sum_dy_xh_gamma));
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

    /// Scaled init for the residual-projection branch (Karpathy/nanoGPT
    /// convention: scale init by `1/sqrt(2 * n_layer)` to keep
    /// activations stable through pre-norm residual stack).
    pub fn new_scaled(d_in: usize, d_out: usize, scale: f32, rng: &mut impl Rng) -> Self {
        let bound = (6.0_f32 / (d_in + d_out) as f32).sqrt() * scale;
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

impl LinearGrads {
    #[must_use]
    pub fn zeros(d_in: usize, d_out: usize) -> Self {
        Self {
            d_w: vec![0.0; d_in * d_out],
        }
    }

    pub fn zero(&mut self) {
        self.d_w.fill(0.0);
    }
}

/// Position-block size for the CPU linear kernels. Each weight row is
/// streamed from memory once per block of positions instead of once per
/// position — the weight matrix is the dominant memory traffic (e.g.
/// d×d_ff = 2.3 MB ≫ L2 per-core slice), so this cuts DRAM reads ~8×.
/// Per-(position, output) dot products are computed in the same order
/// as the unblocked loop, so results are bit-exact.
const LINEAR_BLOCK_T: usize = 8;

/// Forward `y = x @ Wᵀ`. `x: [T * d_in]`, output `y: [T * d_out]`.
pub fn linear_forward(proj: &LinearProj, x: &[f32]) -> Vec<f32> {
    let d_in = proj.d_in;
    let d_out = proj.d_out;
    let t = x.len() / d_in;
    let mut y = vec![0.0_f32; t * d_out];
    let mut t0 = 0;
    while t0 < t {
        let tb = LINEAR_BLOCK_T.min(t - t0);
        for o in 0..d_out {
            let row = &proj.w[o * d_in..(o + 1) * d_in];
            for tt in 0..tb {
                let i = t0 + tt;
                let xr = &x[i * d_in..(i + 1) * d_in];
                let mut acc = 0.0_f32;
                for k in 0..d_in {
                    acc += row[k] * xr[k];
                }
                y[i * d_out + o] = acc;
            }
        }
        t0 += tb;
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
    let d_in = proj.d_in;
    let d_out = proj.d_out;
    let t = x.len() / d_in;
    let mut dx = vec![0.0_f32; x.len()];
    let mut t0 = 0;
    while t0 < t {
        let tb = LINEAR_BLOCK_T.min(t - t0);
        for o in 0..d_out {
            let row = &proj.w[o * d_in..(o + 1) * d_in];
            let drow = &mut grads.d_w[o * d_in..(o + 1) * d_in];
            for tt in 0..tb {
                let i = t0 + tt;
                let g = dy[i * d_out + o];
                let xr = &x[i * d_in..(i + 1) * d_in];
                let dxr = &mut dx[i * d_in..(i + 1) * d_in];
                for k in 0..d_in {
                    drow[k] += g * xr[k];
                    dxr[k] += g * row[k];
                }
            }
        }
        t0 += tb;
    }
    dx
}

// --- Rotary position embedding (RoPE) -----------------------------------

pub const ROPE_BASE: f32 = 10000.0;

/// Rotate adjacent pairs of Q/K features by position-dependent angles
/// (RoFormer-style interleaved layout). `x: [T * n_head * d_head]`
/// full-width rows; rotation is applied per head, in place.
///
/// `pos_offset` is the absolute position of row 0 (non-zero during
/// KV-cached decoding). `inverse` applies the transposed rotation —
/// because each rotation is orthogonal, that is exactly the backward
/// map for gradients: `dx = R(θ)ᵀ dy`.
pub fn rope_rotate(
    x: &mut [f32],
    n_head: usize,
    d_head: usize,
    pos_offset: usize,
    inverse: bool,
) {
    debug_assert_eq!(d_head % 2, 0, "RoPE requires even d_head");
    let d = n_head * d_head;
    let t = x.len() / d;
    let half = d_head / 2;
    // freq_i = BASE^(-2i/d_head) via multiplicative recurrence — one powf
    // per call instead of one per (position, head, pair). The angle is
    // identical across heads, so sin/cos is hoisted out of the head loop.
    let freq_step = ROPE_BASE.powf(-2.0 / d_head as f32);
    for ti in 0..t {
        let pos = (pos_offset + ti) as f32;
        let mut freq = 1.0_f32;
        for i in 0..half {
            let mut theta = pos * freq;
            freq *= freq_step;
            if inverse {
                theta = -theta;
            }
            let (sin, cos) = theta.sin_cos();
            for h in 0..n_head {
                let off = ti * d + h * d_head + 2 * i;
                let x0 = x[off];
                let x1 = x[off + 1];
                x[off] = x0 * cos - x1 * sin;
                x[off + 1] = x0 * sin + x1 * cos;
            }
        }
    }
}

// --- QK-norm --------------------------------------------------------------

pub const QK_NORM_EPS: f32 = 1e-6;

/// Parameter-free RMS normalization of each per-head row of `x`
/// (`[T * n_head * d_head]`), in place. Applied to Q and K after
/// projection, before RoPE.
pub fn qk_norm_forward(x: &mut [f32], n_head: usize, d_head: usize) {
    let d = n_head * d_head;
    let t = x.len() / d;
    let inv_dh = 1.0 / d_head as f32;
    for ti in 0..t {
        for h in 0..n_head {
            let off = ti * d + h * d_head;
            let row = &mut x[off..off + d_head];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() * inv_dh;
            let r = (ms + QK_NORM_EPS).sqrt().recip();
            for v in row.iter_mut() {
                *v *= r;
            }
        }
    }
}

/// Backward of `qk_norm_forward`: maps `dy` (grad w.r.t. the normalized
/// rows) to the grad w.r.t. `x_raw`, in place on `dy`.
pub fn qk_norm_backward(x_raw: &[f32], dy: &mut [f32], n_head: usize, d_head: usize) {
    let d = n_head * d_head;
    let t = x_raw.len() / d;
    let inv_dh = 1.0 / d_head as f32;
    for ti in 0..t {
        for h in 0..n_head {
            let off = ti * d + h * d_head;
            let xr = &x_raw[off..off + d_head];
            let dyr = &mut dy[off..off + d_head];
            let ms: f32 = xr.iter().map(|v| v * v).sum::<f32>() * inv_dh;
            let r = (ms + QK_NORM_EPS).sqrt().recip();
            let mut s = 0.0_f32;
            for j in 0..d_head {
                s += dyr[j] * xr[j] * r;
            }
            s *= inv_dh;
            for j in 0..d_head {
                let y = xr[j] * r;
                dyr[j] = r * (dyr[j] - y * s);
            }
        }
    }
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
/// * `causal`: if true, mask `s[i,j] = -∞` for `j > i` (lower-triangular).
///
/// Returns `(out: [T_q * d_v], cache)` where the cache holds the softmax
/// output for the backward pass.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_k: usize,
    d_v: usize,
    causal: bool,
) -> (Vec<f32>, AttentionCache) {
    scaled_dot_product_attention_windowed(q, k, v, d_k, d_v, causal, 0)
}

/// Causal attention with an optional sliding window: position `i`
/// attends to `j ∈ [i−window+1, i]` (`window == 0` = full causal).
/// The backward pass needs no window — masked positions have
/// `probs == 0` and contribute nothing.
pub fn scaled_dot_product_attention_windowed(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_k: usize,
    d_v: usize,
    causal: bool,
    window: usize,
) -> (Vec<f32>, AttentionCache) {
    let t_q = q.len() / d_k;
    let t_k = k.len() / d_k;
    debug_assert_eq!(v.len(), t_k * d_v);
    let scale = (d_k as f32).sqrt().recip();
    let mut probs = vec![0.0_f32; t_q * t_k];

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
            if causal && (j > i || (window > 0 && j + window <= i)) {
                s = NEG_INF;
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
/// Returns `(dq, dk, dv)`. The causal mask is implicit in `cache.probs` —
/// masked-out positions already have `prob == 0`, so their gradients are
/// naturally zero.
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
