#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Naïve f32 dense linear-algebra primitives used by the CPU
//! transformer path. f32-specialized; generic abstraction is premature.

/// `out = M @ v` for `M: [rows × cols]`, `v: [cols]`.
#[must_use]
pub fn matvec(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), rows * cols);
    debug_assert_eq!(v.len(), cols);
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let row_offset = r * cols;
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += m[row_offset + c] * v[c];
        }
        out[r] = acc;
    }
    out
}

/// `out = Mᵀ @ v` for `M: [rows × cols]`, `v: [rows]` → output of length `cols`.
#[must_use]
pub fn matvec_transpose(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), rows * cols);
    debug_assert_eq!(v.len(), rows);
    let mut out = vec![0.0_f32; cols];
    for r in 0..rows {
        let scale = v[r];
        if scale == 0.0 {
            continue;
        }
        let row_offset = r * cols;
        for c in 0..cols {
            out[c] += m[row_offset + c] * scale;
        }
    }
    out
}

/// Dot product with 8 independent accumulators so LLVM lowers the inner
/// loop to packed AVX2 FMA under `target-cpu=native`. A plain
/// `acc += a*b` reduction stays scalar — f32 add is non-associative, so
/// the compiler may not reassociate it into SIMD lanes. The 8-way
/// partial sum is deterministic but not bit-identical to a sequential
/// reduction.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f32; 8];
    let mut ai = a.chunks_exact(8);
    let mut bi = b.chunks_exact(8);
    for (x, y) in ai.by_ref().zip(bi.by_ref()) {
        // mul_add emits a true FMA (vfmadd) — a plain `acc += x*y` stays
        // mul+add because Rust does not contract f32 ops by default.
        for i in 0..8 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    let mut s =
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ai.remainder().iter().zip(bi.remainder()) {
        s += x * y;
    }
    s
}

/// GELU activation (tanh approximation, used by GPT-2 / nanoGPT).
/// `gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
#[must_use]
pub fn gelu(x: f32) -> f32 {
    let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// `d_gelu/dx` at `x`, used in backward.
#[must_use]
pub fn gelu_backward(x: f32) -> f32 {
    let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner_dx = k * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner_dx
}

/// Squared ReLU (Primer / nanoGPT-speedrun FFN activation).
#[must_use]
pub fn relu2(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

/// `d_relu2/dx` at `x`.
#[must_use]
pub fn relu2_backward(x: f32) -> f32 {
    2.0 * x.max(0.0)
}

/// SiLU / swish activation: `silu(x) = x · sigmoid(x)`. Gate of SwiGLU.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `d_silu/dx` at `x`. With `s = sigmoid(x)`: `silu' = s + x·s·(1−s)`.
#[must_use]
pub fn silu_backward(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}
