#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::similar_names
)]

//! AdamW (Loshchilov & Hutter 2019) + Muon (Keller Jordan 2024).
//!
//! Muon is a hybrid stepper for matrix parameters: SGD-with-momentum
//! whose update is orthogonalized via a quintic Newton–Schulz iteration
//! before being applied. Used together with AdamW for 1D parameters
//! (LayerNorm γ/β, biases) and lookup tables (token/positional embed).

// ───── AdamW ──────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 6e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdamGroup {
    pub m1: Vec<f32>,
    pub m2: Vec<f32>,
}

impl AdamGroup {
    #[must_use]
    pub fn zeros(size: usize) -> Self {
        Self {
            m1: vec![0.0; size],
            m2: vec![0.0; size],
        }
    }
}

/// `apply_wd` selects whether decoupled weight decay is applied to this
/// group — typically off for biases, LayerNorm γ/β, and embeddings.
pub fn adamw_apply(
    params: &mut [f32],
    grads: &[f32],
    group: &mut AdamGroup,
    config: &AdamWConfig,
    bc1: f32,
    bc2: f32,
    apply_wd: bool,
) {
    debug_assert_eq!(params.len(), grads.len());
    debug_assert_eq!(params.len(), group.m1.len());
    debug_assert_eq!(params.len(), group.m2.len());
    let one_minus_b1 = 1.0 - config.beta1;
    let one_minus_b2 = 1.0 - config.beta2;
    for i in 0..params.len() {
        let g = grads[i];
        group.m1[i] = config.beta1 * group.m1[i] + one_minus_b1 * g;
        group.m2[i] = config.beta2 * group.m2[i] + one_minus_b2 * g * g;
        let m1_hat = group.m1[i] / bc1;
        let m2_hat = group.m2[i] / bc2;
        let mut step = config.lr * m1_hat / (m2_hat.sqrt() + config.eps);
        if apply_wd {
            step += config.lr * config.weight_decay * params[i];
        }
        params[i] -= step;
    }
}

#[must_use]
pub fn bias_correction(t: usize, beta1: f32, beta2: f32) -> (f32, f32) {
    let t = t as i32;
    (1.0 - beta1.powi(t), 1.0 - beta2.powi(t))
}

// ───── Muon ───────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub struct MuonConfig {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub weight_decay: f32,
    pub ns_steps: usize,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            weight_decay: 0.0,
            ns_steps: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MuonGroup {
    pub m: Vec<f32>,
}

impl MuonGroup {
    #[must_use]
    pub fn zeros(size: usize) -> Self {
        Self { m: vec![0.0; size] }
    }
}

/// Per-call scratch shared across all Muon param updates within one
/// optimizer step. Caller owns it across steps to avoid re-allocation.
#[derive(Clone, Debug, Default)]
pub struct MuonScratch {
    update: Vec<f32>,
    ns: Vec<f32>,
}

impl MuonScratch {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Quintic Newton–Schulz iteration that pushes the singular values of
/// `g` (`rows × cols`, row-major) toward 1, i.e. orthogonalizes the
/// matrix. The Frobenius norm is bounded first so the iteration starts
/// inside its basin of convergence. Coefficients `(3.4445, -4.7750,
/// 2.0315)` are the Jordan/You quintic.
pub fn newton_schulz5(
    g: &mut [f32],
    rows: usize,
    cols: usize,
    scratch: &mut Vec<f32>,
    n_iter: usize,
) {
    debug_assert_eq!(g.len(), rows * cols);
    let m = rows.min(cols);
    let n = rows.max(cols);
    let needed = 2 * m * m + 2 * m * n;
    if scratch.len() < needed {
        scratch.resize(needed, 0.0);
    }

    // 1. Frobenius normalize — over-estimates singular values but keeps
    //    NS in its basin (true spectral norm would be tighter but costs
    //    a power iteration per call).
    let mut frob_sq = 0.0_f32;
    for &v in g.iter() {
        frob_sq += v * v;
    }
    if frob_sq <= 1e-30 {
        return;
    }
    let inv_scale = 1.0 / (frob_sq.sqrt() + 1e-7);
    for v in g.iter_mut() {
        *v *= inv_scale;
    }

    // 2. Carve scratch: A, B (both m×m), and two m×n ping-pong buffers.
    let (a_buf, rest) = scratch.split_at_mut(m * m);
    let (b_buf, rest) = rest.split_at_mut(m * m);
    let (buf_x, buf_y) = rest.split_at_mut(m * n);
    let buf_x = &mut buf_x[..m * n];
    let buf_y = &mut buf_y[..m * n];

    // Place g into wide orientation [m × n]. If rows > cols, physical
    // transpose; otherwise straight copy.
    let transposed = rows > cols;
    if transposed {
        for r in 0..rows {
            for c in 0..cols {
                buf_x[c * rows + r] = g[r * cols + c];
            }
        }
    } else {
        buf_x.copy_from_slice(g);
    }

    let a_coef = 3.4445_f32;
    let b_coef = -4.7750_f32;
    let c_coef = 2.0315_f32;

    // Ping-pong: cur is read, nxt is written; swap references each iter.
    let mut cur: &mut [f32] = buf_x;
    let mut nxt: &mut [f32] = buf_y;

    for _ in 0..n_iter {
        // A = X X^T  ([m × n] · [n × m] → [m × m]), symmetric.
        for i in 0..m {
            for j in i..m {
                let mut s = 0.0_f32;
                for k in 0..n {
                    s += cur[i * n + k] * cur[j * n + k];
                }
                a_buf[i * m + j] = s;
                if i != j {
                    a_buf[j * m + i] = s;
                }
            }
        }
        // B = b·A + c·A²  ([m × m]).
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0_f32;
                for k in 0..m {
                    s += a_buf[i * m + k] * a_buf[k * m + j];
                }
                b_buf[i * m + j] = c_coef * s + b_coef * a_buf[i * m + j];
            }
        }
        // X_next = a·X + B X  ([m × m] · [m × n] → [m × n]).
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for k in 0..m {
                    s += b_buf[i * m + k] * cur[k * n + j];
                }
                nxt[i * n + j] = a_coef * cur[i * n + j] + s;
            }
        }
        std::mem::swap(&mut cur, &mut nxt);
    }

    // 3. Write back into g (transpose if we transposed at start).
    if transposed {
        for r in 0..rows {
            for c in 0..cols {
                g[r * cols + c] = cur[c * rows + r];
            }
        }
    } else {
        g.copy_from_slice(cur);
    }
}

/// Muon update for one matrix parameter (`rows × cols`, row-major).
///
/// Steps: momentum update → Nesterov lookahead → Newton–Schulz
/// orthogonalization → spectral-aspect scaling → param update with
/// optional decoupled weight decay.
pub fn muon_apply(
    params: &mut [f32],
    grads: &[f32],
    group: &mut MuonGroup,
    rows: usize,
    cols: usize,
    config: &MuonConfig,
    scratch: &mut MuonScratch,
) {
    debug_assert_eq!(params.len(), grads.len());
    debug_assert_eq!(params.len(), rows * cols);
    debug_assert_eq!(group.m.len(), params.len());

    if scratch.update.len() < params.len() {
        scratch.update.resize(params.len(), 0.0);
    }
    let update = &mut scratch.update[..params.len()];

    if config.nesterov {
        for i in 0..params.len() {
            group.m[i] = config.momentum * group.m[i] + grads[i];
            update[i] = grads[i] + config.momentum * group.m[i];
        }
    } else {
        for i in 0..params.len() {
            group.m[i] = config.momentum * group.m[i] + grads[i];
            update[i] = group.m[i];
        }
    }

    newton_schulz5(update, rows, cols, &mut scratch.ns, config.ns_steps);

    // Spectral scale: for square matrices = 1; for d_ff × d (or d × d_ff)
    // FFN matrices it's sqrt(max/min). Keeps per-step update size
    // comparable across aspect ratios.
    let m = rows.min(cols);
    let n = rows.max(cols);
    let scale = (n as f32 / m as f32).sqrt();

    for i in 0..params.len() {
        let mut step = config.lr * scale * update[i];
        if config.weight_decay > 0.0 {
            step += config.lr * config.weight_decay * params[i];
        }
        params[i] -= step;
    }
}
