#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::must_use_candidate
)]

//! AdamW stepper (Loshchilov & Hutter 2019).

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
