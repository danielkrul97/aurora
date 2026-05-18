//! Softmax + cross-entropy with built-in gradient. Identical to the
//! previous `snn_loss` module but renamed and re-homed under
//! `transformer/` because softmax is shared across cells (none of it is
//! SNN-specific in the first place).

#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[must_use]
pub fn cross_entropy_with_grad(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    let p_target = probs[target].max(1e-12);
    let loss = -p_target.ln();
    let mut grad = probs;
    grad[target] -= 1.0;
    (loss, grad)
}

#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}
