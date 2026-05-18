//! Cosine LR schedule with linear warmup.

#[must_use]
pub fn cosine_warmup(
    step: usize,
    warmup_steps: usize,
    max_steps: usize,
    base_lr: f32,
    min_lr: f32,
) -> f32 {
    if step < warmup_steps {
        return base_lr * (step as f32 + 1.0) / warmup_steps.max(1) as f32;
    }
    if step >= max_steps {
        return min_lr;
    }
    let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps).max(1) as f32;
    let cos = (std::f32::consts::PI * progress).cos();
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos)
}
