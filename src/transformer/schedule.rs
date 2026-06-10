//! LR schedules with linear warmup: cosine and warmup–stable–decay.

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

/// Warmup–stable–decay (trapezoid): linear warmup, flat at `base_lr`,
/// then linear decay to `min_lr` over the final `decay_frac` of
/// `max_steps`. The stable phase makes runs extendable — only the decay
/// tail depends on `max_steps`.
#[must_use]
pub fn wsd_warmup(
    step: usize,
    warmup_steps: usize,
    max_steps: usize,
    decay_frac: f32,
    base_lr: f32,
    min_lr: f32,
) -> f32 {
    if step < warmup_steps {
        return base_lr * (step as f32 + 1.0) / warmup_steps.max(1) as f32;
    }
    if step >= max_steps {
        return min_lr;
    }
    let decay_steps = (max_steps as f32 * decay_frac.clamp(0.0, 1.0)).round() as usize;
    let decay_start = max_steps.saturating_sub(decay_steps).max(warmup_steps);
    if step < decay_start {
        return base_lr;
    }
    let progress = (step - decay_start) as f32 / (max_steps - decay_start).max(1) as f32;
    base_lr + (min_lr - base_lr) * progress
}
