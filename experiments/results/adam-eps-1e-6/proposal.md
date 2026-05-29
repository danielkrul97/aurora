# adam-eps-1e-6

**Area**: optimizer

**Motivation**: Aurora's `AdamWConfig::default()` uses `eps = 1e-8`
(PyTorch's default). The AdamW update is
`θ ← θ − lr · m̂₁ / (√m̂₂ + ε)`. The `ε` floor in the denominator
matters most when `√m̂₂` is small — i.e., **in the early steps of
training** before the squared-gradient EMA has stabilized, and for
parameters whose gradient is genuinely tiny (LN γ/β etc.).

With `ε = 1e-8` and an early-step `√m̂₂ ≈ 1e-4`, the effective step is
`lr · m̂₁ / 1e-4 = 10⁴ · lr · m̂₁` — huge. With `ε = 1e-6`, that becomes
`lr · m̂₁ / 1.01e-4 ≈ 9.9e3 · lr · m̂₁` — virtually identical, so the
change is a no-op for "loud" parameters. But for "quiet" parameters
(LN γ early in training, with `√m̂₂ ≈ 1e-7`), `ε = 1e-8` gives step
scale `≈ 1e7 · lr · m̂₁` (overshoot), while `ε = 1e-6` clamps it to
`≈ 1e6 · lr · m̂₁` — a 10× damping of pathological large early updates
on quiet params.

BERT, T5, and several modern LLM recipes use `ε = 1e-6` for this
reason. PaLM kept `1e-8`. At 1-epoch byte-LM scale, where m̂₂ has only
~641 steps to warm up, an early-step damping of pathological updates
should be net-positive.

**Mechanism**: In `src/optimizer.rs::AdamWConfig::default()`, change
`eps: 1e-8` to `eps: 1e-6`. Nothing else changes. Both CPU
(`adamw_apply`) and GPU (`AdamWKernel` reading from
`AdamWParams::eps`) consume `config.eps` directly, so the new default
propagates to both paths.

**Expected Δ BPB**: -0.005 to -0.03. Small positive expected because:
(a) the change only affects "quiet" parameters (LN γ/β), which is a
small fraction of total params; (b) AdamW's m̂₂ converges fast for
loud params, masking ε's effect there. Direction-wise positive
because PaLM's choice is the outlier among modern recipes.

**Why unexplored**: Bank has one `optimizer` win (wsd-tail20-linear,
schedule change). Eps is the *other* AdamW knob worth varying; it
hasn't been touched. Different mechanism from WSD: WSD changes
*global* LR over time, eps changes *per-parameter* update magnitude
based on local gradient statistics. Could compose nicely with WSD in
a future composite.

## Math gate

- **Modified equation(s)**: `θ ← θ − lr · m̂₁ / (√m̂₂ + ε)` with
  `ε: 1e-8 → 1e-6`. No structural change to forward or backward.
- **Backward pass**: Optimizer hyperparameter only; chain rule
  through the model is unchanged.
- **Dim compatibility**: Scalar config field. No tensor shapes
  affected.
- **Causality preserved**: Optimizer doesn't interact with attention
  mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

CPU `adamw_apply` and GPU `AdamWKernel` both read `eps` from the
`AdamWConfig` passed to `step_adamw`. The example builds `adam` from
`AdamWConfig::default()` overridden with CLI lr/wd; eps stays the
default. Changing the default propagates to both paths in lockstep,
so `gpu_parity::full_transformer_training_step_matches_cpu_direction`
should remain within 5e-2 drift.
