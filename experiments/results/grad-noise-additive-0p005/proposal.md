# grad-noise-additive-0p005

**Area**: loss-surface

**Motivation**: Bank has 26 entries across many areas; `loss-surface`
is the last untouched area in the spec's idea-space list besides
curriculum (which requires training-loop changes). Gradient noise
injection (Neelakantan et al. 2015) is the simplest loss-surface
intervention: add small additive noise to gradients during training.
The classical motivation is stochastic regularization — noise lets
the optimizer escape sharp local minima, finding flatter basins.

At 1-epoch byte-LM scale, the model is far from any minimum (val_bpb
2.72 vs goal 1.5), so the "escape sharp minima" argument is weak.
However, noise can also help by:
- Decorrelating per-token gradients across batches (effectively a
  weaker correlation prior).
- Acting as a tiny implicit data augmentation at the loss level.

Net expected effect is uncertain; this is exploration, not exploitation.

**Mechanism**: In `cross_entropy_with_grad`, after computing the
standard CE gradient `d_logits = softmax - 1_target`, add a small
deterministic noise term to each element:
```
seed = hash(target, logits[0])
for each i:
    seed = LCG(seed)         // linear congruential
    unit = (seed >> 32) → f32 in [-1, 1]
    d_logits[i] += 0.005 * unit
```

The noise magnitude (0.005) is 0.5% of the target signal magnitude
(~1.0). Seeded deterministically per (target, logits[0]) so the
experiment is reproducible across runs. Both CPU and GPU paths use
this CE (via cross_entropy_with_grad).

**Expected Δ BPB**: -0.01 to +0.02. Direction uncertain. Most likely
small or noise-level. The 0.5% noise is below the typical "signal"
range so shouldn't dominate; whether it helps via gentle exploration
or hurts via wasted bits-per-step is empirical.

**Why unexplored**: Bank has 26 entries spanning loss-aux ×3,
optimizer ×2, embedding ×4, init ×3, composite ×8, norm-placement ×1,
attention ×1, depth-cond-reg ×2. Zero loss-surface entries.
Fills the last under-represented area in the spec's idea list.

## Math gate

- **Modified equation(s)**: `d_logits[i] = (softmax_i − 1{i=target}) + 0.005 · η_i`
  where `η_i ∈ [−1, 1]` is deterministic LCG-derived noise.
- **Backward pass**: Noise is added to the returned gradient. It does
  not affect the forward loss value. AdamW's bias-correction and
  variance-tracking absorb noise into its m2 estimate, slightly
  inflating the effective denominator and dampening updates. No new
  non-differentiability — noise is an additive constant w.r.t.
  parameters.
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: Noise is per-token, no cross-position
  coupling.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. The test does NOT exercise CE
  directly — it tests MHA, FFN, GELU, embed, tied_lm_head backward
  primitives. Adding noise to CE has no effect on these tests.
- `transformer_smoke`:
  - `step_does_not_nan`: PASS (small additive noise stays finite).
  - `argmax_sanity`: AT MILD RISK. With noise at 0.5% of target
    signal, the model should still learn to predict same token in
    100 steps at lr=3e-2. But the test seed could expose edge cases.
  - `converges_on_copy_task`: AT MILD RISK. 200 steps with 0.5%
    grad noise should still hit 30% loss drop, but tight.
- `gpu_parity`: PASS. Same CPU CE is called from both paths;
  determinism preserved by seeding.

## Determinism guarantee

The noise is seeded from `(target as u64) ^ logits[0].to_bits() as u64`.
Same input → same seed → same noise pattern. Across runs with the same
RNG seed (CLI baseline_config.json), training is fully reproducible.
