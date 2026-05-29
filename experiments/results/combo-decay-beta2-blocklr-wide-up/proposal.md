# combo-decay-beta2-blocklr-wide-up

**Area**: composite (schedule × optimizer × depth-cond-reg)

**Motivation**: Iter 46 (`combo-decay-beta2-blocklr-up`) was the first
win of the post-merge baseline session at Δ = −0.0316, with PERFECT
LINEAR additivity (component sum −0.031, actual −0.0316). Iter 47
showed per-block-lr scales with spread: [0.5, 1.5] → −0.014,
[0.3, 1.7] → −0.0206 (×1.47 vs ×1.4 spread ratio).

This iter substitutes the wider [0.3, 1.7] block-LR variant into the
winning composite. Components:
1. inverse-sqrt-decay (−0.007)
2. AdamW β2 = 0.98 (−0.010)
3. per-block-lr [0.3, 1.7] (−0.0206) — wider than iter 46's [0.5, 1.5]

Linear sum: −0.0376. If linearity holds (as it did at iter 46), this
would be the strongest single-iter composite of the session.

**Mechanism**: Same as iter 46 but the block-LR formula is
`lr_mult = 0.3 + 1.4 * (li / (n_layer-1))` (vs iter 46's
`0.5 + (li / (n_layer-1))`). Other two components identical.
22 LoC total.

**Expected Δ BPB**: −0.030 to −0.040. Lower bound = sub-additive
(20% penalty on top of iter 46's behavior). Upper bound = linear
additivity.

**Why unexplored**: Iter 46 used [0.5, 1.5]; this uses [0.3, 1.7].
Different spread = distinct point in composite space.

## Math gate

Each component's math gate carries over from its individual iter
(31, 32, 47). Composite doesn't introduce new math beyond what each
piece does. Forward and backward unchanged from baseline; only the
optimizer step rules differ.

- **Modified equation(s)**: schedule post-warmup =
  `base_lr * sqrt(warmup_steps / step)`; AdamW β2 = 0.98; per-block
  effective lr = `base_lr * (0.3 + 1.4 * li/(n_layer-1))`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: All three are post-gradient/optimizer.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Iter 46 passed smoke; this just shifts
  block 0 to 0.3× and block 5 (with n_layer=2 in smoke: block 1) to 1.7×.
  Larger effective LR in block 1 might mildly destabilize smoke, but
  iter 47 (same wider spread, solo) passed smoke, so the composite
  should too.
- `gpu_parity`: PASS. CPU and GPU schedule, optimizer, and per-block
  LR all matched.
