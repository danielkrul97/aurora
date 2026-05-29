# combo-decay-beta2-blocklr-extreme-up

**Area**: composite (schedule × optimizer × depth-cond-reg)

**Motivation**: Iter 48 (`combo-decay-beta2-blocklr-wide-up` at [0.3, 1.7])
won at −0.0406 with linear sum −0.0376 → super-add +0.003. Iter 49
showed the per-block-lr-spread axis keeps gaining super-linearly into
[0.2, 1.8] (solo Δ = −0.025).

Substitute the extreme spread variant into the composite:
1. inverse-sqrt-decay (−0.007)
2. AdamW β2 = 0.98 (−0.010)
3. per-block-lr [0.2, 1.8] (−0.025)

Linear sum: −0.042. With the consistent +0.002 to +0.003 super-add
pattern from iters 33, 46, 48, predicted Δ ≈ −0.045.

**Mechanism**: Same as iter 48, but the block-LR formula is
`lr_mult = 0.2 + 1.6 * (li / (n_layer-1))` (vs iter 48's
`0.3 + 1.4*(li/(n_layer-1))`). Other two components identical.
22 LoC total.

**Expected Δ BPB**: −0.040 to −0.048. Win territory by a wide margin.
Sub-additive worst case ~−0.038 — still wins easily.

**Why unexplored**: Iter 48 used [0.3, 1.7]; this uses [0.2, 1.8].
Distinct point in composite space.

## Math gate

Same as iter 48 with [0.2, 1.8] spread substituted. No new math.

- **Modified equation(s)**: post-warmup `base_lr * sqrt(warmup/step)`;
  AdamW β2 = 0.98; per-block lr_mult = `0.2 + 1.6*li/(n_layer-1)`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Both component iters 48 [0.3, 1.7] and
  49 [0.2, 1.8] passed individually. The composite has the same upper
  block multiplier (1.8×); smoke is fine.
- `gpu_parity`: PASS.
