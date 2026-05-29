# per-block-wd-linear-up

**Area**: depth-cond-reg

**Motivation**: Per-block-LR axis is mapped (linear [0.2, 1.8] is
optimal). The companion axis — per-block WEIGHT-DECAY — has never been
varied. Different mechanism: AdamW step =
`lr * m1_hat/sqrt(m2) + lr * wd * param`. Per-block-LR scales both
terms; per-block-WD scales ONLY the second term. So they decouple the
"momentum update" from the "decay shrinkage" with depth.

Hypothesis: deeper blocks specialize more (build high-level features
that overfit faster), so they could benefit from MORE WD. Shallow blocks
learn basic features that need less regularization. This is similar to
`depth-wd-deep-only` (loss) but with linear gradation instead of
binary on/off.

Multipliers: block 0 = 0.5×, block 5 = 1.5×. Mean = 1.0×.

**Mechanism**: Same code pattern as per-block-LR but scaling
`config.weight_decay` per block. CPU step_adamw + GPU dispatch loop.
~16 LoC.

**Expected Δ BPB**: −0.003 to +0.005. WD's per-step impact is small
(lr * wd * |param| ≈ 1.2e-6 typical), so the effect size is bounded.
This is mostly a "map a new axis" iter.

**Why unexplored**: Per-block-WD as a continuous mechanism hasn't been
tested. `depth-wd-deep-only` was binary (loss).

## Math gate

- **Modified equation(s)**: For block li,
  `effective_wd = base_wd * (0.5 + li/(n_layer-1))`. AdamW step uses
  `lr * effective_wd` in the decay term. Other params (w_tok, w_pos,
  LN, final_ln) keep base_wd. (But w_pos, LN already have apply_wd=false,
  so they're unaffected anyway.)
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. WD changes are tiny in magnitude; even
  block 0 at 0.5× WD has minimal effect on smoke convergence.
- `gpu_parity`: PASS. Both backends modified identically.
