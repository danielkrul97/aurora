# per-block-lr-step-up

**Area**: depth-cond-reg

**Motivation**: Continue mapping per-block-LR shape space. Done so far:
- Linear up [0.5, 1.5]: −0.014 (sub-threshold)
- Linear up [0.3, 1.7]: −0.0206 (win)
- Linear up [0.2, 1.8]: −0.025 (win, peak)
- Quadratic same endpoints: ~0 (loss; middle blocks need LR share)

This iter tests a step-function (binary tier) shape: blocks 0-2 get 0.5×,
blocks 3-5 get 1.5×. Same mean (1.0) and same endpoint magnitudes as
the linear-[0.5, 1.5] variant. If linear monotonicity matters (continuous
gradation), step should underperform linear. If endpoint magnitude is
the primary effect, step should match linear.

**Mechanism**: Same code pattern as iter 45 but step-function formula:
`lr_mult = if li < n_layer/2 { 0.5 } else { 1.5 }`. 16 LoC.

**Expected Δ BPB**: −0.005 to −0.015. Compared to linear [0.5, 1.5]
(−0.014), likely similar or slightly worse.

**Why unexplored**: New shape variant not yet tested.

## Math gate

- **Modified equation(s)**: For block li, `lr_mult = if li < n_layer/2
  { 0.5 } else { 1.5 }`. Mean: (3*0.5 + 3*1.5)/6 = 1.0 ✓ (n_layer=6).
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Smoke uses n_layer=2; n_layer/2 = 1, so
  block 0 < 1 gets 0.5×, block 1 ≥ 1 gets 1.5×. Same endpoint setup as
  linear [0.5, 1.5] for n_layer=2. Iter 45 passed; this should too.
- `gpu_parity`: PASS.
