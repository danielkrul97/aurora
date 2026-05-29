# adam-no-bias-correction

**Area**: optimizer

**Motivation**: AdamW bias-correction (`bc1 = 1 - β1^t`, `bc2 = 1 - β2^t`)
divides m1, m2 to make their hat-versions unbiased estimators. At step 1
this gives m1_hat/sqrt(m2_hat) = sign(g), so the first step is
unit-magnitude. Without bias correction, the first step is much smaller
(~0.22× of normal).

The bias correction was designed for Adam's "warmup-free" regime in the
original paper. With Aurora's explicit 200-step linear warmup, bc may
be doing redundant work: warmup already throttles early steps. Skipping
bc would just make the initial gradient steps even gentler than warmup
alone. Could be neutral, slight loss, or possibly slight win if
double-throttling is right.

**Mechanism**: In `src/optimizer.rs::adamw_apply`, change
`group.m1[i] / bc1` to `group.m1[i]` and `group.m2[i] / bc2` to
`group.m2[i]`. Same change in `src/gpu/optimizer.rs` WGSL string. ~4 LoC.

**Expected Δ BPB**: −0.005 to +0.020. Probably neutral or small loss.

**Why unexplored**: bias-correction toggle has never been tested.

## Math gate

- **Modified equation(s)**: `step = lr · m1 / (sqrt(m2) + eps)` (instead
  of `lr · m1_hat / (sqrt(m2_hat) + eps)`).
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Asymptotic equivalence**: At step→∞, bc1, bc2 → 1, so bias-corrected
  and uncorrected coincide. Only early steps differ.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS likely. Smoke uses lr=3e-2 (aggressive).
  Without bc, first step is ~0.224 × normal magnitude — but 0.224 × 3e-2
  = 6.7e-3 effective LR. Still enough to make progress. May be slower
  to hit 30% threshold but 600 steps is plenty.
- `gpu_parity`: PASS (CPU and GPU mirror identical changes).
