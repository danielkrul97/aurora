# adam-beta1-095

**Area**: optimizer

**Motivation**: Iter 32 (`adam-beta2-098`) gave −0.010 — smoother m2.
Iter 31 (`inverse-sqrt-decay`) gave −0.007 — smoother schedule. Iter 33
composite gave −0.019 (just below threshold). All point to a
"smoother dynamics" family that consistently helps marginally on the
new baseline.

The β1 axis hasn't been varied. Currently β1=0.9 (universal default).
β1 controls the first-moment (momentum) averaging window. β1=0.95
extends the effective window from ~10 steps to ~20 steps, making the
direction estimate smoother. Mirrors iter 32's β2 change in mechanism.

Hypothesis: at byte-LM scale with batch=16, momentum estimates from
individual gradients are noisy in direction. A smoother β1 could
help — particularly in the post-warmup region where the schedule
isn't yet decaying.

This is the third independent smoother-dynamics axis (after schedule
curve and m2). If it also gives ~−0.01, a future 3-way composite
(inverse-sqrt × β2=0.98 × β1=0.95) might cross −0.020 win threshold.

**Mechanism**: In `src/optimizer.rs::Default for AdamWConfig`, change
`beta1: 0.9` to `beta1: 0.95`. 1 LoC. Flows to GPU via the same
`AdamWConfig` struct.

**Expected Δ BPB**: −0.003 to −0.012. Lower expected magnitude than
β2 because the m1 contribution to the update is `m1_hat` directly
(not under sqrt), so per-step direction noise reduction has direct
proportional effect — but the noise might already be reduced by m2's
denominator.

**Why unexplored**: Bank has 37 entries. Optimizer area: 3 prior losses
(eps perturbations, WD targeting, adam-beta2-098 marginally below
threshold). β1 alone is the only AdamW hyperparameter not yet varied.

## Math gate

- **Modified equation(s)**: `m1 = 0.95·m1 + 0.05·g` (was
  `0.9·m1 + 0.1·g`). Update rule unchanged otherwise.
- **Backward pass**: AdamW is post-gradient; no autograd change.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Bias correction**: bc1 = 1 - β1^t. At t=1, bc1=0.05 (was 0.1).
  At t=100, bc1 ≈ 0.994 (was ≈ 1). The bias correction term divides
  m1_hat by bc1, so m1_hat is larger when bc1 is small — but m1
  itself is smaller (only 0.05·g₀ at t=1). Net effect: m1_hat
  magnitude is identical at t=1, identical at t→∞; the difference
  is in the transient region.

## Sanity-gate prediction

- `transformer_primitives`: PASS. AdamW config doesn't affect
  gradient computation.
- `transformer_smoke`: PASS. 600-step convergence works under any
  reasonable β1.
- `gpu_parity`: PASS. Both backends read same `config.beta1`.
