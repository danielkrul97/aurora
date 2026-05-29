# combo-inverse-sqrt-beta2-098

**Area**: composite (schedule × optimizer)

**Motivation**: Iters 31 (`inverse-sqrt-decay`, Δ=−0.007) and 32
(`adam-beta2-098`, Δ=−0.010) both gave marginal sub-threshold wins
via the same "smoother dynamics" mechanism — one via slower LR
decay, the other via smoother m2 estimate in AdamW. The mechanisms
are mechanistically INDEPENDENT (one is the LR scalar input to
AdamW, the other is AdamW's internal state) and act on different
time scales (schedule = whole-run macroscopic, β2 = per-step
microscopic smoothing).

Hypothesis: composing them should super-add. Linear sum of effects
would give −0.017; if the mechanisms are truly independent and the
gains aren't both extracting from the same residual headroom, we
could cross the −0.02 win threshold for the first time on the new
post-merge baseline.

Sub-additivity risk: both changes ultimately reduce the variance
in `effective_step = lr · m1_hat / sqrt(m2_hat + eps)`. If most of
each change's benefit comes from a shared "less noisy step
magnitudes" effect, the composite hits diminishing returns and
plateaus at ~−0.012.

**Mechanism**: Two atomic changes:

1. `src/transformer/schedule.rs::cosine_warmup` — replace post-warmup
   cosine branch with `base_lr * sqrt(warmup_steps / step)`. Same
   as iter 31.
2. `src/optimizer.rs::Default for AdamWConfig` — change `beta2: 0.95`
   to `beta2: 0.98`. Same as iter 32.

Total ≈ 4 LoC. No shape, parameter-count, or GPU shader changes.

**Expected Δ BPB**: −0.015 to −0.025 (super-additive scenario). Lower
bound −0.012 (sub-additive plateau). Win threshold is −0.020.

**Why unexplored**: Bank has 34 entries. No prior composite of
"smooth schedule" × "smooth optimizer". Earlier composite work in the
old-baseline era was init × schedule (combo-fixup-wsd) and init ×
embedding (combo-*-half-*). Schedule × optimizer is a new compositional
axis.

## Math gate

Each piece has its own math gate from iters 31 and 32. Both PASSED
sanity gates individually. The composite doesn't introduce new math
beyond what each piece does — the LR seen by AdamW differs (curve
shape) AND the m2 update rule's β2 differs. Both changes are
post-gradient, do not alter forward/backward, and do not touch the
autograd graph.

- **Modified equation(s)**: post-warmup LR = `base · sqrt(warmup/step)`;
  AdamW m2 update uses β2=0.98. Together they give a different
  effective optimizer behavior: at step 400, lr ≈ `6e-4 · 0.707 ≈
  4.24e-4` (vs cosine's ~2.3e-4) and the m2 estimate has ~50-step
  averaging window (vs ~20).
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Both changes are post-gradient.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Neither change touches gradient
  computation paths.
- `transformer_smoke`: PASS. Each piece passed individually. Combined
  effect on 600-step smoke is slightly higher effective LR throughout
  most steps and slightly smoother updates — both helpful for the
  copy task.
- `gpu_parity`: PASS. Both modifications flow through `AdamWConfig`
  and the schedule scalar input; no shader divergence.
