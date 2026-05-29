# wsd-tail20-fresh

**Area**: schedule

**Motivation**: WSD-tail20-linear was a win on OLD baseline (−0.049) but
its worktree was built on pre-merge code. After fixup-mha-only was
merged into main, the schedule axis was tested with inverse-sqrt-decay
(−0.007 on new baseline). These two are different curves with the same
"keep LR near base_lr longer" intuition.

Hypothesis: WSD's flat-then-drop might give different magnitude on new
baseline compared to inverse-sqrt's smooth decay. If similar (~−0.007),
schedule axis is consistent. If much better (~−0.03), WSD-tail20 is a
better schedule curve choice that wasn't yet tested on new baseline.

A WSD result on the new baseline is also a missing data point in the
bank — useful for diagnostic understanding regardless of magnitude.

**Mechanism**: In `src/transformer/schedule.rs::cosine_warmup`, replace
the cosine decay branch with WSD-tail20:
- if step < warmup: linear ramp (unchanged)
- if step < `warmup + 0.8*(max_steps-warmup)`: flat at base_lr
- if step < max_steps: linear decay from base_lr to min_lr
- else: min_lr

For max=641, warmup=200: flat region from step 200 to step 200+0.8*441 = 552.8 ≈ 553. Then linear decay over steps 553-641.

LoC: ~7 in schedule.rs.

**Expected Δ BPB**: −0.005 to −0.020. Likely similar magnitude to
inverse-sqrt-decay (-0.007) but distinct curve. Optimistic: −0.02 (still
sub-threshold, but bigger than inverse-sqrt).

**Why unexplored**: WSD on new baseline never tested; the old worktree
predates the merge.

## Math gate

- **Modified equation(s)**: piecewise linear schedule (warmup → flat
  → linear decay).
- **Backward pass**: External scalar; no gradient flow.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Continuity**: warmup ends at base_lr at step=warmup; flat region
  stays at base_lr; decay region starts at base_lr and ends at min_lr.
  Continuous.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Smoke uses cosine_warmup with 600 steps,
  warmup likely 50-100. The WSD curve gives 80% of post-warmup at full
  base_lr — strictly MORE LR than cosine decay. Convergence should be
  easier.
- `gpu_parity`: PASS. schedule.rs is CPU-only utility.
