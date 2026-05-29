# inverse-sqrt-decay

**Area**: schedule

**Motivation**: Three recent losses on the new baseline (ffn-fc-half-std,
sqrt-warmup, wd-attention-off) suggest the easy single-parameter
perturbations are exhausted. The one prior schedule-area entry that
won — `wsd-tail20-linear` — kept the LR at `base_lr` for ~80% of
training, then dropped linearly. The mechanism: more time at full
LR → more effective optimization steps.

This iteration tests a different "stay-high-longer" curve: the original
Noam Transformer (Vaswani et al 2017) schedule. Post-warmup,
`lr = base_lr * sqrt(warmup_steps / step)`. The decay is power-law
(smooth, slow), and the LR never reaches `min_lr` — at the end of 641
steps with warmup=200, the LR settles around `base_lr * sqrt(200/641)
≈ 0.56 * base_lr`, vs. cosine's `min_lr = 0.1 * base_lr`.

If WSD's win came from "more time at high LR," inverse-sqrt decay
should win similarly via the same mechanism, with a different curve.
If inverse-sqrt wins by ~the same margin as WSD, it confirms the
mechanism. If it wins by MORE, the smooth power-law tail beats the
abrupt flat-then-linear of WSD.

**Mechanism**: In `src/transformer/schedule.rs::cosine_warmup`, replace
the post-warmup cosine branch (lines 14-19) with
`base_lr * sqrt(warmup_steps as f32 / step as f32)`.

The function name becomes misleading (no more cosine) but the public API
stays the same — `min_lr` becomes effectively unused. ~3 LoC change.

**Expected Δ BPB**: −0.02 to −0.06. WSD won at roughly this magnitude
on the old baseline; the new baseline already includes the major init
win, so headroom is smaller, but the schedule axis is independent of
init.

**Why unexplored**: Bank has 32 entries. `schedule` area has 2:
`wsd-tail20-linear` (win) and `sqrt-warmup` (loss, this session).
Both touch one end of the schedule (decay tail, warmup ramp).
Inverse-sqrt is a fundamentally different DECAY curve — power-law
instead of cosine, never reaching `min_lr`. It's the original
Transformer schedule, rarely used in modern code but well-known in
the literature. Spec allows departure from "plain cosine schedulers"
and inverse-sqrt is explicitly not cosine.

## Math gate

- **Modified equation(s)**: For `step >= warmup_steps`:
  `lr(step) = base_lr * sqrt(warmup_steps / step)`.
  At `step = warmup_steps`: `lr = base_lr * sqrt(1) = base_lr`
  (continuous with linear warmup endpoint). At step → ∞:
  `lr → 0`, but the schedule for 641 total steps with warmup=200
  ends at `~0.56 * base_lr`.
- **Backward pass**: LR is an external scalar; no gradient flow.
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: LR cannot leak future info.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Continuity at warmup boundary**: At `step = warmup_steps - 1`,
  linear warmup gives `lr = base_lr * (warmup_steps / warmup_steps)
  = base_lr`. At `step = warmup_steps`, inverse-sqrt gives
  `base_lr * sqrt(warmup_steps/warmup_steps) = base_lr`. Continuous.
- **Behavior**: LR is monotonically decreasing post-warmup. Smooth,
  no flat region. No division by zero (step ≥ warmup_steps > 0).

## Sanity-gate prediction

- `transformer_primitives`: PASS. No LR schedule in unit tests.
- `transformer_smoke`: PASS. Smoke test uses the schedule with
  600 steps; inverse-sqrt gives LARGER LR than cosine at every
  post-warmup step → faster convergence on copy task (smoke test
  threshold is "30% loss drop in 600 steps"; this should hit it
  easier than cosine).
- `gpu_parity`: PASS. schedule.rs is CPU-only utility; GPU just
  consumes the `f32` LR per step.

## Continuity check (numerical)

Old (cosine) at step=200, max=641, base=6e-4, min=6e-5:
  progress = 0/441 = 0, cos(0)=1, lr = 6e-5 + 0.5*(6e-4-6e-5)*2 = 6e-4. OK.
Old at step=640: progress = 440/441 ≈ 1, cos(π) = -1, lr ≈ 6e-5. OK.

New (inverse-sqrt) at step=200: lr = 6e-4 * sqrt(200/200) = 6e-4. OK.
New at step=640: lr = 6e-4 * sqrt(200/640) = 6e-4 * 0.559 ≈ 3.35e-4.

So inverse-sqrt's LR at end of training is **~5.6× higher** than cosine's.
That's a substantial difference in effective training rate.
