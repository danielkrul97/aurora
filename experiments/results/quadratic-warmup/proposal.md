# quadratic-warmup

**Area**: schedule

**Motivation**: Iter 29 (`sqrt-warmup`, concave/faster-early) lost
+0.149. The opposite direction — convex/slower-early — has not been
tested. Linear warmup is the implicit baseline. Quadratic (convex)
warmup gives even gentler early steps, more time for AdamW bias
correction to settle before LR ramps to base_lr.

Formula: `lr(step) = base_lr · ((step+1)/warmup)²`.
At step=0: lr = base/warmup² ≈ 1.5e-8 (tiny)
At step=100/200: lr = base × 0.25 = 1.5e-4 (vs linear's 3e-4)
At step=199: lr = base × 1.0 (same endpoint as linear)

If sqrt's faster-early hurt because momentum hadn't built, quadratic's
slower-early should be neutral or help slightly — momentum has even more
time to converge before the model starts learning aggressively. Risk:
too little learning happens during warmup, model wastes 200 steps.

**Mechanism**: 1 LoC change in `cosine_warmup` warmup branch:
`base_lr * frac * frac` instead of `base_lr * frac`.

**Expected Δ BPB**: −0.005 to +0.015. Wide range. Most likely small
loss (under-trained during warmup window).

**Why unexplored**: Sqrt-warmup (concave) was tested; quadratic
(convex) is the symmetric counterpart.

## Math gate

- **Modified equation(s)**: For `step < warmup_steps`:
  `lr = base_lr · ((step+1)/warmup_steps)²`. Post-warmup unchanged
  (still cosine decay on main baseline).
- **Backward pass**: LR is external scalar; no gradient flow.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Continuity at warmup boundary**: At step = warmup-1, lr = base·1 = base.
  At step = warmup, post-warmup branch starts at progress=0, cos(0)=1,
  lr = base. Continuous.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Smoke uses 600 steps. Quadratic warmup is
  STRICTLY ≤ linear warmup throughout the warmup window, so the early
  LR is smaller. The model converges slightly slower in the first ~200
  steps but has 400 post-warmup steps at full LR. Should hit 30%
  threshold.
- `gpu_parity`: PASS.
