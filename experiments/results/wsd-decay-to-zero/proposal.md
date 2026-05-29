# wsd-decay-to-zero

**Area**: schedule

**Motivation**: The WSD schedule (winning component, iters 58/60/61)
decays from base_lr to `min_lr` (= 6e-5, the CLI floor) over the tail.
The tail-fraction sweep covered 10/20/30%. But the decay TARGET has
never been varied — it's always min_lr. This iter decays to 0 instead.

Hypothesis: a true 0 endpoint gives a slightly more complete anneal in
the final steps. WSD literature (MiniCPM) typically decays toward 0.
The min_lr=6e-5 floor at the end means the last steps still take small
but non-zero updates; decaying to exactly 0 freezes the model fully
at the end. For a 1-epoch run, the last ~130 steps (30% tail) would
end at 0 instead of 6e-5.

This is distinct from the tail-fraction sweep — that varied the tail
WIDTH; this varies the tail ENDPOINT.

**Mechanism**: In the WSD body of `cosine_warmup`, decay
`base_lr → 0` instead of `base_lr → min_lr`. The `min_lr` parameter
becomes unused. ~2 LoC.

**Expected Δ BPB**: −0.005 to +0.005. Small — the difference is only
in the last ~130 steps' LR (0 vs 6e-5), and 6e-5 is already tiny.

**Why unexplored**: WSD decay target never varied (always min_lr).

## Math gate

- **Modified equation(s)**: WSD tail: `lr = base_lr·(1 − decay_progress)`
  (decays to 0) instead of `base_lr + (min_lr−base_lr)·decay_progress`.
- **Backward pass**: External scalar; no gradient flow.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count / VRAM delta**: 0.
- **Continuity**: at decay_start, lr = base_lr; at max_steps, lr = 0.
  Monotone decreasing. No discontinuity.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Decay to 0 only affects the very last
  steps; convergence in the 600-step window is unaffected (the flat
  region and most of the decay are unchanged).
- `gpu_parity`: PASS. schedule.rs is CPU-only utility.
