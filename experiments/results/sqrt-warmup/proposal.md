# sqrt-warmup

**Area**: schedule

**Motivation**: Baseline uses 200 steps of LINEAR warmup out of ~641 total
1-epoch steps = ~31% of training budget in low-LR territory. Linear
warmup spends a lot of time at fractional LR (e.g. step 50 = 0.25·base).
A concave (square-root) warmup curve rises faster while still protecting
the first ~10 steps from gradient shock — at step 50 the LR is already
~0.5·base, freeing up budget for full-rate learning.

Hypothesis: the model wastes the first ~100 steps under linear warmup
when the cross-entropy gradient signal is largest (model essentially
random, predicting near-uniform over 4096-vocab → big consistent
gradient direction). Sqrt warmup unlocks this signal faster while
keeping the very-first-step protection.

This complements (and does not collide with) `wsd-tail20-linear` (iter 2,
schedule area), which modified the DECAY tail. Sqrt-warmup modifies the
opposite end of the schedule — pure addition, no overlap with the WSD
mechanism.

**Mechanism**: In `src/transformer/schedule.rs::cosine_warmup`, replace
the linear warmup branch
`base_lr * (step + 1) / warmup_steps`
with
`base_lr * sqrt((step + 1) / warmup_steps)`.

One LoC. Post-warmup cosine decay is unchanged. The two endpoints match
linear at step 0 (close to 0) and step `warmup_steps-1` (= base_lr), so
the schedule remains continuous.

**Expected Δ BPB**: −0.01 to −0.05 vs new baseline 2.386948. The model
gets more steps near full LR. If the bottleneck is wasted warmup steps,
this helps. Probable upper bound is ~−0.05; could also be near zero if
AdamW's bias correction `bc1, bc2` already absorbs most of the warmup
benefit.

**Why unexplored**: Bank has 30 entries; `schedule` area has only one
(wsd-tail20-linear, win, decay-side). Warmup curve shape has never been
varied. T5-style sqrt warmup is in the literature but uncommon — fits
the spec's "drift toward unexplored corners".

## Math gate

- **Modified equation(s)**: For `step < warmup_steps`:
  `lr(step) = base_lr · sqrt((step + 1) / warmup_steps)`.
  Post-warmup unchanged: `lr = min_lr + 0.5·(base_lr − min_lr)·(1 + cos(π·progress))`.
- **Backward pass**: LR is an external scalar applied at AdamW step
  time. Not part of the autograd graph. No new gradient flow.
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: LR schedule cannot leak future info.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Continuity at warmup boundary**: At `step = warmup_steps - 1`,
  `sqrt(warmup_steps/warmup_steps) = 1`, so `lr = base_lr`. At step
  `warmup_steps`, the post-warmup branch starts at progress=0, cosine=1,
  giving `lr = base_lr`. Continuous.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks don't exercise the LR
  schedule.
- `transformer_smoke`: PASS. Smoke uses cosine_warmup; sqrt curve is
  STRICTLY ≥ linear curve everywhere in warmup → LRs are larger early →
  faster convergence in the same 600-step window. (Worst case: marginally
  faster loss drop. Both endpoints match.)
- `gpu_parity`: PASS. schedule.rs is CPU-only utility; GPU consumes the
  same `f32` LR per step from the training loop.
