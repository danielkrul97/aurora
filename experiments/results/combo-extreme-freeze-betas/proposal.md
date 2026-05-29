# combo-extreme-freeze-betas

**Area**: composite (schedule × optimizer × depth-cond-reg × norm-placement)

**Motivation**: Iter 50 (3-way composite with [0.2, 1.8] block-LR) is
the session bank-best at Δ = −0.0489. Iter 42 (`freeze-ln-betas`) was
neutral solo (+0.001) but suggested LN β is redundant under tied LM
head. This iter adds the β-freeze to the winning composite — testing
whether a "neutral solo" optimizer-state cleanup contributes when
other axes are tuned.

Two plausible outcomes:
- **Independent**: Δ ≈ −0.049 + 0 = −0.049 (no further change)
- **Mildly synergistic**: with other axes smoothed, removing the β
  noise channel reveals a small additional gain (~+0.005 incremental)

The hypothesis is a stretch — freeze-betas was net-zero solo and shows
up here as a free perf bonus only — but it's a clean test.

**Mechanism**: Iter 50's 3 changes plus:
- CPU step_adamw: skip adamw_apply for ln1.beta, ln2.beta, final_ln.beta
- GPU step_adamw: skip dispatch for beta-op indices (3, 5, 7, ...)

LoC: ~32 (iter 50's 22 + iter 42's 10).

**Expected Δ BPB**: −0.045 to −0.055. Win territory; likely marginal
improvement on iter 50.

**Why unexplored**: 4-way composite never tested.

## Math gate

Each component's math gate carries over from its individual iter. The
composite doesn't introduce new math. Forward and backward unchanged.

- **Modified equation(s)**: post-warmup `base_lr * sqrt(warmup/step)`;
  AdamW β2 = 0.98; per-block lr_mult = `0.2 + 1.6*li/(n_layer-1)`;
  LN β values frozen at 0.
- **Backward pass**: Unchanged (β gradients computed but discarded).
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0 effective (β allocated but unused).
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Both iter 50 and iter 42 passed
  individually.
- `gpu_parity`: PASS. Both backends mirror.
