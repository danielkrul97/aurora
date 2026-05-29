# embed-lr-1p2x

**Area**: optimizer

**Motivation**: Iter 62 (`embed-lr-1p5x`) was broken_impl — 1.5×
embedding LR × smoke's aggressive lr=3e-2 = 4.5e-2 effective broke the
100-step argmax sanity test. This iter retries at 1.2×: at smoke's
lr=3e-2 the effective embedding LR is 3.6e-2 — much closer to baseline
3e-2, should pass smoke.

Hypothesis unchanged from iter 62: token + positional embeddings are
critical (tok-embed-zero was catastrophic +0.45) and might benefit
from a modestly faster LR than the transformer blocks.

**Mechanism**: w_tok, w_pos AdamW step uses `base_lr * 1.2`. CPU
`step_adamw` + GPU dispatch (ops 0, 1). ~8 LoC.

**Expected Δ BPB**: −0.005 to +0.005. Smaller multiplier → smaller
effect than 1.5× would have had. Mostly a "map the axis" iter.

**Why unexplored**: Embedding-group LR boost; 1.5× broke smoke, 1.2×
is the milder retry.

## Math gate

- **Modified equation(s)**: w_tok, w_pos AdamW step uses
  `effective_lr = base_lr * 1.2`. Blocks + final_ln use base_lr.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS expected. 1.2× at smoke's lr=3e-2 = 3.6e-2
  effective embedding LR. Iter 62's 1.5× (4.5e-2) broke it; 1.2× is
  20% above baseline — should be within the convergence margin.
- `gpu_parity`: PASS.
