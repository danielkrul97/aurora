# embed-lr-1p5x

**Area**: optimizer

**Motivation**: Per-block-LR (depth-conditioned LR for the 6 transformer
blocks) is a confirmed winning axis. The EMBEDDING parameters (w_tok,
w_pos) are not part of that scaling — they use base lr. But embeddings
are demonstrably critical: `tok-embed-zero-init` caused a catastrophic
+0.45 regression. Token identity is the most information-dense input.

Hypothesis: the token + positional embeddings might benefit from a
LARGER LR than the transformer blocks. They sit "before block 0" and
the per-block-lr pattern (deeper = more LR) doesn't obviously extend
to them — but their criticality argues for faster, not slower, learning.
This iter gives w_tok and w_pos 1.5× base LR.

Note: w_tok is tied (input embed + LM head). Faster w_tok learning
affects both the input representation and the output projection.

**Mechanism**: In `step_adamw`, the w_tok and w_pos `adamw_apply` calls
use a config with `lr = base_lr * 1.5`. CPU `src/transformer/model.rs`
+ GPU `src/gpu/transformer.rs` (ops 0 and 1). ~8 LoC.

**Expected Δ BPB**: −0.010 to +0.010. Token embedding is critical, so
faster learning could help; or it could overshoot. Genuinely uncertain.

**Why unexplored**: Per-param-group LR has been applied to transformer
blocks (per-block-lr) but never to the embedding group.

## Math gate

- **Modified equation(s)**: w_tok, w_pos AdamW step uses
  `effective_lr = base_lr * 1.5`. Blocks and final_ln keep base_lr
  (block scaling unchanged from baseline = 1.0× for this solo test).
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. 1.5× LR on embeddings is modest; smoke
  convergence unaffected materially.
- `gpu_parity`: PASS. CPU and GPU apply the same 1.5× to ops 0, 1.
