# embed-no-wd

**Area**: embedding

**Motivation**: Aurora's AdamW currently applies decoupled weight decay
(0.1) to the token embedding matrix `w_tok` (both `src/transformer/model.rs:394`
for the CPU path and `src/gpu/transformer.rs:527` for the GPU path).
With a tied LM head, `w_tok` receives gradient from *two* sources every
step — the input-embedding lookup and the tied LM head matmul — so it
already updates twice as fast as any other weight. Applying WD on top
of that pushes embedding rows toward zero unnecessarily, especially in
the under-trained 1-epoch byte-LM regime where the model is still
learning the token table.

Standard convention since GPT-2 is to exclude embeddings, positional
embeddings, biases, and LayerNorm γ/β from WD. Aurora already excludes
the latter three but not `w_tok` — the change brings Aurora in line and
removes a small but consistent shrinkage pressure.

**Mechanism**: Change the `apply_wd` flag for the `w_tok` AdamGroup
from `true` to `false` in both the CPU and GPU optimizer wiring. Nothing
else changes: same gradients, same momentum, same LR schedule.

**Expected Δ BPB**: -0.005 to -0.025. Small but directionally consistent.
At WD=0.1 with LR following the WSD-baseline cosine, the per-step shrinkage
is `0.1 · lr ≈ 6e-5` peak, which over 641 steps integrates to a ~1-3%
shrinkage of embedding rows from their trained position. Removing the
shrinkage should slightly increase the effective embedding magnitude
and tighten the input → output coupling via tied LM head.

**Why unexplored**: First two bank entries cover `loss-aux` (loss) and
`optimizer` (win). No `embedding`-area entries yet. The change is
conventional at large scale but is *currently a measurable bug-grade
discrepancy in Aurora*, which makes it under-explored on this specific
codebase.

## Math gate

- **Modified equation(s)**: Only the AdamW step for `w_tok`. Old:
  `w_tok ← w_tok − lr · m̂₁ / (√m̂₂ + ε) − lr · wd · w_tok`. New:
  `w_tok ← w_tok − lr · m̂₁ / (√m̂₂ + ε)` (WD term dropped, same as
  all LayerNorm and positional embedding parameters already do).
- **Backward pass**: No change. WD is a parameter-update term, not a
  gradient. The chain rule through the model is identical.
- **Dim compatibility**: No tensor shapes change. `w_tok` is still
  `[vocab × d_model]`.
- **Causality preserved**: WD is per-parameter scalar; no interaction
  with position or attention. Causality untouched.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

Both paths must be changed in lockstep:
- `src/transformer/model.rs:394` — last arg `true` → `false`
- `src/gpu/transformer.rs:527` — last arg `true` → `false`

`tests/gpu_parity.rs` checks CPU↔GPU drift after one training step. If
both flags flip together, parity is preserved.
