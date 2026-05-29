# wd-attention-off

**Area**: optimizer

**Motivation**: Current AdamW applies decoupled weight decay (λ=0.1)
uniformly to all matmul weights: Q, K, V, O in MHA and fc, proj in FFN,
plus tied token embedding `w_tok`. Two prior optimizer/embedding entries
that selectively removed WD (`embed-no-wd`, `depth-wd-deep-only`) were
both losses, but neither tested the simple split:

- **Attention** parameters tend to learn sparse, head-specialized
  patterns (induction heads, copy heads, etc). WD pulls them toward
  zero, which can erase the very structure attention is meant to
  discover.
- **FFN** parameters behave like a dense MLP; WD is the standard,
  well-justified regularizer for that role.

The new baseline has `w_o = 0` at init (fixup-mha-only baked in). WD
on `w_o` during early steps pulls a parameter that is *just leaving
zero* back toward zero — exactly the wrong direction. Disabling WD
on the attention matmuls (Q/K/V/O) and the token embedding gives those
parameters more freedom to develop structure within the 1-epoch budget,
while keeping FFN regularized.

**Mechanism**: Flip `apply_wd: bool` from `true` to `false` for:
- `w_tok` (token embedding)
- `w_q`, `w_k`, `w_v`, `w_o` (per block, 4 sites × 6 blocks)

Keep `apply_wd: true` for `w_fc`, `w_proj`. (Positional embedding,
LayerNorm γ/β, biases already have WD off.)

Touched in two places:
- `src/transformer/model.rs::step_adamw` (CPU)
- `src/gpu/transformer.rs::build_grad_ops` (GPU dispatch list)

~10 LoC total (5 flag flips × 2 backends).

**Expected Δ BPB**: −0.01 to −0.06. The closest precedent is
`embed-no-wd` (+0.001, neutral) — disabling WD on `w_tok` alone had no
effect. Adding the four attention matmuls *might* matter because of
fixup-mha-only's `w_o=0` start. A loss outcome (no effect) is also
plausible if AdamW's effective WD step is too small to matter in 641
steps regardless.

**Why unexplored**: Bank has 31 entries. `optimizer` area has
`adam-eps-1e-6` (loss) and `ln-eps-1e-3` (loss). `embed-no-wd` (loss)
tested w_tok-only WD removal but not the principled attention split.
`depth-wd-deep-only` (loss) tested depth-conditional WD, a different
axis. The "WD on FFN only" split is a clean, novel test.

## Math gate

- **Modified equation(s)**: AdamW update for affected params drops
  the `(1 − λ·lr)·θ` shrinkage term. Forward and backward unchanged
  per-step.
- **Backward pass**: Identical. WD is applied post-gradient inside
  AdamW. Reachability of grads unaffected.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Optimizer step doesn't interact with
  attention mask.
- **Param count delta**: 0 (only the update rule changes for affected
  groups).
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks don't depend on
  AdamW step at all.
- `transformer_smoke`: PASS. Smoke test trains 600 steps with the
  default AdamW config — removing WD on 5 groups slightly reduces
  the per-step shrinkage, which can only INCREASE effective weight
  magnitudes. The model still has all gradients and reaches the
  copy task's solution. Risk: very slight overshoot, but at lr=3e-2
  and λ=0.1, the per-step WD shrinkage is ~0.3% — negligible to skip.
- `gpu_parity`: PASS only if CPU and GPU `apply_wd` flags are flipped
  in lockstep. Both files have the same fixed list of param groups
  in the same order. The flags must be byte-for-byte equivalent.
