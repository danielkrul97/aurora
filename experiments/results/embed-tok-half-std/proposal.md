# embed-tok-half-std

**Area**: embedding

**Motivation**: Aurora uses tied LM head: the same `w_tok` matrix
serves as input token embedding AND output logit projection. With
init std=0.02, initial logits magnitude is ~ std(h_final) · ||w_tok||
≈ 0.4 (as computed earlier in bank notes). Two earlier results put
a directional constraint on this:

- `logit-scale-2x` (iter 9): multiply logits by 2 in CE → +0.14 BPB
  (regression). Amplifying LM head gradient hurts.
- `pos-embed-half-std` (iter 16): w_pos init at 0.5× → −0.022 BPB.
  Less initial pos signal helps in linear dial.

This iteration tests the **symmetric experiment for w_tok**: halve
w_tok init magnitude. Effects:
1. Initial logits magnitude halved → softer initial softmax → CE
   gradient slightly tamer at LM head pathway (inverse of iter 9
   regression direction).
2. Initial input embeddings half magnitude → first LN renormalizes to
   ~N(0,1) anyway, so downstream layers see ~same scale.

The net is "smaller starting w_tok" — model has to grow embeddings to
get logit magnitude up. With AdamW and bias correction, this is
self-correcting after warmup.

**Mechanism**: In `EmbedWeights::new`, multiply only the w_tok init
`scale` by 0.5: `tok_scale = scale * 0.5`. w_pos init unchanged
(baseline magnitude). 2 LoC. Both CPU and GPU propagate (GPU loads
from CPU via `from_cpu`).

**Expected Δ BPB**: -0.005 to -0.025. Direction-wise positive based
on the "Aurora hates LM head gradient amplification" pattern (iter 9
regression). Magnitude uncertain — w_tok also affects input embedding
path, which is a confound.

**Why unexplored**: Bank has 17 entries. Embedding-area entries cover
`embed-no-wd` (loss), `pos-embed-zero-init` (win), `pos-embed-half-std`
(win). No `w_tok` magnitude experiment yet. The symmetric counterpart
to pos-embed scaling is the natural next probe.

## Math gate

- **Modified equation(s)**:
  - `w_tok[v, j] ∼ uniform[−s·0.5, +s·0.5]` where `s = 0.02·√3 ≈ 0.0346`.
    New range: [−0.0173, +0.0173].
  - Forward: `embed[t] = w_tok[token_t] + w_pos[t]` — w_tok values
    halved at init, w_pos baseline.
  - LM head: logits = LN(h_final) @ w_tok^T — half-magnitude w_tok
    means half-magnitude initial logits.
- **Backward pass**: `tied_lm_head_backward` scatter-adds dh through
  both paths (LM head matmul + embedding lookup). Gradient is
  independent of w_tok's current value. AdamW updates from a smaller
  initial w_tok in the same direction as baseline.
- **Dim compatibility**: No shape changes.
- **Causality preserved**: Embedding values don't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Smoke-gate prediction

- `cpu_transformer_step_does_not_nan`: PASS — smaller init means
  smaller initial activations, less risk of NaN.
- `cpu_transformer_argmax_sanity`: PASS expected. The test trains with
  lr=3e-2 for 100 steps. With halved w_tok, the model needs to grow
  embeddings further to learn discriminable token reps. 100 steps at
  3e-2 should be sufficient — embedding growth is fast under AdamW.
- `cpu_transformer_converges_on_copy_task`: PASS expected. Smaller
  init means softer initial predictions; relative loss drop should be
  similar to baseline (~70%), well above 30% threshold.

## CPU/GPU parity

`from_cpu` copies w_tok to GPU buffer. Both paths see identical
halved values. Standard parity preserved.
