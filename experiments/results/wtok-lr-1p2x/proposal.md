# wtok-lr-1p2x

**Area**: optimizer

**Motivation**: Iter 63 (`embed-lr-1p2x`) won −0.027 by giving BOTH
w_tok and w_pos 1.2× LR. This iter decomposes the win: boost ONLY
w_tok (the tied token embedding) at 1.2×, leave w_pos at base LR.

Hypothesis: w_tok is tied (input embedding + output LM head) and
carries token identity — the most information-dense parameter group.
w_pos is just positional structure (and pos-embed-zero won on the old
baseline, suggesting positional info is less critical). So the embed-LR
gain likely comes mostly from w_tok. If `wtok-lr-1p2x` recovers most
of −0.027, that's confirmed; if it recovers little, w_pos was the
driver.

Diagnostic value: tells us which embedding to prioritize and informs
whether a w_tok-specific LR is the cleaner mechanism than the combined
embed-LR.

**Mechanism**: `step_adamw`: w_tok uses `base_lr·1.2`, w_pos uses
base_lr (reverted from iter 63's combined boost). CPU + GPU (only
op 0). ~6 LoC.

**Expected Δ BPB**: −0.015 to −0.027 if w_tok is the main driver;
−0.005 to −0.015 if w_pos contributed substantially.

**Why unexplored**: Decomposition of iter 63's combined embed-LR win
— distinct experiment (like depth-fixup-mha-shallow decomposed
fixup-mha-only).

## Math gate

- **Modified equation(s)**: w_tok AdamW step uses `lr·1.2`; w_pos and
  all other params use base_lr.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count / VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. 1.2× on w_tok only is milder than iter 63
  (which boosted both and passed smoke).
- `gpu_parity`: PASS. Only op 0 (w_tok) gets the 1.2× multiplier.
