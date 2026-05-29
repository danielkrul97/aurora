# combo-embed-both-half

**Area**: composite (embedding × embedding)

**Motivation**: Bank has two near-identical embedding wins:
- `pos-embed-half-std` (iter 16): Δ −0.022 (w_pos at 0.5×)
- `embed-tok-half-std` (iter 18): Δ −0.022 (w_tok at 0.5×)

The symmetry is striking — both halve a different embedding tensor
and both yield essentially the same gain. Two competing hypotheses:

1. **Additive (different mechanisms)**: pos-half reduces per-position
   noise in residual stream (helping attention learn position from
   data); tok-half reduces initial LM head logit magnitude (helping
   the tied head learn from softer initial state). Predicted combined
   Δ ≈ −0.044.

2. **Redundant (same mechanism)**: both wins are explained by "smaller
   initial embed_forward output magnitude reduces some downstream
   gradient pathology equally". Predicted combined Δ ≈ −0.022.

3. **Super-additive**: rare but possible if the two reductions unlock
   a third effect together. Predicted Δ between −0.04 and −0.06.

This iteration runs the decisive test. Iter 19 showed that wins
**do not always stack** (tok-half + fixup-combo regressed), so this
test is principled even though the outcome is uncertain.

Importantly, this combo has NO fixup and NO WSD — only the two
embedding scales. Both standalone changes passed all smoke gates
(iters 16, 18), so the combo should clear sanity without issue. We
get clean signal on the embedding-area composability.

**Mechanism**: In `src/transformer/embedding.rs::EmbedWeights::new`,
multiply both init scales by 0.5:
```
let tok_scale = scale * 0.5;
let pos_scale = scale * 0.5;
```
Both w_tok and w_pos use the halved bound `s · 0.5 = 0.0173`.

**Expected Δ BPB**: -0.022 to -0.050. Most likely additive (-0.044).
If redundant (-0.022), tells us both wins share the same mechanism
(probably "smaller initial embed magnitude → softer initial logits"
since that's the dominant effect under final LN renormalization).

**Why unexplored**: Bank has 19 entries. Both halved-embedding wins
characterized separately but never combined. Different composite
than iter 19's (which added a third ingredient on top of an existing
winning combo); this combo combines two simple wins to test their
interaction directly.

## Math gate

- **Modified equation(s)**: `w_tok ∼ uniform[±0.0173]` and
  `w_pos ∼ uniform[±0.0173]` instead of both `±0.0346`. embed_forward
  returns half-magnitude per element compared to baseline (linear in
  scale).
- **Backward pass**: `embed_backward` scatter-adds dh unaffected by
  init values. AdamW updates both tensors from smaller starts in the
  same direction as baseline. No new non-differentiability.
- **Dim compatibility**: No shape changes.
- **Causality preserved**: Init doesn't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

All gates expected PASS:
- pos-half passed all gates standalone (iter 16).
- tok-half passed all gates standalone (iter 18).
- Combined still preserves discriminative signal — embeddings are
  halved but sublayers are NOT zeroed, so the model has full
  computational capacity from step 0.
- No interaction with WSD schedule, no zero-init residual branches.
