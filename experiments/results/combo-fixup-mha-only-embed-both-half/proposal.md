# combo-fixup-mha-only-embed-both-half

**Area**: composite (init × embedding)

**Motivation**: Bank's strongest single-change wins so far:
- `fixup-mha-only` (iter 22): Δ −0.332 (zero `w_o` only)
- `combo-embed-both-half` (iter 20): Δ −0.053 (halve both w_tok & w_pos)

Iter 19 found that **full fixup** anti-stacked badly with tok-half
(−0.234 BPB regression from the combo). That happened because full
fixup zeroed BOTH w_o AND w_proj, leaving the model with no
working sublayer pathway at init; halving the only remaining
learning channel (LM head via w_tok) broke optimization.

But iter 22 showed the **harmful piece of full-fixup is w_proj=0**;
just zeroing w_o (`fixup-mha-only`) keeps FFN sublayer non-zero at
init. So **fixup-mha-only + embed-half** should NOT trigger the same
anti-stacking — the FFN pathway provides discriminative signal during
warm-up.

Prediction: if cross-area composability holds (like iter 10 super-add
or iter 21 mild sub-add), Δ ≈ −0.36 to −0.40. If anti-stacking still
fires, Δ would be much weaker than combo-fixup-mha-only-wsd's −0.376.

**Mechanism**: Two changes:
1. `src/transformer/attention.rs::MhaWeights::new` — `w_o = LinearProj::new_scaled(d, d, 0.0, rng)`. FFN's w_proj unchanged.
2. `src/transformer/embedding.rs::EmbedWeights::new` — `tok_scale = pos_scale = scale * 0.5`.

**Expected Δ BPB**: -0.34 to -0.40. Additive prediction: -0.385.
Mild super-additivity (+0.01) is the prior for cross-area composites.
Anti-stacking risk: lower than iter 19 because FFN is preserved.

**Why unexplored**: Bank has 24 entries; this specific combo (best
single init + best embedding combo) hasn't been tested. Combo-fixup-wsd
(iter 23 derivative) tested init+schedule; this tests init+embedding.

## Math gate

- Both component changes are independent at the equation level
  (different files, different stages of init, different tensors).
- Each component validated standalone for all sanity gates.
- No new non-differentiability, shape changes, causality issues.
- Param count delta: 0. VRAM delta: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (fixup-mha-only passed iter 22).
- `transformer_smoke` (all 3): PASS expected. fixup-mha-only alone
  passed; embed-both-half alone passed. With FFN sublayer still
  contributing at init (unlike full fixup), the argmax test has
  full FFN-pathway signal to learn token discrimination from.
- `gpu_parity`: PASS.

## Anti-stacking risk

Iter 19 (combo-fixup-wsd-tok-half) regressed by +0.234 vs the
parent combo. Mechanism: zero w_o AND zero w_proj left the LM head
as the only learning channel; halving w_tok crippled it.

This iteration has **only w_o=0** (not w_proj=0), so the FFN pathway
remains a strong learning channel from step 1 regardless of
embedding-half. The structural condition that caused iter 19's
anti-stacking shouldn't apply.
