# fixup-ffn-only

**Area**: init

**Motivation**: Iter 22's `fixup-mha-only` (Δ −0.332) outperformed the
prior champion `fixup-resid-zero-init` (Δ −0.251) by +0.08 BPB. The
reinterpretation in iter 22's KEEP: full-fixup's FFN-side zero-init
was actually *harmful* by ~+0.08, partially cancelling the MHA-side's
true contribution of −0.332.

This iteration runs the symmetric ablation to settle the
decomposition: zero only `w_proj` (FFN output), keep `w_o` (MHA output)
at baseline GPT-3 residual scaling.

Two predictions for this iter's standalone Δ:

1. **Decomposition is additive** (predicted Δ ≈ +0.081):
   `fixup-full = fixup-mha-only + fixup-ffn-only`
   `−0.251 = −0.332 + (+0.081)`
   → ffn-only standalone is mildly harmful.

2. **Interaction-driven** (Δ ≈ 0 or slightly negative):
   FFN zero-init is innocuous alone but fights with MHA zero-init
   in the combo. The +0.08 gap in iter 4 was an interaction term.

The result distinguishes these. Useful for understanding why fixup
worked AND for principled future composites.

**Mechanism**: In `src/transformer/ffn.rs::FfnWeights::new`, replace
`w_proj = LinearProj::new_scaled(d_ff, d_model, resid_scale, rng)`
with `w_proj = LinearProj::new_scaled(d_ff, d_model, 0.0, rng)`.
`src/transformer/attention.rs` unchanged. 2 LoC.

**Expected Δ BPB**: +0.04 to +0.10 (mild loss) if decomposition is
additive. 0 to −0.05 if mostly an interaction effect.

**Why unexplored**: Bank has 23 entries. Init-area entries: fixup-full
(win), fixup-mha-only (big win), he-uniform (broken). FFN-only ablation
is the symmetric counterpart of fixup-mha-only and the cleanest test
of the iter-22 reinterpretation.

## Math gate

- **Modified equation(s)**: At init, `w_proj = 0`. FFN output =
  `w_proj · GELU(w_fc · LN(h)) = 0`. So FFN sublayer contribution to
  residual stream = 0 at init. MHA sublayer: baseline (non-zero).
- **Backward pass**: w_proj receives gradient via
  `dz @ GELU(w_fc · LN(h))^T` (input to w_proj is non-zero). After
  step 1, w_proj leaves zero. No new non-differentiability.
- **Dim compatibility**: No shape changes.
- **Causality preserved**: Init doesn't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (validated for fixup-full iter 4 and
  fixup-mha-only iter 22; FFN-only is a strict subset).
- `transformer_smoke`: PASS. With MHA still contributing at init, the
  argmax/convergence tests have full attention pathway from step 0.
- `gpu_parity`: PASS.
