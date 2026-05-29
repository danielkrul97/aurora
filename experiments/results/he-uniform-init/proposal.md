# he-uniform-init

**Area**: init

**Motivation**: Aurora's `LinearProj::new` uses Glorot/Xavier-uniform
init: `bound = sqrt(6 / (d_in + d_out))`. Xavier was derived assuming
*linear or tanh-like activations* — it conserves variance assuming the
activation is roughly linear. Aurora's FFN uses GELU, which behaves
roughly like ReLU for positive inputs (zero for negative). He/Kaiming
init was specifically derived for ReLU/GELU: `bound = sqrt(6 / d_in)`
(uniform) or `std = sqrt(2 / d_in)` (normal). For asymmetric activations
that zero out half the input distribution, He preserves variance
through the activation; Xavier under-initializes.

For Aurora's tensor shapes (square Q/K/V/w_o + rectangular w_fc/w_proj),
He yields *wider* init than Xavier:
- Square (d=d): He × Xavier = sqrt(2) ≈ 1.41
- w_fc (384 → 1536): ratio = sqrt(1920/384) ≈ 2.24
- w_proj (1536 → 384): ratio = sqrt(1920/1536) ≈ 1.12

Wider Q/K means sharper initial attention; wider w_fc means louder FFN
intermediate activations. Both could amplify early gradient signal.
Note: `new_scaled` is built on top of `new` and multiplies its `bound`
by `scale`, so the GPT-3 residual scaling on `w_o` and `w_proj` (and
the iter-4 fixup zero-init variant) still works correctly — the scale
factor compounds.

**Mechanism**: In `src/attention.rs::LinearProj::new`, replace
`bound = sqrt(6 / (d_in + d_out))` with `bound = sqrt(6 / d_in)`.
Also update `new_scaled` analogously so the residual-branch scaling
chain stays consistent.

**Expected Δ BPB**: -0.005 to -0.04. The literature on ReLU-family inits
suggests modest wins for short trainings; at 1-epoch byte-LM the
optimization is still dominated by initial-condition effects, so a
better-matched init should help. Direction-wise positive; magnitude
uncertain because GELU is *softer* than pure ReLU, so He's correction
is partial.

**Why unexplored**: Bank has one `init` entry (fixup-resid-zero-init,
big win). That entry changed the *output projections* init only. He
init changes the *input* projections too (and is a different mechanism:
matched-to-activation init width, not zero-init). Aurora's existing
init was nanoGPT-default Xavier-uniform; replacing it with the
GELU-matched He variant is a principled choice that hasn't been
benchmarked at Aurora's scale.

## Math gate

- **Modified equation(s)**: `LinearProj::new(d_in, d_out, rng)` returns
  weights uniform[−bound, +bound] with
  - Old: `bound = sqrt(6 / (d_in + d_out))`.
  - New: `bound = sqrt(6 / d_in)`.
  No change to forward, no change to backward, no shape changes.
- **Backward pass**: Init values don't affect gradient computation
  symbolically. Numerically, with wider Q/K, the chain `dy → d_softmax →
  d_QK → d_Q, d_K` produces slightly larger gradients early; AdamW's
  second-moment buffer normalizes this away after a few steps.
- **Dim compatibility**: Unchanged.
- **Causality preserved**: Init doesn't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

`from_cpu` copies CPU weights to GPU. Both paths see identical He-init
weights. `tests/gpu_parity.rs` should pass: forward and backward are
identical math, only the initial weight magnitudes differ.
