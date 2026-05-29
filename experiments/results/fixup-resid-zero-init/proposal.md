# fixup-resid-zero-init

**Area**: init

**Motivation**: Aurora's residual branches (`w_o` in MHA, `w_proj` in FFN)
use GPT-3 residual scaling `std·1/sqrt(2N)` ≈ `0.02·0.289 ≈ 0.0058` at
init. Fixup (Zhang et al. 2019) and ReZero argue for the **fully zero
initialization** of these output projections: at step 0 every sublayer
output is exactly zero, the residual stream equals the input pass-through,
and the network is bit-identical to a "skip-everything" identity stack.
The sublayer contributions are then learned from scratch with no
"random sublayer noise" fighting the optimization signal early on.

At 12M params / 1 epoch byte-LM, the model is far from converged. Any
init that makes the *early* optimization signal cleaner can land
meaningful BPB. Zero-init exactly satisfies the "residual stream
variance does not grow with depth" desiderata that GPT-3 res-scaling
only approximates.

**Mechanism**: In `MhaWeights::new` replace
`LinearProj::new_scaled(d, d, resid_scale, rng)` with
`LinearProj::zeros(d, d)`. In `FfnWeights::new` replace
`LinearProj::new_scaled(d_ff, d_model, resid_scale, rng)` with
`LinearProj::zeros(d_ff, d_model)`. All other weights init unchanged.

**Expected Δ BPB**: -0.01 to -0.05. Fixup-style res-zero usually wins
modestly on small models (~5–10M) where pre-existing res-scaling already
helps; the gain comes from the *cleaner identity start* and absence of
sublayer noise. At Aurora's depth (6 blocks, 12 sublayers) the prior is
moderate, not huge.

**Why unexplored**: Bank has loss-aux/optimizer/embedding entries. No
`init`-area entries yet. Fixup is folklore for very deep transformers
(100+ layers) but is rarely reported on 12M / 6-layer / byte-LM regimes,
where the conventional wisdom is that depth scaling matters less. The
1-epoch screening setting amplifies any early-training advantage.

## Math gate

- **Modified equation(s)**: At step 0, both `w_o = 0` and `w_proj = 0`.
  Each sublayer's forward output is therefore identically zero (any
  linear projection of arbitrary `h_in` by a zero matrix is zero).
  Residual `y = x + sublayer(LN(x)) = x + 0 = x`. The whole transformer
  collapses to the identity for one step. Gradient computation proceeds
  as usual; gradients to `w_o`, `w_proj` are non-zero because they
  multiply `h_in` (which is non-zero), so after step 1 they leave zero.
- **Backward pass**: `∂L/∂w_o[j,i] = (∂L/∂sublayer_out[j]) · h_in[i]`.
  With `w_o = 0`, sublayer_out = 0 but ∂L/∂sublayer_out is still
  computed from the downstream residual stream — non-zero unless the
  whole loss is constant. So gradient flow is healthy and AdamW updates
  these matrices normally. No new non-differentiability.
- **Dim compatibility**: All shapes are identical to baseline init.
  `LinearProj::zeros(d_in, d_out)` returns a `LinearProj` with the same
  `[d_out · d_in]` flat layout, just filled with 0.0.
- **Causality preserved**: Init values do not interact with causal
  masking. The mask is structural, not learned.
- **Param count delta**: 0. Same parameters, different initial values.
- **VRAM delta**: 0.

## CPU/GPU parity

GPU model loads weights from CPU via `DecoderTransformerGpu::from_cpu()`
(`src/gpu/transformer.rs:605`). Changing CPU init propagates to GPU
through this load. Both paths see identical zero-init.
`tests/gpu_parity.rs` runs a CPU↔GPU drift test after one step — with
zero-init the first step's gradients on `w_o`/`w_proj` are the only
non-trivial difference; AdamW applies them identically on both sides,
so drift should remain below the 5e-2 tolerance.
