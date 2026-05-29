# combo-fixup-pos-zero

**Area**: composite (init + embedding)

**Motivation**: Two independent zero-init wins in the bank — fixup
(`w_o = w_proj = 0` initially, Δ BPB = −0.251) and pos-embed
(`w_pos = 0` initially, Δ BPB = −0.040) — suggest the same underlying
mechanism: when a parameter's "useful initial state" is closer to
zero than to random noise, zero-init plus from-data learning is a free
upgrade. The two changes touch entirely different tensors (residual
output projections vs. positional embeddings), so a priori we expect
the gains to **stack additively** (predicted combined Δ ≈ −0.29). If
they stack, the hypothesis is confirmed; if they don't, the two
mechanisms partially overlap and we learn something new about Aurora's
optimization landscape.

**Mechanism**: Apply both winning changes simultaneously:
1. `src/transformer/attention.rs::MhaWeights::new`:
   `w_o = LinearProj::new_scaled(d_model, d_model, 0.0, rng)`
   (was `1/sqrt(2N)`-scaled).
2. `src/transformer/ffn.rs::FfnWeights::new`:
   `w_proj = LinearProj::new_scaled(d_ff, d_model, 0.0, rng)`
   (was `1/sqrt(2N)`-scaled).
3. `src/transformer/embedding.rs::EmbedWeights::new`:
   `w_pos = vec![0.0; seq_len_max * d_model]` (was uniform).

No other code changes; both effects propagate to GPU via
`DecoderTransformerGpu::from_cpu`.

**Expected Δ BPB**: −0.20 to −0.32. Strict additivity predicts −0.29.
Modest dilution is plausible: at step 0, fixup gives identity-only
residual stream; if `w_pos = 0` also, the residual stream encodes
*pure token identity with no position info*, which the attention
heads see through (now zero) sublayers. The model has nothing useful
to do for the first few steps until at least one of the three
parameter groups starts contributing. So *very early* the combo may
be marginally worse than fixup alone; the question is whether the
later trajectory recovers + exceeds.

**Why unexplored**: Spec explicitly allows composite as an area with
the proviso "justify why the composition is novel and not just a
stack". The justification here is **measuring additivity** — we want
to know if Aurora's zero-init wins compose linearly or saturate. The
bank has no composite entries yet.

## Math gate

- **Modified equation(s)**: All three component changes are
  independent multiplicative-with-zero substitutions at init time;
  forward/backward computation is unchanged at every step (different
  initial values, same kernel). At step 0:
  - `w_pos = 0`, `w_o = 0`, `w_proj = 0`.
  - `embed_forward = w_tok[tokens] + 0 = w_tok[tokens]`.
  - `block_forward = h_in + 0 + 0 = h_in` (residuals only).
  - `final_ln(h_in) @ w_tok` → logits (depends on `w_tok` and final LN).
- **Backward pass**: All three parameter groups receive non-zero
  gradient at step 1:
  - `dw_pos[p] = Σ_{t: pos(t)=p} dh_t` from scatter-add (non-zero).
  - `dw_o[j,i] = (∂L/∂mha_out_j) · h_in_i`, `h_in_i` non-zero (non-zero).
  - `dw_proj[j,i] = (∂L/∂ffn_out_j) · h_post_i`, non-zero.
  None of the chain dependencies on the zero-initialized matrices
  multiply by themselves (the chain rule paths go through `h_in`,
  `dh_attn_out`, etc., which are independent of the zero matrices).
  So every parameter starts moving from step 1 onward.
- **Dim compatibility**: All shapes unchanged from baseline.
- **Causality preserved**: Init values don't interact with causal
  masking. Mask is structural.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

All three changes are init-only and the GPU model loads weights from
the CPU via `from_cpu`. Both paths see identical zero-init for all
three tensors. `tests/gpu_parity.rs` confirmed the same setup works
for fixup-resid-zero-init (iter 4) and pos-embed-zero-init (iter 6);
the union of both changes preserves parity by the same argument.
