# combo-fixup-wsd-tok-half

**Area**: composite (init + optimizer + embedding)

**Motivation**: Bank's best is `combo-fixup-wsd` (Δ −0.308). Recent
single-tensor wins are nearly symmetric:
- `pos-embed-half-std` (iter 16): Δ −0.022 (w_pos at 0.5×)
- `embed-tok-half-std` (iter 18): Δ −0.022 (w_tok at 0.5×)

Iter 17 tried `combo-fixup-wsd-pos-half` and broke smoke argmax
(model predicted wrong token after 100 steps). Argmax failure with
pos-half + fixup means: smaller per-position variation in residual
stream, combined with zero sublayer init, leaves the model unable to
discriminate input tokens via the tied LM head in 100 high-LR steps.

This iteration tries the symmetric pair: `combo-fixup-wsd-tok-half`.
With tok-half, the input embedding magnitudes are halved, but
**pos info is at baseline magnitude** — the residual stream's
per-position variation is preserved. The argmax test should have
more discriminative signal than the pos-half variant.

**Mechanism**: Combine four changes:
1. `src/transformer/attention.rs::MhaWeights::new` — `w_o = LinearProj::new_scaled(d, d, 0.0, rng)`.
2. `src/transformer/ffn.rs::FfnWeights::new` — `w_proj = LinearProj::new_scaled(d_ff, d, 0.0, rng)`.
3. `src/transformer/embedding.rs::EmbedWeights::new` — `tok_scale = scale * 0.5` (w_pos unchanged).
4. `src/transformer/schedule.rs::cosine_warmup` body — WSD (warmup → stable base_lr → linear decay over tail 20%).

**Expected Δ BPB**: -0.30 to -0.34. Combo-fixup-wsd is -0.308; adding
tok-half predicts -0.330 additive. Iter 10 showed mild super-additivity
(+0.008). Range -0.32 to -0.34 if smoke gate clears.

**Why unexplored**: 3rd attempt at 3-ingredient combo. Different
ingredient than iter 7 (w_pos zero), iter 13 (w_pos quarter), iter 17
(w_pos half). Tests the tok-vs-pos symmetry for smoke gate resilience.

## Math gate

- Components independent at the equation level (different files,
  different tensors, different stages).
- All three components passed primitives/smoke standalone (fixup iter 4,
  WSD iter 2, tok-half iter 18).
- No new non-differentiability, shape changes, or causality issues.
- Param count delta: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS — each component validated.
- `transformer_smoke::cpu_transformer_step_does_not_nan`: PASS.
- `transformer_smoke::cpu_transformer_argmax_sanity`: **AT RISK**.
  The single-tensor `embed-tok-half-std` (iter 18) passed argmax,
  but combined with fixup (sublayers=0) the model has less initial
  capacity for token discrimination via the tied LM head. With baseline
  w_pos, residual stream still varies per position so the model can
  use position-+ tok-via-attention as discriminative signal even with
  half-magnitude w_tok.
- `transformer_smoke::cpu_transformer_converges_on_copy_task`: PASS
  expected. fixup alone passes this with ~70% drop. tok-half adds
  minor delay, but well within margin.
- `stability::cosine_warmup_monotone_then_decays`: PASS (WSD verified).
- `gpu_parity`: PASS.
