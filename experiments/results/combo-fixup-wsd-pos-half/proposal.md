# combo-fixup-wsd-pos-half

**Area**: composite (init + optimizer + embedding)

**Motivation**: Bank's current best is `combo-fixup-wsd` (Δ −0.308),
combining fixup zero-init (-0.251) + WSD schedule (-0.049). Iter 16
(`pos-embed-half-std`) standalone showed −0.022 with w_pos at ½
baseline magnitude, characterizing the pos-init dial as approximately
linear in magnitude reduction (vs iter 6's −0.040 at 0× magnitude).

Two prior 3-ingredient attempts hit the smoke gate:
- `combo-fixup-pos-zero` (iter 7): w_pos at 0× → 22% smoke drop.
- `combo-fixup-wsd-pos-quarter` (iter 13): w_pos at 0.25× → 28% drop.

The gate is 30%. Half-magnitude pos init preserves more positional
gradient signal during the 200-step copy task, recovering ~4 pp of
warmup speed vs quarter. The expected smoke drop is ~32-38% — above
the gate but not by much, so this is a probe of whether half-magnitude
is the threshold for "passes smoke while combined with fixup".

**Mechanism**: Combine four changes:
1. `src/transformer/attention.rs::MhaWeights::new` — `w_o = LinearProj::new_scaled(d, d, 0.0, rng)`.
2. `src/transformer/ffn.rs::FfnWeights::new` — `w_proj = LinearProj::new_scaled(d_ff, d, 0.0, rng)`.
3. `src/transformer/embedding.rs::EmbedWeights::new` — `pos_scale = scale * 0.5`.
4. `src/transformer/schedule.rs::cosine_warmup` body — WSD (warmup → stable base_lr → linear decay over tail 20%).

**Expected Δ BPB**: -0.31 to -0.34. Combo-fixup-wsd is -0.308; adding
the half-pos win predicts -0.330 (purely additive). Iter 10 showed
mild super-additivity (+0.008), so realistic estimate is -0.32 to -0.34.

**Why unexplored**: Iter 7 and 13 are the prior 3-ingredient attempts,
both broken_impl on smoke. This is the third probe at a less-aggressive
pos magnitude. Each iteration moves us along the smoke-gate boundary.

## Math gate

All three init components (fixup w_o, fixup w_proj, half pos) and the
schedule change are independent at the equation level — they touch
different parameters / different stages and never interact in the
forward/backward computation. Each verified standalone.

- **Modified equation(s)**: No structural change; only initial values
  and LR schedule shape.
- **Backward pass**: No new non-differentiability (each component
  validated independently in iters 4, 16, 2 respectively).
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: Init / schedule don't interact with the
  causal attention mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (each component passes).
- `transformer_smoke::cpu_transformer_converges_on_copy_task`: **AT RISK**.
  Prior data points:
  - fixup alone: PASS (well above 30%).
  - fixup + pos at 0× (iter 7): FAIL 22%.
  - fixup + pos at 0.25× (iter 13): FAIL 28%.
  - fixup + pos at 0.5× (this): predicted ~32-38% — expect PASS by
    narrow margin but not certain.
- `stability::cosine_warmup_monotone_then_decays`: PASS (WSD verified
  iter 2).
- `gpu_parity`: PASS (CPU/GPU lockstep on init+schedule).
