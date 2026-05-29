# combo-fixup-wsd-pos-quarter

**Area**: composite (init + optimizer + embedding)

**Motivation**: Bank current best is `combo-fixup-wsd` (Δ −0.308) — a
clean orthogonal stack of fixup (init) + WSD (schedule). Bank also has
`pos-embed-zero-init` (Δ −0.040 standalone) which broke smoke when
naively added to fixup (`combo-fixup-pos-zero`: 22% loss drop in 200
steps vs 30% required — three simultaneous zero-inits delayed warmup).

This iteration tests whether a **softened** third ingredient
(`w_pos` init at ¼ original magnitude) preserves enough warmup speed
to clear the smoke gate while still capturing part of the
pos-embed-zero win. The hypothesis is that the smoke failure was
driven by the *cumulative* zero-init mass — quartering one of the
three ingredients should keep the warmup characteristic close to
fixup-alone (which passes smoke).

**Mechanism**: Combine three changes:
1. `src/transformer/attention.rs` — `w_o = LinearProj::new_scaled(d, d, 0.0, rng)`.
2. `src/transformer/ffn.rs` — `w_proj = LinearProj::new_scaled(d_ff, d, 0.0, rng)`.
3. `src/transformer/embedding.rs` — `w_pos = uniform[-0.0087, 0.0087]`
   (= ¼ × baseline uniform range; baseline used `0.02·√3 ≈ 0.0346`).
4. `src/transformer/schedule.rs` — WSD body (warmup → stable base_lr → linear decay over tail 20%).

**Expected Δ BPB**: -0.30 to -0.35. Baseline of combo-fixup-wsd is
−0.308. If quarter-pos captures ~50% of the pos-zero standalone win
(−0.040 × 0.5 ≈ −0.020), expected combined Δ ≈ −0.33. If the smoke
gate still fails, log broken_impl honestly and we learn the cumulative
zero-mass threshold sits below "quarter pos magnitude".

**Why unexplored**: Bank has 2 composite entries (1 win, 1 broken).
The 3-ingredient combo is novel; the "soften one ingredient to clear
sanity gates" pattern is a useful technique to refine wins.

## Math gate

- **Modified equation(s)**: All three component changes are
  independent at the equation level — they touch different
  parameters and never multiply with each other. Forward and backward
  computations are unchanged; only initial values and the LR schedule
  differ.
- **Backward pass**: Each component already verified independently.
  No new non-differentiability.
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: Pre-attention causal mask is structural,
  unaffected by init/schedule.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: all changes affect init values only; the
  test uses a fresh seeded RNG, so changed inits flow through. Math
  is unchanged, primitives gradient checks should pass. PASS expected.
- `transformer_smoke::cpu_transformer_converges_on_copy_task`: fixup
  alone passed (78% drop in 200 steps, iter 4). The earlier 3-zero
  combo failed at 22%. Quartering the pos init recovers much of the
  pos-embed gradient signal early, expected to be near fixup-alone
  warmup speed. PASS expected — but this is the key gate at risk.
- `stability::cosine_warmup_monotone_then_decays`: WSD body verified
  iter 2. PASS.
- `gpu_parity`: all changes are init/schedule, identical math on CPU
  and GPU. PASS expected.
