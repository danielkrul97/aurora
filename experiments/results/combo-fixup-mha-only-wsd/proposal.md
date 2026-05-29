# combo-fixup-mha-only-wsd

**Area**: composite (init + optimizer)

**Motivation**: Iter 22 (`fixup-mha-only`, Δ −0.332) overtook the
prior single-change champion `fixup-resid-zero-init` (Δ −0.251) by
+0.08 BPB, by zeroing *only* `w_o` (MHA output) and leaving `w_proj`
(FFN output) at baseline GPT-3 residual scaling. This experiment
stacks that new champion with the bank's proven schedule win,
`wsd-tail20-linear` (Δ −0.049).

Composability prior from iter 10 (`combo-fixup-wsd`): init × optimizer
composed super-additively (+0.008). If that pattern holds, predicted
Δ ≈ −0.381 (additive) to −0.389 (with super-additivity).

This would be the new bank-best composite, ~+0.07 better than the
current best (combo-fixup-wsd at −0.308). The pattern's hold here
also confirms that the iter-22 finding (MHA zero-init is the real
driver) generalizes to schedule-stacked composites.

**Mechanism**: Two changes:
1. `src/transformer/attention.rs::MhaWeights::new` — w_o init scaled
   by 0.0 (`new_scaled(d, d, 0.0, rng)`). FFN's w_proj keeps baseline
   GPT-3 1/sqrt(2N) residual scaling.
2. `src/transformer/schedule.rs::cosine_warmup` body — WSD: warmup →
   stable base_lr → linear decay over tail 20% of post-warmup steps.

**Expected Δ BPB**: -0.36 to -0.40.

**Why unexplored**: Bank has 22 entries; combo-fixup-mha-only-wsd is
the natural follow-up to iter 22's MHA-only champion. Different
composition than combo-fixup-wsd (which zeroed BOTH w_o and w_proj).

## Math gate

- Both components are independent at the equation level and
  validated standalone.
- All sanity gates pass for each component.
- No new non-differentiability, no shape changes, no causality issues.
- Param count delta: 0. VRAM delta: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (fixup-mha-only passed at iter 22).
- `transformer_smoke` (all 3 sub-tests): PASS. fixup-mha-only alone
  passed; WSD doesn't affect smoke. With FFN sublayer still
  contributing at init (unlike full fixup), warm-up has more signal
  than the iter-4 full-fixup state, easily clearing the 30% gate.
- `stability::cosine_warmup_monotone_then_decays`: PASS (WSD verified
  iter 2).
- `gpu_parity`: PASS.
