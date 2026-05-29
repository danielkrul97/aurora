# combo-fixup-wsd

**Area**: composite (init + optimizer/schedule)

**Motivation**: Two strongest wins in the bank come from independent
mechanisms:
- `fixup-resid-zero-init` (Δ −0.251): zero-init `w_o` and `w_proj`.
- `wsd-tail20-linear` (Δ −0.049): WSD LR schedule (stable then linear
  decay only over last 20% of post-warmup).

Earlier composite `combo-fixup-pos-zero` (fixup + pos-embed-zero) hit
the smoke convergence gate (22% drop vs 30% required) because **three**
simultaneous zero-inits delayed warmup beyond the 200-step copy task
window. This combo deliberately *omits* `pos-embed-zero-init` and adds
a schedule change instead — schedule changes do not affect the
warmup-shape characteristic that the smoke test probes. Fixup alone
already passed smoke (verified iter 4), and WSD is a pure schedule
modification (verified iter 2). Their union should pass smoke and gives
us a clean additivity test between init-quality and LR-schedule wins.

**Mechanism**:
1. `src/transformer/attention.rs::MhaWeights::new` — `w_o` init scaled
   by 0.0 (fixup-style).
2. `src/transformer/ffn.rs::FfnWeights::new` — `w_proj` init scaled by
   0.0 (fixup-style).
3. `src/transformer/schedule.rs::cosine_warmup` body — WSD: warmup →
   constant base_lr → linear decay over tail 20% of post-warmup.

**Expected Δ BPB**: -0.22 to -0.31. If the two wins are perfectly
additive, predicted ≈ −0.30. Partial saturation is plausible because
fixup's gain comes from cleaner early-trajectory and WSD's gain from
later-trajectory LR; the mechanisms aren't obviously overlapping, so
near-additivity is the prior.

**Why unexplored**: Bank has one composite entry (broken_impl), but
that was a different combination. Fixup × WSD is a novel pairing of
two different-area wins, and the additivity test is the highest-value
information we can extract from the bank's current state.

## Math gate

- **Modified equation(s)**: Both component changes are independent at
  the equation level (different files, no shared state):
  - Fixup: at step 0, `w_o = w_proj = 0`, residual stream = identity.
    Sublayer output = 0; gradient still flows. (Validated iter 4.)
  - WSD: schedule has piecewise definition (warmup, stable, linear
    decay tail); `cosine_warmup` signature preserved. (Validated iter 2.)
- **Backward pass**: Neither change introduces non-differentiability.
  Both validated independently. Their composition does not interact.
- **Dim compatibility**: Both changes are init/schedule-only; no
  tensor shape changes.
- **Causality preserved**: Neither touches the attention causal mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: fixup-zero-init passes (verified iter 4),
  schedule change irrelevant. PASS.
- `transformer_smoke::cpu_transformer_converges_on_copy_task`: fixup
  alone passed (iter 4). Schedule change irrelevant to this 200-step
  test (it uses fixed LR, not the cosine_warmup schedule — let me
  re-check). PASS expected.
- `stability::cosine_warmup_monotone_then_decays`: WSD body satisfies
  the test's assertions (verified iter 2). PASS.
- `gpu_parity`: both changes pass independently. PASS expected.
