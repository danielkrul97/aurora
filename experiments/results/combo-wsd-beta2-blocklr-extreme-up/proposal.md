# combo-wsd-beta2-blocklr-extreme-up

**Area**: composite (schedule × optimizer × depth-cond-reg)

**Motivation**: Iter 58 (`wsd-tail20-fresh`) revealed that WSD-tail20
gives Δ = −0.044 SOLO on the new baseline — 6× larger than
inverse-sqrt-decay (−0.007) at the same schedule axis position. Iter 50's
winning composite (Δ = −0.049) used the WEAKER inverse-sqrt-decay.
Substituting WSD-tail20 should produce the strongest composite yet.

Components:
1. WSD-tail20 schedule (−0.044 solo) — flat at base_lr for 80%
   post-warmup, linear decay to min_lr in last 20%
2. AdamW β2 = 0.98 (−0.010 solo)
3. per-block lr_mult = 0.2 + 1.6·li/(n_layer-1) (−0.025 solo)

Linear sum: −0.079. Pattern of super-additivity from 3-way composites:
+0.003 (iter 48 wide) to +0.007 (iter 50 extreme). Expected combined Δ:
−0.082 to −0.086.

**Mechanism**: Same as iter 50 with the schedule body swapped from
inverse-sqrt-decay to WSD-tail20. 26 LoC total (11 for WSD schedule,
1 for β2, 16 for per-block-lr CPU+GPU).

**Expected Δ BPB**: −0.075 to −0.090. Strongly above threshold; new
bank-best. Risk: sub-additive interaction between WSD's flat region
and the other axes could reduce the gain.

**Why unexplored**: Iter 50 used inverse-sqrt; this uses WSD-tail20.
Distinct composite-space point.

## Math gate

Each component's math gate carries over from its individual iter
(58, 32, 49). Composite doesn't introduce new math.

- **Modified equation(s)**: schedule = piecewise (warmup → flat
  base_lr → linear decay to min_lr); AdamW β2 = 0.98; per-block effective
  lr = base_lr × (0.2 + 1.6·li/(n_layer−1)).
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: All post-gradient modifications.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Both iter 58 (WSD solo) and iter 50
  (3-way with inverse-sqrt) passed smoke. The substitution just
  changes the schedule curve.
- `gpu_parity`: PASS.
