# combo-wsd-beta2-blocklr-embed-lr

**Area**: composite (schedule × optimizer × depth-cond-reg × embedding-LR)

**Motivation**: Iter 59 (`combo-wsd-beta2-blocklr-extreme-up`) is the
bank-best at −0.0825. Iter 63 (`embed-lr-1p2x`) revealed a NEW
independent axis: 1.2× LR for the embedding group (w_tok, w_pos) gives
−0.027 solo. The embedding-LR axis is orthogonal to per-block-LR
(transformer blocks) — they touch disjoint parameter groups.

4-way composite:
1. WSD-tail20 schedule (−0.044 solo)
2. AdamW β2 = 0.98 (−0.010 solo)
3. per-block lr_mult [0.2×, 1.8×] (−0.025 solo)
4. embed-lr 1.2× (−0.027 solo)

Linear sum: −0.106. The 3-way composites showed +0.003 to +0.007
super-additivity. A 4-way with all-independent axes should give
≈ −0.106 to −0.113.

**Mechanism**: Iter 59's 3 changes plus the embed-LR 1.2× multiplier
for w_tok/w_pos. The per-block-LR (ops 4+) and embed-LR (ops 0,1) are
applied to disjoint op ranges, so they don't conflict. ~36 LoC.

**Expected Δ BPB**: −0.095 to −0.115. New bank-best by a wide margin.
Risk: sub-additive interaction (the per-block + embed LR changes both
touch the optimizer step) could reduce the gain.

**Why unexplored**: 4-way composite never tested; embed-LR is a
brand-new axis as of iter 63.

## Math gate

Each component's math gate carries over from its individual iter
(58, 32, 49, 63). No new math.

- **Modified equation(s)**: schedule = WSD-tail20; AdamW β2 = 0.98;
  per-block effective lr = base·(0.2+1.6·li/(n_layer−1)); embedding
  effective lr = base·1.2.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: All post-gradient.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. All four components passed smoke
  individually (embed-lr at 1.2× passed iter 63; 1.5× was the one that
  broke). No capacity-starvation stacking (unlike iter 53's
  freeze-betas combo): embed-LR is a boost, per-block-LR has block 0 at
  0.2× but that passed in iter 50/59 composites.
- `gpu_parity`: PASS.
