# combo-wsd-beta2-embedlr

**Area**: composite (schedule × optimizer × embedding-LR)

**Motivation**: Two 3-way composites are now of interest:
- iter 59: WSD + β2 + per-block-lr [0.2,1.8] → −0.0825
- this:    WSD + β2 + embed-lr 1.2×        → ?

Both swap the "third axis" between the two per-group-LR mechanisms.
per-block-lr (−0.025 solo) scales transformer blocks; embed-lr
(−0.027 solo) scales embeddings. Question: does embed-lr substitute
cleanly for per-block-lr in the 3-way? If this gives ≈−0.082, the two
per-group-LR axes are interchangeable as the third component. If
different, one composes better with WSD+β2.

The 4-way (iter 65, all four) gave −0.0868 — embed-lr added only
+0.004 over the 3-way iter 59, confirming embed-lr and per-block-lr
overlap. This iter isolates embed-lr's standalone composability.

**Mechanism**: WSD-tail20 schedule + β2=0.98 + embed-lr 1.2× (w_tok,
w_pos). NO per-block-lr. ~26 LoC.

**Expected Δ BPB**: −0.075 to −0.085. Likely close to iter 59.

**Why unexplored**: Distinct 3-way axis-set; iter 59 used per-block-lr,
this uses embed-lr.

## Math gate

Components from iters 58, 32, 63. No new math.
- **Modified equation(s)**: WSD-tail20 schedule; β2=0.98; embedding lr
  = base·1.2.
- **Backward pass / dims / causality / params / VRAM**: unchanged, 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. No per-block-lr → no block-0/embed LR
  imbalance (which broke the iter-64 4-way). embed-lr-1.2x + WSD + β2
  each passed smoke; this combination has no known interference.
- `gpu_parity`: PASS.
