# combo-4way-embed-1p1

**Area**: composite (schedule × optimizer × depth-cond-reg × embedding-LR)

**Motivation**: Iter 64 (`combo-wsd-beta2-blocklr-embed-lr` with embed
at 1.2×) was broken_impl — the 4-way stack failed smoke's copy-task
convergence (24.4% drop, needs 30%). All 4 components and the 3-way
subset (iter 59) pass smoke individually; the 4-way's "embeddings fast
(1.2×) + block 0 slow (0.2×)" LR imbalance (6× ratio) tipped the tiny
smoke model below threshold.

This retry uses embed-lr 1.1× — reducing the embed/block-0 LR ratio
from 6× to 5.5×. Marginal, but the smoke failure was only 5.6
percentage points short; a gentler embed boost might recover it.

**Mechanism**: Same as iter 64 but embed multiplier 1.1 instead of 1.2.
WSD-tail20 + β2=0.98 + per-block-lr [0.2,1.8] + embed-lr 1.1×. ~36 LoC.

**Expected Δ BPB**: −0.095 to −0.110 IF smoke passes. Linear sum:
−0.044 + −0.010 + −0.025 + ~−0.024 (embed 1.1× scaled from 1.2×'s
−0.027) = −0.103. If smoke fails again, the 4-way line is confirmed
smoke-incompatible and iter 59 (−0.0825) stands as the bank-best.

**Why unexplored**: Iter 64 used embed 1.2×; this is the milder retry.

## Math gate

Same as iter 64 with embed multiplier 1.1.

- **Modified equation(s)**: schedule WSD-tail20; β2=0.98; per-block
  lr = base·(0.2+1.6·li/(n_layer−1)); embedding lr = base·1.1.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta / VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: UNCERTAIN. Iter 64 at embed 1.2× failed by 5.6
  points. 1.1× is a small reduction in the imbalance — may or may not
  recover the gap. ~50/50.
- `gpu_parity`: PASS.
