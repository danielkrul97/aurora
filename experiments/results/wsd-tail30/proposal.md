# wsd-tail30

**Area**: schedule

**Motivation**: Bracket the WSD tail-fraction peak.
- tail10 (iter 60): −0.033
- tail20 (iter 58): −0.044
20% beats 10%. This probes 30%: if tail30 < tail20, the peak is
confirmed at ~20%. If tail30 > tail20, longer annealing helps and
the peak is higher.

WSD literature (MiniCPM) uses ~10-20% decay; 30% is on the long end.
For a 1-epoch run where each step is precious, a 30% decay tail means
only 70% of post-warmup at flat base_lr — less high-LR time, more
annealing.

**Mechanism**: WSD body with `0.3` decay fraction. ~3 LoC in
`src/transformer/schedule.rs`.

**Expected Δ BPB**: −0.035 to −0.045. Most likely slightly below
tail20 (−0.044) — 20% is probably near-optimal.

**Why unexplored**: Completes the tail-fraction sweep (10/20/30).

## Math gate

- **Modified equation(s)**: post-warmup flat at base_lr until
  `decay_start = max_steps − ceil(0.3·post_warmup)`, then linear decay.
- **Backward pass**: External scalar; no gradient flow.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. 30% tail still leaves 70% post-warmup at
  base_lr; convergence is fine.
- `gpu_parity`: PASS.
