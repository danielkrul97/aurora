# wsd-tail10

**Area**: schedule

**Motivation**: Iter 58 (`wsd-tail20-fresh`) won −0.044 with a 20%
linear-decay tail. The "keep LR near base_lr longer" theme has won
consistently this session (inverse-sqrt-decay, WSD-tail20). The WSD
tail fraction is a tunable: a SHORTER tail means more time at flat
base_lr.

This iter tests 10% tail: flat at base_lr for 90% of post-warmup, then
linear decay to min_lr in the last 10%. If "stay-high-longer" continues
to help, tail10 beats tail20. If 20% was already near-optimal (the
last-phase annealing matters), tail10 underperforms.

For max=641, warmup=200: flat from step 200 to step ~597, decay over
steps 597-641 (~44 steps).

**Mechanism**: In `cosine_warmup`, WSD body with `0.1` decay fraction
instead of `0.2`. ~3 LoC.

**Expected Δ BPB**: −0.040 to −0.050. If shorter tail helps, beats
wsd-tail20's −0.044. If it loses, 20% is near-optimal.

**Why unexplored**: WSD tail fraction tuning — wsd-tail20 is in bank,
this probes a different fraction (allowed: same mechanism, different
parameter, like per-block-lr spread variants).

## Math gate

- **Modified equation(s)**: post-warmup: flat at base_lr until
  `decay_start = max_steps - ceil(0.1·post_warmup)`, then linear decay
  to min_lr.
- **Backward pass**: External scalar; no gradient flow.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Continuity**: warmup ends at base_lr; flat region holds base_lr;
  decay starts at base_lr, ends at min_lr. Continuous.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Shorter tail → even more LR at base; smoke
  convergence (30% drop in 600 steps) is easier with more high-LR time.
- `gpu_parity`: PASS. schedule.rs is CPU-only utility.
