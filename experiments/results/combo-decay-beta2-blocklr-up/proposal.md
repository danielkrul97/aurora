# combo-decay-beta2-blocklr-up

**Area**: composite (schedule × optimizer × depth-cond-reg)

**Motivation**: Three independent smoother-dynamics axes in the
sub-threshold-win bucket:
1. `inverse-sqrt-decay`: Δ = −0.007 (smoother schedule, more time at base_lr)
2. `adam-beta2-098`: Δ = −0.010 (smoother m2 estimate)
3. `per-block-lr-linear-up`: Δ = −0.014 (deep blocks larger LR)

Each axis is mechanically distinct: schedule (whole-run LR curve),
optimizer internal (m2 averaging), depth-cond per-param (block-LR
multiplier). They operate on different time-scales and parameter
groups; should compose with minimal interference.

Linear sum: −0.031. Iter 33 (decay × β2 composite) was super-additive
by 0.002. If similar super-add applies here, expected ~−0.033. Even
with sub-additive penalty of 20%, ~−0.025 still crosses −0.020 win
threshold.

This is the highest-leverage composite this session can build with
available bank-mapped axes.

**Mechanism**: Three atomic changes:

1. `src/transformer/schedule.rs::cosine_warmup` — post-warmup branch =
   `base_lr * sqrt(warmup_steps / step)` (matches iter 31).
2. `src/optimizer.rs::Default for AdamWConfig` — `beta2: 0.95 → 0.98`
   (matches iter 32).
3. `src/transformer/model.rs::step_adamw` and
   `src/gpu/transformer.rs::step_adamw` — per-block lr_mult =
   `0.5 + li/(n_layer-1)` (matches iter 45).

Total ≈ 22 LoC. No GPU shader changes; all changes flow through
existing dispatch parameters.

**Expected Δ BPB**: −0.020 to −0.035. Win-threshold-crossing scenario.

**Why unexplored**: Each of the three components has its own bank
entry; the 3-way combination is novel.

## Math gate

- **Modified equation(s)**: Each piece's math is unchanged from its
  individual iter (31, 32, 45). The composite simply applies all three
  simultaneously. Forward unchanged; backward unchanged; only the
  optimizer step rules differ.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: All three are post-gradient/optimizer
  modifications.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Each piece passed individually;
  composite doesn't affect grad computation.
- `transformer_smoke`: PASS. Each piece passed smoke (600-step
  convergence). The 3-way composite gives shallow block 0 = 0.5×
  effective LR — gentler — but smoke uses n_layer=2 where block 0 = 0.5×
  and block 1 = 1.5×; mean preserved. Should converge.
- `gpu_parity`: PASS. CPU and GPU schedule, optimizer, and per-block
  LR all matched.
