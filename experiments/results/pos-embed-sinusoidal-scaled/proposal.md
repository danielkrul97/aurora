# pos-embed-sinusoidal-scaled

**Area**: embedding

**Motivation**: Iter 34 (`pos-embed-sinusoidal-init`) was broken_impl —
sinusoidal magnitudes in [-1, 1] overwhelmed token signal in the
small-dim smoke test (~28× larger than original random init's std
of 0.020). Math was sound; the bug was magnitude balance, not the
structural prior itself. This iteration fixes the magnitude.

The original random init: `U(-0.02·√3, 0.02·√3) → std ≈ 0.020`.
Raw sin/cos values: `std ≈ 0.707` (uniform over circle). Scale
factor 0.020 / 0.707 ≈ 0.028 ≈ **0.03**.

Mechanism: same as iter 34 (Vaswani PE pattern) but each sin/cos
output multiplied by 0.03. The frequency-vs-distance structural
prior is preserved (relative pattern across positions and dims is
identical) while absolute magnitudes match the well-tuned init
magnitude that the baseline expects.

**Mechanism**: In `EmbedWeights::new`,
`w_pos[pos*d + 2i] = 0.03 * sin(pos / 10000^(2i/d))`
`w_pos[pos*d + 2i+1] = 0.03 * cos(pos / 10000^(2i/d))`
~12 LoC pure CPU. GPU inherits via from_cpu.

**Expected Δ BPB**: −0.005 to −0.020. Structural prior gives the
model "starting positional information" while still allowing
gradient-driven refinement. The magnitude is now safe. Optimistic
end could even cross the −0.020 win threshold — first time on the
new baseline if it does. Pessimistic: 641 steps is enough for AdamW
to learn position from scratch, structural-init advantage erased.

**Why unexplored**: Iter 34 broke; this is the magnitude-corrected
follow-up — a genuinely different attempt, not a duplicate.

## Math gate

- **Modified equation(s)**:
  - At init: `w_pos[pos*d + 2i] = 0.03 * sin(pos / 10000^(2i/d))`
  - At init: `w_pos[pos*d + 2i+1] = 0.03 * cos(pos / 10000^(2i/d))`
- **Backward pass**: Identical. Init values don't affect gradient
  reachability.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: w_pos is summed with token embedding
  at each position; structural prior doesn't introduce leakage.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Magnitude check**: scaled init has std ≈ 0.03 * 0.707 ≈ 0.021,
  matching the old random init's std ≈ 0.020. Smoke test should
  pass.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Math unchanged.
- `transformer_smoke`: PASS. Magnitude is now matched to old init;
  argmax sanity should not be affected. (Iter 34 failed only because
  magnitude was 28× too large.)
- `gpu_parity`: PASS. CPU init copied to GPU buffer byte-for-byte.
