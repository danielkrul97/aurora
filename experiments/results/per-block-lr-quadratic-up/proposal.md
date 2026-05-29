# per-block-lr-quadratic-up

**Area**: depth-cond-reg

**Motivation**: Linear per-block-LR has peaked at [0.2, 1.8] (iter 49,
Δ=−0.025). Extending to [0.1, 1.9] regressed sharply (iter 51, Δ=−0.013)
— the linear shape has saturated at this LR base.

But the SHAPE of the LR distribution along depth is unexplored. Linear
spreads the LR boost evenly across all blocks. A QUADRATIC shape with
the same endpoints concentrates boost in the deepest blocks (and
allocates more LR to middle layers that linear left at ~1.0×).

Formula: `lr_mult = 0.536 + 1.264 · (li/(n_layer-1))²`
- block 0: 0.536
- block 1: 0.586
- block 2: 0.738
- block 3: 0.991 (~1.0)
- block 4: 1.345
- block 5: 1.800
- Mean: 1.0 ✓ (preserved)
- Endpoints: 0.536 / 1.8 (slightly less extreme than linear's 0.2/1.8)

Hypothesis: if deepest blocks are the bottleneck (where representation
becomes task-specific), quadratic boost there might compound. Or it
might lose because middle blocks need more help than quadratic gives.

**Mechanism**: Same code pattern as iter 49 but quadratic formula.
~16 LoC.

**Expected Δ BPB**: −0.018 to −0.030. Comparable to linear [0.2, 1.8]
or slightly better/worse depending on which blocks matter.

**Why unexplored**: Linear is the only shape tested. Quadratic is the
natural next probe.

## Math gate

- **Modified equation(s)**: `lr_mult = 0.536 + 1.264 · (li/(n_layer-1))²`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Mean preservation**: For uniform li ∈ [0, n_layer-1], mean of
  (li/(n_layer-1))² = sum_{i=0}^{n_layer-1} (i/(n_layer-1))² / n_layer.
  For n_layer=6: sum/6 = (0+1+4+9+16+25)/(25·6) = 55/150 = 0.367.
  Verified: 0.536 + 1.264 · 0.367 = 0.536 + 0.464 = 1.0 ✓.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. Smoke uses n_layer=2 (block 0 = 0.536×,
  block 1 = 1.8×). At lr=3e-2: block 1 effective ≈ 5.4e-2. Iter 49 had
  identical block 1 at 1.8× and passed.
- `gpu_parity`: PASS.
