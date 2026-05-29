# per-block-lr-linear-up

**Area**: depth-cond-reg

**Motivation**: Direct follow-up to iter 44 (`per-block-lr-linear-down`,
+0.018 loss). That iter tested "shallow blocks get more LR"
(counter-intuitive direction); it hurt. This iter tests the reverse:
"deep blocks get more LR" — closer to conventional intuition that
deeper layers do higher-level abstraction and benefit from larger
update magnitudes.

LR ramp: block 0 = 0.5×, block 1 = 0.7×, block 2 = 0.9×, block 3 = 1.1×,
block 4 = 1.3×, block 5 = 1.5×. Mean = 1.0 (same as iter 44, just
inverted).

Hypothesis: if iter 44 (shallow-more) lost ~0.018, the symmetric
deep-more direction should gain a similar amount. If both lose, the
finding is "per-block asymmetry hurts in any direction" — also useful.

**Mechanism**: Same code structure as iter 44 but with flipped formula:
`lr_mult = 0.5 + li as f32 / (n_layer - 1) as f32`.
CPU `step_adamw` + GPU `step_adamw` loop both modified. ~16 LoC.

**Expected Δ BPB**: −0.005 to −0.025. Mild win expected if the
asymmetry direction matters. Or near-zero if iter 44's regression was
due to per-block asymmetry per se (not direction).

**Why unexplored**: Iter 44's inverse direction is a natural follow-up
not yet in bank.

## Math gate

- **Modified equation(s)**: For block `li`, `lr_mult = 0.5 + li/(n_layer-1)`.
  AdamW step uses `base_lr * lr_mult`. Other params unchanged.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (no AdamW in unit tests).
- `transformer_smoke`: PASS. Iter 44's symmetric variant passed; this
  one applies smaller LR to block 0 and larger to block 1 in n_layer=2
  case, also reasonable.
- `gpu_parity`: PASS. Lockstep CPU+GPU.
