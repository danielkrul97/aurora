# per-block-lr-extreme-up

**Area**: depth-cond-reg

**Motivation**: Map the saturation curve of per-block-LR asymmetry.
Bank data points:
- iter 44 [1.5, 0.5] (counter-direction): Δ = +0.018
- iter 45 [0.5, 1.5]: Δ = −0.014
- iter 47 [0.3, 1.7]: Δ = −0.0206

Scaling so far: [0.5, 1.5] → [0.3, 1.7] = spread 1.0 → 1.4, gain 1.47×.
Roughly linear in spread. Probe [0.2, 1.8] (spread 1.6, 14% wider than
iter 47). Linear scaling predicts solo Δ ≈ −0.023. Saturation predicts
plateau near −0.021.

Block multipliers: 0.2, 0.52, 0.84, 1.16, 1.48, 1.8. Mean = 1.0.

**Mechanism**: Same code as iter 47 but formula
`lr_mult = 0.2 + 1.6 * (li/(n_layer-1))`. 16 LoC.

**Expected Δ BPB**: −0.021 to −0.026. Wins the threshold solo.

**Why unexplored**: Different spread variant; iter 47 was [0.3, 1.7],
iter 45 was [0.5, 1.5]. This probes the next point on the saturation
curve.

## Math gate

- **Modified equation(s)**: For block li,
  `lr_mult = 0.2 + 1.6 · li/(n_layer-1)`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. With n_layer=2, block 0 = 0.2× and block 1
  = 1.8×. At smoke's lr=3e-2, block 1 effective ≈ 5.4e-2 — getting high
  but iter 47 [block 1 = 1.7×, effective 5.1e-2] passed; the marginal
  delta should still be safe.
- `gpu_parity`: PASS.
