# per-block-lr-megaextreme-up

**Area**: depth-cond-reg

**Motivation**: Continue mapping the per-block-LR saturation curve.
Data so far:
- [0.5, 1.5] iter 45: Δ = −0.014 (sub-threshold)
- [0.3, 1.7] iter 47: Δ = −0.0206 (win)
- [0.2, 1.8] iter 49: Δ = −0.025 (win)

Per-unit-spread gain: 0.014, 0.0147, 0.0156 → mild super-linearity.
At [0.1, 1.9] (spread 1.8) the linear extrapolation predicts Δ ≈ −0.028
solo; the super-linear extrapolation predicts Δ ≈ −0.030.

Boundary check: block 5 effective LR = 1.9 × 6e-4 = 1.14e-3 — high but
likely stable given that AdamW normalizes by sqrt(m2). Block 0 = 0.1×
= 6e-5 = same as min_lr, very slow but not zero.

Smoke risk: at lr=3e-2, block 1 (in n_layer=2 smoke) = 1.9× = 5.7e-2.
Iter 47 [1.7×] passed smoke at 5.1e-2, so 5.7e-2 should be borderline-OK.

**Mechanism**: Same code as iter 49 with formula
`lr_mult = 0.1 + 1.8 * (li/(n_layer-1))`. 16 LoC.

**Expected Δ BPB**: −0.025 to −0.032 if linear/super-linear continues.
Saturated near −0.025 if curve flattens. Broken_impl if 1.9× × smoke lr
destabilizes.

**Why unexplored**: Probes the next point on the spread-saturation curve.

## Math gate

- **Modified equation(s)**: `lr_mult = 0.1 + 1.8 · li/(n_layer-1)`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PROBABLY PASS. Block 1 at 1.9× × 3e-2 = 5.7e-2 —
  iter 47's 1.7× × 3e-2 = 5.1e-2 passed; 12% higher should be borderline.
  Some risk of marginal smoke failure.
- `gpu_parity`: PASS.
