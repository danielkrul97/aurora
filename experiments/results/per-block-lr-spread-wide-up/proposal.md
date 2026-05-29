# per-block-lr-spread-wide-up

**Area**: depth-cond-reg

**Motivation**: Iter 45 (`per-block-lr-linear-up` at [0.5, 1.5]) gave
Δ = −0.014 (sub-threshold win). Iter 46 just verified the mechanism
composes linearly with two other smooth-dynamics axes for a clean win.
This iter probes whether the per-block-LR effect SATURATES at [0.5, 1.5]
or scales further with wider spread.

Mechanism: same as iter 45 but range [0.3, 1.7] (vs [0.5, 1.5]). Mean
preserved at 1.0; spread is 2× wider. Block multipliers: 0.3, 0.58,
0.86, 1.14, 1.42, 1.7.

If the depth-asymmetric LR effect is roughly LINEAR in spread, expected
Δ ≈ −0.028 solo. If saturated, Δ stays near −0.015. If unstable (too
extreme), might hurt.

Note: block 0 = 0.3× lr = 1.8e-4 effective. Block 5 = 1.7× lr = 1.02e-3.
The deepest block is operating significantly above the original lr=6e-4.
That's substantial — possible overshoot risk.

**Mechanism**: Same code pattern as iter 45 but formula:
`lr_mult = 0.3 + 1.4 * (li / (n_layer - 1))` so [0, 1] → [0.3, 1.7].
22 LoC across CPU + GPU.

**Expected Δ BPB**: −0.020 to −0.030. Above threshold solo if linear
in spread; sub-threshold if saturating.

**Why unexplored**: Iter 45 tested [0.5, 1.5]; this tests wider spread.
The "shape of the depth-LR response" is a meaningful axis to probe.

## Math gate

- **Modified equation(s)**: For block li, `lr_mult = 0.3 + 1.4 · li/(n_layer-1)`.
  AdamW step uses `base_lr * lr_mult`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (no AdamW in unit tests).
- `transformer_smoke`: PASS hopefully. With n_layer=2, block 0 = 0.3×
  and block 1 = 1.7×. At smoke's lr=3e-2, block 1 gets effective
  5.1e-2 which is high but Aurora handles 6e-2-ish in fix-mha tests
  without issue. Should converge.
- `gpu_parity`: PASS. Both backends modified identically.
