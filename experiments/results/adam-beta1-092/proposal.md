# adam-beta1-092

**Area**: optimizer

**Motivation**: Iter 36 (`adam-beta1-095`) broke smoke (argmax sanity
failed under high-lr=3e-2 regime — momentum builds too slowly to
converge in 100 steps). β1=0.92 sits between the passing 0.9 default
and the failing 0.95. It probes the smoke-compatibility boundary while
still moving in the smoother-dynamics direction.

Effective momentum window: at β1=0.92, ~12 steps (vs 10 at 0.9, 20 at
0.95). Smaller delta from baseline than 0.95 but still slightly
smoother.

If smoke passes:
- Expected screening Δ: tiny (−0.001 to −0.005). The
  smoother-dynamics gains have been correlating with how much the
  momentum window shifts; +20% window expansion is modest.
- Confirms β1 axis is alive and informs future composites.

If smoke fails:
- Confirms the boundary is below 0.92; β1=0.9 is essentially the
  smoothest momentum compatible with the test infrastructure at
  lr=3e-2. Useful info for future tweaks.

**Mechanism**: In `src/optimizer.rs::Default for AdamWConfig`, change
`beta1: 0.9` to `beta1: 0.92`. 1 LoC. Flows to GPU.

**Expected Δ BPB**: −0.001 to −0.005 if smoke passes. n/a if broken.

**Why unexplored**: Iter 36 left this boundary unmapped. β1 axis
remains under-probed.

## Math gate

- **Modified equation(s)**: `m1 = 0.92·m1 + 0.08·g`.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PROBABLY PASS. The β1=0.95 break gave argmax=0
  (vs expected 2) after 100 steps at lr=3e-2; β1=0.92 has a 20%
  smaller momentum-buildup deficit. Likely converges before the
  100-step check.
- `gpu_parity`: PASS.
