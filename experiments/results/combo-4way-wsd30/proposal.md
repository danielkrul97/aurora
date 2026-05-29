# combo-4way-wsd30

**Area**: composite (schedule × optimizer × depth-cond-reg × embedding-LR)

**Motivation**: The bank-best is iter 65 (`combo-4way-embed-1p1`,
−0.0868) using WSD-tail20. The tail-fraction sweep (iters 58/60/61)
found tail30 (−0.045) marginally beats tail20 (−0.044) solo — within
noise but slightly favorable. This iter consolidates the best-known
4-way recipe with the marginally-better tail fraction.

Components:
1. WSD-tail**30** schedule (−0.045 solo)
2. AdamW β2 = 0.98 (−0.010 solo)
3. per-block lr_mult [0.2×, 1.8×] (−0.025 solo)
4. embed-lr 1.1× (−0.024 solo)

If tail30 transfers its ~0.001 solo edge into the composite, this
edges past −0.0868. If not, it confirms tail20≈tail30 holds in
composite too.

**Mechanism**: Identical to iter 65 but the WSD decay fraction is
0.3 instead of 0.2. ~36 LoC.

**Expected Δ BPB**: −0.085 to −0.090. Likely ≈ iter 65 (−0.0868) ±
0.002.

**Why unexplored**: Distinct composite — iter 65 used tail20, this
uses tail30.

## Math gate

Each component's math gate carries from its individual iter (61, 32,
49, 63). No new math.

- **Modified equation(s)**: WSD-tail30 schedule; β2=0.98; per-block
  lr = base·(0.2+1.6·li/(n_layer−1)); embed lr = base·1.1.
- **Backward / dims / causality / params / VRAM**: unchanged, 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. iter 65 (same recipe, tail20) passed all
  gates; tail30 only changes the decay-fraction constant.
- `gpu_parity`: PASS.
