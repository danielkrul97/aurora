# adam-beta2-099

**Area**: optimizer

**Motivation**: β2 sweep to bracket the optimum.
- β2 = 0.95 (baseline)
- β2 = 0.98 (iter 32): Δ = −0.010 (win-direction, sub-threshold)
0.98 beats 0.95. This probes 0.99: if it beats 0.98, the optimum is
higher; if it loses, 0.98 is near-peak. Same bracketing logic used
for the WSD-tail (10/20/30) and per-block-lr spread sweeps.

β2 = 0.99 gives an effective m2 averaging window of ~100 steps (vs
~50 at 0.98, ~20 at 0.95). At 641 total steps that's a substantial
fraction — the m2 estimate becomes very slow-moving.

**Mechanism**: `src/optimizer.rs::Default for AdamWConfig`,
`beta2: 0.95 → 0.99`. 1 LoC. Flows to CPU + GPU via the config struct.

**Expected Δ BPB**: −0.005 to +0.010. If 0.99's very-slow m2 is too
sluggish to track early-training gradient-variance shifts, it could
underperform 0.98.

**Why unexplored**: β2 sweep — 0.98 is in bank, 0.99 is the next
bracket point.

## Math gate

- **Modified equation(s)**: `m2 = 0.99·m2 + 0.01·g²`. Update rule
  otherwise identical.
- **Backward pass**: AdamW is post-gradient; no autograd change.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count / VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS. β2=0.99 smooths m2 more; gradient
  direction unchanged. 600-step convergence fine.
- `gpu_parity`: PASS.
