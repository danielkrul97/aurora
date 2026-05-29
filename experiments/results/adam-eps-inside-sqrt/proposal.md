# adam-eps-inside-sqrt

**Area**: optimizer

**Motivation**: Current AdamW: `step = lr * m1_hat / (sqrt(m2_hat) + eps)`.
Alternative ("eps-inside"): `step = lr * m1_hat / sqrt(m2_hat + eps²)`.

These are mathematically NEAR-equivalent for any m2_hat much larger
than eps² (= 1e-16 at default eps=1e-8). They differ meaningfully only
when m2_hat is at or near zero, where:
- Standard: denominator ≈ eps (so step ≈ lr * m1_hat / eps — very large)
- Inside: denominator ≈ eps (same magnitude, sign matters)

In practice, m2_hat is bounded below by 1-β2 fraction of g² accumulated.
The first step has bc2 = 1-β2 = 0.05, m2 = 0.05 * g², m2_hat = m2/bc2 =
g². So m2_hat ≈ g² >> eps² at step 1 already. The two formulas should
give numerically identical updates after step 1.

But — eps appears differently in the update rule. Standard ADD form
contributes a floor of `eps` to the denominator. Inside-sqrt contributes
a floor of `sqrt(eps²) = eps` only when m2_hat ≤ eps². So the inside
form has a sharper transition. Some numerical analysis papers prefer
this form for stability.

Expected effect: essentially zero in practice. This is a data-point
iteration to map a previously untouched optimizer axis, not a serious
attempt at a win.

**Mechanism**: 
- CPU: `src/optimizer.rs` line 70 — change
  `let mut step = config.lr * m1_hat / (m2_hat.sqrt() + config.eps);`
  to `let mut step = config.lr * m1_hat / (m2_hat + config.eps * config.eps).sqrt();`
- GPU: `src/gpu/optimizer.rs` line 44 of WGSL string — change
  `var step = p.lr * m1_hat / (sqrt(m2_hat) + p.eps);`
  to `var step = p.lr * m1_hat / sqrt(m2_hat + p.eps * p.eps);`
2 LoC.

**Expected Δ BPB**: −0.001 to +0.002. Mostly noise.

**Why unexplored**: Bank has 41 entries. Optimizer area: 5 prior
(2 eps-magnitude, β1/β2, WD-target). Eps PLACEMENT is a distinct axis
that hasn't been tested.

## Math gate

- **Modified equation(s)**: AdamW step = `lr * m1_hat / sqrt(m2_hat + eps²)`.
- **Backward pass**: Optimizer step is post-gradient; no autograd change.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **GPU parity**: CPU and GPU formula must match — both edited in lockstep.

## Sanity-gate prediction

- `transformer_primitives`: PASS.
- `transformer_smoke`: PASS.
- `gpu_parity`: PASS provided CPU/GPU formulas match byte-for-byte.
