# silu-activation

**Area**: loss-surface

**Motivation**: The FFN uses GELU (tanh approximation). The activation
function shapes the loss-surface curvature and gradient flow. SiLU
(a.k.a. Swish, `x·sigmoid(x)`) is the modern-LLM choice (Llama, Gemma,
PaLM use SiLU or its GLU variant). SiLU and GELU are very similar
(both smooth, non-monotonic near 0, ≈ReLU for large x) but differ
slightly in the transition region.

The bank has never varied the activation function — it's an unexplored
axis. SiLU's slightly different curvature could shift loss-surface
geometry favorably (or be neutral, as GELU and SiLU are close).

**Mechanism**: Replace the GELU computation with SiLU in both
backends:
- CPU `src/linalg.rs::gelu` → `silu(x) = x / (1 + exp(-x))`
- CPU `src/linalg.rs::gelu_backward` → `silu'(x) = s·(1 + x·(1−s))`
  where `s = sigmoid(x)`
- GPU `src/gpu/gelu.rs` GELU_FORWARD_WGSL + GELU_BACKWARD_WGSL strings

Function names stay `gelu`/`gelu_backward` (renaming is cosmetic and
out of scope) — only the bodies change. ~14 LoC.

**Expected Δ BPB**: −0.010 to +0.010. GELU and SiLU are close; the
effect is likely small. Could be a marginal win or neutral.

**Why unexplored**: Activation function has never been changed in any
bank entry.

## Math gate

- **Modified equation(s)**: FFN hidden activation
  `h_post = silu(h_pre)` where `silu(x) = x·σ(x)`, `σ(x)=1/(1+e^{−x})`.
  Backward: `∂h_post/∂h_pre = σ(x)·(1 + x·(1−σ(x)))`.
- **Backward pass**: SiLU is smooth and differentiable everywhere;
  the derivative is finite and non-zero for all finite x (∂/∂x → 1 as
  x→+∞, → 0 as x→−∞). Gradient reaches w_fc via the same chain.
- **Dim compatibility**: 0 changes — element-wise, same shapes.
- **Causality preserved**: Element-wise position-wise; no cross-position
  coupling.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. The numerical gradient check verifies
  `silu_backward ≈ finite-diff(silu)`. Both are computed consistently
  (analytic derivative matches the forward); finite-diff check holds.
- `transformer_smoke`: PASS. SiLU is a valid smooth activation; the
  copy task converges with it just as with GELU.
- `gpu_parity`: PASS provided CPU and GPU compute identical SiLU. The
  drift threshold is 5e-2; SiLU is numerically simple (one exp, one
  div), CPU and GPU f32 agree well within tolerance.
