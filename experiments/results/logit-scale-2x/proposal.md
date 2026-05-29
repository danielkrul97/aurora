# logit-scale-2x

**Area**: loss-aux

**Motivation**: At Aurora's init (w_tok std=0.02, h_final std≈1 after
final LN), the LM-head logits have std ≈ 0.4 across 4096 classes. The
resulting softmax is essentially uniform, so the gradient signal `(p −
1_target)` is small in magnitude per logit (~1/V on non-target, ~−(V−1)/V
at target — net norm ≈ 1). This is the dominant gradient signal for the
*entire* tied LM-head/embedding tensor, which receives gradient through
both the input embedding path *and* the output projection path.

In the byte-LM 1-epoch regime, the model is severely under-trained; any
boost to the dominant gradient pathway should speed up early learning,
provided it doesn't destabilize the rest. Scaling logits by `T = 2`
before softmax produces:
- Same softmax distribution shape (slightly sharper because logits doubled)
- Gradient `dL/d(raw_logit_j) = T · (p_j − 1[j=t])` — gradient magnitude
  ~doubled
- This effectively boosts the LM-head gradient (and thereby the
  tied-embedding gradient via the LM head path) without changing global
  LR (and so without affecting the much smaller LN/positional gradients).

**Mechanism**: In `cross_entropy_with_grad`, before computing softmax,
scale logits by `T = 2`. Compute loss on the scaled logits. The gradient
returned must equal `dL/d(raw_logit)`, which by chain rule is
`T · (softmax(T·logit) − 1_target)`. Implementation is a couple of
multiplications in the body of `cross_entropy_with_grad`; the signature
stays the same so both CPU and GPU call sites pick it up.

**Expected Δ BPB**: -0.01 to -0.05. Direction-wise positive because we
amplify the strongest learning signal. Magnitude uncertain because: (a)
AdamW's second-moment buffer partially compensates for global gradient
magnitude changes after a few warmup steps, (b) at byte-LM scale the
*absolute* loss-floor is still high (~2.7 BPB), so per-step learning
matters but isn't strongly bottlenecked by LM-head signal strength.

**Why unexplored**: Bank has loss-aux × 2 (z-loss, focal — both losses).
Both prior attempts modified the loss VALUE; this one modifies the
loss-GRADIENT magnitude via input scaling. Different mechanism: a
constant pre-softmax temperature is essentially "extra LR for the LM
head pathway" without touching the LR schedule. Plain temperature
scaling at the LM head is folklore for inference but rarely used as a
training mechanism.

## Math gate

- **Modified equation(s)**: Let `T = 2`, `s = T · logit`, `p = softmax(s)`.
  - Forward: `L = −log p_target`.
  - Per-logit grad via chain rule:
    `dL/d(logit_j) = (dL/d s_j) · (d s_j / d logit_j) = (p_j − 𝟙[j=t]) · T`.
  - So `grad = T · (softmax(T·logit) − 1_target)`. Note the loss VALUE
    is computed on scaled logits (same softmax, same `p_target`), so
    `L` differs from raw CE — but `evaluate()` uses raw softmax for
    val_bpb, so the *reported* `val_bpb` remains directly comparable
    to baseline.
- **Backward pass**: `p_t` is finite as long as inputs are finite.
  Standard CE backward extended by constant `T`. No new
  non-differentiability or detach.
- **Dim compatibility**: No tensor shapes change.
- **Causality preserved**: Pointwise per token, no cross-position
  coupling.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Smoke-test interaction

Smoke convergence test asserts `last_loss < first_loss · 0.7`. With
T=2, the *recorded loss values* will differ from baseline (because the
softmax now operates on scaled logits), but the *relative* drop pattern
is expected to be similar or slightly better (because gradient is 2×).
Should pass with comfortable margin.

## Primitives test interaction

`mha_causal_backward_matches_numerical` does NOT exercise
`cross_entropy_with_grad` — it tests MHA forward/backward directly with
synthetic gradients. No tolerance risk from this change.
