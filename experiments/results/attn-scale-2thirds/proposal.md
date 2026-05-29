# attn-scale-2thirds

**Area**: attention

**Motivation**: Aurora's scaled dot-product attention uses the canonical
Vaswani scale `1/sqrt(d_k)` where `d_k = 64` (for d_model=384, 6 heads).
This scale was derived to keep the variance of `QK^T` close to 1 at
*standard* init, ensuring the softmax doesn't saturate. With Aurora's
Xavier init and current dimensions, that variance assumption holds at
init — but Aurora is severely under-trained (val BPB 2.72 after 1 epoch
vs goal <1.5). In the under-trained regime, *attention sharpness* is a
core bottleneck: heads need many gradient updates to learn to focus.

A *softer* scale (e.g. `2/(3·sqrt(d_k))`) makes the initial softmax more
uniform — each query sees a broader average of past tokens. This is
suboptimal for late-stage prediction (where focusing matters), but in
the early-training regime the model is updating most weights for the
first time and *broad context averaging* gives more stable gradient
signal through attention.

A related observation: `logit-scale-2x` (iter 9) hurt by +0.14 BPB,
showing Aurora is sensitive to amplifying the dominant gradient signal.
Softening *attention* — orthogonal pathway — tests whether the inverse
direction (lower-magnitude attention scores) helps via the smoother
loss landscape.

**Mechanism**: Replace
`(d_k as f32).sqrt().recip()` with
`(d_k as f32).sqrt().recip() * (2.0 / 3.0)` at every site where attention
scale is computed:
- `src/attention.rs::scaled_dot_product_attention` (CPU forward)
- `src/gpu/attention.rs` lines 413, 475, 597, 765 (4 GPU dispatch sites
  covering training fwd/bwd and kv-cache prefill/append)

Backward derivative scaling cascades naturally — the GPU backward
kernels multiply `d_scores` by the same `scale`, so the chain rule is
self-consistent.

**Expected Δ BPB**: -0.005 to -0.04. Direction-wise positive based on
the "Aurora over-amplifies dominant gradients" pattern; magnitude
uncertain because softer attention is also weaker information for late
predictions, partial cancellation likely.

**Why unexplored**: Bank has 13 entries, no `attention` entries yet.
This is the first probe of the attention area. Attention temperature is
in the spec's idea list ("attention temperature schedule") — a constant
softening is the simplest variant.

## Math gate

- **Modified equation(s)**: scores = (Q·Kᵀ) · (2 / (3·√d_k)) instead of
  (Q·Kᵀ) · (1/√d_k). Forward softmax(scores) is computed on the new
  scores. Backward: `d_QKt = d_scores_softmax · (2/(3·√d_k))` (same
  factor cascades).
- **Backward pass**: Scale is a positive scalar; chain rule preserves
  differentiability. No new non-differentiability.
- **Dim compatibility**: Scalar factor; no tensor shape changes.
- **Causality preserved**: Scale operates on already-masked scores;
  the causal mask is structural.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

CPU `scaled_dot_product_attention` and GPU MHA kernels both compute
scale independently from `d_k`. If I change both sides consistently
(5 sites total), they remain in sync; `gpu_parity::causal_mha_*` should
pass with the same drift tolerance.

## Primitives test interaction

`mha_causal_backward_matches_numerical` uses central-difference at
d_model=8, EPS=1e-3, TOL=5e-2. A *softer* scale produces *smoother*
softmax (less peaked), which improves finite-difference precision
relative to baseline — opposite direction from the He-init failure
(iter 8, which sharpened softmax and exceeded TOL). Expected PASS.
