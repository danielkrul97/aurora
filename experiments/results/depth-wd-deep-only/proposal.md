# depth-wd-deep-only

**Area**: depth-cond-reg

**Motivation**: AdamW currently applies weight decay (wd=0.1 from CLI)
uniformly to every weight matrix in every block (12 of them: 4 attention
projections + 2 FFN projections × 6 blocks). Conventional intuition
from transfer-learning literature (LLRD, RoBERTa-style depth-aware WD)
is that **shallow layers learn general representations early** and
benefit from less perturbation, while **deep layers are more task-
specific** and can absorb more regularization. In a from-scratch
1-epoch byte-LM run, shallow layers are still grinding to find basic
byte-trigram patterns — extra shrinkage from WD slows them down.

This experiment tests a step-function variant of LLRD: turn WD *off*
for the first half of blocks (layers 0-2) and keep WD *on* for the
deeper half (layers 3-5). Existing LayerNorm γ/β, positional embeddings,
and token embeddings all stay with their current apply_wd settings.

**Mechanism**: In both CPU `step_adamw` and GPU `build_grad_ops`,
iterate over blocks with index. For block index `li < n_layer/2`, pass
`apply_wd = false` for the 6 weight tensors (w_q, w_k, w_v, w_o, w_fc,
w_proj). For `li >= n_layer/2`, keep `apply_wd = true`. LN params
already use `apply_wd = false` regardless of depth.

**Expected Δ BPB**: -0.005 to -0.025. Small expected gain because at
WD=0.1 and the WSD-not-applied baseline schedule, integrated shrinkage
per parameter over 641 steps is modest. Direction-wise positive based
on LLRD prior; could go negative if shallow layers benefit from
regularization more than I think.

**Why unexplored**: Bank has 14 entries, no `depth-cond-reg` entries
yet. Depth-dependent WD is in the spec's explicitly-suggested ideas
list. Different mechanism from any prior entry: per-tensor regularization
asymmetry, not init/schedule/loss/attention scaling.

## Math gate

- **Modified equation(s)**: AdamW step is unchanged structurally. The
  decoupled WD term `+ lr · wd · param` is applied conditionally:
  - Layers 0..N/2-1: WD off for w_{q,k,v,o,fc,proj} (same as LN γ/β).
  - Layers N/2..N-1: WD on for these tensors (same as baseline).
- **Backward pass**: WD is an optimizer update term, not a gradient.
  Backward through the model is unchanged.
- **Dim compatibility**: No tensor shapes change.
- **Causality preserved**: WD is per-parameter scalar; no spatial
  interaction.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

Both paths must apply the same apply_wd pattern per block. CPU's
`step_adamw` iterates blocks with `enumerate()`; GPU's `build_grad_ops`
needs the same. If pattern matches, `gpu_parity::full_transformer_training_step_matches_cpu_direction`
preserves drift under 5e-2.
