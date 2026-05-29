# freeze-ln-betas

**Area**: norm-placement

**Motivation**: Bank has 43 entries, only 1 in norm-placement
(norm-final-half-gamma, loss). The γ axis is unfavorable (backward
gradient scaling). The β axis remains unexplored. β is initialized
to 0 and AdamW updates it; freezing it to 0 makes all LayerNorms
"shift-free": forward becomes `y = γ * (x - μ)/σ`, which is RMSNorm
semantics with mean-centering still applied.

Hypothesis: β contributes a per-channel bias to the post-LN output.
For Aurora (decoder-only pre-norm + tied LM head), constant per-channel
biases in residual stream activations are largely absorbable by w_tok
columns (constants shift logits proportionally). Removing β slightly
reduces parameter count (13 × 384 = 4992 params) and one degree of
freedom that might be redundant under the LM-head tying.

For 1-epoch undertrained byte-LM, redundant params waste a small
fraction of the gradient/AdamW budget. Eliminating them might give a
marginal improvement.

Counter-risk: β is part of standard LN. Modern LLMs (Llama, Gemini)
use RMSNorm specifically and report fine results. Hugging Face GPT-2
uses standard LN with β and also works fine. So β's role is moderate
either way.

**Mechanism**: In `src/transformer/model.rs::step_adamw`, skip the
adamw_apply calls for ln1.beta, ln2.beta, and final_ln.beta (3 sites,
all blocks for the per-block ones). Equivalent CPU change: don't call
adamw_apply for these groups. In `src/gpu/transformer.rs::step_adamw`,
skip the dispatch for beta-ops by indexing (4 ln-betas per non-block
+ 2 per block = 12 ops total to skip).

LoC: ~10. β values stay at their init = 0 forever.

**Expected Δ BPB**: −0.005 to +0.005. Modest. The 1-epoch budget
isn't a strong test of regularization effects.

**Why unexplored**: Bank has 43 entries. Norm-placement: 1 entry.
β freezing is the most surgical possible test of "norm-bias necessity",
sitting between standard LN (β learnable) and full RMSNorm (no β at
all). Hasn't been tested.

## Math gate

- **Modified equation(s)**: All LN forward unchanged: `y = γ * (x - μ)/σ + β`,
  but β is hardcoded to 0 (the init value). Effectively `y = γ * (x - μ)/σ`.
- **Backward pass**: β gradient is still COMPUTED (sum over batch of dy[c]).
  But the AdamW UPDATE is skipped. Net effect: β stays at zero forever.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes (LN is per-position).
- **Param count delta**: Effectively −4992 (β is unused). At construction
  it still takes memory; the gradient buffer still computes d_beta. Could
  be optimized later but for the test, this is fine.
- **VRAM delta**: 0 (params are still allocated, just not updated).

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks don't involve AdamW
  step; β=0 throughout the test doesn't break finite-difference.
- `transformer_smoke`: PASS. Model still has γ (scale) for each LN,
  which is sufficient for the copy task and identity prediction. β=0
  is the standard init anyway; skipping updates doesn't break learning.
- `gpu_parity`: PASS. Both backends skip β AdamW dispatch identically.
