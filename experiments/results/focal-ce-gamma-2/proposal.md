# focal-ce-gamma-2

**Area**: loss-aux

**Motivation**: Standard cross-entropy weighs every token-loss equally:
the gradient for an example with target probability `p_t = 0.001` and
one with `p_t = 0.5` differs only via softmax — both contribute to the
loss / gradient with magnitude `~ 1 − p_t`. In a 1-epoch byte-LM at
vocab=4096, the model spends most of the epoch in a regime where many
positions are "moderately easy" (the model knows likely next-byte
patterns from local context) while a long tail of positions is "hard"
(rare bytes after unusual context). CE smears effort across both;
Focal CE (Lin et al. 2017, retina-net) reweights with `(1 − p_t)^γ`,
which leaves the gradient on hard examples (small p_t) ~unchanged and
sharply downweights easy ones (large p_t). At byte-LM scale this is
folklore-rare but principled — the easy-tail problem is essentially
the same as in image classification.

**Mechanism**: Replace per-token CE
`L_t = -log p_t` with focal CE
`L_t = (1 − p_t)^γ · (-log p_t)` with γ=2. The gradient at logit `j`
becomes `s · (p_j − 𝟙[j=t])` where `s = γ · (1 − p_t)^(γ−1) · log p_t +
(1 − p_t)^γ · (-1/p_t)` is the (signed) per-token weighting factor.
Reuses softmax (and `p_t`) already materialized inside CE — adds ~3
multiplies per logit.

**Expected Δ BPB**: -0.01 to -0.05. Modest expected gain because at 1
epoch most byte predictions are *not* yet "easy" — p_t is small almost
everywhere — so focal's downweighting bites only late in the run.
Direction-wise it should help when paired with WSD's stable-LR tail.

**Why unexplored**: Bank has 4 entries; loss-aux has one (z-loss-1e4
loss, +0.001). z-loss anchors the log-partition; focal reshapes the
per-token weight curve — different mechanism, different math. The
implementation modifies the body of `cross_entropy_with_grad` (same
signature) so both CPU and GPU training call sites pick it up
automatically.

## Math gate

- **Modified equation(s)**: Let `p = softmax(logits)`, `t` = target.
  - Forward: `L = (1 − p_t)^γ · (-log p_t)`.
  - Backward, by chain rule with `dp_t/dlogit_j = p_t (𝟙[j=t] − p_j)`:
    - `dL/dp_t = γ(1−p_t)^(γ−1)(log p_t) + (1−p_t)^γ · (-1/p_t)`
    - `dL/dlogit_j = (dL/dp_t) · p_t · (𝟙[j=t] − p_j)`
    - Let `s = -(dL/dp_t) · p_t`. Then
      `dL/dlogit_j = s · (p_j − 𝟙[j=t])` — i.e., the standard CE
      gradient pre-multiplied by the per-token scalar `s`.
  - At p_t → 0 (random init): `s → 1` (recovers CE). At p_t → 1
    (perfectly learned): `s → 0` (no gradient). Continuous in between.
- **Backward pass**: `s` is finite as long as `p_t > 0`, which we
  already guard with `p_t = max(p_t, 1e-12)`. Every parameter still
  receives gradient via the standard softmax-CE chain, scaled
  per-position by `s`. No new non-differentiability or detach.
- **Dim compatibility**: No tensor shapes change. `s` is a per-token
  scalar (B·T total, negligible memory).
- **Causality preserved**: Loss is pointwise per token. No cross-position
  mixing introduced.
- **Param count delta**: 0. γ=2 is a fixed constant.
- **VRAM delta**: 0.

## Compliance notes

- `cross_entropy_with_grad` keeps its signature
  `(logits: &[f32], target: usize) -> (f32, Vec<f32>)`. Both call
  sites — `src/transformer/model.rs:335` and
  `src/gpu/transformer.rs:992` — compile unchanged.
- The function name stays the same; the body computes focal CE rather
  than plain CE. Slightly misleading naming but acceptable for a
  screening (matches the WSD `cosine_warmup→WSD` precedent).
- `tests/transformer_smoke.rs` asserts `last_loss < first_loss · 0.7`
  on a 200-step copy task. Focal loss has the same shape early on
  (`s ≈ 1` while p_t small) and *decreases more aggressively* as p_t
  grows, so the test should pass with margin.
- No test reads exact loss values; only relative drops.
