# fixup-w-k-zero

**Area**: init

**Motivation**: The merged baseline already has `w_o = 0` (fixup-mha-only,
the dominant single-iter win, −0.335 in multi-seed validation). The
question is whether ADDING a second zero-init within the MHA — on `w_k`
this time — provides further benefit, anti-stacks, or breaks.

Why w_k specifically (not w_v or w_q)?
- **w_v = 0 + w_o = 0** is a complete deadlock: `mha_out = probs·V·w_o
  = probs·0·0 = 0`, and the backward gradients to both w_v and w_o
  vanish symmetrically. Rejected by math gate.
- **w_q = 0** has symmetric properties to w_k = 0 (both produce
  scores=0). Picking w_k arbitrarily; either choice produces uniform
  attention at init.
- **w_k = 0** makes scores=Q·K^T=Q·0=0 → `softmax(0) = uniform` →
  attention is uniform at init. Combined with w_o=0 in the baseline:
  mha_out = uniform·V·0 = 0 (residual passes through unchanged at
  step 0).

Gradient flow at step 1:
- d_w_o = heads_out^T · d_mha_out where heads_out = uniform·V ≠ 0 →
  d_w_o ≠ 0. w_o starts moving away from zero.
- d_w_v = ln1_out^T · d_V where d_V = probs^T · d_heads_out = uniform^T
  · (d_mha_out · w_o^T) = uniform^T · 0 = 0 at step 1.
- d_scores: d_probs = d_heads_out · V^T = 0 · V^T = 0 → d_scores = 0
  → d_Q, d_K = 0 at step 1.
- At step 2, w_o ≠ 0, so the cascade unblocks: d_V ≠ 0, d_scores ≠ 0,
  all params start learning.

So there's a one-step "warmup" where only w_o updates; from step 2
onward, everything trains. The benefit (if any): w_o specializes
slightly before the attention pattern (Q·K) commits to a structure.
Intuitively, that's the "right" order — first decide HOW heads
combine, then decide WHAT they attend to.

**Mechanism**: In `src/transformer/attention.rs::MhaWeights::new`,
replace `w_k: LinearProj::new(d_model, d_model, rng)` with
`w_k: LinearProj::new_scaled(d_model, d_model, 0.0, rng)`. 1 LoC.
GPU inherits via from_cpu.

**Expected Δ BPB**: −0.005 to −0.025. The mechanism is principled
(staged specialization) but the size of the effect is uncertain
in a 1-epoch byte-LM regime. Anti-stack risk: w_o=0 and w_k=0
might over-soften the MHA branch in a way that anti-stacks (the
"signal-strength floor" pattern from iters 19, 25, 28). But w_k=0
is structurally different from "magnitude reduction" — attention
becomes UNIFORM, not silent.

**Why unexplored**: Bank has 40 entries. Init area: 6 prior
(zero-init residual, fixup-ffn-only, fixup-mha-only, fixup-resid,
he-uniform, ffn-fc-half). Q/K-side init has never been touched —
zero-init on the attention-score-producing branch is genuinely
novel relative to the bank.

## Math gate

- **Modified equation(s)**: At init, `w_k = 0` → `K = X · 0 = 0` →
  `scores = Q·K^T / √(d_head) = 0` → `softmax(0) = uniform(1/T)` →
  `heads_out = uniform · V` (= row-wise average of V). Combined with
  baseline w_o=0: `mha_out = heads_out · w_o = uniform·V · 0 = 0`.
  Residual: `h_new = h_in + 0 = h_in` at init.
- **Backward pass**: All gradient paths exist via the chain analyzed
  above. d_w_q, d_w_k, d_w_v have zero gradient at step 1 but non-zero
  from step 2 onward (one-step deadlock that breaks because w_o
  updates first).
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Uniform attention at init still respects
  the causal mask — softmax of -inf (masked positions) is 0, so
  "uniform" is uniform over allowed (≤ t) positions only.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks are at a small
  scale; w_k=0 produces uniform attention which doesn't break finite-
  difference verification.
- `transformer_smoke`: PASS. The 600-step convergence test should hit
  the 30% threshold even with one-step w_k deadlock. The argmax test
  uses 100 steps which is plenty to recover. (fixup-mha-only passed
  smoke under the same constraints.)
- `gpu_parity`: PASS. GPU uses CPU-init weights via from_cpu; uniform
  attention is deterministic.
