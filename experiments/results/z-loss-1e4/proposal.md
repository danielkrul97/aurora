# z-loss-1e4

**Area**: loss-aux

**Motivation**: At byte-level enwik8 with vocab=4096 and tied LM head,
the softmax log-partition `z = log Σ exp(logit_j)` can drift to large
absolute values mid-training, which makes attention/embedding gradients
into the LM head unstable and slows convergence in the first epoch. PaLM
and Chinchilla add `λ · z²` as an auxiliary loss to anchor the partition
near zero; it costs ~zero compute because `z` is already materialized
inside softmax. Aurora's 12M-param, 1-epoch screening regime is exactly
where the *initial* loss-landscape conditioning matters most, so a
PaLM-style anchor should give a small but reliable BPB improvement.

**Mechanism**: Replace the per-token CE loss `L_ce = -log p_target` with
`L = L_ce + λ · z²`, where `z = log Σ_j exp(logit_j)` (computed via the
existing log-sum-exp). The gradient w.r.t. `logit_j` gains an extra
`2λ z · softmax_j` term added to the existing `softmax_j - 1{j=target}`.
λ = 1e-4 (PaLM's value). Applied to *training* loss only; the
`evaluate()` path keeps pure cross-entropy so reported `val_bpb` stays
comparable to baseline.

**Expected Δ BPB**: -0.005 to -0.030 (small but positive sign expected;
PaLM reports stability gains, BPB delta at this scale is likely modest).

**Why unexplored**: Idea bank is empty — first iteration of the loop.
Z-loss is folklore at 100B+ scale (PaLM, Chinchilla, Gopher) but rarely
benchmarked on byte-level enwik8 at 10M params, where the regime
(short training, no overfitting, byte-level vocab) differs sharply from
the original setting.

## Math gate

- **Modified equation(s)**:
  - Per-token: `L_t = -log p_{t,y_t} + λ · z_t²`,
    where `z_t = log Σ_j exp(s_{t,j})`, `s_{t,j}` is logit at position t,
    class j, `p_{t,j} = exp(s_{t,j} - z_t)`.
  - Mean batch loss: `L = (1/T) Σ_t L_t`.
- **Backward pass**:
  - `∂L_ce/∂s_{t,j} = p_{t,j} − 1{j=y_t}` (existing).
  - `∂(λ z_t²)/∂s_{t,j} = 2λ z_t · ∂z_t/∂s_{t,j} = 2λ z_t · p_{t,j}`
    (because `∂ log Σ exp(s_j) / ∂s_j = softmax_j`).
  - Total: `d_logits_{t,j} = (p_{t,j} − 1{j=y_t} + 2λ z_t · p_{t,j}) / T`.
  - `p_{t,j}` is finite and non-zero (softmax output), `z_t` is finite
    for finite logits, so `∂L/∂θ` reaches every parameter via the same
    chain as the existing CE path. No new detach/non-differentiability.
- **Dim compatibility**: With B=16, T=256, d=384, h=6, ff=1536, vocab=V:
  - logits: `(B·T, V)` already materialized for CE — no change.
  - `z_t` per-token scalar `(B·T,)` — already computed inside softmax
    (we just hoist it out). No new tensor; reuses existing log-sum-exp.
  - d_logits: `(B·T, V)` — same shape, additive term applied in-place.
  - No new intermediate tensor of significant size.
- **Causality preserved**: Loss is computed pointwise per (t, target_t)
  with no cross-position mixing. Adding a per-token scalar `z_t²` to
  the loss cannot couple position `t'` into position `t < t'`. The
  attention causal mask upstream is untouched.
- **Param count delta**: 0. λ is a fixed constant (no learned scalar).
- **VRAM delta**: 0 meaningful change. `z_t` for the whole batch is
  `B·T · 4 bytes = 16 KB` — negligible vs. activation footprint.
