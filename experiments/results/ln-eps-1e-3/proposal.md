# ln-eps-1e-3

**Area**: norm-placement

**Motivation**: Aurora's LayerNorm uses `eps = 1e-5` (the same value
nanoGPT, PyTorch, and most LLM codebases use). The role of `eps` is
to floor the variance denominator `1/sqrt(var + eps)`. For typical
post-LN hidden states with variance ~1, eps is negligible and serves
only to prevent division-by-zero in edge cases. But for *very low
variance* dimensions (which can happen during early training, when a
specific channel's values collapse to near-constant), a small eps
produces a huge `1/sqrt(var)` factor, amplifying the channel's
contribution and the corresponding gradient.

A larger `eps = 1e-3` clamps this amplification: when a channel's
variance is small, the normalization is gentler. The forward output's
magnitude is closer to the raw input, and gradients through that
channel are tamer. This is a *mild* regularizer on the normalization
operation itself — it discourages the model from relying on any
particular channel being near-constant.

**Mechanism**: Change `eps` from `1e-5` to `1e-3` in:
- `src/attention.rs::LayerNorm::new` (CPU constant).
- `src/gpu/transformer.rs` — all 8 dispatch sites for `ln_fwd`
  (3 in the main training path, 3 in `prefill_kv`, 2 in
  generate-loop forward).

LN backward kernel does **not** take eps — it reuses the
pre-computed `inv_std` from forward (which has the new eps baked in),
so backward needs no edit.

**Expected Δ BPB**: -0.005 to -0.02. Small expected gain because the
"low-variance channel" pathology is rare at well-behaved init (Aurora
uses GPT-3 res-scaling, no exotic activations). Direction-wise
positive because the looser eps is documented as helpful in some
training stability papers; downside is bounded because eps is only
additive in the denominator.

**Why unexplored**: Bank has 11 entries, no `norm-placement` entries
yet. Eps tuning is a knob, but it's a structural knob on the
normalization operation — different mechanism from init / schedule /
loss changes.

## Math gate

- **Modified equation(s)**: LayerNorm forward becomes
  `y = γ · (x − μ) / √(var + 1e-3) + β`. Backward inherits eps via
  the stored `inv_std = 1/√(var + 1e-3)`. No structural change.
- **Backward pass**: Same chain rule as baseline; `inv_std` is the
  only place eps enters, and it's already computed in forward and
  reused. Continues to be finite and non-zero for any input.
- **Dim compatibility**: No tensor shapes change.
- **Causality preserved**: LN is per-token; no cross-position mixing.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

The CPU `LayerNorm::new` sets `eps` once per LN instance, used in
`layer_norm_forward` and `_backward`. The GPU receives eps as a
push-constant per dispatch (8 sites, all currently hardcoded `1e-5`).
Changing all 9 in lockstep keeps CPU and GPU consistent.
`tests/gpu_parity.rs::layer_norm_forward_backward_matches_cpu` should
pass: both paths use the same eps and the LN math is the same.
