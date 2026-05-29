# ffn-fc-half-std

**Area**: init

**Motivation**: The new post-merge baseline (main @ 651c643) bakes
fixup-mha-only in: `w_o = 0` so the MHA sublayer is identity at init,
and `w_proj` in FFN already uses GPT-3 residual scaling `1/sqrt(2N)`.
The FFN's INPUT projection `w_fc` (the d_model → d_ff expansion) is the
last unperturbed init in the residual branches — it still uses plain
Xavier-uniform.

Hypothesis: halving the init magnitude of `w_fc` reduces the initial
magnitude of pre-GELU activations by 2×. Because GELU is near-linear
around zero, the FFN sublayer output becomes ~4× smaller at init
(0.5 from w_fc × 0.5-ish from GELU's flatter region near zero × baseline
w_proj scale). This further softens the FFN branch at init, in the
same spirit as fixup-mha-only softened the MHA branch.

Empirically, the closest prior data point is **embed-tok-half-std**
(iter 18, won at −0.069 on the OLD baseline). It halved the init of
an upstream embedding; this halves the init of an intermediate FFN
projection. Different mechanism (no signal-strength-at-LM-head
attenuation, which was the iter-19 anti-stacking failure mode), but
same "softer init" family.

**Mechanism**: In `src/transformer/ffn.rs::FfnWeights::new`, replace
`LinearProj::new(d_model, d_ff, rng)` with
`LinearProj::new_scaled(d_model, d_ff, 0.5, rng)`. One line. GPU
inherits via `from_cpu`.

**Expected Δ BPB**: −0.01 to −0.04 vs the new baseline (2.386948).
The new baseline is already strong, so headroom is smaller. A loss
(no-effect to +0.01) is also plausible if w_fc's init magnitude is
already in the "good" range and AdamW absorbs the change within the
1-epoch budget.

**Why unexplored**: Bank has 28 entries plus a multi-seed validation.
The FFN sublayer has been touched once (fixup-ffn-only iter 24,
zeroing w_proj, loss). No iteration has tried softening (not zeroing)
the FFN INPUT projection. Distinct from embedding-magnitude halves
(those happen upstream, not in a sublayer branch).

## Math gate

- **Modified equation(s)**: `w_fc ~ U(−b/2, b/2)` where
  `b = sqrt(6 / (d_model + d_ff))` is the Xavier bound. Forward pass
  unchanged: `h_pre = w_fc · x`, `h_post = GELU(h_pre)`,
  `y = w_proj · h_post`. At init `h_pre` magnitude scales linearly with
  w_fc magnitude → halved.
- **Backward pass**: Standard chain rule. ∂L/∂w_fc = ∂L/∂h_pre · xᵀ.
  Reaches w_fc via the same path; init magnitude does not affect
  reachability or smoothness. No new non-differentiability.
- **Dim compatibility**: No tensor shape changes. h_pre, h_post still
  `[T, d_ff] = [256, 1536]` per sequence at the default batch.
- **Causality preserved**: FFN is position-wise; init magnitude doesn't
  introduce cross-position dependencies.
- **Param count delta**: 0. Same number of trainable scalars in w_fc.
- **VRAM delta**: 0. No new tensors or caches.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Numerical gradient checks use a
  small dim and don't depend on absolute init magnitude. The check
  asserts `∂L/∂w ≈ finite-diff(L, w)` — halving w doesn't break this.
- `transformer_smoke`: PASS. Copy task. The model still has full
  expressive capacity; halved w_fc just means FFN starts with smaller
  outputs and ramps up via AdamW. The smoke window was bumped to 600
  steps (per meta-smoke-window-bump entry), and earlier softer-init
  experiments (fixup-mha-only, pos-half) all passed.
- `gpu_parity`: PASS. The change is in `FfnWeights::new` only; both
  CPU and GPU consume the same constructed weights (CPU model is built,
  then `from_cpu` mirrors to GPU buffers).
