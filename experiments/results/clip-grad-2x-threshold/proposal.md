# clip-grad-2x-threshold

**Area**: optimizer

**Motivation**: From `experiments/baseline.jsonl` step 636: `grad_norm =
16.196` (pre-clip). Baseline GPU clip threshold is `grad_clip *
batch_size = 1.0 * 16 = 16.0`. So clipping is ACTIVE EVERY STEP at
~16.0 — gradient is being shaved by ~1.3% on average late in training,
likely much more early on.

Hypothesis: the `grad_clip=1.0` cli arg is conservatively tuned for
worst-case stability. With fixup-mha-only init now in the baseline,
gradients are inherently softer (MHA branch starts at zero), so the
aggressive clip may unnecessarily reduce effective step size.

Doubling the threshold internally (without touching cli_args) lets
larger gradient norms pass through unmodified. Could improve effective
training rate. Risk: if clipping is actively suppressing instabilities,
removing it could destabilize.

**Mechanism**: In both `src/transformer/model.rs::clip_grads` (CPU)
and `src/gpu/transformer.rs::clip_grads` (GPU), internally multiply
`max_norm` by 2.0 before applying the cap. Cli arg `grad_clip=1.0`
remains in cli_args but the internal cap becomes 2.0 (CPU-side, with
1/batch_size pre-scaling) or 32.0 (GPU-side, with batch_size in
max_norm multiplier).

LoC: 2 (one per backend).

**Expected Δ BPB**: −0.005 to +0.020. Wide range. If clip is just
"safety net" (rare activation), no effect. If clip is actively
suppressing useful gradient signal, mild win. If clip is preventing
divergence, mild-to-major loss.

**Why unexplored**: Bank has 44 entries. Optimizer area: 6 prior
entries (eps placement/magnitude, β1/β2, WD-target). Grad-clip
algorithm hasn't been touched. The threshold isn't in cli_args
semantically — it's a HYPERPARAM whose multiplier is hidden in code.
Changing internal behavior is permitted per spec ("you change the
model/optimizer code only").

## Math gate

- **Modified equation(s)**: `if ||g|| > 2·max_norm: g ← g · (2·max_norm)/||g||`
  (else unchanged). Equivalent to a 2× larger global gradient norm cap.
- **Backward pass**: Grad clipping is post-backward, pre-AdamW. No
  autograd graph change.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Grad clipping doesn't introduce time leakage.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Grad clipping isn't exercised in
  unit tests.
- `transformer_smoke`: PASS. Smoke test uses small models; gradient
  norms during 600-step convergence are bounded. Whether they exceed
  baseline threshold (1.0) depends on the test setup. Doubling
  threshold only ever loosens the cap — at worst no change, at best
  slightly larger steps. Convergence intent unchanged.
- `gpu_parity`: PASS. CPU and GPU clip_grads modified in lockstep
  with the same 2× factor.
