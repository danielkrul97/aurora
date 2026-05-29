# adam-beta2-098

**Area**: optimizer

**Motivation**: Three optimizer-area attempts have failed on the new
baseline (adam-eps-1e-6, ln-eps-1e-3, wd-attention-off — all losses).
But β2 has never been varied. The current value 0.95 is GPT-3's
byte-LM-tuned choice; standard AdamW uses 0.999. β2 = 0.98 is a
middle-ground used by Llama and PaLM that lets the second-moment
estimate retain more history without freezing it (effective averaging
window ≈ 50 steps at 0.98 vs ≈ 20 at 0.95).

Hypothesis: at 641 total steps with batch=16, gradient estimates are
noisy. A smoother m2 estimate could damp the per-step variance in the
effective update rule (m1_hat / sqrt(m2_hat + eps)), giving more
consistent step magnitudes — especially in the post-warmup region
where LR is decaying.

**Mechanism**: In `src/optimizer.rs::Default for AdamWConfig`, change
`beta2: 0.95` to `beta2: 0.98`. The constructor in
`examples/aurora_train.rs::TrainContext` uses `..AdamWConfig::default()`,
so this flows through to both CPU and GPU AdamW (GPU reads
`config.beta2` from the same struct). 1 LoC change.

**Expected Δ BPB**: −0.01 to +0.01. Modest expected effect; AdamW
config tweaks have been mostly neutral on this baseline. The most
likely outcome is "no win" with delta within noise. Negative tail
(loss): if 0.98 is too sluggish to track the rapid early-training
gradient shifts, the model trains less effectively in warmup.

**Why unexplored**: Bank has 33 entries. Optimizer area: 3 losses
(eps perturbations, WD targeting). β2 is the only major AdamW
hyperparameter not yet tested. Spec area suggestion
"second-moment alternatives" covers this.

## Math gate

- **Modified equation(s)**: `m2 = 0.98·m2 + 0.02·g²` (was
  `0.95·m2 + 0.05·g²`). Update rule otherwise identical:
  `θ -= lr·(m1_hat/(sqrt(m2_hat)+eps) + λ·θ·is_wd)`.
- **Backward pass**: AdamW is a post-gradient operation; no change
  to autograd graph.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes (optimizer step has no temporal coupling).
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Bias correction**: bc2 = 1 - β2^t. At t=1, bc2 = 0.02 (was 0.05).
  At t=100, bc2 ≈ 0.867 (was 0.994). Bias correction divides m2 by
  bc2, so m2_hat starts much larger when bc2 is small — but this is
  cancelled by m2's smaller magnitude (only 0.02·g² accumulated at
  step 1). Net effective magnitude at early steps is the same.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks don't invoke
  AdamW step.
- `transformer_smoke`: PASS. Smoke test uses AdamW for 600 steps;
  β2=0.98 just smooths m2 more — gradient direction unchanged.
  Convergence test should pass easily.
- `gpu_parity`: PASS. CPU and GPU read the same `config.beta2` via
  push constants; no shader code path divergence.
