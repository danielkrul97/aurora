# wsd-tail20-linear

**Area**: optimizer (LR schedule)

**Motivation**: Aurora's baseline uses cosine decay from peak 6e-4 to
min 6e-5 over the entire post-warmup region (441 of 641 steps). Cosine
spends roughly half its budget at LR < base_lr/2, which is well-known
to be over-conservative for short, under-converged trainings. WSD
(warmup-stable-decay) — also called the "minimal recipe" by
MiniCPM/StableLM — holds at peak LR for most of training and only
decays in the final tail. The average LR is substantially higher,
giving more effective optimization signal per step. At 12M params /
1 epoch / under-trained regime, more learning per step is the bottleneck.

**Mechanism**: Replace cosine decay with linear decay over only the
final 20% of post-warmup steps:
- step < warmup_steps: linear warmup from 0 → base_lr (unchanged)
- warmup_steps ≤ step < decay_start: constant base_lr (NEW stable phase)
- decay_start ≤ step < max_steps: linear from base_lr → min_lr (NEW)
- step ≥ max_steps: min_lr (unchanged)
- decay_start = warmup_steps + ⌈0.8 · (max_steps − warmup_steps)⌉

For the baseline config (warmup=200, max=641), this gives:
- 0–199: warmup (linear 0 → 6e-4)
- 200–552: stable (6e-4 for 353 steps)
- 553–640: linear decay (6e-4 → 6e-5 over 88 steps)

**Expected Δ BPB**: -0.02 to -0.08. WSD typically wins by margins
matching ~1 extra epoch's worth of cosine training in 1-epoch budgets.
Aurora is far from saturation (BPB 2.72 vs goal 1.5), so any extra
effective optimization should land in val_bpb.

**Why unexplored**: Bank has one entry (z-loss-1e4, loss-aux). No prior
optimizer-class entries. WSD is folklore at large pretraining scale but
rarely reported on 1-epoch byte-level enwik8 at 10M params. The
implementation is fully contained in `src/transformer/schedule.rs`
(modify the body of `cosine_warmup`) — no caller changes needed, the
example still calls the function by name.

## Math gate

- **Modified equation(s)**: Schedule is replaced piecewise. Let
  `T_w = warmup_steps`, `T = max_steps`, `T_d = T_w + ⌈0.8·(T − T_w)⌉`.
  ```
  lr(step) = {
    base · (step+1) / T_w,                       0 ≤ step < T_w
    base,                                        T_w ≤ step < T_d
    base + (min − base) · (step − T_d)/(T − T_d), T_d ≤ step < T
    min,                                         step ≥ T
  }
  ```
- **Backward pass**: LR appears in the AdamW update step
  `θ ← θ − lr · m̂ / (√v̂ + ε)`. Backward through *the model* is
  completely unaffected — only the optimizer scales the parameter
  update by lr. No new differentiability concerns.
- **Dim compatibility**: LR is scalar; no tensor shapes change.
- **Causality preserved**: Schedule is per-step and does not interact
  with attention or position-mixing. Causality untouched.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Compliance notes

- The function signature of `cosine_warmup` is preserved
  (`cosine_warmup(step, warmup_steps, max_steps, base_lr, min_lr) → f32`)
  so all callers (example, tests) compile and run unchanged.
- The function's NAME is now slightly misleading (does WSD instead of
  cosine). This is acceptable for a screening experiment; a winning
  result would warrant a clean rename in a real PR.
- `tests/stability.rs::cosine_warmup_monotone_then_decays` asserts:
  (a) strict monotone increasing during warmup ✓ (unchanged),
  (b) plateau at peak at end of warmup ✓ (now holds longer, but the
      test only checks `step = warmup_steps` directly),
  (c) weak monotone decreasing post-warmup (`lr ≤ prev + 1e-6`) ✓
      (stable phase is equal-to-prev, decay phase is strictly less),
  (d) at-or-after `max_steps` returns `min_lr` ✓ (unchanged).
  Test passes without modification.
