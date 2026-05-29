# combo-wsd-embed-both-half

**Area**: composite (optimizer × embedding)

**Motivation**: Bank's composability map so far:
- `combo-fixup-wsd` (iter 10): init × optimizer = super-additive (+0.008)
- `combo-embed-both-half` (iter 20): embedding × embedding = super-additive (+0.009)
- `combo-fixup-wsd-tok-half` (iter 19): init × init = ANTI-stacks (−0.234)

Two areas where wins **do compose**: init+schedule, embedding+embedding.
One where they **don't**: when fixup zeros residuals, additional
embedding-magnitude reductions fight for the same broken-channel
gradient pathway.

This iteration tests the cleanest remaining orthogonal composition:
**optimizer (WSD) × embedding (both halves)** — schedule is purely
a per-step LR multiplier, and embedding init is per-tensor scaling
at step 0. Zero conceptual overlap. If the composability map's
"orthogonal areas stack" pattern holds, predicted Δ ≈ −0.102 (additive)
or up to −0.11 with super-additive boost.

If this loses or stacks weakly, we learn that "orthogonal areas
stack" is too coarse — WSD's effect interacts with init magnitude
somehow (perhaps the stable-LR phase amplifies whatever advantage
small embeddings give, double-counting the gain).

**Mechanism**: Three changes:
1. `src/transformer/embedding.rs::EmbedWeights::new` — `tok_scale = scale * 0.5`, `pos_scale = scale * 0.5`.
2. `src/transformer/schedule.rs::cosine_warmup` body — WSD (warmup → stable base_lr → linear decay over tail 20%).

(2 files, 1 src/ tree.)

**Expected Δ BPB**: -0.08 to -0.12. Predicted additive: -0.102. Mild
super-additivity (+0.01) is the prior based on iters 10 & 20.

**Why unexplored**: Bank has 20 entries; the WSD × embedding composite
hasn't been tested. Each component validated standalone. Sanity gates
should pass cleanly (both standalone passed all gates; the failure
modes that hit broken-impl in prior 3-ingredient combos all involved
fixup or extreme pos reductions).

## Math gate

- All three components are mathematically independent (different
  parameters / different stages of training).
- No new non-differentiability, shape changes, or causality issues.
- Param count delta: 0. VRAM delta: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (each component verified).
- `transformer_smoke` (all 3 sub-tests): PASS expected. Both
  standalone passed; combo has no zero-init residuals (no fixup),
  so warm-up speed isn't bottlenecked.
- `stability::cosine_warmup_monotone_then_decays`: PASS (WSD shape
  satisfies its assertions; verified iter 2).
- `gpu_parity`: PASS.
