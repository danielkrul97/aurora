# tok-embed-zero-init

**Area**: embedding

**Motivation**: `pos-embed-zero-init` (iter 6 era) won −0.105 on the
OLD baseline by zeroing positional embedding at init. Its sibling
"zero the OTHER embedding" — `tok-embed-zero-init` — has never been
tested. The bank has 7 embedding entries; none zero w_tok.

Hypothesis: under the same "head-start hurts" intuition that made
pos-embed-zero win — random spurious init signals are worse than
gradient-driven discovery — zeroing w_tok forces all tokens to start
identical, with only positional context distinguishing them. The
model has to learn token identity from gradient via the LM head
pathway (which is tied to w_tok).

At init:
- `embed_out[t] = w_tok[tok] + w_pos[t] = 0 + w_pos[t]` → positional only
- `logits = h_final · w_tok = h_final · 0 = 0` → uniform softmax
- All tokens look identical to the model

Gradient flow at step 0:
- `d_logits = softmax(0) - target_onehot = 1/V - δ_target`
- `d_w_tok = h_final · d_logits` → non-zero (h_final has positional
  signal flowing through)
- So w_tok unblocks immediately from step 1.

This is **structurally different from pos-embed-zero**: 
- pos-embed-zero: positional structure must be learned from scratch
- tok-embed-zero: token-identity structure must be learned from scratch

For byte-LM with vocab=4096, token identity is more critical than
position (a "specific byte" carries more info than "byte at position
t"). So this could either:
- Win (head-start spurious-init pattern applies broader than position)
- Lose (token info is too critical to leave zero at init)

**Mechanism**: In `src/transformer/embedding.rs::EmbedWeights::new`,
replace the random init for `w_tok` with zeros. ~3 LoC. GPU inherits
via from_cpu.

**Expected Δ BPB**: −0.05 to +0.05. High variance prediction. The
asymmetry between pos and tok is real, and we don't know which way it
cuts on byte-LM.

**Why unexplored**: Bank has 42 entries across 8 areas. 7 embedding
entries — magnitudes (zero pos, half pos, half tok, no-WD) and pattern
(sinusoidal scaled/raw). None zero w_tok. The dual of pos-embed-zero
that should have been a natural day-1 idea but never made it in.

## Math gate

- **Modified equation(s)**: At init, `w_tok = 0`. Forward unchanged:
  `out = w_tok[tok] + w_pos[t]`, `logits = h_final · w_tok`. Both
  evaluate to `0 + w_pos[t]` and `0` at init.
- **Backward pass**: All gradient paths preserved.
  `d_w_tok` accumulates from BOTH the embed pathway (scatter-add from
  `d_h`) AND the LM head pathway (`h_final · d_logits`). Both
  non-zero at step 0 because h is non-zero (via w_pos).
- **Dim compatibility**: 0 changes.
- **Causality preserved**: w_tok = 0 doesn't introduce position
  leakage.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks involve small d_model
  but the gradient computations are well-defined; w_tok=0 at the start
  of a finite-diff step doesn't break the check.
- `transformer_smoke`: PASS. The argmax sanity test (100 steps at
  lr=3e-2) should easily learn the identity prediction — gradient
  signal flows directly through the LM head to update w_tok. The
  convergence test (600 steps, 30% drop) also expected to pass —
  initial loss is exactly log(vocab) = log(8) ≈ 2.08 for the smoke
  test's tiny vocab, with rapid drop expected. Caveat: if the smoke
  vocab is very small and h_final is also small, the gradient updates
  might be slow — but lr=3e-2 is very aggressive and should compensate.
- `gpu_parity`: PASS. CPU-init w_tok = 0 flows to GPU buffer
  byte-for-byte.
