# pos-embed-half-std

**Area**: embedding

**Motivation**: Iter 6 (`pos-embed-zero-init`) won with Δ −0.040 by
setting w_pos init to all zeros. The win was attributed to "cleaner
initial signal" — at step 0 the residual stream carries token identity
only, with position info learned from gradient. But that interpretation
is ambiguous: is the gain *linear* in init magnitude (suggesting any
reduction helps proportionally), or *step-function* at exactly zero
(suggesting only complete elimination of random pos noise matters)?

This iteration is a clean differential test: w_pos init at ½ baseline
magnitude. Expected outcomes:
- **Linear**: Δ ≈ −0.020 (half the iter-6 gain).
- **Step-function at zero**: Δ ≈ 0 (similar to baseline; no benefit
  unless fully zero).
- **Sweet-spot middle**: Δ between −0.04 and −0.05 (some intermediate
  init magnitude is optimal).

The test also generates ammunition for combo refinements: if
half-magnitude has near-zero standalone effect, it's not a useful
composite ingredient. If it captures most of the win, it could work
in a `combo-fixup-wsd-pos-half` (different from the iter-13 broken
quarter variant — half-magnitude restores more pos info for warm-up).

**Mechanism**: In `EmbedWeights::new`, multiply the w_pos init `scale`
by 0.5: `pos_scale = scale * 0.5`. w_tok init unchanged. Backward/
forward computation unchanged; only initial values differ.

**Expected Δ BPB**: -0.005 to -0.030. Most likely linear → ~−0.020,
based on the prior of "zero-init effects scale with init magnitude
reduction".

**Why unexplored**: Bank has pos-embed-zero (extreme: 0×) but no
intermediate point. This is a 1-LoC datapoint that disambiguates the
mechanism.

## Math gate

- **Modified equation(s)**: `w_pos[i, j] ∼ uniform[−s·0.5, +s·0.5]`
  where `s = 0.02·√3 ≈ 0.0346`. New uniform range: [−0.0173, +0.0173].
- **Backward pass**: `embed_backward` scatter-adds dh into w_pos rows.
  Gradient is independent of w_pos's current value, so embedding moves
  from its smaller init in the same direction as baseline, just from a
  different starting point. No new non-differentiability.
- **Dim compatibility**: w_pos shape unchanged.
- **Causality preserved**: Position embeddings don't interact with
  causal mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

CPU `EmbedWeights::new` constructs w_pos values; GPU loads via
`from_cpu`. Both paths see identical half-magnitude pos init. Parity
tests pass (same math, just smaller initial values).
