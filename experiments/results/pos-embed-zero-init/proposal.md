# pos-embed-zero-init

**Area**: embedding (mechanically), init (philosophically)

**Motivation**: Aurora's position embeddings `w_pos` are initialized with
the same `uniform[-0.02, 0.02] · √3` distribution as token embeddings.
At step 0 every position contributes random noise to the residual stream
that the model must learn to either suppress or interpret. But for a
**causal** decoder, sequential order is already encoded structurally
via the causal attention mask — positions are distinguishable *before*
position embeddings come into play. The position embedding is only
needed to break the bag-of-words symmetry *within* a causal window
(which positions can attend to which past tokens), which the model can
learn from data.

This parallels the fixup-resid-zero-init win (iter 4, Δ BPB = −0.25):
if a parameter's "useful initial state" is *cleaner than its random
init*, zero-init plus learn-from-data is a free upgrade. For position
embeddings specifically, the useful prior is "tokens at different
positions should produce different attention patterns", which the model
can find via gradient signal.

**Mechanism**: In `EmbedWeights::new`, replace
`w_pos = uniform[-0.02·√3, 0.02·√3]^{seq_len_max × d_model}` with
`w_pos = 0^{seq_len_max × d_model}`. At step 0, `embed_forward` returns
`w_tok[token_id] + 0 = w_tok[token_id]`. The transformer (with
fixup-style or default residual init) processes a *purely token-identity*
input stream initially; positional bias appears as `w_pos` learns
from `embed_backward`'s scatter-add gradient.

**Expected Δ BPB**: -0.005 to -0.040. Smaller than fixup's because the
information content of zero-init pos embeddings (zero) is closer to
what the model needs (low-magnitude position biases) than zero-init
residual branches were to their useful state (small but nonzero
contributions). Direction-wise positive, magnitude uncertain.

**Why unexplored**: Bank has 5 entries spanning loss-aux, optimizer,
embedding, init. embed-no-wd was the only direct touch of `w_pos`-adjacent
behavior, and it was a loss. Zero-init of *learned* position embeddings
is rare in literature — most papers either use *sinusoidal* positions
(which are fixed, equivalent to zero-trainable-init at first glance but
different math) or random-init learned positions. Aurora uses learned
positions with conventional init; switching to zero-init is a small
but meaningful experiment.

## Math gate

- **Modified equation(s)**:
  - Step 0: `embed_forward(tokens) = w_tok[tokens] + 0 = w_tok[tokens]`.
    All other forward computation unchanged.
  - Step k > 0: `w_pos` has accumulated gradient via standard
    `embed_backward` scatter-add. Forward becomes
    `w_tok[tokens] + w_pos[positions]` with `w_pos` learned.
- **Backward pass**: `embed_backward` already computes
  `∂L/∂w_pos[p] = sum_{t : t_pos = p} ∂L/∂h_t`. With `w_pos = 0`, this
  gradient is non-zero (it depends on `∂L/∂h_t`, not on `w_pos`'s
  current value). So AdamW moves `w_pos` away from zero starting at
  step 1. No new non-differentiability.
- **Dim compatibility**: `w_pos` is still
  `[seq_len_max × d_model]`. Same shape, different initial value.
- **Causality preserved**: Position embedding addition is per-token
  and does not interact with the causal attention mask, which is
  applied later. Causality is structural, not learned.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## CPU/GPU parity

`DecoderTransformerGpu::from_cpu()` copies CPU's `w_pos` to GPU
(`src/gpu/transformer.rs:613`). Changing CPU init propagates. Parity
test should pass: it runs CPU + GPU on identical (zero-initialized)
weights for `w_pos` and confirms forward / backward / step drift stays
under 5e-2.
