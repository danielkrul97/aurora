# pos-embed-sinusoidal-init

**Area**: embedding

**Motivation**: Bank embedding area has 4 entries — three magnitude
perturbations (pos-embed-zero, pos-embed-half, embed-tok-half — all wins
on old baseline) and one WD-off (loss). All these modulate the MAGNITUDE
of embedding rows. Sinusoidal-init is structurally different: it varies
the SHAPE / pattern of pos embeddings while keeping magnitude unchanged.

Hypothesis: random uniform w_pos init means the model must discover
positional structure from gradient signal alone over ~641 steps. The
classical Transformer (Vaswani 2017) sinusoidal pattern encodes position
with rotation-frequency structure — neighboring positions are similar,
distant positions are dissimilar in a smooth, deterministic way. Even
if the model gradient-updates w_pos away from sinusoidal during training,
the head-start advantage might matter in an undertrained 1-epoch regime.

For byte-level enwik8 where order is paramount (causal LM depends entirely
on sequence), a structured positional prior could specifically help.

**Mechanism**: In `src/transformer/embedding.rs::EmbedWeights::new`,
replace the uniform random init of `w_pos` with the sinusoidal formula:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

`w_tok` initialization unchanged. ~12 LoC of pure CPU init code. GPU
inherits via existing `from_cpu` upload path — no shader changes.

**Expected Δ BPB**: −0.005 to −0.030. Optimistic but feasible: this is
genuinely a new mechanism axis. Pessimistic outcome: after 641 steps
the position-info is learned anyway, init advantage erased → no effect
or tiny loss. Worst case if AdamW pushes w_pos away from sinusoidal
in a destabilizing way (unlikely; init magnitudes are unit-scale).

**Why unexplored**: Bank has 35 entries. Embedding area touches were all
magnitude (zero, half, no-WD) — none ever changed the PATTERN of init.
Sinusoidal PE is well-known historically but rarely seen in modern
codebases (most use learned random pos embed or RoPE). For Aurora's
learned pos embed, sinusoidal init is a clean, novel starting point.

## Math gate

- **Modified equation(s)**:
  - At init: `w_pos[pos*d + 2i] = sin(pos / 10000^(2i/d))`
  - At init: `w_pos[pos*d + 2i+1] = cos(pos / 10000^(2i/d))`
  - Forward unchanged: `emb[t,c] = w_tok[tok][c] + w_pos[t][c]`
  - Backward unchanged: `grad.w_pos[t][c] += dy[t][c]`
- **Backward pass**: Identical. Init values don't affect gradient
  reachability or smoothness.
- **Dim compatibility**: No tensor shape changes; `w_pos` is still
  `[seq_len_max * d_model] = [256 * 384]`.
- **Causality preserved**: w_pos is summed into the embedding at each
  position based on the token's absolute time index — already causal.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Magnitude check**: Sin/cos init values are bounded in [-1, 1].
  Old random init was `U(-0.02√3, 0.02√3) ≈ U(-0.035, 0.035)`. So
  sinusoidal init is ~28× larger in magnitude. This is INTENTIONAL —
  it matches the d_model-scale used in the original Transformer (which
  also scales token embedding by sqrt(d_model)). Aurora doesn't scale
  tok embed, so the imbalance pos/tok is larger than the paper's
  ratio. But the LN inside each block normalizes back to unit scale,
  so downstream stays well-behaved.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Gradient checks use a tiny d_model;
  the init values are different but the gradient computations are the
  same operations.
- `transformer_smoke`: PASS. Copy task still solvable; model still has
  full expressiveness. Different init = different but reasonable
  starting state.
- `gpu_parity`: PASS. GPU `from_cpu` copies the CPU-initialized
  `w_pos` buffer byte-for-byte. No shader divergence.

## Init values sanity check (concrete)

For pos=0: PE[0, 2i] = sin(0) = 0; PE[0, 2i+1] = cos(0) = 1.
  → row 0 alternates [0, 1, 0, 1, ...].
For pos=1, low freq (2i=0): PE[1,0] = sin(1) ≈ 0.841, PE[1,1] = cos(1) ≈ 0.540.
For pos=1, high freq (2i=d-2=382): PE[1,382] = sin(1/10000^(382/384)) ≈ sin(1.16e-4) ≈ 1.16e-4.
  → high-freq dims barely change between pos=0 and pos=1.
For pos=255 (max), low freq: PE[255,0] = sin(255) ≈ wraps to ~-0.176.
  → covers low-freq dims through many cycles, high-freq through a few.

So the pattern has the expected "different frequencies attend to different
distance scales" property.
