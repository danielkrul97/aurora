# norm-final-half-gamma

**Area**: norm-placement

**Motivation**: `norm-placement` is the only spec-suggested area with
ZERO bank entries. Most ideas in this area (peri-norm, sandwich-norm,
RMSNorm replacement, DeepNorm) require either new params or GPU
shader changes — both expensive. This iter probes the cheapest
norm-placement angle: tweak the INIT of the existing final LayerNorm's
γ scale.

Standard LN init: γ=1.0, β=0.0. Output `y = γ * (x - μ)/σ + β` produces
unit-magnitude features. Setting γ=0.5 halves the h_final magnitude at
init, which halves the initial logit magnitudes (via tied LM head).
Effect: slightly softer initial softmax → broader distribution over
vocab → smaller per-step gradient signal in the very early steps.

This is mechanistically distinct from `logit-scale-2x` (loss, sharpens
softmax via fixed scaling of logits): here γ is LEARNABLE — AdamW
adjusts it during training. By iter 100, γ_final might already be
back near 1.0 or beyond. The init merely shapes the trajectory through
logit space, similar to how fixup-mha-only shapes attention init.

**Mechanism**: In `src/transformer/model.rs::DecoderTransformer::new`,
after constructing `final_ln = LayerNorm::new(d)`, override
`final_ln.gamma.fill(0.5)`. 1 LoC override. CPU+GPU share the same
weights via from_cpu.

**Expected Δ BPB**: −0.005 to +0.005. Subtle warmup-like effect. The
modal outcome is "no significant change" because γ recovers within
~20 steps. Optimistic: smoother early trajectory through logit space
gives marginal benefit (similar to how the warmup-LR has a
trajectory-shaping role).

**Why unexplored**: Bank has 39 entries across 8 areas; norm-placement
has 0. This is the first probe of that axis with a minimum-viable
2-LoC change.

## Math gate

- **Modified equation(s)**: At init, `final_ln.γ = 0.5` (vector of d
  elements). Forward unchanged: `y = γ * (x - μ)/σ`. Backward unchanged:
  `∂L/∂γ_c = sum_t (∂L/∂y[t,c]) * (x[t,c]-μ[t])/σ[t]`.
- **Backward pass**: All gradient paths preserved. γ is updated by
  AdamW normally; no apply_wd issue (gamma's apply_wd is false already).
- **Dim compatibility**: 0 changes.
- **Causality preserved**: LN is per-position; γ doesn't introduce
  cross-position effects.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS. Different γ init doesn't affect
  gradient math.
- `transformer_smoke`: PASS. The smoke test uses a tiny model where
  the LM head pathway is fully exercised. Halving γ at init means
  logits start at half magnitude → softer softmax → marginally
  smaller initial CE gradient. The model still converges; γ
  recovers via AdamW within the 600-step window.
- `gpu_parity`: PASS. GPU mirrors CPU weights via from_cpu; γ values
  flow over byte-for-byte.
