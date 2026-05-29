# fixup-mha-only

**Area**: init

**Motivation**: `fixup-resid-zero-init` (iter 4, Δ −0.251) zeroed both
residual-branch projections: `w_o` (MHA output) AND `w_proj` (FFN
output). The combined effect made every sublayer output exactly zero
at init, turning the network into an identity stack. The win was
huge, but the two changes were entangled — we don't know whether MHA
zero-init, FFN zero-init, or specifically their combination drives
the gain.

This iteration ablates the MHA side: zero-init w_o, leave w_proj at
baseline (GPT-3 residual scaling `1/sqrt(2N)`). Diagnostic outcomes:

- **MHA-driven (Δ ≈ −0.20 to −0.25)**: most of fixup's win comes from
  zero-init MHA. FFN's residual-scaling at init is already good
  enough; the dominant noise was MHA.
- **FFN-driven (Δ ≈ −0.02 to −0.05)**: most of the gain was FFN; MHA
  zero-init alone isn't much.
- **Joint (Δ ≈ −0.10 to −0.15)**: roughly half the win — additive
  with FFN side at ~50% each. A future `fixup-ffn-only` would confirm.

The result decides whether to recommend MHA-only fixup as a simpler
variant, or whether fixup needs both changes.

**Mechanism**: In `src/transformer/attention.rs::MhaWeights::new`,
replace `w_o = LinearProj::new_scaled(d, d, resid_scale, rng)` with
`w_o = LinearProj::new_scaled(d, d, 0.0, rng)`. FFN's `w_proj`
construction in `src/transformer/ffn.rs` is unchanged. 2 LoC.

**Expected Δ BPB**: -0.10 to -0.20. Based on prior: MHA's role in
residual stream is "long-range mixing" via attention, which is the
dominant later-layer feature; FFN is "per-position non-linearity",
which is somewhat less critical initially. So MHA-only should
capture most but not all of fixup's gain.

**Why unexplored**: Bank has 21 entries. `init`-area has 2: fixup
(big win) and he-uniform (broken_impl). This is a clean ablation
of the biggest win in the bank.

## Math gate

- **Modified equation(s)**: At init, `w_o = 0`. Therefore MHA's
  output = `softmax(QK^T/√d_k) @ V @ 0 = 0`. Residual stream after
  MHA: `h_in + 0 = h_in` (identity). FFN at init: full
  `w_proj · GELU(w_fc · LN(h)) ~ small random noise` (GPT-3 res scale).
  So sublayer contributions: MHA=0, FFN=non-trivial.
- **Backward pass**: w_o gets gradient via `dh_z @ x_in` (validated
  for fixup iter 4). w_proj gets gradient as in baseline. No new
  non-differentiability.
- **Dim compatibility**: No shape changes.
- **Causality preserved**: Init doesn't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

All gates expected PASS — this is a subset of iter 4's fixup change
which passed all gates. With FFN still contributing at init, the
warm-up speed is faster than full fixup (zero w_o + zero w_proj),
so smoke convergence is easier.
