# depth-fixup-mha-shallow

**Area**: depth-cond-reg

**Motivation**: `fixup-mha-only` (iter 22, Δ −0.332) zeroed `w_o` in
**all 6 blocks**. The gain is large but undifferentiated: we don't
know which blocks contribute most. Two competing intuitions:

- **Shallow blocks dominate**: Early MHA outputs feed into all
  downstream layers; their initial noise propagates and gets
  amplified. Zeroing shallow MHA cleans up the most-amplified noise.
- **Deep blocks dominate**: Late MHA outputs are closer to the LM
  head and directly shape logits. Zeroing deep MHA cleans up the
  gradient signal that matters most.

This iteration tests the **shallow-only** half: zero `w_o` in blocks
0, 1, 2; keep baseline GPT-3 1/sqrt(2N) scaling in blocks 3, 4, 5.
Compares to the full version (-0.332) and the symmetric deep-only
(future). Useful diagnostic and potentially a cheaper recipe if
shallow alone captures most of the gain.

**Mechanism**: In `src/transformer/model.rs::DecoderTransformer::new`,
after constructing all blocks via the existing map, iterate with
enumerate and zero out `mha.w_o.w` for blocks with `li < n_layer/2`.
~6 LoC. Both CPU + GPU follow via `from_cpu`.

**Expected Δ BPB**: -0.10 to -0.25. If gain is roughly proportional
to number of zeroed blocks (3/6), expect -0.17. If shallow dominates,
closer to -0.25. If deep dominates, closer to -0.10.

**Why unexplored**: Bank has 25 entries; `depth-cond-reg` has one
entry (depth-wd-deep-only, loss). Depth-asymmetric init zeroing is
a different mechanism. The result also informs whether the win
is "structural at every block" or "specific to certain depths".

## Math gate

- **Modified equation(s)**: For blocks 0-2: `w_o = 0` at init → MHA
  sublayer output = 0 → residual unchanged. For blocks 3-5: `w_o`
  uses baseline `1/sqrt(2N) * Xavier` init.
- **Backward pass**: Standard chain rule for both block subsets.
  Each w_o gradient is `∂L/∂mha_out · h_in`; in shallow blocks,
  initial mha_out = 0 but `∂L/∂mha_out` is non-zero, so w_o starts
  leaving zero from step 1. No new non-differentiability.
- **Dim compatibility**: No tensor shape changes.
- **Causality preserved**: Init doesn't interact with mask.
- **Param count delta**: 0.
- **VRAM delta**: 0.

## Sanity-gate prediction

- `transformer_primitives`: PASS (uses single block, n_layer=2 in
  test; shallow cutoff = 1 → block 0 has w_o=0). Gradient checks
  are still valid; same math as fixup-mha-only iter 22.
- `transformer_smoke`: PASS. The standalone fixup-mha-only passed;
  having only HALF the blocks zeroed (and a single-layer model in
  the argmax test) is less aggressive.
- `gpu_parity`: PASS.

Note: `transformer_smoke` uses single-layer models (cfg.n_layer=1).
With shallow cutoff `1 / 2 = 0`, NO blocks get zeroed. So smoke is
identical to baseline. Safe.
