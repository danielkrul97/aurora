# per-block-lr-linear-down

**Area**: depth-cond-reg

**Motivation**: Bank's depth-cond-reg has 2 prior entries
(depth-wd-deep-only loss, depth-fixup-mha-shallow broken_impl). Both
modified the WD or init asymmetrically by depth; neither modified
LR. Per-block LR scaling is the most direct depth-cond mechanism.

Hypothesis: shallow blocks (close to embeddings) carry the heaviest
representational burden — they must transform raw bytes-with-pos into
useful features that deeper blocks can specialize on. Deeper blocks
build on top; their updates have higher leverage but ride on shallow
features. Giving shallow blocks MORE LR lets them establish stable
features faster; reducing deep LR prevents them from overshooting on
features that are still being learned below.

Linear ramp from 1.5× at block 0 to 0.5× at block 5 (mean = 1.0, so
total optimization budget unchanged on average). The blocks see:
- Block 0: 1.5× LR
- Block 1: 1.3× LR
- Block 2: 1.1× LR
- Block 3: 0.9× LR
- Block 4: 0.7× LR
- Block 5: 0.5× LR

Non-block params (w_tok, w_pos, final_ln) keep 1.0× LR.

**Mechanism**: Modify `step_adamw` in both backends to scale config.lr
per block before dispatch.

CPU `src/transformer/model.rs::step_adamw`: in the
`for (li, blk) in self.blocks.iter_mut().enumerate()` loop, compute
`lr_mult = 1.5 - (1.0 * li as f32 / (n_layer-1) as f32)` and pass a
modified AdamWConfig with lr scaled.

GPU `src/gpu/transformer.rs::step_adamw`: in the
`for (i, op) in self.grad_ops.iter().enumerate()` loop, derive
block index from op index (block_i = (i - 4) / 10 for i >= 4) and
apply lr_mult.

~15 LoC across both backends.

**Expected Δ BPB**: −0.005 to +0.030. Wide. Counter-conventional
direction (deep gets less LR), but for from-scratch training (not
fine-tuning) the rationale could work.

**Why unexplored**: Bank has 45 entries. Depth-cond-reg: 2 (both
non-LR axes). Per-block LR has never been tested.

## Math gate

- **Modified equation(s)**: For block `li`,
  `effective_lr = base_lr · (1.5 - 1.0·li/(n_layer-1))`. Each block's
  AdamW step uses this effective lr. Other params (w_tok, w_pos, LN)
  use base_lr unchanged.
- **Backward pass**: Unchanged.
- **Dim compatibility**: 0 changes.
- **Causality preserved**: Yes.
- **Param count delta**: 0.
- **VRAM delta**: 0.
- **Mean LR preserved**: mean over 6 blocks of (1.5, 1.3, 1.1, 0.9,
  0.7, 0.5) = 1.0. Total block-LR budget unchanged.

## Sanity-gate prediction

- `transformer_primitives`: PASS. AdamW step doesn't affect gradient
  computation.
- `transformer_smoke`: PASS. Smoke uses n_layer=1 (some tests) or
  n_layer=2 (others). With n_layer=2: block 0 gets 1.5×, block 1 gets
  0.5×. With n_layer=1: only block 0, gets 1.5×. Higher LR could
  destabilize at smoke's aggressive lr=3e-2, but 1.5× isn't extreme.
  Should converge.
- `gpu_parity`: PASS. Both backends modified in lockstep.
