#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

//! KV cache for fast autoregressive sampling.
//!
//! During generation, only the newest token needs a fresh forward pass —
//! the previous K and V projections are unchanged. The cache stores K
//! and V per block; each `generate_step` appends a single row and
//! computes attention of the new query against the full K/V history.
//!
//! With cache:    each new token is O(N) work (N = current context len).
//! Without cache: each new token is O(N²) (re-runs the whole forward).
//!
//! Output of `generate_step` is identical to running `forward()` over
//! the full context up to that point (asserted in tests).

use crate::attention::{linear_forward, qk_norm_forward};
use crate::linalg::{gelu, relu2};
use crate::transformer::block::TransformerBlock;
use crate::transformer::embedding::tied_lm_head_forward;
use crate::transformer::model::DecoderTransformer;

#[derive(Clone, Debug)]
pub struct KvBlock {
    /// `[current_len * d_model]`
    pub k_history: Vec<f32>,
    /// `[current_len * d_model]`
    pub v_history: Vec<f32>,
    pub current_len: usize,
}

impl KvBlock {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            k_history: Vec::new(),
            v_history: Vec::new(),
            current_len: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KvCache {
    pub blocks: Vec<KvBlock>,
    pub d_model: usize,
    pub current_len: usize,
}

impl KvCache {
    #[must_use]
    pub fn new(n_layer: usize, d_model: usize) -> Self {
        Self {
            blocks: (0..n_layer).map(|_| KvBlock::empty()).collect(),
            d_model,
            current_len: 0,
        }
    }

    pub fn reset(&mut self) {
        for b in &mut self.blocks {
            b.k_history.clear();
            b.v_history.clear();
            b.current_len = 0;
        }
        self.current_len = 0;
    }

    /// Pre-fill the cache by running the full sequence up through each
    /// block. Returns the final hidden state at the last position.
    pub fn prefill(&mut self, model: &DecoderTransformer, tokens: &[u32]) -> Vec<f32> {
        self.reset();
        let d = self.d_model;
        let t = tokens.len();
        // Run full forward once (cheaper than incremental for cold start).
        let (_logits, cache) = model.forward(tokens);
        // Extract per-block K and V from `cache.blocks`.
        for li in 0..model.blocks.len() {
            let bc = &cache.blocks[li];
            self.blocks[li].k_history = bc.mha_cache.k.clone();
            self.blocks[li].v_history = bc.mha_cache.v.clone();
            self.blocks[li].current_len = t;
        }
        self.current_len = t;
        // Return last position's h_final.
        cache.h_final[(t - 1) * d..t * d].to_vec()
    }
}

/// Run one new token through the entire stack using the cache. Returns
/// the logits for the next token prediction.
pub fn generate_step(
    model: &DecoderTransformer,
    cache: &mut KvCache,
    new_token: u32,
) -> Vec<f32> {
    let d = model.config.d_model;
    let v_dim = model.config.vocab;
    let t_new_pos = cache.current_len; // position index of the new token

    // 1. Embed the new token + add positional encoding for its position.
    let tok = new_token as usize;
    let mut h = vec![0.0_f32; d];
    let tok_row = &model.embed.w_tok[tok * d..(tok + 1) * d];
    h.copy_from_slice(tok_row);
    if model.embed.use_pos {
        let pos_row = &model.embed.w_pos[t_new_pos * d..(t_new_pos + 1) * d];
        for c in 0..d {
            h[c] += pos_row[c];
        }
    }

    // 2. Run through each block with single-query attention against history.
    for li in 0..model.blocks.len() {
        let blk = &model.blocks[li];
        let kv = &mut cache.blocks[li];
        h = block_step(blk, kv, &h, t_new_pos);
    }

    // 3. Final LN + LM head over single position.
    let (h_final, _) = crate::attention::layer_norm_forward(&model.final_ln, &h);
    let head_w = model.w_head.as_deref().unwrap_or(&model.embed.w_tok);
    let mut logits = tied_lm_head_forward(&h_final, head_w, v_dim, d);
    let cap = model.config.softcap;
    if cap > 0.0 {
        for l in &mut logits {
            *l = cap * (*l / cap).tanh();
        }
    }

    cache.current_len += 1;
    logits
}

/// Single-step block forward using KV cache. Appends the new q/k/v to the
/// cache, runs causal attention against history.
fn block_step(
    blk: &TransformerBlock,
    kv: &mut KvBlock,
    h_in: &[f32],
    t_new_pos: usize,
) -> Vec<f32> {
    let d = blk.d_model;
    let n_head = blk.mha.n_head;
    let d_head = blk.mha.d_head;

    // PreLN
    let (ln1_out, _) = crate::attention::layer_norm_forward(&blk.ln1, h_in);
    // Project new q/k/v (single token, so each is length D)
    let mut q_new = linear_forward(&blk.mha.w_q, &ln1_out);
    let mut k_new = linear_forward(&blk.mha.w_k, &ln1_out);
    let v_new = linear_forward(&blk.mha.w_v, &ln1_out);
    if blk.mha.qk_norm {
        qk_norm_forward(&mut q_new, n_head, d_head);
        qk_norm_forward(&mut k_new, n_head, d_head);
    }
    if blk.mha.rope {
        crate::attention::rope_rotate(&mut q_new, n_head, d_head, t_new_pos, false);
        crate::attention::rope_rotate(&mut k_new, n_head, d_head, t_new_pos, false);
    }

    // Append to history (k and v)
    kv.k_history.extend_from_slice(&k_new);
    kv.v_history.extend_from_slice(&v_new);
    kv.current_len = t_new_pos + 1;
    let t_total = kv.current_len;

    // For each head, run scaled dot-product against full K/V history.
    let mut heads_out = vec![0.0_f32; d];
    let scale = (d_head as f32).sqrt().recip();
    for h in 0..n_head {
        let h_off = h * d_head;
        let q_h = &q_new[h_off..h_off + d_head];
        // Compute s[j] for j in 0..t_total
        let mut scores = vec![0.0_f32; t_total];
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..t_total {
            let k_row = &kv.k_history[j * d + h_off..j * d + h_off + d_head];
            let mut s = 0.0_f32;
            for c in 0..d_head {
                s += q_h[c] * k_row[c];
            }
            s *= scale;
            scores[j] = s;
            if s > row_max {
                row_max = s;
            }
        }
        let mut sum = 0.0_f32;
        for j in 0..t_total {
            scores[j] = (scores[j] - row_max).exp();
            sum += scores[j];
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for j in 0..t_total {
            scores[j] *= inv;
        }
        // out = scores @ V_history
        for j in 0..t_total {
            let p = scores[j];
            if p == 0.0 {
                continue;
            }
            let v_row = &kv.v_history[j * d + h_off..j * d + h_off + d_head];
            for c in 0..d_head {
                heads_out[h_off + c] += p * v_row[c];
            }
        }
    }

    // Output projection
    let mha_out = linear_forward(&blk.mha.w_o, &heads_out);

    // Residual
    let mut post_attn = vec![0.0_f32; d];
    for i in 0..d {
        post_attn[i] = h_in[i] + mha_out[i];
    }

    // FFN
    let (ln2_out, _) = crate::attention::layer_norm_forward(&blk.ln2, &post_attn);
    let h_pre = linear_forward(&blk.ffn.w_fc, &ln2_out);
    let h_post: Vec<f32> = if blk.ffn.relu2 {
        h_pre.iter().map(|&v| relu2(v)).collect()
    } else {
        h_pre.iter().map(|&v| gelu(v)).collect()
    };
    let ffn_out = linear_forward(&blk.ffn.w_proj, &h_post);

    let mut z = vec![0.0_f32; d];
    for i in 0..d {
        z[i] = post_attn[i] + ffn_out[i];
    }
    z
}

