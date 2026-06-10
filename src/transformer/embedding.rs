#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::doc_markdown
)]

//! Token + positional embedding (learned), plus tied LM head helpers.
//!
//! Embedding matrices are stored row-major:
//! * `w_tok: [vocab * d_model]` — row `v` is the embedding of token `v`
//! * `w_pos: [seq_len_max * d_model]` — row `t` is the positional vec for index `t`
//!
//! Forward looks both up and sums. Backward scatter-adds gradient into
//! the corresponding rows. The tied LM head computes
//! `logits[t, v] = sum_c h[t, c] * w_tok[v, c]` — same `w_tok` as input.

use rand::Rng;

#[derive(Clone, Debug)]
pub struct EmbedWeights {
    pub w_tok: Vec<f32>,
    pub w_pos: Vec<f32>,
    pub vocab: usize,
    pub d_model: usize,
    pub seq_len_max: usize,
    /// False when RoPE supplies position information — w_pos stays
    /// allocated (checkpoint layout unchanged) but is never added and
    /// receives no gradient.
    pub use_pos: bool,
}

impl EmbedWeights {
    pub fn new<R: Rng>(vocab: usize, d_model: usize, seq_len_max: usize, rng: &mut R) -> Self {
        // nanoGPT init: token + positional ~ N(0, 0.02²). We approximate
        // with uniform[-0.02, 0.02] * sqrt(3) ≈ same variance.
        let scale: f32 = 0.02_f32 * (3.0_f32).sqrt();
        let w_tok: Vec<f32> = (0..vocab * d_model)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let w_pos: Vec<f32> = (0..seq_len_max * d_model)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            w_tok,
            w_pos,
            vocab,
            d_model,
            seq_len_max,
            use_pos: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbedGrads {
    pub d_w_tok: Vec<f32>,
    pub d_w_pos: Vec<f32>,
    /// Untied LM head gradient; empty when the head is tied.
    pub d_w_head: Vec<f32>,
}

impl EmbedGrads {
    #[must_use]
    pub fn zeros(vocab: usize, d_model: usize, seq_len_max: usize) -> Self {
        Self {
            d_w_tok: vec![0.0; vocab * d_model],
            d_w_pos: vec![0.0; seq_len_max * d_model],
            d_w_head: Vec::new(),
        }
    }

    pub fn zero(&mut self) {
        self.d_w_tok.fill(0.0);
        self.d_w_pos.fill(0.0);
        self.d_w_head.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbedCache {
    pub tokens: Vec<u32>,
    pub t: usize,
}

/// Forward: `out[ti, c] = w_tok[tokens[ti], c] + w_pos[ti, c]`.
pub fn embed_forward(w: &EmbedWeights, tokens: &[u32]) -> (Vec<f32>, EmbedCache) {
    let t = tokens.len();
    let d = w.d_model;
    debug_assert!(t <= w.seq_len_max, "seq_len {t} > seq_len_max {}", w.seq_len_max);
    let mut out = vec![0.0_f32; t * d];
    for ti in 0..t {
        let tok = tokens[ti] as usize;
        debug_assert!(tok < w.vocab, "token {tok} out of vocab");
        let dst = &mut out[ti * d..(ti + 1) * d];
        let tok_row = &w.w_tok[tok * d..(tok + 1) * d];
        if w.use_pos {
            let pos_row = &w.w_pos[ti * d..(ti + 1) * d];
            for c in 0..d {
                dst[c] = tok_row[c] + pos_row[c];
            }
        } else {
            dst.copy_from_slice(tok_row);
        }
    }
    (
        out,
        EmbedCache {
            tokens: tokens.to_vec(),
            t,
        },
    )
}

/// Backward: scatter `dy` rows into `g_w_tok[token[ti]]` and `g_w_pos[ti]`.
pub fn embed_backward(w: &EmbedWeights, cache: &EmbedCache, dy: &[f32], grads: &mut EmbedGrads) {
    let d = w.d_model;
    assert_eq!(grads.d_w_tok.len(), w.vocab * d);
    assert_eq!(grads.d_w_pos.len(), w.seq_len_max * d);
    for ti in 0..cache.t {
        let tok = cache.tokens[ti] as usize;
        let dy_row = &dy[ti * d..(ti + 1) * d];
        let tok_row = &mut grads.d_w_tok[tok * d..(tok + 1) * d];
        for c in 0..d {
            tok_row[c] += dy_row[c];
        }
        if w.use_pos {
            let pos_row = &mut grads.d_w_pos[ti * d..(ti + 1) * d];
            for c in 0..d {
                pos_row[c] += dy_row[c];
            }
        }
    }
}

/// Position-block size: each `w_tok` row is streamed once per block of
/// positions instead of once per position (`w_tok` is the largest matrix
/// in the model — vocab × d_model ≫ cache). Dot products are computed in
/// the unblocked order, so results are bit-exact.
const HEAD_BLOCK_T: usize = 8;

/// Forward: `logits[t, v] = sum_c h[t, c] * w_tok[v, c]`. Tied LM head.
#[must_use]
pub fn tied_lm_head_forward(h: &[f32], w_tok: &[f32], vocab: usize, d_model: usize) -> Vec<f32> {
    let t = h.len() / d_model;
    let mut logits = vec![0.0_f32; t * vocab];
    let mut t0 = 0;
    while t0 < t {
        let tb = HEAD_BLOCK_T.min(t - t0);
        for v in 0..vocab {
            let wr = &w_tok[v * d_model..(v + 1) * d_model];
            for tt in 0..tb {
                let ti = t0 + tt;
                let hr = &h[ti * d_model..(ti + 1) * d_model];
                let mut acc = 0.0_f32;
                for c in 0..d_model {
                    acc += hr[c] * wr[c];
                }
                logits[ti * vocab + v] = acc;
            }
        }
        t0 += tb;
    }
    logits
}

/// Backward through the LM head matmul. Returns `d_h` and accumulates
/// the head-weight gradient into `d_w`. With a tied head `d_w` is the
/// **same** buffer as the input-embedding gradient (`d_w_tok`) — the
/// caller is responsible for clearing once per step.
pub fn lm_head_backward(
    h: &[f32],
    w: &[f32],
    d_logits: &[f32],
    vocab: usize,
    d_model: usize,
    d_w: &mut [f32],
) -> Vec<f32> {
    let t = h.len() / d_model;
    let mut d_h = vec![0.0_f32; h.len()];
    assert_eq!(d_w.len(), vocab * d_model);
    let mut t0 = 0;
    while t0 < t {
        let tb = HEAD_BLOCK_T.min(t - t0);
        for v in 0..vocab {
            let wr = &w[v * d_model..(v + 1) * d_model];
            let dwr = &mut d_w[v * d_model..(v + 1) * d_model];
            for tt in 0..tb {
                let ti = t0 + tt;
                let g = d_logits[ti * vocab + v];
                let hr = &h[ti * d_model..(ti + 1) * d_model];
                let dhr = &mut d_h[ti * d_model..(ti + 1) * d_model];
                for c in 0..d_model {
                    dwr[c] += g * hr[c];
                    dhr[c] += g * wr[c];
                }
            }
        }
        t0 += tb;
    }
    d_h
}

/// Tied-head wrapper: accumulates into `grads.d_w_tok`.
pub fn tied_lm_head_backward(
    h: &[f32],
    w_tok: &[f32],
    d_logits: &[f32],
    vocab: usize,
    d_model: usize,
    grads: &mut EmbedGrads,
) -> Vec<f32> {
    lm_head_backward(h, w_tok, d_logits, vocab, d_model, &mut grads.d_w_tok)
}
