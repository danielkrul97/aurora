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
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbedGrads {
    pub d_w_tok: Vec<f32>,
    pub d_w_pos: Vec<f32>,
}

impl EmbedGrads {
    #[must_use]
    pub fn zeros(vocab: usize, d_model: usize, seq_len_max: usize) -> Self {
        Self {
            d_w_tok: vec![0.0; vocab * d_model],
            d_w_pos: vec![0.0; seq_len_max * d_model],
        }
    }

    pub fn zero(&mut self) {
        self.d_w_tok.fill(0.0);
        self.d_w_pos.fill(0.0);
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
        let pos_row = &w.w_pos[ti * d..(ti + 1) * d];
        for c in 0..d {
            dst[c] = tok_row[c] + pos_row[c];
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
        let pos_row = &mut grads.d_w_pos[ti * d..(ti + 1) * d];
        for c in 0..d {
            pos_row[c] += dy_row[c];
        }
    }
}

/// Forward: `logits[t, v] = sum_c h[t, c] * w_tok[v, c]`. Tied LM head.
#[must_use]
pub fn tied_lm_head_forward(h: &[f32], w_tok: &[f32], vocab: usize, d_model: usize) -> Vec<f32> {
    let t = h.len() / d_model;
    let mut logits = vec![0.0_f32; t * vocab];
    for ti in 0..t {
        let hr = &h[ti * d_model..(ti + 1) * d_model];
        let lr = &mut logits[ti * vocab..(ti + 1) * vocab];
        for v in 0..vocab {
            let wr = &w_tok[v * d_model..(v + 1) * d_model];
            let mut acc = 0.0_f32;
            for c in 0..d_model {
                acc += hr[c] * wr[c];
            }
            lr[v] = acc;
        }
    }
    logits
}

/// Backward through the tied LM head. Returns `d_h` and accumulates
/// `d_w_tok` into `grads.d_w_tok`. Crucially, this is the **same**
/// gradient buffer as the input embedding path — the caller is
/// responsible for clearing once per step.
pub fn tied_lm_head_backward(
    h: &[f32],
    w_tok: &[f32],
    d_logits: &[f32],
    vocab: usize,
    d_model: usize,
    grads: &mut EmbedGrads,
) -> Vec<f32> {
    let t = h.len() / d_model;
    let mut d_h = vec![0.0_f32; h.len()];
    assert_eq!(grads.d_w_tok.len(), vocab * d_model);
    for ti in 0..t {
        let hr = &h[ti * d_model..(ti + 1) * d_model];
        let dlr = &d_logits[ti * vocab..(ti + 1) * vocab];
        let dhr = &mut d_h[ti * d_model..(ti + 1) * d_model];
        for v in 0..vocab {
            let g = dlr[v];
            let wr = &w_tok[v * d_model..(v + 1) * d_model];
            let dwr = &mut grads.d_w_tok[v * d_model..(v + 1) * d_model];
            for c in 0..d_model {
                dwr[c] += g * hr[c];
                dhr[c] += g * wr[c];
            }
        }
    }
    d_h
}
