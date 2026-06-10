//! Pre-norm transformer decoder block.

use rand::Rng;

use crate::attention::{
    layer_norm_backward, layer_norm_forward, LayerNorm, LayerNormCache, LayerNormGrads,
};
use crate::transformer::attention::{
    causal_mha_backward_full, causal_mha_forward_full, MhaCache, MhaGrads, MhaWeights,
};
use crate::transformer::ffn::{ffn_backward, ffn_forward, FfnCache, FfnGrads, FfnWeights};

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub mha: MhaWeights,
    pub ln2: LayerNorm,
    pub ffn: FfnWeights,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new<R: Rng>(
        d_model: usize,
        n_head: usize,
        d_ff: usize,
        n_layer: usize,
        rng: &mut R,
    ) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            mha: MhaWeights::new(d_model, n_head, rng),
            ln2: LayerNorm::new(d_model),
            ffn: FfnWeights::new(d_model, d_ff, n_layer, rng),
            d_model,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformerBlockGrads {
    pub d_ln1: LayerNormGrads,
    pub d_mha: MhaGrads,
    pub d_ln2: LayerNormGrads,
    pub d_ffn: FfnGrads,
}

impl TransformerBlockGrads {
    #[must_use]
    pub fn zeros(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_ln1: LayerNormGrads::zeros(d_model),
            d_mha: MhaGrads::zeros(d_model),
            d_ln2: LayerNormGrads::zeros(d_model),
            d_ffn: FfnGrads::zeros(d_model, d_ff),
        }
    }

    pub fn zero(&mut self) {
        self.d_ln1.zero();
        self.d_mha.zero();
        self.d_ln2.zero();
        self.d_ffn.zero();
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformerBlockCache {
    pub x_in: Vec<f32>,
    pub ln1_in: Vec<f32>,
    pub ln1_cache: LayerNormCache,
    pub ln1_out: Vec<f32>,
    pub mha_cache: MhaCache,
    pub mha_out: Vec<f32>,
    pub post_attn: Vec<f32>,
    pub ln2_cache: LayerNormCache,
    pub ln2_out: Vec<f32>,
    pub ffn_cache: FfnCache,
    pub ffn_out: Vec<f32>,
}

/// PreLN block: `y = x + MHA(LN(x))`, `z = y + FFN(LN(y))`.
/// `window` is this layer's sliding-attention width (0 = full causal);
/// `v0` is layer 0's value stream for the value residual.
pub fn block_forward(
    blk: &TransformerBlock,
    x: &[f32],
    window: usize,
    v0: Option<&[f32]>,
) -> (Vec<f32>, TransformerBlockCache) {
    let (ln1_out, ln1_cache) = layer_norm_forward(&blk.ln1, x);
    let (mha_out, mha_cache) = causal_mha_forward_full(&blk.mha, &ln1_out, window, v0);

    let mut post_attn = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
        post_attn[i] = x[i] + mha_out[i];
    }

    let (ln2_out, ln2_cache) = layer_norm_forward(&blk.ln2, &post_attn);
    let (ffn_out, ffn_cache) = ffn_forward(&blk.ffn, &ln2_out);

    let mut z = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
        z[i] = post_attn[i] + ffn_out[i];
    }

    (
        z,
        TransformerBlockCache {
            x_in: x.to_vec(),
            ln1_in: x.to_vec(),
            ln1_cache,
            ln1_out,
            mha_cache,
            mha_out,
            post_attn,
            ln2_cache,
            ln2_out,
            ffn_cache,
            ffn_out,
        },
    )
}

/// Backward through the block. Returns `dx`. The value-residual
/// plumbing (`v0`, `extra_d_v`, `d_v0_accum`) passes straight through
/// to the MHA backward.
pub fn block_backward(
    blk: &TransformerBlock,
    cache: &TransformerBlockCache,
    dz: &[f32],
    grads: &mut TransformerBlockGrads,
    v0: Option<&[f32]>,
    extra_d_v: Option<&[f32]>,
    d_v0_accum: Option<&mut Vec<f32>>,
) -> Vec<f32> {
    // FFN branch: dz contributes to both `post_attn` (residual) and FFN input.
    let d_ffn_out = dz.to_vec();
    let d_ln2_out = ffn_backward(&blk.ffn, &cache.ffn_cache, &d_ffn_out, &mut grads.d_ffn);
    let d_post_attn_from_ffn =
        layer_norm_backward(&blk.ln2, &cache.ln2_cache, &d_ln2_out, &mut grads.d_ln2);

    let mut d_post_attn = vec![0.0_f32; dz.len()];
    for i in 0..dz.len() {
        d_post_attn[i] = dz[i] + d_post_attn_from_ffn[i];
    }

    // MHA branch: d_post_attn → both `x_in` (residual) and MHA output.
    let d_mha_out = d_post_attn.clone();
    let d_ln1_out = causal_mha_backward_full(
        &blk.mha, &cache.mha_cache, &d_mha_out, &mut grads.d_mha,
        v0, extra_d_v, d_v0_accum,
    );
    let d_x_from_mha =
        layer_norm_backward(&blk.ln1, &cache.ln1_cache, &d_ln1_out, &mut grads.d_ln1);

    let mut dx = vec![0.0_f32; dz.len()];
    for i in 0..dz.len() {
        dx[i] = d_post_attn[i] + d_x_from_mha[i];
    }
    dx
}
