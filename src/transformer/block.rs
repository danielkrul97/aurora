//! Pre-norm transformer decoder block.

use rand::Rng;

use crate::attention::{
    layer_norm_backward, layer_norm_forward, LayerNorm, LayerNormCache, LayerNormGrads,
};
use crate::transformer::attention::{
    causal_mha_backward, causal_mha_forward, MhaCache, MhaGrads, MhaWeights,
};
use crate::transformer::ffn::{ffn_backward, ffn_forward, FfnCache, FfnGrads, FfnWeights};

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub mha: MhaWeights,
    pub ln2: LayerNorm,
    pub ffn: FfnWeights,
    pub d_model: usize,
    /// Parallel (GPT-J) topology: FFN reads the block input instead of
    /// the post-attention residual. See `block_forward`.
    pub parallel: bool,
    /// Per-channel LayerScale gates on the attention / FFN branch outputs
    /// (CaiT). Empty when off; length `d_model` when on, init ~1e-4.
    pub ls1: Vec<f32>,
    pub ls2: Vec<f32>,
    pub layerscale: bool,
    /// Canon causal depthwise-conv taps `[d_model * canon_k]`; empty when
    /// off. Mixed into the residual stream ahead of the block.
    pub canon_w: Vec<f32>,
    pub canon_k: usize,
}

impl TransformerBlock {
    pub fn new<R: Rng>(
        d_model: usize,
        n_head: usize,
        d_ff: usize,
        n_layer: usize,
        swiglu: bool,
        rng: &mut R,
    ) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            mha: MhaWeights::new(d_model, n_head, rng),
            ln2: LayerNorm::new(d_model),
            ffn: FfnWeights::new(d_model, d_ff, n_layer, swiglu, rng),
            d_model,
            parallel: false,
            ls1: Vec::new(),
            ls2: Vec::new(),
            layerscale: false,
            canon_w: Vec::new(),
            canon_k: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformerBlockGrads {
    pub d_ln1: LayerNormGrads,
    pub d_mha: MhaGrads,
    pub d_ln2: LayerNormGrads,
    pub d_ffn: FfnGrads,
    /// LayerScale gate grads `[d_model]`; empty when layerscale is off.
    pub d_ls1: Vec<f32>,
    pub d_ls2: Vec<f32>,
    /// Canon conv-tap grads `[d_model * canon_k]`; empty when off.
    pub d_canon: Vec<f32>,
}

impl TransformerBlockGrads {
    #[must_use]
    pub fn zeros(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_ln1: LayerNormGrads::zeros(d_model),
            d_mha: MhaGrads::zeros(d_model),
            d_ln2: LayerNormGrads::zeros(d_model),
            d_ffn: FfnGrads::zeros(d_model, d_ff),
            d_ls1: Vec::new(),
            d_ls2: Vec::new(),
            d_canon: Vec::new(),
        }
    }

    pub fn zero(&mut self) {
        self.d_ln1.zero();
        self.d_mha.zero();
        self.d_ln2.zero();
        self.d_ffn.zero();
        self.d_ls1.fill(0.0);
        self.d_ls2.fill(0.0);
        self.d_canon.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformerBlockCache {
    /// Original block input before the Canon conv (for canon backward);
    /// empty when Canon is off.
    pub canon_in: Vec<f32>,
    pub x_in: Vec<f32>,
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

/// Causal depthwise conv mixed into the residual stream: returns
/// `x + dwconv(x)` where `out[t,c] = Σ_{j<k, t≥j} w[c,j]·x[t−j,c]`.
/// Per-channel, looks only backward. Zero weights = identity.
fn canon_forward(x: &[f32], w: &[f32], k: usize, d: usize) -> Vec<f32> {
    let t = x.len() / d;
    let mut out = x.to_vec();
    for ti in 0..t {
        for c in 0..d {
            let mut acc = 0.0_f32;
            for j in 0..k.min(ti + 1) {
                acc += w[c * k + j] * x[(ti - j) * d + c];
            }
            out[ti * d + c] += acc;
        }
    }
    out
}

/// Backward of `canon_forward`: returns `d_x` (residual + conv) and
/// accumulates the per-channel tap gradient into `d_w`.
fn canon_backward(
    x: &[f32],
    d_out: &[f32],
    w: &[f32],
    k: usize,
    d: usize,
    d_w: &mut [f32],
) -> Vec<f32> {
    let t = x.len() / d;
    let mut d_x = d_out.to_vec();
    for ti in 0..t {
        for c in 0..d {
            let g = d_out[ti * d + c];
            for j in 0..k.min(ti + 1) {
                d_w[c * k + j] += g * x[(ti - j) * d + c];
                d_x[(ti - j) * d + c] += g * w[c * k + j];
            }
        }
    }
    d_x
}

/// PreLN block: `y = x + MHA(LN(x))`, `z = y + FFN(LN(y))`.
pub fn block_forward(blk: &TransformerBlock, x_in: &[f32]) -> (Vec<f32>, TransformerBlockCache) {
    // Canon: causal depthwise conv mixed into the residual stream ahead of
    // the block (lightweight local mixing, 2025). Zero-init taps = identity.
    let canon_in = if blk.canon_k > 0 { x_in.to_vec() } else { Vec::new() };
    let canon_out =
        (blk.canon_k > 0).then(|| canon_forward(x_in, &blk.canon_w, blk.canon_k, blk.d_model));
    let x: &[f32] = canon_out.as_deref().unwrap_or(x_in);

    let (ln1_out, ln1_cache) = layer_norm_forward(&blk.ln1, x);
    let (mha_out, mha_cache) = causal_mha_forward(&blk.mha, &ln1_out);

    let d = blk.d_model;
    let mut post_attn = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
        let s = if blk.layerscale { blk.ls1[i % d] } else { 1.0 };
        post_attn[i] = x[i] + s * mha_out[i];
    }

    // Parallel (GPT-J) topology: the FFN reads the block input `x`, so
    // attention and FFN are independent branches off the residual stream.
    // Sequential: the FFN reads the post-attention residual.
    let ffn_in: &[f32] = if blk.parallel { x } else { &post_attn };
    let (ln2_out, ln2_cache) = layer_norm_forward(&blk.ln2, ffn_in);
    let (ffn_out, ffn_cache) = ffn_forward(&blk.ffn, &ln2_out);

    let mut z = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
        let s = if blk.layerscale { blk.ls2[i % d] } else { 1.0 };
        z[i] = post_attn[i] + s * ffn_out[i];
    }

    (
        z,
        TransformerBlockCache {
            canon_in,
            x_in: x.to_vec(),
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

/// Backward through the block. Returns `dx`.
pub fn block_backward(
    blk: &TransformerBlock,
    cache: &TransformerBlockCache,
    dz: &[f32],
    grads: &mut TransformerBlockGrads,
) -> Vec<f32> {
    // FFN branch: dz contributes to both `post_attn` (residual) and FFN input.
    // LayerScale on the FFN branch: z = post_attn + ls2 ⊙ ffn_out, so
    // d ffn_out = dz ⊙ ls2 and d ls2[c] = Σ_t dz[t,c]·ffn_out[t,c].
    let d_ln2_out = if blk.layerscale {
        let d = blk.d_model;
        let mut d_ffn_out = dz.to_vec();
        for i in 0..dz.len() {
            let c = i % d;
            grads.d_ls2[c] += dz[i] * cache.ffn_out[i];
            d_ffn_out[i] = dz[i] * blk.ls2[c];
        }
        ffn_backward(&blk.ffn, &cache.ffn_cache, &d_ffn_out, &mut grads.d_ffn)
    } else {
        ffn_backward(&blk.ffn, &cache.ffn_cache, dz, &mut grads.d_ffn)
    };
    // Gradient w.r.t. the FFN input — post_attn (sequential) or x (parallel).
    let d_ffn_in =
        layer_norm_backward(&blk.ln2, &cache.ln2_cache, &d_ln2_out, &mut grads.d_ln2);

    // In sequential mode the FFN feeds the post-attention residual, so its
    // gradient joins `d_post_attn`. In parallel mode it feeds `x` directly,
    // so it joins `dx` further down instead.
    let mut d_post_attn = vec![0.0_f32; dz.len()];
    for i in 0..dz.len() {
        d_post_attn[i] = dz[i];
        if !blk.parallel {
            d_post_attn[i] += d_ffn_in[i];
        }
    }

    // MHA branch: d_post_attn → both `x_in` (residual) and MHA output.
    // LayerScale: post_attn = x + ls1 ⊙ mha_out, so the MHA output gradient
    // is d_post_attn ⊙ ls1 and d ls1[c] = Σ_t d_post_attn[t,c]·mha_out[t,c].
    // The x residual (d_post_attn straight through) is unaffected.
    let d_ln1_out = if blk.layerscale {
        let d = blk.d_model;
        let mut d_mha_out = d_post_attn.clone();
        for i in 0..d_post_attn.len() {
            let c = i % d;
            grads.d_ls1[c] += d_post_attn[i] * cache.mha_out[i];
            d_mha_out[i] = d_post_attn[i] * blk.ls1[c];
        }
        causal_mha_backward(&blk.mha, &cache.mha_cache, &d_mha_out, &mut grads.d_mha)
    } else {
        causal_mha_backward(&blk.mha, &cache.mha_cache, &d_post_attn, &mut grads.d_mha)
    };
    let d_x_from_mha =
        layer_norm_backward(&blk.ln1, &cache.ln1_cache, &d_ln1_out, &mut grads.d_ln1);

    let mut dx = vec![0.0_f32; dz.len()];
    for i in 0..dz.len() {
        dx[i] = d_post_attn[i] + d_x_from_mha[i];
        if blk.parallel {
            dx[i] += d_ffn_in[i];
        }
    }
    // Canon backward: dx is w.r.t. the conv-mixed input; map it back to the
    // original block input (residual passthrough + conv-tap gradient).
    if blk.canon_k > 0 {
        canon_backward(
            &cache.canon_in,
            &dx,
            &blk.canon_w,
            blk.canon_k,
            blk.d_model,
            &mut grads.d_canon,
        )
    } else {
        dx
    }
}
