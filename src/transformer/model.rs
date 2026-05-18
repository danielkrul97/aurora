#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

//! Full decoder-only transformer assembly + AdamW training step.

use rand::Rng;

use crate::attention::{layer_norm_backward, layer_norm_forward, LayerNorm, LayerNormCache, LayerNormGrads};
use crate::optimizer::{adamw_apply, bias_correction, AdamGroup, AdamWConfig};
use crate::transformer::block::{block_backward, block_forward, TransformerBlock, TransformerBlockCache, TransformerBlockGrads};
use crate::transformer::embedding::{embed_backward, embed_forward, tied_lm_head_backward, tied_lm_head_forward, EmbedCache, EmbedGrads, EmbedWeights};
use crate::transformer::loss::cross_entropy_with_grad;

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub seq_len_max: usize,
}

impl TransformerConfig {
    #[must_use]
    pub fn nano(vocab: usize) -> Self {
        Self {
            vocab,
            d_model: 384,
            n_layer: 6,
            n_head: 6,
            d_ff: 1536,
            seq_len_max: 256,
        }
    }

    #[must_use]
    pub fn tiny(vocab: usize) -> Self {
        Self {
            vocab,
            d_model: 64,
            n_layer: 2,
            n_head: 2,
            d_ff: 128,
            seq_len_max: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecoderTransformer {
    pub config: TransformerConfig,
    pub embed: EmbedWeights,
    pub blocks: Vec<TransformerBlock>,
    pub final_ln: LayerNorm,
    pub adam: AdamState,
    pub step: u64,
}

#[derive(Clone, Debug)]
pub struct AdamState {
    pub g_w_tok: AdamGroup,
    pub g_w_pos: AdamGroup,
    pub blocks: Vec<BlockAdam>,
    pub g_final_gamma: AdamGroup,
    pub g_final_beta: AdamGroup,
}

#[derive(Clone, Debug)]
pub struct BlockAdam {
    pub ln1_gamma: AdamGroup,
    pub ln1_beta: AdamGroup,
    pub w_q: AdamGroup,
    pub w_k: AdamGroup,
    pub w_v: AdamGroup,
    pub w_o: AdamGroup,
    pub ln2_gamma: AdamGroup,
    pub ln2_beta: AdamGroup,
    pub w_fc: AdamGroup,
    pub w_proj: AdamGroup,
}

impl DecoderTransformer {
    pub fn new<R: Rng>(config: TransformerConfig, rng: &mut R) -> Self {
        let embed = EmbedWeights::new(config.vocab, config.d_model, config.seq_len_max, rng);
        let blocks: Vec<TransformerBlock> = (0..config.n_layer)
            .map(|_| {
                TransformerBlock::new(
                    config.d_model,
                    config.n_head,
                    config.d_ff,
                    config.n_layer,
                    rng,
                )
            })
            .collect();
        let final_ln = LayerNorm::new(config.d_model);

        let d = config.d_model;
        let d_ff = config.d_ff;
        let block_adam = || BlockAdam {
            ln1_gamma: AdamGroup::zeros(d),
            ln1_beta: AdamGroup::zeros(d),
            w_q: AdamGroup::zeros(d * d),
            w_k: AdamGroup::zeros(d * d),
            w_v: AdamGroup::zeros(d * d),
            w_o: AdamGroup::zeros(d * d),
            ln2_gamma: AdamGroup::zeros(d),
            ln2_beta: AdamGroup::zeros(d),
            w_fc: AdamGroup::zeros(d * d_ff),
            w_proj: AdamGroup::zeros(d_ff * d),
        };
        let adam = AdamState {
            g_w_tok: AdamGroup::zeros(config.vocab * d),
            g_w_pos: AdamGroup::zeros(config.seq_len_max * d),
            blocks: (0..config.n_layer).map(|_| block_adam()).collect(),
            g_final_gamma: AdamGroup::zeros(d),
            g_final_beta: AdamGroup::zeros(d),
        };
        Self {
            config,
            embed,
            blocks,
            final_ln,
            adam,
            step: 0,
        }
    }

    /// Total number of trainable scalar parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        let d = self.config.d_model;
        let d_ff = self.config.d_ff;
        let per_block = 2 * d + 2 * d + 4 * d * d + 2 * d * d_ff;
        let embed = self.config.vocab * d + self.config.seq_len_max * d;
        embed + self.config.n_layer * per_block + 2 * d
    }
}

#[derive(Clone, Debug, Default)]
pub struct ForwardCache {
    pub embed: EmbedCache,
    pub blocks: Vec<TransformerBlockCache>,
    pub final_ln_cache: LayerNormCache,
    pub h_final: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ModelGrads {
    pub embed: EmbedGrads,
    pub blocks: Vec<TransformerBlockGrads>,
    pub final_ln: LayerNormGrads,
}

impl ModelGrads {
    #[must_use]
    pub fn zeros(config: &TransformerConfig) -> Self {
        Self {
            embed: EmbedGrads::zeros(config.vocab, config.d_model, config.seq_len_max),
            blocks: (0..config.n_layer)
                .map(|_| TransformerBlockGrads::zeros(config.d_model, config.d_ff))
                .collect(),
            final_ln: LayerNormGrads::zeros(config.d_model),
        }
    }

    pub fn zero(&mut self) {
        self.embed.zero();
        for b in &mut self.blocks {
            b.zero();
        }
        self.final_ln.zero();
    }

    /// Global L2 norm of all gradients. For clipping.
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        let mut s = 0.0_f32;
        for v in &self.embed.d_w_tok {
            s += v * v;
        }
        for v in &self.embed.d_w_pos {
            s += v * v;
        }
        for b in &self.blocks {
            for v in &b.d_ln1.d_gamma {
                s += v * v;
            }
            for v in &b.d_ln1.d_beta {
                s += v * v;
            }
            for v in &b.d_mha.d_w_q.d_w {
                s += v * v;
            }
            for v in &b.d_mha.d_w_k.d_w {
                s += v * v;
            }
            for v in &b.d_mha.d_w_v.d_w {
                s += v * v;
            }
            for v in &b.d_mha.d_w_o.d_w {
                s += v * v;
            }
            for v in &b.d_ln2.d_gamma {
                s += v * v;
            }
            for v in &b.d_ln2.d_beta {
                s += v * v;
            }
            for v in &b.d_ffn.d_w_fc.d_w {
                s += v * v;
            }
            for v in &b.d_ffn.d_w_proj.d_w {
                s += v * v;
            }
        }
        for v in &self.final_ln.d_gamma {
            s += v * v;
        }
        for v in &self.final_ln.d_beta {
            s += v * v;
        }
        s.sqrt()
    }

    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.embed.d_w_tok {
            *v *= factor;
        }
        for v in &mut self.embed.d_w_pos {
            *v *= factor;
        }
        for b in &mut self.blocks {
            for v in &mut b.d_ln1.d_gamma {
                *v *= factor;
            }
            for v in &mut b.d_ln1.d_beta {
                *v *= factor;
            }
            for v in &mut b.d_mha.d_w_q.d_w {
                *v *= factor;
            }
            for v in &mut b.d_mha.d_w_k.d_w {
                *v *= factor;
            }
            for v in &mut b.d_mha.d_w_v.d_w {
                *v *= factor;
            }
            for v in &mut b.d_mha.d_w_o.d_w {
                *v *= factor;
            }
            for v in &mut b.d_ln2.d_gamma {
                *v *= factor;
            }
            for v in &mut b.d_ln2.d_beta {
                *v *= factor;
            }
            for v in &mut b.d_ffn.d_w_fc.d_w {
                *v *= factor;
            }
            for v in &mut b.d_ffn.d_w_proj.d_w {
                *v *= factor;
            }
        }
        for v in &mut self.final_ln.d_gamma {
            *v *= factor;
        }
        for v in &mut self.final_ln.d_beta {
            *v *= factor;
        }
    }
}

impl DecoderTransformer {
    /// Forward a single sequence of length `T`. Returns `(logits[T*V], cache)`.
    pub fn forward(&self, tokens: &[u32]) -> (Vec<f32>, ForwardCache) {
        let (mut h, embed_cache) = embed_forward(&self.embed, tokens);

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (h_next, cache) = block_forward(blk, &h);
            h = h_next;
            block_caches.push(cache);
        }
        let (h_final, final_ln_cache) = layer_norm_forward(&self.final_ln, &h);
        let logits = tied_lm_head_forward(
            &h_final,
            &self.embed.w_tok,
            self.config.vocab,
            self.config.d_model,
        );

        (
            logits,
            ForwardCache {
                embed: embed_cache,
                blocks: block_caches,
                final_ln_cache,
                h_final,
            },
        )
    }

    /// Compute per-token cross-entropy loss (averaged) and accumulate
    /// gradients into `grads`. `cache` must come from a recent
    /// `forward()`. `targets[t]` is the desired next token at position `t`.
    pub fn backward(
        &self,
        cache: &ForwardCache,
        targets: &[u32],
        grads: &mut ModelGrads,
    ) -> f32 {
        let t = targets.len();
        let v = self.config.vocab;
        let d = self.config.d_model;
        debug_assert_eq!(cache.h_final.len(), t * d);

        let mut total_loss = 0.0_f32;
        let mut d_logits = vec![0.0_f32; t * v];

        // Re-derive logits + d_logits per timestep using softmax CE.
        // (We could cache softmax in forward but the memory cost is V*T
        // which dominates everything else; cheaper to recompute.)
        let logits = tied_lm_head_forward(&cache.h_final, &self.embed.w_tok, v, d);
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let (loss, mut grad) = cross_entropy_with_grad(row, targets[ti] as usize);
            total_loss += loss;
            for c in 0..v {
                d_logits[ti * v + c] = grad[c] / t as f32;
            }
            grad.clear();
        }

        // Tied LM head backward.
        let d_h_final = tied_lm_head_backward(
            &cache.h_final,
            &self.embed.w_tok,
            &d_logits,
            v,
            d,
            &mut grads.embed,
        );
        // Final LayerNorm backward.
        let d_pre_final = layer_norm_backward(
            &self.final_ln,
            &cache.final_ln_cache,
            &d_h_final,
            &mut grads.final_ln,
        );

        // Block stack backward.
        let mut d_h = d_pre_final;
        for li in (0..self.blocks.len()).rev() {
            d_h = block_backward(
                &self.blocks[li],
                &cache.blocks[li],
                &d_h,
                &mut grads.blocks[li],
            );
        }

        // Embedding backward (token + positional). Note: token
        // gradients ALSO accrue from the tied LM head above — same
        // grads.embed buffer, additive.
        embed_backward(&self.embed, &cache.embed, &d_h, &mut grads.embed);

        total_loss / t as f32
    }

    /// Apply AdamW step. Weight decay is applied to matmul weights
    /// (Q/K/V/O/fc/proj/tok_embed) but NOT to LayerNorm γ/β or
    /// positional embeddings. (Bias parameters are absent — no biases
    /// in our linears; LayerNorm β acts in that role.)
    pub fn step_adamw(&mut self, grads: &ModelGrads, config: &AdamWConfig) {
        self.step += 1;
        let (bc1, bc2) = bias_correction(self.step as usize, config.beta1, config.beta2);

        adamw_apply(
            &mut self.embed.w_tok,
            &grads.embed.d_w_tok,
            &mut self.adam.g_w_tok,
            config,
            bc1,
            bc2,
            true,
        );
        adamw_apply(
            &mut self.embed.w_pos,
            &grads.embed.d_w_pos,
            &mut self.adam.g_w_pos,
            config,
            bc1,
            bc2,
            false,
        );
        for (li, blk) in self.blocks.iter_mut().enumerate() {
            let ba = &mut self.adam.blocks[li];
            let bg = &grads.blocks[li];
            adamw_apply(&mut blk.ln1.gamma, &bg.d_ln1.d_gamma, &mut ba.ln1_gamma, config, bc1, bc2, false);
            adamw_apply(&mut blk.ln1.beta, &bg.d_ln1.d_beta, &mut ba.ln1_beta, config, bc1, bc2, false);
            adamw_apply(&mut blk.mha.w_q.w, &bg.d_mha.d_w_q.d_w, &mut ba.w_q, config, bc1, bc2, true);
            adamw_apply(&mut blk.mha.w_k.w, &bg.d_mha.d_w_k.d_w, &mut ba.w_k, config, bc1, bc2, true);
            adamw_apply(&mut blk.mha.w_v.w, &bg.d_mha.d_w_v.d_w, &mut ba.w_v, config, bc1, bc2, true);
            adamw_apply(&mut blk.mha.w_o.w, &bg.d_mha.d_w_o.d_w, &mut ba.w_o, config, bc1, bc2, true);
            adamw_apply(&mut blk.ln2.gamma, &bg.d_ln2.d_gamma, &mut ba.ln2_gamma, config, bc1, bc2, false);
            adamw_apply(&mut blk.ln2.beta, &bg.d_ln2.d_beta, &mut ba.ln2_beta, config, bc1, bc2, false);
            adamw_apply(&mut blk.ffn.w_fc.w, &bg.d_ffn.d_w_fc.d_w, &mut ba.w_fc, config, bc1, bc2, true);
            adamw_apply(&mut blk.ffn.w_proj.w, &bg.d_ffn.d_w_proj.d_w, &mut ba.w_proj, config, bc1, bc2, true);
        }
        adamw_apply(
            &mut self.final_ln.gamma,
            &grads.final_ln.d_gamma,
            &mut self.adam.g_final_gamma,
            config,
            bc1,
            bc2,
            false,
        );
        adamw_apply(
            &mut self.final_ln.beta,
            &grads.final_ln.d_beta,
            &mut self.adam.g_final_beta,
            config,
            bc1,
            bc2,
            false,
        );
    }

    /// Clip gradient global norm. Returns the pre-clip norm for logging.
    pub fn clip_grads(grads: &mut ModelGrads, max_norm: f32) -> f32 {
        let n = grads.l2_norm();
        if n > max_norm && n > 0.0 {
            grads.scale(max_norm / n);
        }
        n
    }
}
