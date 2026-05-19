//! GPU mirror of `DecoderTransformer` — runs forward, backward, and AdamW
//! entirely on the GPU.
//!
//! Strategy: store all weight + gradient + Adam moment tensors as GPU
//! buffers; intermediate activations (LN cache, attn probs, FFN h_pre,
//! tied head logits) also stay GPU-resident. The host only downloads
//! the final logits to compute cross-entropy (cheap O(B·T·V)) and
//! re-uploads d_logits.
//!
//! Parity with CPU is asserted in `tests/gpu_parity.rs`.

#![allow(clippy::too_many_arguments, clippy::too_many_lines, clippy::similar_names)]

use super::buffers::GpuBuffer;
use super::kv_cache::{KvCacheGpu, MhaKvCacheKernel, SingleTokenEmbedKernel};
use super::{
    AdamWKernel, CausalMhaBackwardKernel, CausalMhaForwardKernel, EmbedScatterAddKernel,
    EmbedScatterReduceKernel, GeluBackwardKernel, GeluKernel, GemmKernel, GemmTransposeBKernel,
    GpuContext, GradNormFinalizeKernel, GradNormSqReduceKernel, LayerNormBackwardKernel,
    LayerNormForwardKernel, OuterProductAccumKernel, RowReduceAddKernel, ScaleByBufferKernel,
    SoftmaxCrossEntropyKernel, TokenPosEmbedKernel, VectorAddKernel, ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig};
use crate::transformer::{DecoderTransformer, TransformerConfig};

const GN_GROUPS_PER_TENSOR: u32 = 8;

pub struct DecoderTransformerGpu {
    pub ctx: GpuContext,
    pub config: TransformerConfig,
    pub step: u64,

    // Kernels (cheap to keep around — pipelines are Arc<>).
    pub gemm: GemmKernel,
    pub gemm_t: GemmTransposeBKernel,
    pub gelu_fwd: GeluKernel,
    pub gelu_bwd: GeluBackwardKernel,
    pub ln_fwd: LayerNormForwardKernel,
    pub ln_bwd: LayerNormBackwardKernel,
    pub mha_fwd: CausalMhaForwardKernel,
    pub mha_bwd: CausalMhaBackwardKernel,
    pub embed_fwd: TokenPosEmbedKernel,
    pub embed_bwd: EmbedScatterAddKernel,
    pub adamw: AdamWKernel,
    pub vadd: VectorAddKernel,
    pub zerob: ZeroBufferKernel,
    pub outer_accum: OuterProductAccumKernel,
    pub row_reduce: RowReduceAddKernel,
    pub gn_sq: GradNormSqReduceKernel,
    pub gn_finalize: GradNormFinalizeKernel,
    pub scaleb_buf: ScaleByBufferKernel,
    pub embed_scatter_reduce: EmbedScatterReduceKernel,
    pub single_embed: SingleTokenEmbedKernel,
    pub mha_kv: MhaKvCacheKernel,
    pub softmax_ce: SoftmaxCrossEntropyKernel,

    // Pure-GPU clip_grads scratch:
    //   gn_partials_flat — concatenated per-tensor partial sums of squares
    //   gn_result        — [norm, clip_factor] written by GradNormFinalize
    pub gn_partials_flat: GpuBuffer,
    pub gn_result: GpuBuffer,

    // Parameters + Adam moments.
    pub p: GpuParams,

    // Cached bind groups for the per-grad-tensor work in zero/clip/adam.
    // The buffers these reference are stable for the lifetime of the
    // model, so the bind groups can be built once and reused every step.
    // Eliminates the dominant per-step cost: ~64 BindGroup creations per
    // each of zero_grad / clip_grads / step_adamw.
    pub(crate) grad_ops: Vec<GradOps>,
    pub(crate) gn_finalize_bind: wgpu::BindGroup,

    // Persistent activation arena. All buffers sized to `seq_len_max`;
    // train_step reuses them every step (no per-step allocation).
    pub(crate) acts: Activations,

    // Cached bind groups for the entire forward + backward hot path.
    // Built once at construction since arena buffers + param buffers
    // are stable for the lifetime of the model.
    pub(crate) fwd_ops: ForwardOps,
    pub(crate) bwd_ops: BackwardOps,
}

/// Cached forward bind groups. One `LayerForwardOps` per layer +
/// top-level (embed, final ln, final gemm). Built once at construction.
pub(crate) struct ForwardOps {
    pub embed: wgpu::BindGroup,
    pub layers: Vec<LayerForwardOps>,
    pub final_ln: wgpu::BindGroup,
    pub final_gemm: wgpu::BindGroup,
}

pub(crate) struct LayerForwardOps {
    pub ln1: wgpu::BindGroup,
    pub gemm_q: wgpu::BindGroup,
    pub gemm_k: wgpu::BindGroup,
    pub gemm_v: wgpu::BindGroup,
    pub mha: wgpu::BindGroup,
    pub gemm_o: wgpu::BindGroup,
    pub vadd_post_attn: wgpu::BindGroup,
    pub ln2: wgpu::BindGroup,
    pub gemm_fc: wgpu::BindGroup,
    pub gelu: wgpu::BindGroup,
    pub gemm_proj: wgpu::BindGroup,
    pub vadd_z: wgpu::BindGroup,
}

pub(crate) struct BackwardOps {
    pub gemm_d_h_final: wgpu::BindGroup,
    pub outer_g_w_tok: wgpu::BindGroup,
    pub final_ln_bwd: wgpu::BindGroup,
    pub layers: Vec<LayerBackwardOps>,
    pub embed_bwd: wgpu::BindGroup,
    pub embed_scatter_reduce: wgpu::BindGroup,
}

pub(crate) struct LayerBackwardOps {
    pub gemm_d_h_post: wgpu::BindGroup,
    pub outer_g_proj: wgpu::BindGroup,
    pub gelu_bwd: wgpu::BindGroup,
    pub gemm_d_ln2: wgpu::BindGroup,
    pub outer_g_fc: wgpu::BindGroup,
    pub ln2_bwd: wgpu::BindGroup,
    pub vadd_d_post_attn: wgpu::BindGroup,
    pub gemm_d_heads_out: wgpu::BindGroup,
    pub outer_g_o: wgpu::BindGroup,
    pub zerob_dq: wgpu::BindGroup,
    pub zerob_dk: wgpu::BindGroup,
    pub zerob_dv: wgpu::BindGroup,
    pub mha_bwd_dscores: wgpu::BindGroup,
    pub mha_bwd_dq: wgpu::BindGroup,
    pub mha_bwd_dkv: wgpu::BindGroup,
    pub gemm_dx_q: wgpu::BindGroup,
    pub gemm_dx_k: wgpu::BindGroup,
    pub gemm_dx_v: wgpu::BindGroup,
    pub vadd_qk: wgpu::BindGroup,
    pub vadd_qkv: wgpu::BindGroup,
    pub outer_g_q: wgpu::BindGroup,
    pub outer_g_k: wgpu::BindGroup,
    pub outer_g_v: wgpu::BindGroup,
    pub ln1_bwd: wgpu::BindGroup,
    pub vadd_dx: wgpu::BindGroup,
}

pub(crate) struct GradOps {
    pub n: u32,
    pub apply_wd: bool,
    pub gn_offset: u32,
    pub zerob_bind: wgpu::BindGroup,
    pub adamw_bind: wgpu::BindGroup,
    pub gn_sq_bind: wgpu::BindGroup,
    pub scale_bind: wgpu::BindGroup,
}

/// Per-layer activations needed by both forward and backward.
/// All sized to `seq_len_max`; the kernel takes the current `t` via
/// push constant and indexes within the prefix.
pub(crate) struct LayerCache {
    pub ln1_out: GpuBuffer,
    pub ln1_xhat: GpuBuffer,
    pub ln1_inv: GpuBuffer,
    pub q: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub heads_out: GpuBuffer,
    pub head_probs: GpuBuffer,
    pub ln2_out: GpuBuffer,
    pub ln2_xhat: GpuBuffer,
    pub ln2_inv: GpuBuffer,
    pub h_pre: GpuBuffer,
    pub h_post: GpuBuffer,
}

/// All activation buffers used by `train_step`, allocated once at
/// `from_cpu` and reused every step. Replaces ~80 `GpuBuffer::empty`
/// calls per step (each of which costs a Vulkan vkAllocateMemory).
pub(crate) struct Activations {
    pub layers: Vec<LayerCache>,
    /// `z_chain[li]` is the output of block `li`, becomes input of `li+1`.
    pub z_chain: Vec<GpuBuffer>,

    // Forward top-level.
    pub tok_buf: GpuBuffer,
    pub h0: GpuBuffer,
    pub ln1_mean: GpuBuffer,
    pub ln2_mean: GpuBuffer,
    pub mha_out: GpuBuffer,
    pub post_attn: GpuBuffer,
    pub ffn_out: GpuBuffer,
    pub f_mean: GpuBuffer,
    pub f_inv: GpuBuffer,
    pub f_xhat: GpuBuffer,
    pub h_final: GpuBuffer,
    pub logits_buf: GpuBuffer,
    pub d_logits_buf: GpuBuffer,
    /// Per-token NLL written by `SoftmaxCrossEntropyKernel`. Sized for `seq_len_max`.
    pub loss_seq_buf: GpuBuffer,
    /// u32 target token IDs uploaded once per call to train_step / eval_step.
    pub targets_buf: GpuBuffer,

    // Backward scratch (shared across blocks since bwd is sequential).
    pub d_h_final: GpuBuffer,
    pub d_pre_final: GpuBuffer,
    pub d_h_post: GpuBuffer,
    pub d_h_pre: GpuBuffer,
    pub d_ln2_out: GpuBuffer,
    pub d_post_attn_ffn: GpuBuffer,
    pub d_post_attn: GpuBuffer,
    pub d_heads_out: GpuBuffer,
    pub dq: GpuBuffer,
    pub dk: GpuBuffer,
    pub dv: GpuBuffer,
    pub d_scores_scratch: GpuBuffer,
    pub dx_q: GpuBuffer,
    pub dx_k: GpuBuffer,
    pub dx_v: GpuBuffer,
    pub d_ln1_qk: GpuBuffer,
    pub d_ln1_out: GpuBuffer,
    pub d_x_mha: GpuBuffer,
    pub dx: GpuBuffer,
    pub d_w_tok_per_pos: GpuBuffer,
}

impl Activations {
    fn new(ctx: &GpuContext, cfg: &TransformerConfig) -> Self {
        let t = cfg.seq_len_max;
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        let n_head = cfg.n_head;
        let v = cfg.vocab;
        let td = t * d;
        let td_ff = t * d_ff;
        let probs_cap = n_head * t * t;

        let layers: Vec<LayerCache> = (0..cfg.n_layer)
            .map(|_| LayerCache {
                ln1_out: GpuBuffer::empty(ctx, "ln1_out", td),
                ln1_xhat: GpuBuffer::empty(ctx, "ln1_xhat", td),
                ln1_inv: GpuBuffer::empty(ctx, "ln1_inv", t),
                q: GpuBuffer::empty(ctx, "q", td),
                k: GpuBuffer::empty(ctx, "k", td),
                v: GpuBuffer::empty(ctx, "v", td),
                heads_out: GpuBuffer::empty(ctx, "heads", td),
                head_probs: GpuBuffer::empty(ctx, "probs", probs_cap),
                ln2_out: GpuBuffer::empty(ctx, "ln2_out", td),
                ln2_xhat: GpuBuffer::empty(ctx, "ln2_xhat", td),
                ln2_inv: GpuBuffer::empty(ctx, "ln2_inv", t),
                h_pre: GpuBuffer::empty(ctx, "h_pre", td_ff),
                h_post: GpuBuffer::empty(ctx, "h_post", td_ff),
            })
            .collect();
        let z_chain: Vec<GpuBuffer> = (0..cfg.n_layer)
            .map(|_| GpuBuffer::empty(ctx, "z", td))
            .collect();

        Self {
            layers,
            z_chain,
            tok_buf: GpuBuffer::empty(ctx, "tok", t),
            h0: GpuBuffer::empty(ctx, "h0", td),
            ln1_mean: GpuBuffer::empty(ctx, "ln1_mean", t),
            ln2_mean: GpuBuffer::empty(ctx, "ln2_mean", t),
            mha_out: GpuBuffer::empty(ctx, "mha_out", td),
            post_attn: GpuBuffer::empty(ctx, "post_attn", td),
            ffn_out: GpuBuffer::empty(ctx, "ffn_out", td),
            f_mean: GpuBuffer::empty(ctx, "f_mean", t),
            f_inv: GpuBuffer::empty(ctx, "f_inv", t),
            f_xhat: GpuBuffer::empty(ctx, "f_xhat", td),
            h_final: GpuBuffer::empty(ctx, "h_final", td),
            logits_buf: GpuBuffer::empty(ctx, "logits", t * v),
            d_logits_buf: GpuBuffer::empty(ctx, "d_logits", t * v),
            loss_seq_buf: GpuBuffer::empty(ctx, "loss_seq", t),
            targets_buf: GpuBuffer::empty(ctx, "targets", t),
            d_h_final: GpuBuffer::empty(ctx, "d_h_final", td),
            d_pre_final: GpuBuffer::empty(ctx, "d_pre_final", td),
            d_h_post: GpuBuffer::empty(ctx, "d_h_post", td_ff),
            d_h_pre: GpuBuffer::empty(ctx, "d_h_pre", td_ff),
            d_ln2_out: GpuBuffer::empty(ctx, "d_ln2", td),
            d_post_attn_ffn: GpuBuffer::empty(ctx, "d_pa_ffn", td),
            d_post_attn: GpuBuffer::empty(ctx, "d_post_attn", td),
            d_heads_out: GpuBuffer::empty(ctx, "d_heads_out", td),
            dq: GpuBuffer::empty(ctx, "dq", td),
            dk: GpuBuffer::empty(ctx, "dk", td),
            dv: GpuBuffer::empty(ctx, "dv", td),
            d_scores_scratch: GpuBuffer::empty(ctx, "d_scores", probs_cap),
            dx_q: GpuBuffer::empty(ctx, "dx_q", td),
            dx_k: GpuBuffer::empty(ctx, "dx_k", td),
            dx_v: GpuBuffer::empty(ctx, "dx_v", td),
            d_ln1_qk: GpuBuffer::empty(ctx, "d_ln1_qk", td),
            d_ln1_out: GpuBuffer::empty(ctx, "d_ln1", td),
            d_x_mha: GpuBuffer::empty(ctx, "d_x_mha", td),
            dx: GpuBuffer::empty(ctx, "dx_blk", td),
            d_w_tok_per_pos: GpuBuffer::empty(ctx, "dwt_pr", td),
        }
    }
}

pub struct GpuParams {
    pub w_tok: GpuBuffer,
    pub w_pos: GpuBuffer,
    pub blocks: Vec<GpuBlock>,
    pub final_gamma: GpuBuffer,
    pub final_beta: GpuBuffer,

    pub g_w_tok: GpuBuffer,
    pub g_w_pos: GpuBuffer,
    pub g_final_gamma: GpuBuffer,
    pub g_final_beta: GpuBuffer,

    pub m1_w_tok: GpuBuffer,
    pub m2_w_tok: GpuBuffer,
    pub m1_w_pos: GpuBuffer,
    pub m2_w_pos: GpuBuffer,
    pub m1_final_gamma: GpuBuffer,
    pub m2_final_gamma: GpuBuffer,
    pub m1_final_beta: GpuBuffer,
    pub m2_final_beta: GpuBuffer,
}

pub struct GpuBlock {
    pub ln1_gamma: GpuBuffer,
    pub ln1_beta: GpuBuffer,
    pub w_q: GpuBuffer,
    pub w_k: GpuBuffer,
    pub w_v: GpuBuffer,
    pub w_o: GpuBuffer,
    pub ln2_gamma: GpuBuffer,
    pub ln2_beta: GpuBuffer,
    pub w_fc: GpuBuffer,
    pub w_proj: GpuBuffer,

    pub g_ln1_gamma: GpuBuffer,
    pub g_ln1_beta: GpuBuffer,
    pub g_w_q: GpuBuffer,
    pub g_w_k: GpuBuffer,
    pub g_w_v: GpuBuffer,
    pub g_w_o: GpuBuffer,
    pub g_ln2_gamma: GpuBuffer,
    pub g_ln2_beta: GpuBuffer,
    pub g_w_fc: GpuBuffer,
    pub g_w_proj: GpuBuffer,

    pub m1: BlockAdamMoments,
    pub m2: BlockAdamMoments,
}

pub struct BlockAdamMoments {
    pub ln1_gamma: GpuBuffer,
    pub ln1_beta: GpuBuffer,
    pub w_q: GpuBuffer,
    pub w_k: GpuBuffer,
    pub w_v: GpuBuffer,
    pub w_o: GpuBuffer,
    pub ln2_gamma: GpuBuffer,
    pub ln2_beta: GpuBuffer,
    pub w_fc: GpuBuffer,
    pub w_proj: GpuBuffer,
}

#[allow(clippy::too_many_arguments)]
fn build_forward_ops(
    ctx: &GpuContext,
    p: &GpuParams,
    a: &Activations,
    cfg: &TransformerConfig,
    embed_fwd: &TokenPosEmbedKernel,
    ln_fwd: &LayerNormForwardKernel,
    gemm_t: &GemmTransposeBKernel,
    mha_fwd: &CausalMhaForwardKernel,
    gelu_fwd: &GeluKernel,
    vadd: &VectorAddKernel,
) -> ForwardOps {
    let embed = embed_fwd.build_bind(ctx, &a.tok_buf, &p.w_tok, &p.w_pos, &a.h0);
    let layers: Vec<LayerForwardOps> = (0..cfg.n_layer)
        .map(|li| {
            let blk = &p.blocks[li];
            let lc = &a.layers[li];
            let x = if li == 0 { &a.h0 } else { &a.z_chain[li - 1] };
            LayerForwardOps {
                ln1: ln_fwd.build_bind(
                    ctx, x, &blk.ln1_gamma, &blk.ln1_beta, &lc.ln1_out, &a.ln1_mean,
                    &lc.ln1_inv, &lc.ln1_xhat,
                ),
                gemm_q: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_q, &lc.q),
                gemm_k: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_k, &lc.k),
                gemm_v: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_v, &lc.v),
                mha: mha_fwd.build_bind(ctx, &lc.q, &lc.k, &lc.v, &lc.heads_out, &lc.head_probs),
                gemm_o: gemm_t.build_bind(ctx, &lc.heads_out, &blk.w_o, &a.mha_out),
                vadd_post_attn: vadd.build_bind(ctx, x, &a.mha_out, &a.post_attn),
                ln2: ln_fwd.build_bind(
                    ctx, &a.post_attn, &blk.ln2_gamma, &blk.ln2_beta, &lc.ln2_out, &a.ln2_mean,
                    &lc.ln2_inv, &lc.ln2_xhat,
                ),
                gemm_fc: gemm_t.build_bind(ctx, &lc.ln2_out, &blk.w_fc, &lc.h_pre),
                gelu: gelu_fwd.build_bind(ctx, &lc.h_pre, &lc.h_post),
                gemm_proj: gemm_t.build_bind(ctx, &lc.h_post, &blk.w_proj, &a.ffn_out),
                vadd_z: vadd.build_bind(ctx, &a.post_attn, &a.ffn_out, &a.z_chain[li]),
            }
        })
        .collect();
    let final_ln = ln_fwd.build_bind(
        ctx, &a.z_chain[cfg.n_layer - 1], &p.final_gamma, &p.final_beta,
        &a.h_final, &a.f_mean, &a.f_inv, &a.f_xhat,
    );
    let final_gemm = gemm_t.build_bind(ctx, &a.h_final, &p.w_tok, &a.logits_buf);
    ForwardOps { embed, layers, final_ln, final_gemm }
}

#[allow(clippy::too_many_arguments)]
fn build_backward_ops(
    ctx: &GpuContext,
    p: &GpuParams,
    a: &Activations,
    cfg: &TransformerConfig,
    gemm: &GemmKernel,
    ln_bwd: &LayerNormBackwardKernel,
    mha_bwd: &CausalMhaBackwardKernel,
    gelu_bwd: &GeluBackwardKernel,
    vadd: &VectorAddKernel,
    outer_accum: &OuterProductAccumKernel,
    embed_bwd_k: &EmbedScatterAddKernel,
    embed_scatter_reduce: &EmbedScatterReduceKernel,
    zerob: &ZeroBufferKernel,
) -> BackwardOps {
    let gemm_d_h_final = gemm.build_bind(ctx, &a.d_logits_buf, &p.w_tok, &a.d_h_final);
    let outer_g_w_tok = outer_accum.build_bind(ctx, &a.h_final, &a.d_logits_buf, &p.g_w_tok);
    let final_ln_bwd = ln_bwd.build_bind(
        ctx, &a.d_h_final, &p.final_gamma, &a.f_xhat, &a.f_inv,
        &a.d_pre_final, &p.g_final_gamma, &p.g_final_beta,
    );

    let n = cfg.n_layer;
    let layers: Vec<LayerBackwardOps> = (0..n)
        .map(|li| {
            let blk = &p.blocks[li];
            let lc = &a.layers[li];
            // d_z is d_pre_final only for the LAST backward iteration
            // (which is li = n-1 since we iterate in reverse).
            let d_z = if li == n - 1 { &a.d_pre_final } else { &a.dx };
            LayerBackwardOps {
                gemm_d_h_post: gemm.build_bind(ctx, d_z, &blk.w_proj, &a.d_h_post),
                outer_g_proj: outer_accum.build_bind(ctx, &lc.h_post, d_z, &blk.g_w_proj),
                gelu_bwd: gelu_bwd.build_bind(ctx, &lc.h_pre, &a.d_h_post, &a.d_h_pre),
                gemm_d_ln2: gemm.build_bind(ctx, &a.d_h_pre, &blk.w_fc, &a.d_ln2_out),
                outer_g_fc: outer_accum.build_bind(ctx, &lc.ln2_out, &a.d_h_pre, &blk.g_w_fc),
                ln2_bwd: ln_bwd.build_bind(
                    ctx, &a.d_ln2_out, &blk.ln2_gamma, &lc.ln2_xhat, &lc.ln2_inv,
                    &a.d_post_attn_ffn, &blk.g_ln2_gamma, &blk.g_ln2_beta,
                ),
                vadd_d_post_attn: vadd.build_bind(ctx, d_z, &a.d_post_attn_ffn, &a.d_post_attn),
                gemm_d_heads_out: gemm.build_bind(ctx, &a.d_post_attn, &blk.w_o, &a.d_heads_out),
                outer_g_o: outer_accum.build_bind(ctx, &lc.heads_out, &a.d_post_attn, &blk.g_w_o),
                zerob_dq: zerob.build_bind(ctx, &a.dq),
                zerob_dk: zerob.build_bind(ctx, &a.dk),
                zerob_dv: zerob.build_bind(ctx, &a.dv),
                mha_bwd_dscores: mha_bwd.build_bind_dscores(
                    ctx, &lc.v, &lc.head_probs, &a.d_heads_out, &a.d_scores_scratch,
                ),
                mha_bwd_dq: mha_bwd.build_bind_dq(ctx, &lc.k, &a.d_scores_scratch, &a.dq),
                mha_bwd_dkv: mha_bwd.build_bind_dkv(
                    ctx, &lc.q, &lc.head_probs, &a.d_heads_out, &a.d_scores_scratch,
                    &a.dk, &a.dv,
                ),
                gemm_dx_q: gemm.build_bind(ctx, &a.dq, &blk.w_q, &a.dx_q),
                gemm_dx_k: gemm.build_bind(ctx, &a.dk, &blk.w_k, &a.dx_k),
                gemm_dx_v: gemm.build_bind(ctx, &a.dv, &blk.w_v, &a.dx_v),
                vadd_qk: vadd.build_bind(ctx, &a.dx_q, &a.dx_k, &a.d_ln1_qk),
                vadd_qkv: vadd.build_bind(ctx, &a.d_ln1_qk, &a.dx_v, &a.d_ln1_out),
                outer_g_q: outer_accum.build_bind(ctx, &lc.ln1_out, &a.dq, &blk.g_w_q),
                outer_g_k: outer_accum.build_bind(ctx, &lc.ln1_out, &a.dk, &blk.g_w_k),
                outer_g_v: outer_accum.build_bind(ctx, &lc.ln1_out, &a.dv, &blk.g_w_v),
                ln1_bwd: ln_bwd.build_bind(
                    ctx, &a.d_ln1_out, &blk.ln1_gamma, &lc.ln1_xhat, &lc.ln1_inv,
                    &a.d_x_mha, &blk.g_ln1_gamma, &blk.g_ln1_beta,
                ),
                vadd_dx: vadd.build_bind(ctx, &a.d_post_attn, &a.d_x_mha, &a.dx),
            }
        })
        .collect();
    let embed_bwd = embed_bwd_k.build_bind(
        ctx, &a.tok_buf, &a.dx, &a.d_w_tok_per_pos, &p.g_w_pos,
    );
    let embed_scatter_reduce = embed_scatter_reduce.build_bind(
        ctx, &a.tok_buf, &a.d_w_tok_per_pos, &p.g_w_tok,
    );
    BackwardOps {
        gemm_d_h_final,
        outer_g_w_tok,
        final_ln_bwd,
        layers,
        embed_bwd,
        embed_scatter_reduce,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_grad_ops(
    ctx: &GpuContext,
    p: &GpuParams,
    cfg: &TransformerConfig,
    zerob: &ZeroBufferKernel,
    adamw: &AdamWKernel,
    gn_sq: &GradNormSqReduceKernel,
    scale: &ScaleByBufferKernel,
    gn_partials_flat: &GpuBuffer,
    gn_result: &GpuBuffer,
) -> Vec<GradOps> {
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;
    let v = cfg.vocab;
    let seq_max = cfg.seq_len_max;

    let mut ops: Vec<GradOps> = Vec::with_capacity(4 + cfg.n_layer * 10);
    let mut push = |params: &GpuBuffer,
                    grad: &GpuBuffer,
                    m1: &GpuBuffer,
                    m2: &GpuBuffer,
                    n: u32,
                    apply_wd: bool| {
        let gn_offset = ops.len() as u32 * GN_GROUPS_PER_TENSOR;
        ops.push(GradOps {
            n,
            apply_wd,
            gn_offset,
            zerob_bind: zerob.build_bind(ctx, grad),
            adamw_bind: adamw.build_bind(ctx, params, grad, m1, m2),
            gn_sq_bind: gn_sq.build_bind(ctx, grad, gn_partials_flat),
            scale_bind: scale.build_bind(ctx, grad, gn_result),
        });
    };

    push(&p.w_tok, &p.g_w_tok, &p.m1_w_tok, &p.m2_w_tok, (v * d) as u32, true);
    push(&p.w_pos, &p.g_w_pos, &p.m1_w_pos, &p.m2_w_pos, (seq_max * d) as u32, false);
    push(&p.final_gamma, &p.g_final_gamma, &p.m1_final_gamma, &p.m2_final_gamma, d as u32, false);
    push(&p.final_beta, &p.g_final_beta, &p.m1_final_beta, &p.m2_final_beta, d as u32, false);
    for blk in &p.blocks {
        push(&blk.ln1_gamma, &blk.g_ln1_gamma, &blk.m1.ln1_gamma, &blk.m2.ln1_gamma, d as u32, false);
        push(&blk.ln1_beta, &blk.g_ln1_beta, &blk.m1.ln1_beta, &blk.m2.ln1_beta, d as u32, false);
        push(&blk.w_q, &blk.g_w_q, &blk.m1.w_q, &blk.m2.w_q, (d * d) as u32, true);
        push(&blk.w_k, &blk.g_w_k, &blk.m1.w_k, &blk.m2.w_k, (d * d) as u32, true);
        push(&blk.w_v, &blk.g_w_v, &blk.m1.w_v, &blk.m2.w_v, (d * d) as u32, true);
        push(&blk.w_o, &blk.g_w_o, &blk.m1.w_o, &blk.m2.w_o, (d * d) as u32, true);
        push(&blk.ln2_gamma, &blk.g_ln2_gamma, &blk.m1.ln2_gamma, &blk.m2.ln2_gamma, d as u32, false);
        push(&blk.ln2_beta, &blk.g_ln2_beta, &blk.m1.ln2_beta, &blk.m2.ln2_beta, d as u32, false);
        push(&blk.w_fc, &blk.g_w_fc, &blk.m1.w_fc, &blk.m2.w_fc, (d * d_ff) as u32, true);
        push(&blk.w_proj, &blk.g_w_proj, &blk.m1.w_proj, &blk.m2.w_proj, (d * d_ff) as u32, true);
    }
    ops
}

fn upload_block(ctx: &GpuContext, cpu: &crate::transformer::TransformerBlock) -> GpuBlock {
    let d = cpu.d_model;
    let d_ff = cpu.ffn.d_ff;
    let zeros_d = || GpuBuffer::zeros(ctx, "z", d);
    let zeros_dd = || GpuBuffer::zeros(ctx, "z", d * d);
    let zeros_dff = || GpuBuffer::zeros(ctx, "z", d * d_ff);
    let m1 = BlockAdamMoments {
        ln1_gamma: zeros_d(),
        ln1_beta: zeros_d(),
        w_q: zeros_dd(),
        w_k: zeros_dd(),
        w_v: zeros_dd(),
        w_o: zeros_dd(),
        ln2_gamma: zeros_d(),
        ln2_beta: zeros_d(),
        w_fc: zeros_dff(),
        w_proj: zeros_dff(),
    };
    let m2 = BlockAdamMoments {
        ln1_gamma: zeros_d(),
        ln1_beta: zeros_d(),
        w_q: zeros_dd(),
        w_k: zeros_dd(),
        w_v: zeros_dd(),
        w_o: zeros_dd(),
        ln2_gamma: zeros_d(),
        ln2_beta: zeros_d(),
        w_fc: zeros_dff(),
        w_proj: zeros_dff(),
    };
    GpuBlock {
        ln1_gamma: GpuBuffer::from_slice(ctx, "ln1_g", &cpu.ln1.gamma),
        ln1_beta: GpuBuffer::from_slice(ctx, "ln1_b", &cpu.ln1.beta),
        w_q: GpuBuffer::from_slice(ctx, "wq", &cpu.mha.w_q.w),
        w_k: GpuBuffer::from_slice(ctx, "wk", &cpu.mha.w_k.w),
        w_v: GpuBuffer::from_slice(ctx, "wv", &cpu.mha.w_v.w),
        w_o: GpuBuffer::from_slice(ctx, "wo", &cpu.mha.w_o.w),
        ln2_gamma: GpuBuffer::from_slice(ctx, "ln2_g", &cpu.ln2.gamma),
        ln2_beta: GpuBuffer::from_slice(ctx, "ln2_b", &cpu.ln2.beta),
        w_fc: GpuBuffer::from_slice(ctx, "wfc", &cpu.ffn.w_fc.w),
        w_proj: GpuBuffer::from_slice(ctx, "wp", &cpu.ffn.w_proj.w),
        g_ln1_gamma: zeros_d(),
        g_ln1_beta: zeros_d(),
        g_w_q: zeros_dd(),
        g_w_k: zeros_dd(),
        g_w_v: zeros_dd(),
        g_w_o: zeros_dd(),
        g_ln2_gamma: zeros_d(),
        g_ln2_beta: zeros_d(),
        g_w_fc: zeros_dff(),
        g_w_proj: zeros_dff(),
        m1,
        m2,
    }
}

impl DecoderTransformerGpu {
    /// Upload a CPU `DecoderTransformer` to the GPU. The two then run
    /// in parallel; `to_cpu()` snapshots the GPU back.
    pub fn from_cpu(cpu: &DecoderTransformer, ctx: GpuContext) -> Self {
        let cfg = cpu.config.clone();
        let v = cfg.vocab;
        let d = cfg.d_model;
        let seq_max = cfg.seq_len_max;

        let p = GpuParams {
            w_tok: GpuBuffer::from_slice(&ctx, "w_tok", &cpu.embed.w_tok),
            w_pos: GpuBuffer::from_slice(&ctx, "w_pos", &cpu.embed.w_pos),
            blocks: cpu.blocks.iter().map(|b| upload_block(&ctx, b)).collect(),
            final_gamma: GpuBuffer::from_slice(&ctx, "f_g", &cpu.final_ln.gamma),
            final_beta: GpuBuffer::from_slice(&ctx, "f_b", &cpu.final_ln.beta),

            g_w_tok: GpuBuffer::zeros(&ctx, "g_w_tok", v * d),
            g_w_pos: GpuBuffer::zeros(&ctx, "g_w_pos", seq_max * d),
            g_final_gamma: GpuBuffer::zeros(&ctx, "g_f_g", d),
            g_final_beta: GpuBuffer::zeros(&ctx, "g_f_b", d),

            m1_w_tok: GpuBuffer::zeros(&ctx, "m1_tok", v * d),
            m2_w_tok: GpuBuffer::zeros(&ctx, "m2_tok", v * d),
            m1_w_pos: GpuBuffer::zeros(&ctx, "m1_pos", seq_max * d),
            m2_w_pos: GpuBuffer::zeros(&ctx, "m2_pos", seq_max * d),
            m1_final_gamma: GpuBuffer::zeros(&ctx, "m1_fg", d),
            m2_final_gamma: GpuBuffer::zeros(&ctx, "m2_fg", d),
            m1_final_beta: GpuBuffer::zeros(&ctx, "m1_fb", d),
            m2_final_beta: GpuBuffer::zeros(&ctx, "m2_fb", d),
        };

        // 4 top-level grad tensors + 10 per block. N_GROUPS_PER_TENSOR=8.
        let total_partials = (4 + cfg.n_layer * 10) * GN_GROUPS_PER_TENSOR as usize;
        let gn_partials_flat = GpuBuffer::zeros(&ctx, "gn_partials_flat", total_partials);
        let gn_result = GpuBuffer::zeros(&ctx, "gn_result", 2);

        let zerob = ZeroBufferKernel::new(&ctx);
        let adamw = AdamWKernel::new(&ctx);
        let gn_sq = GradNormSqReduceKernel::new(&ctx);
        let gn_finalize = GradNormFinalizeKernel::new(&ctx);
        let scaleb_buf = ScaleByBufferKernel::new(&ctx);
        let gemm = GemmKernel::new(&ctx);
        let gemm_t = GemmTransposeBKernel::new(&ctx);
        let gelu_fwd = GeluKernel::new(&ctx);
        let gelu_bwd = GeluBackwardKernel::new(&ctx);
        let ln_fwd = LayerNormForwardKernel::new(&ctx);
        let ln_bwd = LayerNormBackwardKernel::new(&ctx);
        let mha_fwd = CausalMhaForwardKernel::new(&ctx);
        let mha_bwd = CausalMhaBackwardKernel::new(&ctx);
        let embed_fwd = TokenPosEmbedKernel::new(&ctx);
        let embed_bwd = EmbedScatterAddKernel::new(&ctx);
        let vadd = VectorAddKernel::new(&ctx);
        let outer_accum = OuterProductAccumKernel::new(&ctx);
        let row_reduce = RowReduceAddKernel::new(&ctx);
        let embed_scatter_reduce = EmbedScatterReduceKernel::new(&ctx);
        let single_embed = SingleTokenEmbedKernel::new(&ctx);
        let mha_kv = MhaKvCacheKernel::new(&ctx);
        let softmax_ce = SoftmaxCrossEntropyKernel::new(&ctx);

        let grad_ops =
            build_grad_ops(&ctx, &p, &cfg, &zerob, &adamw, &gn_sq, &scaleb_buf,
                           &gn_partials_flat, &gn_result);
        let gn_finalize_bind = gn_finalize.build_bind(&ctx, &gn_partials_flat, &gn_result);
        let acts = Activations::new(&ctx, &cfg);
        let fwd_ops = build_forward_ops(
            &ctx, &p, &acts, &cfg,
            &embed_fwd, &ln_fwd, &gemm_t, &mha_fwd, &gelu_fwd, &vadd,
        );
        let bwd_ops = build_backward_ops(
            &ctx, &p, &acts, &cfg,
            &gemm, &ln_bwd, &mha_bwd, &gelu_bwd, &vadd, &outer_accum,
            &embed_bwd, &embed_scatter_reduce, &zerob,
        );

        Self {
            gemm,
            gemm_t,
            gelu_fwd,
            gelu_bwd,
            ln_fwd,
            ln_bwd,
            mha_fwd,
            mha_bwd,
            embed_fwd,
            embed_bwd,
            adamw,
            vadd,
            zerob,
            outer_accum,
            row_reduce,
            gn_sq,
            gn_finalize,
            scaleb_buf,
            embed_scatter_reduce,
            single_embed,
            mha_kv,
            softmax_ce,
            gn_partials_flat,
            gn_result,
            ctx,
            config: cfg,
            step: cpu.step,
            p,
            grad_ops,
            gn_finalize_bind,
            acts,
            fwd_ops,
            bwd_ops,
        }
    }

    /// Read GPU weights back into a fresh CPU `DecoderTransformer`.
    /// Adam state is **not** transferred (CPU model keeps fresh moments).
    pub fn to_cpu(&self) -> DecoderTransformer {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);
        let mut cpu = DecoderTransformer::new(self.config.clone(), &mut rng);
        cpu.embed.w_tok = self.p.w_tok.download(&self.ctx);
        cpu.embed.w_pos = self.p.w_pos.download(&self.ctx);
        cpu.final_ln.gamma = self.p.final_gamma.download(&self.ctx);
        cpu.final_ln.beta = self.p.final_beta.download(&self.ctx);
        for (li, blk) in cpu.blocks.iter_mut().enumerate() {
            let gb = &self.p.blocks[li];
            blk.ln1.gamma = gb.ln1_gamma.download(&self.ctx);
            blk.ln1.beta = gb.ln1_beta.download(&self.ctx);
            blk.mha.w_q.w = gb.w_q.download(&self.ctx);
            blk.mha.w_k.w = gb.w_k.download(&self.ctx);
            blk.mha.w_v.w = gb.w_v.download(&self.ctx);
            blk.mha.w_o.w = gb.w_o.download(&self.ctx);
            blk.ln2.gamma = gb.ln2_gamma.download(&self.ctx);
            blk.ln2.beta = gb.ln2_beta.download(&self.ctx);
            blk.ffn.w_fc.w = gb.w_fc.download(&self.ctx);
            blk.ffn.w_proj.w = gb.w_proj.download(&self.ctx);
        }
        cpu.step = self.step;
        cpu
    }

    /// Zero all gradient buffers. One compute pass, cached bind groups.
    pub fn zero_grad(&self) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero_grad") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_grad pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.zerob.pipeline);
            for op in &self.grad_ops {
                self.zerob.dispatch_with_bind(&mut pass, &op.zerob_bind, op.n);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Compute global L2 norm of all grad tensors. Pure-GPU pipeline:
    /// per-tensor partial-sum-of-squares → one finalize kernel → 1 scalar
    /// download. Uses cached bind groups (no per-call allocations).
    pub fn grad_norm(&self) -> f32 {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grad_norm") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gn_sq pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_sq.pipeline);
            for op in &self.grad_ops {
                self.gn_sq.dispatch_with_bind(
                    &mut pass, &op.gn_sq_bind, op.n, op.gn_offset, GN_GROUPS_PER_TENSOR,
                );
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gn_finalize pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_finalize.pipeline);
            let n_partials = self.grad_ops.len() as u32 * GN_GROUPS_PER_TENSOR;
            self.gn_finalize
                .dispatch_with_bind(&mut pass, &self.gn_finalize_bind, n_partials, f32::INFINITY);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        let r = self.gn_result.download(&self.ctx);
        r[0]
    }

    /// Clip global gradient norm to `max_norm`. Pure-GPU; cached bind
    /// groups. Three passes: gn_sq (concurrent) → gn_finalize → scale
    /// (concurrent). One ~8-byte download at the end for the logged
    /// pre-clip norm.
    pub fn clip_grads(&self, max_norm: f32) -> f32 {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clip") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip gn_sq"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_sq.pipeline);
            for op in &self.grad_ops {
                self.gn_sq.dispatch_with_bind(
                    &mut pass, &op.gn_sq_bind, op.n, op.gn_offset, GN_GROUPS_PER_TENSOR,
                );
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_finalize.pipeline);
            let n_partials = self.grad_ops.len() as u32 * GN_GROUPS_PER_TENSOR;
            self.gn_finalize.dispatch_with_bind(
                &mut pass, &self.gn_finalize_bind, n_partials, max_norm,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scaleb_buf.pipeline);
            for op in &self.grad_ops {
                self.scaleb_buf.dispatch_with_bind(&mut pass, &op.scale_bind, op.n, 1);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
        let r = self.gn_result.download(&self.ctx);
        r[0]
    }

    /// Encode the forward pass (embed → N blocks → final LN → tied LM
    /// head) into `enc`. Writes logits into `self.acts.logits_buf` at
    /// offset 0, length `t * vocab`. Caller must have uploaded tokens
    /// to `self.acts.tok_buf` first. Used by both `train_step` and
    /// `eval_step` so the two paths share the exact same dispatch chain.
    fn encode_forward(&self, enc: &mut wgpu::CommandEncoder, t: usize) {
        let cfg = &self.config;
        let d = cfg.d_model;
        let v = cfg.vocab;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        let t_u = t as u32;
        let d_u = d as u32;
        let d_ff_u = d_ff as u32;

        // Embed (1 pass).
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.embed_fwd.pipeline);
            self.embed_fwd.dispatch_with_bind(
                &mut pass, &self.fwd_ops.embed,
                t_u, d_u, v as u32, cfg.seq_len_max as u32,
            );
        }

        for li in 0..cfg.n_layer {
            let fop = &self.fwd_ops.layers[li];

            // ln1
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ln1"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.ln_fwd.pipeline);
                self.ln_fwd.dispatch_with_bind(&mut pass, &fop.ln1, t_u, d_u, 1e-5);
            }
            // QKV — independent gemm_t writes; batch into one pass (1.2).
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("qkv"), timestamp_writes: None,
                });
                pass.set_pipeline(self.gemm_t.pipeline());
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_q, t_u, d_u, d_u);
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_k, t_u, d_u, d_u);
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_v, t_u, d_u, d_u);
            }
            // mha
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mha"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.mha_fwd.pipeline);
                self.mha_fwd.dispatch_with_bind(
                    &mut pass, &fop.mha, t_u, n_head as u32, d_head as u32,
                );
            }
            // W_O
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("wo"), timestamp_writes: None,
                });
                pass.set_pipeline(self.gemm_t.pipeline());
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_o, t_u, d_u, d_u);
            }
            // residual: post_attn = x + mha_out
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("res1"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.vadd.pipeline);
                self.vadd.dispatch_with_bind(&mut pass, &fop.vadd_post_attn, (t * d) as u32);
            }
            // ln2
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ln2"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.ln_fwd.pipeline);
                self.ln_fwd.dispatch_with_bind(&mut pass, &fop.ln2, t_u, d_u, 1e-5);
            }
            // fc + gelu + proj (each in its own pass — sequential deps).
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fc"), timestamp_writes: None,
                });
                pass.set_pipeline(self.gemm_t.pipeline());
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_fc, t_u, d_ff_u, d_u);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gelu"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.gelu_fwd.pipeline);
                self.gelu_fwd.dispatch_with_bind(&mut pass, &fop.gelu, (t * d_ff) as u32);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("proj"), timestamp_writes: None,
                });
                pass.set_pipeline(self.gemm_t.pipeline());
                self.gemm_t.dispatch_with_bind(&mut pass, &fop.gemm_proj, t_u, d_u, d_ff_u);
            }
            // z = post_attn + ffn_out
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("res2"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.vadd.pipeline);
                self.vadd.dispatch_with_bind(&mut pass, &fop.vadd_z, (t * d) as u32);
            }
        }

        // Final LN
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("final ln"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.ln_fwd.pipeline);
            self.ln_fwd.dispatch_with_bind(&mut pass, &self.fwd_ops.final_ln, t_u, d_u, 1e-5);
        }
        // Tied LM head (logits = h_final × w_tokᵀ)
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lm head"), timestamp_writes: None,
            });
            pass.set_pipeline(self.gemm_t.pipeline());
            self.gemm_t.dispatch_with_bind(&mut pass, &self.fwd_ops.final_gemm, t_u, v as u32, d_u);
        }
    }

    /// Full forward + backward + loss computation. Single-sequence
    /// version (B=1) to keep state minimal. Caller is expected to
    /// reset gradients via `zero_grad()` between minibatches.
    ///
    /// Activation buffers come from `self.acts` and bind groups from
    /// `self.fwd_ops` / `self.bwd_ops` — all built once at construction.
    /// Loss + d_logits are computed on the GPU by `softmax_ce` in the
    /// same encoder as fwd/bwd; only T floats (per-token NLL) come back
    /// to the host. Returns the per-token mean cross-entropy loss.
    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> f32 {
        let cfg = self.config.clone();
        let t = tokens.len();
        let d = cfg.d_model;
        let v = cfg.vocab;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        let t_u = t as u32;
        let d_u = d as u32;
        debug_assert_eq!(targets.len(), t);

        self.acts.tok_buf.upload_u32_prefix(&self.ctx, tokens);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd") });

        self.encode_forward(&mut enc, t);

        // GPU softmax + cross-entropy: writes per-token NLL into
        // `loss_seq_buf` (length T) and pre-scaled gradient (probs −
        // one_hot)/T into `d_logits_buf`. Eliminates the T·V logits
        // download + per-token CPU loss + T·V d_logits upload that the
        // earlier path paid every step.
        self.acts.targets_buf.upload_u32_prefix(&self.ctx, targets);
        let loss_scale = 1.0_f32 / t as f32;
        self.softmax_ce.dispatch_into(
            &self.ctx,
            &mut enc,
            &self.acts.logits_buf,
            &self.acts.d_logits_buf,
            &self.acts.loss_seq_buf,
            &self.acts.targets_buf,
            v as u32,
            t_u,
            1,
            loss_scale,
        );

        // d_h_final = d_logits × w_tok  AND  outer_accum into g_w_tok.
        // The two reads of d_logits are concurrent (read-only) so we
        // can put them in one pass.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lm head bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm.pipeline);
            self.gemm.dispatch_with_bind(
                &mut pass, &self.bwd_ops.gemm_d_h_final, t_u, d_u, v as u32,
            );
            pass.set_pipeline(&self.outer_accum.pipeline);
            self.outer_accum.dispatch_with_bind(
                &mut pass, &self.bwd_ops.outer_g_w_tok, v as u32, d_u, t_u,
            );
        }
        // Final LN bwd
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("final ln bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.ln_bwd.pipeline);
            self.ln_bwd.dispatch_with_bind(
                &mut pass, &self.bwd_ops.final_ln_bwd, t_u, d_u,
            );
        }

        // Block backward chain.
        for li in (0..cfg.n_layer).rev() {
            self.block_backward(&mut enc, li, t, d, n_head, d_head, d_ff);
        }

        // Embed bwd: per-pos grad + scatter-reduce to g_w_tok / g_w_pos.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.embed_bwd.pipeline);
            self.embed_bwd.dispatch_with_bind(
                &mut pass, &self.bwd_ops.embed_bwd,
                t_u, d_u, v as u32, cfg.seq_len_max as u32,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed scatter"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.embed_scatter_reduce.pipeline);
            self.embed_scatter_reduce.dispatch_with_bind(
                &mut pass, &self.bwd_ops.embed_scatter_reduce, t_u, d_u,
            );
        }
        self.ctx.queue.submit(Some(enc.finish()));

        // Per-token NLL was written by softmax_ce; reduce on host.
        // Download is T floats (small) instead of the legacy T·V logits.
        let loss_per_t = self.acts.loss_seq_buf.download_prefix(&self.ctx, t);
        loss_per_t.iter().sum::<f32>() / t as f32
    }

    /// Forward-only path for validation. Encodes the same forward
    /// dispatch chain as `train_step`, runs GPU softmax_ce to produce
    /// per-token NLL, downloads T floats and returns their **sum**
    /// (not mean — caller aggregates across windows). `d_logits_buf`
    /// is still bound by softmax_ce but its contents are discarded.
    ///
    /// `&self` (no mutation): safe to call any number of times between
    /// train steps without disturbing optimizer / gradient state.
    pub fn eval_step(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let cfg = self.config.clone();
        let t = tokens.len();
        let v = cfg.vocab;
        let t_u = t as u32;
        debug_assert_eq!(targets.len(), t);

        self.acts.tok_buf.upload_u32_prefix(&self.ctx, tokens);
        self.acts.targets_buf.upload_u32_prefix(&self.ctx, targets);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eval fwd"),
            });

        self.encode_forward(&mut enc, t);

        // loss_scale is irrelevant for eval (d_logits not used), but
        // softmax_ce still writes it; pass 1.0 to keep raw probabilities.
        self.softmax_ce.dispatch_into(
            &self.ctx,
            &mut enc,
            &self.acts.logits_buf,
            &self.acts.d_logits_buf,
            &self.acts.loss_seq_buf,
            &self.acts.targets_buf,
            v as u32,
            t_u,
            1,
            1.0,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        let loss_per_t = self.acts.loss_seq_buf.download_prefix(&self.ctx, t);
        loss_per_t.iter().sum()
    }

    fn block_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        block_idx: usize,
        t: usize,
        d: usize,
        n_head: usize,
        d_head: usize,
        d_ff: usize,
    ) {
        let bop = &self.bwd_ops.layers[block_idx];
        let t_u = t as u32;
        let d_u = d as u32;
        let d_ff_u = d_ff as u32;
        let td = (t * d) as u32;
        let td_ff = (t * d_ff) as u32;

        // FFN backward chain. gemm + outer_accum share `d_z` (read-only)
        // and write disjoint outputs — safe in one pass.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bwd ffn proj"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm.pipeline);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_d_h_post, t_u, d_ff_u, d_u);
            pass.set_pipeline(&self.outer_accum.pipeline);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_proj, d_u, d_ff_u, t_u);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gelu bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gelu_bwd.pipeline);
            self.gelu_bwd.dispatch_with_bind(&mut pass, &bop.gelu_bwd, td_ff);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bwd fc"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm.pipeline);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_d_ln2, t_u, d_u, d_ff_u);
            pass.set_pipeline(&self.outer_accum.pipeline);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_fc, d_ff_u, d_u, t_u);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln2 bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.ln_bwd.pipeline);
            self.ln_bwd.dispatch_with_bind(&mut pass, &bop.ln2_bwd, t_u, d_u);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("d_post_attn"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.vadd.pipeline);
            self.vadd.dispatch_with_bind(&mut pass, &bop.vadd_d_post_attn, td);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bwd o"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm.pipeline);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_d_heads_out, t_u, d_u, d_u);
            pass.set_pipeline(&self.outer_accum.pipeline);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_o, d_u, d_u, t_u);
        }

        // Zero dq/dk/dv in one pass (disjoint outputs, no dep).
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero dqkv"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.zerob.pipeline);
            self.zerob.dispatch_with_bind(&mut pass, &bop.zerob_dq, td);
            self.zerob.dispatch_with_bind(&mut pass, &bop.zerob_dk, td);
            self.zerob.dispatch_with_bind(&mut pass, &bop.zerob_dv, td);
        }
        // mha bwd (3 internal sequential passes due to d_scores dep).
        self.mha_bwd.dispatch_with_binds(
            enc, &bop.mha_bwd_dscores, &bop.mha_bwd_dq, &bop.mha_bwd_dkv,
            t_u, n_head as u32, d_head as u32,
        );

        // dx_q, dx_k, dx_v — all read distinct dq/dk/dv, write disjoint
        // dx_q/k/v. Safe in one pass.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bwd dx qkv"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm.pipeline);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_dx_q, t_u, d_u, d_u);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_dx_k, t_u, d_u, d_u);
            self.gemm.dispatch_with_bind(&mut pass, &bop.gemm_dx_v, t_u, d_u, d_u);
        }
        // outer_accum for w_q, w_k, w_v — disjoint outputs.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bwd g_w_qkv"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.outer_accum.pipeline);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_q, d_u, d_u, t_u);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_k, d_u, d_u, t_u);
            self.outer_accum.dispatch_with_bind(&mut pass, &bop.outer_g_v, d_u, d_u, t_u);
        }
        // d_ln1_qk = dx_q + dx_k, then d_ln1_out = d_ln1_qk + dx_v.
        // Sequential — separate passes.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vadd qk"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.vadd.pipeline);
            self.vadd.dispatch_with_bind(&mut pass, &bop.vadd_qk, td);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vadd qkv"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.vadd.pipeline);
            self.vadd.dispatch_with_bind(&mut pass, &bop.vadd_qkv, td);
        }
        // ln1 bwd
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln1 bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.ln_bwd.pipeline);
            self.ln_bwd.dispatch_with_bind(&mut pass, &bop.ln1_bwd, t_u, d_u);
        }
        // Final residual: dx = d_post_attn + d_x_mha. Writes a.dx which
        // becomes d_z for the next (earlier) block's backward.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vadd dx"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.vadd.pipeline);
            self.vadd.dispatch_with_bind(&mut pass, &bop.vadd_dx, td);
        }
    }

    /// Apply AdamW step to all parameter tensors. Single compute pass,
    /// cached bind groups.
    pub fn step_adamw(&mut self, config: &AdamWConfig) {
        self.step += 1;
        let (bc1, bc2) = bias_correction(self.step as usize, config.beta1, config.beta2);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora adamw step"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adamw pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adamw.pipeline);
            for op in &self.grad_ops {
                self.adamw.dispatch_with_bind(
                    &mut pass,
                    &op.adamw_bind,
                    op.n,
                    op.apply_wd,
                    config.lr,
                    config.beta1,
                    config.beta2,
                    config.eps,
                    bc1,
                    bc2,
                    config.weight_decay,
                );
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Populate a fresh KV cache from a seed sequence. Runs full GPU
    /// forward, extracts each block's K/V buffers, and copies them into
    /// the cache's history buffers at offset 0.
    pub fn prefill_kv(&self, tokens: &[u32]) -> KvCacheGpu {
        let cfg = self.config.clone();
        let kv = self.forward_kv_only(tokens);
        let mut cache = KvCacheGpu::new(&self.ctx, cfg.n_layer, cfg.d_model, cfg.seq_len_max);
        let bytes = (tokens.len() * cfg.d_model * std::mem::size_of::<f32>()) as u64;

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prefill_kv copy"),
            });
        for (li, (k_buf, v_buf)) in kv.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &k_buf.inner,
                0,
                &cache.blocks[li].k_history.inner,
                0,
                bytes,
            );
            enc.copy_buffer_to_buffer(
                &v_buf.inner,
                0,
                &cache.blocks[li].v_history.inner,
                0,
                bytes,
            );
        }
        self.ctx.queue.submit(Some(enc.finish()));
        cache.current_len = tokens.len();
        cache
    }

    /// Single-position forward through the full stack using the cache.
    /// Appends K/V for the new token + computes attention against full
    /// history. Returns logits Vec<f32> of length vocab.
    pub fn generate_step_kv(&self, new_token: u32, cache: &mut KvCacheGpu) -> Vec<f32> {
        let cfg = self.config.clone();
        let d = cfg.d_model;
        let v_dim = cfg.vocab;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        let pos = cache.current_len;
        assert!(pos < cache.max_len, "KV cache full ({}≥{})", pos, cache.max_len);

        let h_cur = GpuBuffer::empty(&self.ctx, "step_h", d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        self.single_embed.dispatch_into(
            &self.ctx, &mut enc, &self.p.w_tok, &self.p.w_pos, &h_cur,
            d as u32, v_dim as u32, cfg.seq_len_max as u32, new_token, pos as u32,
        );

        let mha_scratch = GpuBuffer::empty(&self.ctx, "step_scratch", n_head * cache.max_len);

        let mut h = h_cur;
        let t_total_after = pos + 1;
        for li in 0..cfg.n_layer {
            let blk = &self.p.blocks[li];
            let kv = &cache.blocks[li];

            let ln1_out = GpuBuffer::empty(&self.ctx, "ln1_out", d);
            let ln1_mean = GpuBuffer::empty(&self.ctx, "ln1_mean", 1);
            let ln1_inv = GpuBuffer::empty(&self.ctx, "ln1_inv", 1);
            let ln1_xhat = GpuBuffer::empty(&self.ctx, "ln1_xhat", d);
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &h, &blk.ln1_gamma, &blk.ln1_beta, &ln1_out, &ln1_mean,
                &ln1_inv, &ln1_xhat, 1, d as u32, 1e-5,
            );

            let q_new = GpuBuffer::empty(&self.ctx, "step_q", d);
            let k_new = GpuBuffer::empty(&self.ctx, "step_k", d);
            let v_new = GpuBuffer::empty(&self.ctx, "step_v", d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &v_new, 1, d as u32, d as u32);
            let row_bytes = (d * std::mem::size_of::<f32>()) as u64;
            let dst_offset = (pos * d * std::mem::size_of::<f32>()) as u64;
            enc.copy_buffer_to_buffer(&k_new.inner, 0, &kv.k_history.inner, dst_offset, row_bytes);
            enc.copy_buffer_to_buffer(&v_new.inner, 0, &kv.v_history.inner, dst_offset, row_bytes);

            let heads_out = GpuBuffer::empty(&self.ctx, "step_heads", d);
            self.mha_kv.dispatch_into(
                &self.ctx, &mut enc, &q_new, &kv.k_history, &kv.v_history,
                &heads_out, &mha_scratch,
                t_total_after as u32, n_head as u32, d_head as u32, cache.max_len as u32,
            );

            let mha_out = GpuBuffer::empty(&self.ctx, "step_mha_out", d);
            let post_attn = GpuBuffer::empty(&self.ctx, "step_post_attn", d);
            let ln2_out = GpuBuffer::empty(&self.ctx, "step_ln2_out", d);
            let ln2_mean = GpuBuffer::empty(&self.ctx, "step_ln2_mean", 1);
            let ln2_inv = GpuBuffer::empty(&self.ctx, "step_ln2_inv", 1);
            let ln2_xhat = GpuBuffer::empty(&self.ctx, "step_ln2_xhat", d);
            let h_pre = GpuBuffer::empty(&self.ctx, "step_h_pre", d_ff);
            let h_post = GpuBuffer::empty(&self.ctx, "step_h_post", d_ff);
            let ffn_out = GpuBuffer::empty(&self.ctx, "step_ffn_out", d);
            let z = GpuBuffer::empty(&self.ctx, "step_z", d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &heads_out, &blk.w_o, &mha_out, 1, d as u32, d as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &h, &mha_out, &post_attn, d as u32);
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &post_attn, &blk.ln2_gamma, &blk.ln2_beta, &ln2_out,
                &ln2_mean, &ln2_inv, &ln2_xhat, 1, d as u32, 1e-5,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln2_out, &blk.w_fc, &h_pre, 1, d_ff as u32, d as u32);
            self.gelu_fwd.dispatch_into(&self.ctx, &mut enc, &h_pre, &h_post, d_ff as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &h_post, &blk.w_proj, &ffn_out, 1, d as u32, d_ff as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &post_attn, &ffn_out, &z, d as u32);

            h = z;
        }

        let h_final = GpuBuffer::empty(&self.ctx, "step_h_final", d);
        let f_mean = GpuBuffer::empty(&self.ctx, "step_f_mean", 1);
        let f_inv = GpuBuffer::empty(&self.ctx, "step_f_inv", 1);
        let f_xhat = GpuBuffer::empty(&self.ctx, "step_f_xhat", d);
        let logits = GpuBuffer::empty(&self.ctx, "step_logits", v_dim);
        self.ln_fwd.dispatch_into(
            &self.ctx, &mut enc, &h, &self.p.final_gamma, &self.p.final_beta, &h_final,
            &f_mean, &f_inv, &f_xhat, 1, d as u32, 1e-5,
        );
        self.gemm_t.dispatch_into(
            &self.ctx, &mut enc, &h_final, &self.p.w_tok, &logits, 1, v_dim as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        cache.current_len = t_total_after;
        logits.download(&self.ctx)
    }

    /// Forward-only path; returns per-block (K, V) GPU buffers. Used by
    /// `prefill_kv`. Identical to the forward portion of `train_step`
    /// up through and including each block's MHA projections; the
    /// remaining downstream work (output proj, residuals, FFN) is still
    /// run to keep block-to-block state consistent, but the returned
    /// vec only points at the K/V buffers.
    fn forward_kv_only(&self, tokens: &[u32]) -> Vec<(GpuBuffer, GpuBuffer)> {
        let cfg = self.config.clone();
        let t = tokens.len();
        let d = cfg.d_model;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        let v = cfg.vocab;

        let tok_buf = GpuBuffer::from_u32_slice(&self.ctx, "tok", tokens);
        let h0 = GpuBuffer::empty(&self.ctx, "h0", t * d);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prefill embed"),
            });
        self.embed_fwd.dispatch_into(
            &self.ctx, &mut enc, &tok_buf, &self.p.w_tok, &self.p.w_pos, &h0,
            t as u32, d as u32, v as u32, cfg.seq_len_max as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        let mut x = h0;
        let mut kv_per_block: Vec<(GpuBuffer, GpuBuffer)> = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            let blk = &self.p.blocks[li];

            let ln1_out = GpuBuffer::empty(&self.ctx, "ln1_out", t * d);
            let ln1_mean = GpuBuffer::empty(&self.ctx, "ln1_mean", t);
            let ln1_inv = GpuBuffer::empty(&self.ctx, "ln1_inv", t);
            let ln1_xhat = GpuBuffer::empty(&self.ctx, "ln1_xhat", t * d);
            let q = GpuBuffer::empty(&self.ctx, "q", t * d);
            let k = GpuBuffer::empty(&self.ctx, "k", t * d);
            let val = GpuBuffer::empty(&self.ctx, "v", t * d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prefill block fwd"),
                });
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &x, &blk.ln1_gamma, &blk.ln1_beta, &ln1_out, &ln1_mean,
                &ln1_inv, &ln1_xhat, t as u32, d as u32, 1e-5,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &val, t as u32, d as u32, d as u32);

            // Heads_out via single MHA dispatch — n_head heads collapsed into wg.y.
            let heads_out = GpuBuffer::empty(&self.ctx, "heads", t * d);
            let pb = GpuBuffer::empty(&self.ctx, "probs", n_head * t * t);
            self.mha_fwd.dispatch_into(
                &self.ctx, &mut enc, &q, &k, &val, &heads_out, &pb,
                t as u32, n_head as u32, d_head as u32,
            );
            let _ = pb;

            let mha_out = GpuBuffer::empty(&self.ctx, "mha_out", t * d);
            let post_attn = GpuBuffer::empty(&self.ctx, "post_attn", t * d);
            let ln2_out = GpuBuffer::empty(&self.ctx, "ln2_out", t * d);
            let ln2_mean = GpuBuffer::empty(&self.ctx, "ln2_mean", t);
            let ln2_inv = GpuBuffer::empty(&self.ctx, "ln2_inv", t);
            let ln2_xhat = GpuBuffer::empty(&self.ctx, "ln2_xhat", t * d);
            let h_pre = GpuBuffer::empty(&self.ctx, "h_pre", t * d_ff);
            let h_post = GpuBuffer::empty(&self.ctx, "h_post", t * d_ff);
            let ffn_out = GpuBuffer::empty(&self.ctx, "ffn_out", t * d);
            let z = GpuBuffer::empty(&self.ctx, "z", t * d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &heads_out, &blk.w_o, &mha_out, t as u32, d as u32, d as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &x, &mha_out, &post_attn, (t * d) as u32);
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &post_attn, &blk.ln2_gamma, &blk.ln2_beta, &ln2_out,
                &ln2_mean, &ln2_inv, &ln2_xhat, t as u32, d as u32, 1e-5,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln2_out, &blk.w_fc, &h_pre, t as u32, d_ff as u32, d as u32);
            self.gelu_fwd.dispatch_into(&self.ctx, &mut enc, &h_pre, &h_post, (t * d_ff) as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &h_post, &blk.w_proj, &ffn_out, t as u32, d as u32, d_ff as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &post_attn, &ffn_out, &z, (t * d) as u32);
            self.ctx.queue.submit(Some(enc.finish()));

            kv_per_block.push((k, val));
            x = z;
        }
        kv_per_block
    }
}

