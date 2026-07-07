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
    AdamWKernel, CanonDwKernel, CanonDxKernel, CanonForwardKernel, CausalMhaBackwardKernel,
    CausalMhaForwardKernel, EmbedScatterAddKernel,
    EmbedScatterReduceKernel, GeluBackwardKernel, GeluKernel, GemmKernel, GemmTransposeBKernel,
    GpuContext, GradNormFinalizeKernel, GradNormSqReduceKernel, LayerNormBackwardKernel,
    LayerNormForwardKernel, MuonApplyKernel, MuonCombineBKernel, MuonFrobInvKernel,
    MuonLoadStoreKernel, MuonMomentumKernel, MuonStepXKernel, OuterProductAccumKernel,
    QkNormBackwardKernel, QkNormKernel, RopeKernel, RowReduceAddKernel, ScaleByBufferKernel,
    SoftcapBackwardKernel, SoftcapKernel, SoftmaxCrossEntropyKernel, TokenPosEmbedKernel,
    VectorAddKernel, ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig, MuonConfig};
use crate::transformer::{DecoderTransformer, TransformerConfig};

const GN_GROUPS_PER_TENSOR: u32 = 8;

pub struct DecoderTransformerGpu {
    pub ctx: GpuContext,
    pub config: TransformerConfig,
    pub step: u64,
    /// Capacity of the shared B·T token/target/loss buffers — the max
    /// number of sequences `train_step_batch` / `eval_step_batch` can
    /// take per call.
    pub max_batch: usize,

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
    pub muon_momentum: MuonMomentumKernel,
    pub muon_frob_inv: MuonFrobInvKernel,
    pub muon_load_store: MuonLoadStoreKernel,
    pub muon_combine_b: MuonCombineBKernel,
    pub muon_step_x: MuonStepXKernel,
    pub muon_apply: MuonApplyKernel,
    pub embed_scatter_reduce: EmbedScatterReduceKernel,
    pub single_embed: SingleTokenEmbedKernel,
    pub mha_kv: MhaKvCacheKernel,
    pub softmax_ce: SoftmaxCrossEntropyKernel,
    pub rope: RopeKernel,
    pub qknorm_fwd: QkNormKernel,
    pub qknorm_bwd: QkNormBackwardKernel,
    pub softcap_fwd: SoftcapKernel,
    pub softcap_bwd: SoftcapBackwardKernel,
    pub canon_fwd: CanonForwardKernel,
    pub canon_dx: CanonDxKernel,
    pub canon_dw: CanonDwKernel,

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
    // Zeroes `gn_partials_flat` on-GPU at the start of the subset clip so
    // finalize (which sums the whole buffer) never picks up stale
    // matrix-slot partials — replaces a per-step host zero-fill + upload.
    pub(crate) gn_partials_zero_bind: wgpu::BindGroup,

    // Muon hybrid step: scratch buffers (sized for the largest matrix),
    // per-matrix cached bind groups, and one shared set of bind groups
    // for the scratch-only stages. `muon_scratch` owns the buffers that
    // `muon_shared` bind groups reference — the compiler can't see the
    // indirect ownership, hence the allow.
    #[allow(dead_code)]
    pub(crate) muon_scratch: MuonScratchGpu,
    pub(crate) muon_ops: Vec<MuonOps>,
    pub(crate) muon_shared: MuonSharedOps,

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
    /// In-place softcap on the logits buffer (None when softcap == 0).
    pub softcap: Option<wgpu::BindGroup>,
}

pub(crate) struct LayerForwardOps {
    pub ln1: wgpu::BindGroup,
    pub gemm_q: wgpu::BindGroup,
    pub gemm_k: wgpu::BindGroup,
    pub gemm_v: wgpu::BindGroup,
    /// q_raw → q / k_raw → k RMS row norm (None when qk_norm off; the
    /// QKV gemms then write q/k directly).
    pub qknorm_q: Option<wgpu::BindGroup>,
    pub qknorm_k: Option<wgpu::BindGroup>,
    pub rope_q: wgpu::BindGroup,
    pub rope_k: wgpu::BindGroup,
    pub mha: wgpu::BindGroup,
    pub gemm_o: wgpu::BindGroup,
    pub vadd_post_attn: wgpu::BindGroup,
    pub ln2: wgpu::BindGroup,
    pub gemm_fc: wgpu::BindGroup,
    pub gelu: wgpu::BindGroup,
    pub gemm_proj: wgpu::BindGroup,
    pub vadd_z: wgpu::BindGroup,
    /// Canon conv `x → canon_out` (None when canon off). ln1 + post_attn
    /// residual read canon_out instead of the raw block input.
    pub canon_fwd: Option<wgpu::BindGroup>,
}

pub(crate) struct BackwardOps {
    pub gemm_d_h_final: wgpu::BindGroup,
    /// Accumulates the LM-head weight grad — into g_w_head when untied,
    /// else into g_w_tok (tied).
    pub outer_g_w_tok: wgpu::BindGroup,
    pub final_ln_bwd: wgpu::BindGroup,
    pub layers: Vec<LayerBackwardOps>,
    pub embed_bwd: wgpu::BindGroup,
    pub embed_scatter_reduce: wgpu::BindGroup,
    /// dq/dk are shared backward scratch, so one inverse-rope bind each
    /// serves every layer.
    pub rope_dq: wgpu::BindGroup,
    pub rope_dk: wgpu::BindGroup,
    /// d_logits *= 1 − tanh² chain factor (None when softcap == 0).
    pub softcap_bwd: Option<wgpu::BindGroup>,
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
    /// In-place qk-norm backward on dq/dk against this layer's cached
    /// q_raw/k_raw (None when qk_norm off).
    pub qknorm_dq: Option<wgpu::BindGroup>,
    pub qknorm_dk: Option<wgpu::BindGroup>,
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
    /// Canon backward (None when off): `vadd_dx` writes `canon_din`, then
    /// canon_dw accumulates g_canon_w and canon_dx maps canon_din → dx.
    pub canon_dw: Option<wgpu::BindGroup>,
    pub canon_dx: Option<wgpu::BindGroup>,
}

pub(crate) struct GradOps {
    pub n: u32,
    pub apply_wd: bool,
    /// `true` for the 6 matrix params per block that Muon owns in
    /// hybrid mode (Q/K/V/O, FC, Proj). Used by `step_muon_hybrid` to
    /// skip the AdamW dispatch for matrix params and by
    /// `clip_grads_adamw_subset` to exclude them from the clip norm.
    pub is_matrix: bool,
    /// Embedding/head group (w_tok/w_pos/w_head): its AdamW LR is scaled
    /// by `config.embed_lr_mult`.
    pub embed: bool,
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
    /// Pre-qk-norm Q/K, kept for the norm's backward (1-element dummies
    /// when qk_norm is off).
    pub q_raw: GpuBuffer,
    pub k_raw: GpuBuffer,
    pub heads_out: GpuBuffer,
    pub head_probs: GpuBuffer,
    pub ln2_out: GpuBuffer,
    pub ln2_xhat: GpuBuffer,
    pub ln2_inv: GpuBuffer,
    pub h_pre: GpuBuffer,
    pub h_post: GpuBuffer,
    /// Canon-mixed block input `x + dwconv(x)` (1-elem dummy when off).
    pub canon_out: GpuBuffer,
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
    /// Canon backward scratch: d w.r.t. the canon-mixed input (1-elem
    /// dummy when canon off). canon_dx then maps it to `dx`.
    pub canon_din: GpuBuffer,
}

impl Activations {
    fn new(ctx: &GpuContext, cfg: &TransformerConfig, max_batch: usize) -> Self {
        let t = cfg.seq_len_max;
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        let n_head = cfg.n_head;
        let v = cfg.vocab;
        let td = t * d;
        let td_ff = t * d_ff;
        let probs_cap = n_head * t * t;
        // Token / target / per-token-loss buffers hold the whole batch
        // (B sequences laid out contiguously) so `train_step_batch` can
        // run all sequences in one submission with a single readback.
        let bt = max_batch.max(1) * t;

        let qk_raw_len = if cfg.qk_norm { td } else { 1 };
        let canon_len = if cfg.canon_kernel > 0 { td } else { 1 };
        let layers: Vec<LayerCache> = (0..cfg.n_layer)
            .map(|_| LayerCache {
                ln1_out: GpuBuffer::empty(ctx, "ln1_out", td),
                ln1_xhat: GpuBuffer::empty(ctx, "ln1_xhat", td),
                ln1_inv: GpuBuffer::empty(ctx, "ln1_inv", t),
                q: GpuBuffer::empty(ctx, "q", td),
                k: GpuBuffer::empty(ctx, "k", td),
                v: GpuBuffer::empty(ctx, "v", td),
                q_raw: GpuBuffer::empty(ctx, "q_raw", qk_raw_len),
                k_raw: GpuBuffer::empty(ctx, "k_raw", qk_raw_len),
                heads_out: GpuBuffer::empty(ctx, "heads", td),
                head_probs: GpuBuffer::empty(ctx, "probs", probs_cap),
                ln2_out: GpuBuffer::empty(ctx, "ln2_out", td),
                ln2_xhat: GpuBuffer::empty(ctx, "ln2_xhat", td),
                ln2_inv: GpuBuffer::empty(ctx, "ln2_inv", t),
                h_pre: GpuBuffer::empty(ctx, "h_pre", td_ff),
                h_post: GpuBuffer::empty(ctx, "h_post", td_ff),
                canon_out: GpuBuffer::empty(ctx, "canon_out", canon_len),
            })
            .collect();
        let z_chain: Vec<GpuBuffer> = (0..cfg.n_layer)
            .map(|_| GpuBuffer::empty(ctx, "z", td))
            .collect();

        Self {
            layers,
            z_chain,
            tok_buf: GpuBuffer::empty(ctx, "tok", bt),
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
            loss_seq_buf: GpuBuffer::empty(ctx, "loss_seq", bt),
            targets_buf: GpuBuffer::empty(ctx, "targets", bt),
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
            canon_din: GpuBuffer::empty(ctx, "canon_din", canon_len),
        }
    }
}

/// Device-side snapshot of all persistent model state (weights + AdamW
/// moments + Muon momentum), captured on-device for the PHAGE governor's
/// reflex rollback and paired-rollout trials. Buffer order mirrors
/// `DecoderTransformerGpu::state_buffers`.
pub struct GpuStateSnapshot {
    bufs: Vec<GpuBuffer>,
    step: u64,
}

pub struct GpuParams {
    pub w_tok: GpuBuffer,
    pub w_pos: GpuBuffer,
    /// Untied LM head (None = tied to w_tok), plus its grad and moments.
    pub w_head: Option<GpuBuffer>,
    pub blocks: Vec<GpuBlock>,
    pub final_gamma: GpuBuffer,
    pub final_beta: GpuBuffer,

    pub g_w_tok: GpuBuffer,
    pub g_w_pos: GpuBuffer,
    pub g_w_head: Option<GpuBuffer>,
    pub g_final_gamma: GpuBuffer,
    pub g_final_beta: GpuBuffer,

    pub m1_w_tok: GpuBuffer,
    pub m2_w_tok: GpuBuffer,
    pub m1_w_pos: GpuBuffer,
    pub m2_w_pos: GpuBuffer,
    pub m1_w_head: Option<GpuBuffer>,
    pub m2_w_head: Option<GpuBuffer>,
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
    /// Canon conv taps `[d_model * canon_k]` (1-elem dummy when off).
    pub canon_w: GpuBuffer,

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
    pub g_canon_w: GpuBuffer,

    pub m1: BlockAdamMoments,
    pub m2: BlockAdamMoments,
    pub mu: BlockMuonMoments,
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
    pub canon_w: GpuBuffer,
}

/// Muon momentum buffers per block — only for the 6 matrix params.
/// LN γ/β + biases stay on AdamW even in hybrid mode.
pub struct BlockMuonMoments {
    pub w_q: GpuBuffer,
    pub w_k: GpuBuffer,
    pub w_v: GpuBuffer,
    pub w_o: GpuBuffer,
    pub w_fc: GpuBuffer,
    pub w_proj: GpuBuffer,
}

/// GPU scratch shared across all matrix params for one Muon step.
/// Buffers are sized for the largest matrix in the model so a single
/// allocation covers every parameter.
pub(crate) struct MuonScratchGpu {
    pub update: GpuBuffer,
    pub x_cur: GpuBuffer,
    pub bx_buf: GpuBuffer,
    pub a_buf: GpuBuffer,
    pub a_sq: GpuBuffer,
    pub b_buf: GpuBuffer,
    pub frob_partials: GpuBuffer,
    pub frob_result: GpuBuffer,
}

/// One `MuonOps` per matrix parameter holds cached bind groups that
/// reference per-matrix buffers (param, grad, momentum). All bind
/// groups that touch only scratch buffers live in `MuonSharedOps` and
/// are reused across matrices.
pub(crate) struct MuonOps {
    pub rows: u32,
    pub cols: u32,
    pub n: u32,
    pub apply_wd: bool,
    pub momentum_bind: wgpu::BindGroup,
    pub apply_bind: wgpu::BindGroup,
}

/// Bind groups for the scratch-only stages of the Muon step (Frobenius
/// reduce + inv, NS5 matmul/combine, transpose load/store). One set
/// reused across every matrix in `step_muon_hybrid`.
pub(crate) struct MuonSharedOps {
    pub frob_reduce_bind: wgpu::BindGroup,   // update → frob_partials
    pub frob_inv_bind: wgpu::BindGroup,      // frob_partials → frob_result
    pub scale_update_bind: wgpu::BindGroup,  // update *= frob_result[1]
    pub load_bind: wgpu::BindGroup,          // update → x_cur (transpose-aware)
    pub store_bind: wgpu::BindGroup,         // x_cur → update (transpose-aware)
    pub xxt_bind: wgpu::BindGroup,           // X @ X^T → a_buf (gemm_t)
    pub aa_bind: wgpu::BindGroup,            // A @ A → a_sq (gemm)
    pub combine_b_bind: wgpu::BindGroup,     // B = b·A + c·A_sq
    pub bx_bind: wgpu::BindGroup,            // B @ X → bx_buf (gemm)
    pub step_x_bind: wgpu::BindGroup,        // X = a·X + BX
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
    rope: &RopeKernel,
    qknorm_fwd: &QkNormKernel,
    softcap_fwd: &SoftcapKernel,
    canon_fwd: &CanonForwardKernel,
) -> ForwardOps {
    let embed = embed_fwd.build_bind(ctx, &a.tok_buf, &p.w_tok, &p.w_pos, &a.h0);
    let layers: Vec<LayerForwardOps> = (0..cfg.n_layer)
        .map(|li| {
            let blk = &p.blocks[li];
            let lc = &a.layers[li];
            let x = if li == 0 { &a.h0 } else { &a.z_chain[li - 1] };
            // Canon mixes the block input first; ln1 + the post-attn
            // residual then read canon_out instead of the raw input.
            let canon_x = if cfg.canon_kernel > 0 { &lc.canon_out } else { x };
            // With qk-norm the QKV gemms write into q_raw/k_raw and the
            // norm produces q/k; without it they write q/k directly.
            let (q_dst, k_dst) = if cfg.qk_norm {
                (&lc.q_raw, &lc.k_raw)
            } else {
                (&lc.q, &lc.k)
            };
            LayerForwardOps {
                ln1: ln_fwd.build_bind(
                    ctx, canon_x, &blk.ln1_gamma, &blk.ln1_beta, &lc.ln1_out, &a.ln1_mean,
                    &lc.ln1_inv, &lc.ln1_xhat,
                ),
                gemm_q: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_q, q_dst),
                gemm_k: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_k, k_dst),
                gemm_v: gemm_t.build_bind(ctx, &lc.ln1_out, &blk.w_v, &lc.v),
                qknorm_q: cfg
                    .qk_norm
                    .then(|| qknorm_fwd.build_bind(ctx, &lc.q_raw, &lc.q)),
                qknorm_k: cfg
                    .qk_norm
                    .then(|| qknorm_fwd.build_bind(ctx, &lc.k_raw, &lc.k)),
                rope_q: rope.build_bind(ctx, &lc.q),
                rope_k: rope.build_bind(ctx, &lc.k),
                mha: mha_fwd.build_bind(ctx, &lc.q, &lc.k, &lc.v, &lc.heads_out, &lc.head_probs),
                gemm_o: gemm_t.build_bind(ctx, &lc.heads_out, &blk.w_o, &a.mha_out),
                vadd_post_attn: vadd.build_bind(ctx, canon_x, &a.mha_out, &a.post_attn),
                ln2: ln_fwd.build_bind(
                    ctx, &a.post_attn, &blk.ln2_gamma, &blk.ln2_beta, &lc.ln2_out, &a.ln2_mean,
                    &lc.ln2_inv, &lc.ln2_xhat,
                ),
                gemm_fc: gemm_t.build_bind(ctx, &lc.ln2_out, &blk.w_fc, &lc.h_pre),
                gelu: gelu_fwd.build_bind(ctx, &lc.h_pre, &lc.h_post),
                gemm_proj: gemm_t.build_bind(ctx, &lc.h_post, &blk.w_proj, &a.ffn_out),
                vadd_z: vadd.build_bind(ctx, &a.post_attn, &a.ffn_out, &a.z_chain[li]),
                canon_fwd: (cfg.canon_kernel > 0)
                    .then(|| canon_fwd.build_bind(ctx, x, &blk.canon_w, &lc.canon_out)),
            }
        })
        .collect();
    let final_ln = ln_fwd.build_bind(
        ctx, &a.z_chain[cfg.n_layer - 1], &p.final_gamma, &p.final_beta,
        &a.h_final, &a.f_mean, &a.f_inv, &a.f_xhat,
    );
    let head_w = p.w_head.as_ref().unwrap_or(&p.w_tok);
    let final_gemm = gemm_t.build_bind(ctx, &a.h_final, head_w, &a.logits_buf);
    let softcap = (cfg.softcap > 0.0).then(|| softcap_fwd.build_bind(ctx, &a.logits_buf));
    ForwardOps { embed, layers, final_ln, final_gemm, softcap }
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
    rope: &RopeKernel,
    qknorm_bwd: &QkNormBackwardKernel,
    softcap_bwd: &SoftcapBackwardKernel,
    canon_dx: &CanonDxKernel,
    canon_dw: &CanonDwKernel,
) -> BackwardOps {
    let head_w = p.w_head.as_ref().unwrap_or(&p.w_tok);
    let g_head = p.g_w_head.as_ref().unwrap_or(&p.g_w_tok);
    let gemm_d_h_final = gemm.build_bind(ctx, &a.d_logits_buf, head_w, &a.d_h_final);
    let outer_g_w_tok = outer_accum.build_bind(ctx, &a.h_final, &a.d_logits_buf, g_head);
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
            let x_in = if li == 0 { &a.h0 } else { &a.z_chain[li - 1] };
            // With canon, vadd_dx writes canon_din (= d w.r.t. the canon-
            // mixed input); canon backward then maps it to dx (d w.r.t. the
            // raw block input) and accumulates g_canon_w.
            let dx_target = if cfg.canon_kernel > 0 { &a.canon_din } else { &a.dx };
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
                qknorm_dq: cfg
                    .qk_norm
                    .then(|| qknorm_bwd.build_bind(ctx, &lc.q_raw, &a.dq)),
                qknorm_dk: cfg
                    .qk_norm
                    .then(|| qknorm_bwd.build_bind(ctx, &lc.k_raw, &a.dk)),
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
                vadd_dx: vadd.build_bind(ctx, &a.d_post_attn, &a.d_x_mha, dx_target),
                canon_dw: (cfg.canon_kernel > 0)
                    .then(|| canon_dw.build_bind(ctx, &a.canon_din, x_in, &blk.g_canon_w)),
                canon_dx: (cfg.canon_kernel > 0)
                    .then(|| canon_dx.build_bind(ctx, &a.canon_din, &blk.canon_w, &a.dx)),
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
        rope_dq: rope.build_bind(ctx, &a.dq),
        rope_dk: rope.build_bind(ctx, &a.dk),
        softcap_bwd: (cfg.softcap > 0.0)
            .then(|| softcap_bwd.build_bind(ctx, &a.logits_buf, &a.d_logits_buf)),
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
                    apply_wd: bool,
                    is_matrix: bool| {
        let gn_offset = ops.len() as u32 * GN_GROUPS_PER_TENSOR;
        ops.push(GradOps {
            n,
            apply_wd,
            is_matrix,
            embed: false,
            gn_offset,
            zerob_bind: zerob.build_bind(ctx, grad),
            adamw_bind: adamw.build_bind(ctx, params, grad, m1, m2),
            gn_sq_bind: gn_sq.build_bind(ctx, grad, gn_partials_flat),
            scale_bind: scale.build_bind(ctx, grad, gn_result),
        });
    };

    push(&p.w_tok, &p.g_w_tok, &p.m1_w_tok, &p.m2_w_tok, (v * d) as u32, true, false);
    push(&p.w_pos, &p.g_w_pos, &p.m1_w_pos, &p.m2_w_pos, (seq_max * d) as u32, false, false);
    if let (Some(w_head), Some(g_w_head), Some(m1), Some(m2)) =
        (&p.w_head, &p.g_w_head, &p.m1_w_head, &p.m2_w_head)
    {
        // Untied head: AdamW-managed (not Muon), decoupled WD on.
        push(w_head, g_w_head, m1, m2, (v * d) as u32, true, false);
    }
    push(&p.final_gamma, &p.g_final_gamma, &p.m1_final_gamma, &p.m2_final_gamma, d as u32, false, false);
    push(&p.final_beta, &p.g_final_beta, &p.m1_final_beta, &p.m2_final_beta, d as u32, false, false);
    for blk in &p.blocks {
        push(&blk.ln1_gamma, &blk.g_ln1_gamma, &blk.m1.ln1_gamma, &blk.m2.ln1_gamma, d as u32, false, false);
        push(&blk.ln1_beta, &blk.g_ln1_beta, &blk.m1.ln1_beta, &blk.m2.ln1_beta, d as u32, false, false);
        push(&blk.w_q, &blk.g_w_q, &blk.m1.w_q, &blk.m2.w_q, (d * d) as u32, true, true);
        push(&blk.w_k, &blk.g_w_k, &blk.m1.w_k, &blk.m2.w_k, (d * d) as u32, true, true);
        push(&blk.w_v, &blk.g_w_v, &blk.m1.w_v, &blk.m2.w_v, (d * d) as u32, true, true);
        push(&blk.w_o, &blk.g_w_o, &blk.m1.w_o, &blk.m2.w_o, (d * d) as u32, true, true);
        push(&blk.ln2_gamma, &blk.g_ln2_gamma, &blk.m1.ln2_gamma, &blk.m2.ln2_gamma, d as u32, false, false);
        push(&blk.ln2_beta, &blk.g_ln2_beta, &blk.m1.ln2_beta, &blk.m2.ln2_beta, d as u32, false, false);
        push(&blk.w_fc, &blk.g_w_fc, &blk.m1.w_fc, &blk.m2.w_fc, (d * d_ff) as u32, true, true);
        push(&blk.w_proj, &blk.g_w_proj, &blk.m1.w_proj, &blk.m2.w_proj, (d * d_ff) as u32, true, true);
        if cfg.canon_kernel > 0 {
            // Canon taps: AdamW-managed (1D), no decay, not a Muon matrix.
            push(&blk.canon_w, &blk.g_canon_w, &blk.m1.canon_w, &blk.m2.canon_w,
                 (d * cfg.canon_kernel) as u32, false, false);
        }
    }
    // w_tok/w_pos (+ untied w_head) lead the op list; their AdamW LR is
    // scaled by embed_lr_mult at dispatch time.
    let n_embed = 2 + usize::from(p.w_head.is_some());
    for op in &mut ops[..n_embed] {
        op.embed = true;
    }
    ops
}

/// Sized for the largest matrix in the model. All Muon params reuse
/// this scratch in sequence — fine because the orchestrator runs them
/// sequentially anyway.
fn alloc_muon_scratch(ctx: &GpuContext, cfg: &TransformerConfig) -> MuonScratchGpu {
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;
    let max_mn = d * d_ff;
    // For Aurora's nano layout, min(rows, cols) == d for every matrix
    // (Q/K/V/O is d×d, W_fc is d×d_ff, W_proj is d_ff×d), so the m×m
    // NS work buffers are always d² floats.
    let max_mm = d * d;
    MuonScratchGpu {
        update: GpuBuffer::empty(ctx, "muon_update", max_mn),
        x_cur: GpuBuffer::empty(ctx, "muon_x_cur", max_mn),
        bx_buf: GpuBuffer::empty(ctx, "muon_bx", max_mn),
        a_buf: GpuBuffer::empty(ctx, "muon_a", max_mm),
        a_sq: GpuBuffer::empty(ctx, "muon_a_sq", max_mm),
        b_buf: GpuBuffer::empty(ctx, "muon_b", max_mm),
        frob_partials: GpuBuffer::zeros(ctx, "muon_frob_partials", 8),
        frob_result: GpuBuffer::zeros(ctx, "muon_frob_result", 2),
    }
}

fn build_muon_shared_ops(
    ctx: &GpuContext,
    scratch: &MuonScratchGpu,
    gn_sq: &GradNormSqReduceKernel,
    frob_inv: &MuonFrobInvKernel,
    scaleb_buf: &ScaleByBufferKernel,
    load_store: &MuonLoadStoreKernel,
    gemm: &GemmKernel,
    gemm_t: &GemmTransposeBKernel,
    combine_b: &MuonCombineBKernel,
    step_x: &MuonStepXKernel,
) -> MuonSharedOps {
    MuonSharedOps {
        frob_reduce_bind: gn_sq.build_bind(ctx, &scratch.update, &scratch.frob_partials),
        frob_inv_bind: frob_inv.build_bind(ctx, &scratch.frob_partials, &scratch.frob_result),
        scale_update_bind: scaleb_buf.build_bind(ctx, &scratch.update, &scratch.frob_result),
        load_bind: load_store.build_bind(ctx, &scratch.update, &scratch.x_cur),
        store_bind: load_store.build_bind(ctx, &scratch.x_cur, &scratch.update),
        xxt_bind: gemm_t.build_bind(ctx, &scratch.x_cur, &scratch.x_cur, &scratch.a_buf),
        aa_bind: gemm.build_bind(ctx, &scratch.a_buf, &scratch.a_buf, &scratch.a_sq),
        combine_b_bind: combine_b.build_bind(ctx, &scratch.a_buf, &scratch.a_sq, &scratch.b_buf),
        bx_bind: gemm.build_bind(ctx, &scratch.b_buf, &scratch.x_cur, &scratch.bx_buf),
        step_x_bind: step_x.build_bind(ctx, &scratch.x_cur, &scratch.bx_buf),
    }
}

fn build_muon_ops(
    ctx: &GpuContext,
    p: &GpuParams,
    cfg: &TransformerConfig,
    momentum: &MuonMomentumKernel,
    apply: &MuonApplyKernel,
    scratch: &MuonScratchGpu,
) -> Vec<MuonOps> {
    let d = cfg.d_model as u32;
    let d_ff = cfg.d_ff as u32;
    let mut ops = Vec::with_capacity(cfg.n_layer * 6);
    let mut push = |param: &GpuBuffer, grad: &GpuBuffer, m: &GpuBuffer, rows: u32, cols: u32| {
        ops.push(MuonOps {
            rows,
            cols,
            n: rows * cols,
            // All Muon-managed tensors are matrices; decoupled WD applies
            // to every one. With weight_decay == 0 the shader add is a
            // no-op, so the trainer's --muon-wd flag fully controls it.
            apply_wd: true,
            momentum_bind: momentum.build_bind(ctx, grad, m, &scratch.update),
            apply_bind: apply.build_bind(ctx, param, &scratch.update),
        });
    };
    for blk in &p.blocks {
        push(&blk.w_q, &blk.g_w_q, &blk.mu.w_q, d, d);
        push(&blk.w_k, &blk.g_w_k, &blk.mu.w_k, d, d);
        push(&blk.w_v, &blk.g_w_v, &blk.mu.w_v, d, d);
        push(&blk.w_o, &blk.g_w_o, &blk.mu.w_o, d, d);
        push(&blk.w_fc, &blk.g_w_fc, &blk.mu.w_fc, d, d_ff);
        push(&blk.w_proj, &blk.g_w_proj, &blk.mu.w_proj, d_ff, d);
    }
    ops
}

fn upload_block(ctx: &GpuContext, cpu: &crate::transformer::TransformerBlock) -> GpuBlock {
    let d = cpu.d_model;
    let d_ff = cpu.ffn.d_ff;
    let zeros_d = || GpuBuffer::zeros(ctx, "z", d);
    let zeros_dd = || GpuBuffer::zeros(ctx, "z", d * d);
    let zeros_dff = || GpuBuffer::zeros(ctx, "z", d * d_ff);
    let cw = (cpu.canon_k * d).max(1);
    let zeros_canon = || GpuBuffer::zeros(ctx, "z", cw);
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
        canon_w: zeros_canon(),
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
        canon_w: zeros_canon(),
    };
    let mu = BlockMuonMoments {
        w_q: zeros_dd(),
        w_k: zeros_dd(),
        w_v: zeros_dd(),
        w_o: zeros_dd(),
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
        canon_w: if cpu.canon_k > 0 {
            GpuBuffer::from_slice(ctx, "canon", &cpu.canon_w)
        } else {
            GpuBuffer::zeros(ctx, "canon", 1)
        },
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
        g_canon_w: zeros_canon(),
        m1,
        m2,
        mu,
    }
}

impl DecoderTransformerGpu {
    /// Upload a CPU `DecoderTransformer` to the GPU. The two then run
    /// in parallel; `to_cpu()` snapshots the GPU back.
    pub fn from_cpu(cpu: &DecoderTransformer, ctx: GpuContext, max_batch: usize) -> Self {
        let cfg = cpu.config.clone();
        let v = cfg.vocab;
        let d = cfg.d_model;
        let seq_max = cfg.seq_len_max;

        let p = GpuParams {
            w_tok: GpuBuffer::from_slice(&ctx, "w_tok", &cpu.embed.w_tok),
            w_pos: GpuBuffer::from_slice(&ctx, "w_pos", &cpu.embed.w_pos),
            w_head: cpu
                .w_head
                .as_ref()
                .map(|h| GpuBuffer::from_slice(&ctx, "w_head", h)),
            blocks: cpu.blocks.iter().map(|b| upload_block(&ctx, b)).collect(),
            final_gamma: GpuBuffer::from_slice(&ctx, "f_g", &cpu.final_ln.gamma),
            final_beta: GpuBuffer::from_slice(&ctx, "f_b", &cpu.final_ln.beta),

            g_w_tok: GpuBuffer::zeros(&ctx, "g_w_tok", v * d),
            g_w_pos: GpuBuffer::zeros(&ctx, "g_w_pos", seq_max * d),
            g_w_head: cpu
                .w_head
                .is_some()
                .then(|| GpuBuffer::zeros(&ctx, "g_w_head", v * d)),
            g_final_gamma: GpuBuffer::zeros(&ctx, "g_f_g", d),
            g_final_beta: GpuBuffer::zeros(&ctx, "g_f_b", d),

            m1_w_tok: GpuBuffer::zeros(&ctx, "m1_tok", v * d),
            m2_w_tok: GpuBuffer::zeros(&ctx, "m2_tok", v * d),
            m1_w_pos: GpuBuffer::zeros(&ctx, "m1_pos", seq_max * d),
            m2_w_pos: GpuBuffer::zeros(&ctx, "m2_pos", seq_max * d),
            m1_w_head: cpu
                .w_head
                .is_some()
                .then(|| GpuBuffer::zeros(&ctx, "m1_head", v * d)),
            m2_w_head: cpu
                .w_head
                .is_some()
                .then(|| GpuBuffer::zeros(&ctx, "m2_head", v * d)),
            m1_final_gamma: GpuBuffer::zeros(&ctx, "m1_fg", d),
            m2_final_gamma: GpuBuffer::zeros(&ctx, "m2_fg", d),
            m1_final_beta: GpuBuffer::zeros(&ctx, "m1_fb", d),
            m2_final_beta: GpuBuffer::zeros(&ctx, "m2_fb", d),
        };

        // 4 top-level grad tensors (+1 untied head) + 10 per block.
        let n_top = 4 + usize::from(cfg.untied_head);
        let total_partials = (n_top + cfg.n_layer * 10) * GN_GROUPS_PER_TENSOR as usize;
        let gn_partials_flat = GpuBuffer::zeros(&ctx, "gn_partials_flat", total_partials);
        let gn_result = GpuBuffer::zeros(&ctx, "gn_result", 2);

        let zerob = ZeroBufferKernel::new(&ctx);
        let adamw = AdamWKernel::new(&ctx);
        let gn_sq = GradNormSqReduceKernel::new(&ctx);
        let gn_finalize = GradNormFinalizeKernel::new(&ctx);
        let scaleb_buf = ScaleByBufferKernel::new(&ctx);
        let muon_momentum = MuonMomentumKernel::new(&ctx);
        let muon_frob_inv = MuonFrobInvKernel::new(&ctx);
        let muon_load_store = MuonLoadStoreKernel::new(&ctx);
        let muon_combine_b = MuonCombineBKernel::new(&ctx);
        let muon_step_x = MuonStepXKernel::new(&ctx);
        let muon_apply = MuonApplyKernel::new(&ctx);
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
        let rope = RopeKernel::new(&ctx);
        let qknorm_fwd = QkNormKernel::new(&ctx);
        let qknorm_bwd = QkNormBackwardKernel::new(&ctx);
        let softcap_fwd = SoftcapKernel::new(&ctx);
        let softcap_bwd = SoftcapBackwardKernel::new(&ctx);
        let canon_fwd = CanonForwardKernel::new(&ctx);
        let canon_dx = CanonDxKernel::new(&ctx);
        let canon_dw = CanonDwKernel::new(&ctx);

        let grad_ops =
            build_grad_ops(&ctx, &p, &cfg, &zerob, &adamw, &gn_sq, &scaleb_buf,
                           &gn_partials_flat, &gn_result);
        let gn_finalize_bind = gn_finalize.build_bind(&ctx, &gn_partials_flat, &gn_result);
        let gn_partials_zero_bind = zerob.build_bind(&ctx, &gn_partials_flat);
        let muon_scratch = alloc_muon_scratch(&ctx, &cfg);
        let muon_shared = build_muon_shared_ops(
            &ctx, &muon_scratch, &gn_sq, &muon_frob_inv, &scaleb_buf, &muon_load_store,
            &gemm, &gemm_t, &muon_combine_b, &muon_step_x,
        );
        let muon_ops = build_muon_ops(&ctx, &p, &cfg, &muon_momentum, &muon_apply, &muon_scratch);
        let acts = Activations::new(&ctx, &cfg, max_batch);
        let fwd_ops = build_forward_ops(
            &ctx, &p, &acts, &cfg,
            &embed_fwd, &ln_fwd, &gemm_t, &mha_fwd, &gelu_fwd, &vadd, &rope,
            &qknorm_fwd, &softcap_fwd, &canon_fwd,
        );
        let bwd_ops = build_backward_ops(
            &ctx, &p, &acts, &cfg,
            &gemm, &ln_bwd, &mha_bwd, &gelu_bwd, &vadd, &outer_accum,
            &embed_bwd, &embed_scatter_reduce, &zerob, &rope,
            &qknorm_bwd, &softcap_bwd, &canon_dx, &canon_dw,
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
            muon_momentum,
            muon_frob_inv,
            muon_load_store,
            muon_combine_b,
            muon_step_x,
            muon_apply,
            embed_scatter_reduce,
            single_embed,
            mha_kv,
            softmax_ce,
            rope,
            qknorm_fwd,
            qknorm_bwd,
            softcap_fwd,
            softcap_bwd,
            canon_fwd,
            canon_dx,
            canon_dw,
            gn_partials_flat,
            gn_result,
            ctx,
            config: cfg,
            step: cpu.step,
            max_batch: max_batch.max(1),
            p,
            grad_ops,
            gn_finalize_bind,
            gn_partials_zero_bind,
            muon_scratch,
            muon_ops,
            muon_shared,
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
        cpu.w_head = self.p.w_head.as_ref().map(|h| h.download(&self.ctx));
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
            if self.config.canon_kernel > 0 {
                blk.canon_w = gb.canon_w.download(&self.ctx);
            }
        }
        cpu.step = self.step;
        cpu
    }

    /// Ordered list of all persistent state buffers — weights + AdamW
    /// moments + Muon momentum (gradients are excluded; they are recomputed
    /// each step). `snapshot`/`restore` walk this in the same order, so the
    /// pairing is stable as long as it comes from the same model.
    fn state_buffers(&self) -> Vec<&GpuBuffer> {
        let p = &self.p;
        let mut v: Vec<&GpuBuffer> = Vec::with_capacity(14 + p.blocks.len() * 36);
        v.push(&p.w_tok);
        v.push(&p.w_pos);
        if let Some(b) = &p.w_head {
            v.push(b);
        }
        v.push(&p.final_gamma);
        v.push(&p.final_beta);
        v.push(&p.m1_w_tok);
        v.push(&p.m2_w_tok);
        v.push(&p.m1_w_pos);
        v.push(&p.m2_w_pos);
        if let Some(b) = &p.m1_w_head {
            v.push(b);
        }
        if let Some(b) = &p.m2_w_head {
            v.push(b);
        }
        v.push(&p.m1_final_gamma);
        v.push(&p.m2_final_gamma);
        v.push(&p.m1_final_beta);
        v.push(&p.m2_final_beta);
        for blk in &p.blocks {
            for b in [
                &blk.ln1_gamma, &blk.ln1_beta, &blk.w_q, &blk.w_k, &blk.w_v, &blk.w_o,
                &blk.ln2_gamma, &blk.ln2_beta, &blk.w_fc, &blk.w_proj,
                &blk.m1.ln1_gamma, &blk.m1.ln1_beta, &blk.m1.w_q, &blk.m1.w_k, &blk.m1.w_v,
                &blk.m1.w_o, &blk.m1.ln2_gamma, &blk.m1.ln2_beta, &blk.m1.w_fc, &blk.m1.w_proj,
                &blk.m2.ln1_gamma, &blk.m2.ln1_beta, &blk.m2.w_q, &blk.m2.w_k, &blk.m2.w_v,
                &blk.m2.w_o, &blk.m2.ln2_gamma, &blk.m2.ln2_beta, &blk.m2.w_fc, &blk.m2.w_proj,
                &blk.mu.w_q, &blk.mu.w_k, &blk.mu.w_v, &blk.mu.w_o, &blk.mu.w_fc, &blk.mu.w_proj,
                &blk.canon_w, &blk.m1.canon_w, &blk.m2.canon_w,
            ] {
                v.push(b);
            }
        }
        v
    }

    fn copy_state(&self, dst: &[GpuBuffer], src: &[&GpuBuffer], label: &str) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        for (d, s) in dst.iter().zip(src) {
            enc.copy_buffer_to_buffer(&s.inner, 0, &d.inner, 0, (s.len * 4) as u64);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Capture all persistent state into fresh device buffers (one
    /// buffer-to-buffer copy each — stays on-device, no host readback).
    /// For the PHAGE governor's reflex rollback and paired-rollout trials.
    pub fn snapshot_state(&self) -> GpuStateSnapshot {
        let live = self.state_buffers();
        let bufs: Vec<GpuBuffer> = live
            .iter()
            .map(|b| GpuBuffer::empty(&self.ctx, "phage snapshot", b.len))
            .collect();
        self.copy_state(&bufs, &live, "phage snapshot");
        GpuStateSnapshot { bufs, step: self.step }
    }

    /// Re-capture current state into an existing snapshot — no allocation.
    pub fn refresh_state(&self, snap: &mut GpuStateSnapshot) {
        let live = self.state_buffers();
        self.copy_state(&snap.bufs, &live, "phage refresh");
        snap.step = self.step;
    }

    /// Restore weights + optimizer state from a snapshot.
    pub fn restore_state(&mut self, snap: &GpuStateSnapshot) {
        {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("phage restore"),
                });
            for (live, backup) in self.state_buffers().iter().zip(&snap.bufs) {
                enc.copy_buffer_to_buffer(&backup.inner, 0, &live.inner, 0, (backup.len * 4) as u64);
            }
            self.ctx.queue.submit(Some(enc.finish()));
        }
        self.step = snap.step;
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
    fn encode_forward(&self, enc: &mut wgpu::CommandEncoder, t: usize, tok_offset: u32) {
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
                t_u, d_u, v as u32, cfg.seq_len_max as u32, tok_offset, !cfg.rope,
            );
        }

        for li in 0..cfg.n_layer {
            let fop = &self.fwd_ops.layers[li];

            // Canon: canon_out = x + dwconv(x). ln1 + the post-attn residual
            // read canon_out (wired into their bind groups in build_forward_ops).
            if let Some(cf) = &fop.canon_fwd {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("canon fwd"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.canon_fwd.pipeline);
                self.canon_fwd.dispatch_with_bind(&mut pass, cf, t_u, d_u, cfg.canon_kernel as u32);
            }

            // ln1
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ln1"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.ln_fwd.pipeline);
                self.ln_fwd.dispatch_with_bind(&mut pass, &fop.ln1, t_u, d_u, 1e-5, cfg.rmsnorm);
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
            // QK-norm — q_raw → q, k_raw → k (disjoint outputs, one pass).
            if let (Some(qn_q), Some(qn_k)) = (&fop.qknorm_q, &fop.qknorm_k) {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("qk norm"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.qknorm_fwd.pipeline);
                self.qknorm_fwd.dispatch_with_bind(
                    &mut pass, qn_q, t_u, n_head as u32, d_head as u32,
                );
                self.qknorm_fwd.dispatch_with_bind(
                    &mut pass, qn_k, t_u, n_head as u32, d_head as u32,
                );
            }
            // RoPE — in-place rotation of q and k (disjoint buffers, one pass).
            if cfg.rope {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rope qk"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.rope.pipeline);
                self.rope.dispatch_with_bind(
                    &mut pass, &fop.rope_q, t_u, n_head as u32, d_head as u32, 0, false,
                );
                self.rope.dispatch_with_bind(
                    &mut pass, &fop.rope_k, t_u, n_head as u32, d_head as u32, 0, false,
                );
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
                self.ln_fwd.dispatch_with_bind(&mut pass, &fop.ln2, t_u, d_u, 1e-5, cfg.rmsnorm);
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
                self.gelu_fwd.dispatch_with_bind(
                    &mut pass, &fop.gelu, (t * d_ff) as u32, cfg.relu2,
                );
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
            self.ln_fwd.dispatch_with_bind(
                &mut pass, &self.fwd_ops.final_ln, t_u, d_u, 1e-5, cfg.rmsnorm,
            );
        }
        // LM head (logits = h_final × w_headᵀ, w_head = w_tok when tied)
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lm head"), timestamp_writes: None,
            });
            pass.set_pipeline(self.gemm_t.pipeline());
            self.gemm_t.dispatch_with_bind(&mut pass, &self.fwd_ops.final_gemm, t_u, v as u32, d_u);
        }
        // Logit softcap (in place on the logits buffer).
        if let Some(sc) = &self.fwd_ops.softcap {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softcap"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.softcap_fwd.pipeline);
            self.softcap_fwd.dispatch_with_bind(
                &mut pass, sc, (t * v) as u32, cfg.softcap,
            );
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
        let t = tokens.len();
        debug_assert_eq!(targets.len(), t);

        self.acts.tok_buf.upload_u32_prefix(&self.ctx, tokens);
        self.acts.targets_buf.upload_u32_prefix(&self.ctx, targets);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd") });
        self.encode_train_element(&mut enc, t, 0, 0);
        self.ctx.queue.submit(Some(enc.finish()));

        // Per-token NLL was written by softmax_ce; reduce on host.
        // Download is T floats (small) instead of the legacy T·V logits.
        let loss_per_t = self.acts.loss_seq_buf.download_prefix(&self.ctx, t);
        loss_per_t.iter().sum::<f32>() / t as f32
    }

    /// Batched train step: runs all `inputs.len()` sequences through
    /// forward + backward in a SINGLE command encoder, SINGLE submit, and
    /// SINGLE loss readback, accumulating gradients across the batch.
    /// (Muon expects summed — not averaged — grads; the caller scales the
    /// clip threshold by the batch size to compensate, exactly as the old
    /// per-element loop did.)
    ///
    /// Each sequence reuses the shared activation arena; wgpu's automatic
    /// hazard tracking inserts the barriers that serialize one element's
    /// fwd/bwd before the next overwrites the arena, so the gradients are
    /// identical to `inputs.len()` separate `train_step` calls — but the
    /// GPU↔host round-trip (a full device stall) is paid once for the whole
    /// batch instead of once per element.
    ///
    /// Returns the sum of per-sequence mean cross-entropies (divide by
    /// `inputs.len()` for the batch-mean loss).
    pub fn train_step_batch(&mut self, inputs: &[&[u32]], targets: &[&[u32]]) -> f32 {
        let b = inputs.len();
        debug_assert_eq!(targets.len(), b);
        if b == 0 {
            return 0.0;
        }
        let t = inputs[0].len();

        // Flatten into the B·T token/target buffers. Both uploads happen
        // before any dispatch, so there is no write-after-write hazard on
        // the shared token/target buffers within the single submission.
        let mut tok_flat: Vec<u32> = Vec::with_capacity(b * t);
        let mut tgt_flat: Vec<u32> = Vec::with_capacity(b * t);
        for i in 0..b {
            debug_assert_eq!(inputs[i].len(), t);
            debug_assert_eq!(targets[i].len(), t);
            tok_flat.extend_from_slice(inputs[i]);
            tgt_flat.extend_from_slice(targets[i]);
        }
        self.acts.tok_buf.upload_u32_prefix(&self.ctx, &tok_flat);
        self.acts.targets_buf.upload_u32_prefix(&self.ctx, &tgt_flat);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd batch"),
            });
        for bi in 0..b {
            let off = (bi * t) as u32;
            self.encode_train_element(&mut enc, t, off, off);
        }
        self.ctx.queue.submit(Some(enc.finish()));

        // Σ over all B·T NLLs / T = Σ_b mean_t(NLL_b): the sum of
        // per-sequence means (caller divides by B for the batch mean).
        let loss_per = self.acts.loss_seq_buf.download_prefix(&self.ctx, b * t);
        loss_per.iter().sum::<f32>() / t as f32
    }

    /// Encode one sequence's forward + softmax-CE + backward into `enc`,
    /// reading its tokens at `tok_offset` and reading targets / writing its
    /// per-token NLL at `bt_offset` within the shared `B·T` buffers.
    /// Gradients accumulate (+=) into the shared grad tensors, so the
    /// caller must `zero_grad()` once before the batch. No submit, no
    /// readback — the caller batches those.
    fn encode_train_element(
        &self,
        enc: &mut wgpu::CommandEncoder,
        t: usize,
        tok_offset: u32,
        bt_offset: u32,
    ) {
        let cfg = &self.config;
        let d = cfg.d_model;
        let v = cfg.vocab;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        let t_u = t as u32;
        let d_u = d as u32;

        self.encode_forward(enc, t, tok_offset);

        // GPU softmax + cross-entropy: per-token NLL → loss_seq[bt_offset..],
        // pre-scaled gradient (probs − one_hot)/T → d_logits (offset 0).
        let loss_scale = 1.0_f32 / t as f32;
        self.softmax_ce.dispatch_into(
            &self.ctx,
            enc,
            &self.acts.logits_buf,
            &self.acts.d_logits_buf,
            &self.acts.loss_seq_buf,
            &self.acts.targets_buf,
            v as u32,
            t_u,
            1,
            loss_scale,
            bt_offset,
        );

        // Softcap chain rule: d_logits *= 1 − tanh² (tanh recovered from
        // the capped logits still sitting in logits_buf).
        if let Some(sc) = &self.bwd_ops.softcap_bwd {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softcap bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.softcap_bwd.pipeline);
            self.softcap_bwd.dispatch_with_bind(
                &mut pass, sc, (t * v) as u32, cfg.softcap,
            );
        }

        // d_h_final = d_logits × w_head  AND  outer_accum into the head grad.
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
                &mut pass, &self.bwd_ops.final_ln_bwd, t_u, d_u, cfg.rmsnorm,
            );
        }

        // Block backward chain.
        for li in (0..cfg.n_layer).rev() {
            self.block_backward(enc, li, t, d, n_head, d_head, d_ff);
        }

        // Embed bwd: per-pos grad + scatter-reduce to g_w_tok / g_w_pos.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.embed_bwd.pipeline);
            self.embed_bwd.dispatch_with_bind(
                &mut pass, &self.bwd_ops.embed_bwd,
                t_u, d_u, v as u32, cfg.seq_len_max as u32, tok_offset, !cfg.rope,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed scatter"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.embed_scatter_reduce.pipeline);
            self.embed_scatter_reduce.dispatch_with_bind(
                &mut pass, &self.bwd_ops.embed_scatter_reduce, t_u, d_u, tok_offset,
            );
        }
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
        self.eval_step_per_token(tokens, targets).iter().sum()
    }

    /// Batched eval: forward + softmax_ce for up to `max_batch` windows
    /// in a single command encoder / submit / readback, mirroring
    /// `train_step_batch`. Each window reuses the shared activation
    /// arena (wgpu hazard tracking serializes them); per-token NLLs land
    /// at window offsets in `loss_seq_buf`. Returns B·T nats — caller
    /// slices per window. ~B× fewer device round-trips than calling
    /// `eval_step_per_token` per window.
    pub fn eval_step_batch(&self, inputs: &[&[u32]], targets: &[&[u32]]) -> Vec<f32> {
        let b = inputs.len();
        debug_assert_eq!(targets.len(), b);
        debug_assert!(b <= self.max_batch, "eval batch {b} > capacity {}", self.max_batch);
        if b == 0 {
            return Vec::new();
        }
        let t = inputs[0].len();
        let v = self.config.vocab;

        let mut tok_flat: Vec<u32> = Vec::with_capacity(b * t);
        let mut tgt_flat: Vec<u32> = Vec::with_capacity(b * t);
        for i in 0..b {
            debug_assert_eq!(inputs[i].len(), t);
            debug_assert_eq!(targets[i].len(), t);
            tok_flat.extend_from_slice(inputs[i]);
            tgt_flat.extend_from_slice(targets[i]);
        }
        self.acts.tok_buf.upload_u32_prefix(&self.ctx, &tok_flat);
        self.acts.targets_buf.upload_u32_prefix(&self.ctx, &tgt_flat);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("eval batch"),
            });
        for bi in 0..b {
            let off = (bi * t) as u32;
            self.encode_forward(&mut enc, t, off);
            self.softmax_ce.dispatch_into(
                &self.ctx,
                &mut enc,
                &self.acts.logits_buf,
                &self.acts.d_logits_buf,
                &self.acts.loss_seq_buf,
                &self.acts.targets_buf,
                v as u32,
                t as u32,
                1,
                1.0,
                off,
            );
        }
        self.ctx.queue.submit(Some(enc.finish()));

        self.acts.loss_seq_buf.download_prefix(&self.ctx, b * t)
    }

    /// Per-position NLL (nats) for one window. Lets callers score only
    /// a suffix of the window, as overlapping-window eval requires.
    pub fn eval_step_per_token(&self, tokens: &[u32], targets: &[u32]) -> Vec<f32> {
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

        self.encode_forward(&mut enc, t, 0);

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
            0,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        self.acts.loss_seq_buf.download_prefix(&self.ctx, t)
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
        let k_u = self.config.canon_kernel as u32;

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
            self.gelu_bwd.dispatch_with_bind(&mut pass, &bop.gelu_bwd, td_ff, self.config.relu2);
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
            self.ln_bwd.dispatch_with_bind(&mut pass, &bop.ln2_bwd, t_u, d_u, self.config.rmsnorm);
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

        // dq/dk are gradients w.r.t. the rotated q/k — apply the inverse
        // rotation before they feed the projection backward below.
        if self.config.rope {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope bwd dqk"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.rope.pipeline);
            self.rope.dispatch_with_bind(
                &mut pass, &self.bwd_ops.rope_dq, t_u, n_head as u32, d_head as u32, 0, true,
            );
            self.rope.dispatch_with_bind(
                &mut pass, &self.bwd_ops.rope_dk, t_u, n_head as u32, d_head as u32, 0, true,
            );
        }
        // QK-norm backward: dq/dk (w.r.t. normalized rows) → w.r.t. raw,
        // in place, using this layer's cached q_raw/k_raw.
        if let (Some(qn_dq), Some(qn_dk)) = (&bop.qknorm_dq, &bop.qknorm_dk) {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qk norm bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.qknorm_bwd.pipeline);
            self.qknorm_bwd.dispatch_with_bind(
                &mut pass, qn_dq, t_u, n_head as u32, d_head as u32,
            );
            self.qknorm_bwd.dispatch_with_bind(
                &mut pass, qn_dk, t_u, n_head as u32, d_head as u32,
            );
        }

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
            self.ln_bwd.dispatch_with_bind(&mut pass, &bop.ln1_bwd, t_u, d_u, self.config.rmsnorm);
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
        // Canon backward: vadd_dx wrote canon_din (d w.r.t. canon-mixed
        // input); accumulate g_canon_w (dw) and map canon_din → dx (dx).
        if let (Some(dw), Some(dx)) = (&bop.canon_dw, &bop.canon_dx) {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("canon bwd"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.canon_dw.pipeline);
            self.canon_dw.dispatch_with_bind(&mut pass, dw, t_u, d_u, k_u);
            pass.set_pipeline(&self.canon_dx.pipeline);
            self.canon_dx.dispatch_with_bind(&mut pass, dx, t_u, d_u, k_u);
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
                let lr = if op.embed {
                    config.lr * self.config.embed_lr_mult
                } else {
                    config.lr
                };
                self.adamw.dispatch_with_bind(
                    &mut pass,
                    &op.adamw_bind,
                    op.n,
                    op.apply_wd,
                    lr,
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

    /// Hybrid Muon + AdamW step. Matrix params (Q/K/V/O, FC, Proj) go
    /// through per-matrix Newton–Schulz5 orthogonalization on
    /// momentum-modified gradients; everything else (embed lookups +
    /// LayerNorm γ/β) uses AdamW. Per-matrix pipeline = ~32 compute
    /// passes; full-step dispatch overhead is ~1.1k passes for nano.
    pub fn step_muon_hybrid(&mut self, muon_cfg: &MuonConfig, adam_cfg: &AdamWConfig) {
        self.step += 1;
        let (bc1, bc2) = bias_correction(self.step as usize, adam_cfg.beta1, adam_cfg.beta2);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora muon hybrid step"),
            });

        // AdamW pass for all non-matrix grad tensors (one compute pass,
        // many dispatches — independent reads/writes).
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("muon-hybrid adamw subset"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adamw.pipeline);
            for op in &self.grad_ops {
                if op.is_matrix {
                    continue;
                }
                let lr = if op.embed {
                    adam_cfg.lr * self.config.embed_lr_mult
                } else {
                    adam_cfg.lr
                };
                self.adamw.dispatch_with_bind(
                    &mut pass,
                    &op.adamw_bind,
                    op.n,
                    op.apply_wd,
                    lr,
                    adam_cfg.beta1,
                    adam_cfg.beta2,
                    adam_cfg.eps,
                    bc1,
                    bc2,
                    adam_cfg.weight_decay,
                );
            }
        }

        // Per matrix: momentum → Frobenius bound → load → 5×NS5 → store → apply.
        for op in &self.muon_ops {
            let transposed = op.rows > op.cols;
            let (m, n_dim) = if transposed { (op.cols, op.rows) } else { (op.rows, op.cols) };
            let mn = m * n_dim;
            let mm = m * m;

            // momentum + nesterov → update
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon momentum"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.muon_momentum.pipeline);
                self.muon_momentum.dispatch_with_bind(
                    &mut pass,
                    &op.momentum_bind,
                    op.n,
                    muon_cfg.nesterov,
                    muon_cfg.momentum,
                );
            }
            // Frobenius squared partials over update
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon frob reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gn_sq.pipeline);
                self.gn_sq.dispatch_with_bind(
                    &mut pass,
                    &self.muon_shared.frob_reduce_bind,
                    op.n,
                    0,
                    8,
                );
            }
            // finalize: frob_result[0] = sqrt(sum), [1] = 1/(sqrt+eps)
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon frob inv"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.muon_frob_inv.pipeline);
                self.muon_frob_inv.dispatch_with_bind(
                    &mut pass,
                    &self.muon_shared.frob_inv_bind,
                    8,
                );
            }
            // update *= inv_scale (factor_idx=1)
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon scale"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scaleb_buf.pipeline);
                self.scaleb_buf.dispatch_with_bind(
                    &mut pass,
                    &self.muon_shared.scale_update_bind,
                    op.n,
                    1,
                );
            }
            // load update → x_cur (transpose if rows > cols)
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon load"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.muon_load_store.pipeline);
                self.muon_load_store.dispatch_with_bind(
                    &mut pass,
                    &self.muon_shared.load_bind,
                    op.rows,
                    op.cols,
                    transposed,
                    false,
                );
            }
            // NS5 iterations
            for _ in 0..muon_cfg.ns_steps {
                // A = X X^T  ([m × n] · [n × m] → [m × m])
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ns xxt"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(self.gemm_t.pipeline());
                    self.gemm_t.dispatch_with_bind(
                        &mut pass,
                        &self.muon_shared.xxt_bind,
                        m,
                        m,
                        n_dim,
                    );
                }
                // A_sq = A @ A  ([m × m] · [m × m] → [m × m])
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ns aa"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.gemm.pipeline);
                    self.gemm.dispatch_with_bind(
                        &mut pass,
                        &self.muon_shared.aa_bind,
                        m,
                        m,
                        m,
                    );
                }
                // B = b·A + c·A_sq
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ns combine_b"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.muon_combine_b.pipeline);
                    self.muon_combine_b.dispatch_with_bind(
                        &mut pass,
                        &self.muon_shared.combine_b_bind,
                        mm,
                    );
                }
                // BX = B @ X  ([m × m] · [m × n] → [m × n])
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ns bx"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.gemm.pipeline);
                    self.gemm.dispatch_with_bind(
                        &mut pass,
                        &self.muon_shared.bx_bind,
                        m,
                        n_dim,
                        m,
                    );
                }
                // X = a·X + BX (in-place on x_cur)
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ns step_x"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.muon_step_x.pipeline);
                    self.muon_step_x.dispatch_with_bind(
                        &mut pass,
                        &self.muon_shared.step_x_bind,
                        mn,
                    );
                }
            }
            // store x_cur → update (transpose back if needed)
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon store"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.muon_load_store.pipeline);
                self.muon_load_store.dispatch_with_bind(
                    &mut pass,
                    &self.muon_shared.store_bind,
                    op.rows,
                    op.cols,
                    transposed,
                    true,
                );
            }
            // params -= lr · sqrt(n/m) · update  (+ optional decoupled wd)
            let scale = (n_dim as f32 / m as f32).sqrt();
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("muon apply"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.muon_apply.pipeline);
                self.muon_apply.dispatch_with_bind(
                    &mut pass,
                    &op.apply_bind,
                    op.n,
                    op.apply_wd,
                    muon_cfg.lr,
                    scale,
                    muon_cfg.weight_decay,
                );
            }
        }

        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Clip only the AdamW-managed subset (embed lookups + LayerNorm
    /// γ/β). Matrix grads pass through untouched — Newton–Schulz inside
    /// `step_muon_hybrid` already bounds their update magnitude. The clip
    /// scale is applied entirely on-GPU; `want_norm` controls whether the
    /// pre-clip norm is read back to the host (a full device stall) — pass
    /// `false` on steps that won't log it and the return value is `NaN`.
    pub fn clip_grads_adamw_subset(&self, max_norm: f32, want_norm: bool) -> f32 {
        let n_partials = self.grad_ops.len() as u32 * GN_GROUPS_PER_TENSOR;

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clip_subset"),
            });
        // Zero the partial-sum buffer on-GPU. The subset gn_sq below
        // writes only the non-matrix slots; finalize sums the whole
        // buffer, so the matrix slots must read as 0. (Replaces a per-step
        // host zero-fill + PCIe upload.)
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip_subset zero"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.zerob.pipeline);
            self.zerob
                .dispatch_with_bind(&mut pass, &self.gn_partials_zero_bind, n_partials);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip_subset gn_sq"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_sq.pipeline);
            for op in &self.grad_ops {
                if op.is_matrix {
                    continue;
                }
                self.gn_sq.dispatch_with_bind(
                    &mut pass,
                    &op.gn_sq_bind,
                    op.n,
                    op.gn_offset,
                    GN_GROUPS_PER_TENSOR,
                );
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip_subset finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gn_finalize.pipeline);
            self.gn_finalize.dispatch_with_bind(
                &mut pass,
                &self.gn_finalize_bind,
                n_partials,
                max_norm,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clip_subset scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scaleb_buf.pipeline);
            for op in &self.grad_ops {
                if op.is_matrix {
                    continue;
                }
                self.scaleb_buf
                    .dispatch_with_bind(&mut pass, &op.scale_bind, op.n, 1);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));

        if want_norm {
            self.gn_result.download(&self.ctx)[0]
        } else {
            f32::NAN
        }
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
            d as u32, v_dim as u32, cfg.seq_len_max as u32, new_token, pos as u32, !cfg.rope,
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
                &ln1_inv, &ln1_xhat, 1, d as u32, 1e-5, cfg.rmsnorm,
            );

            let mut q_new = GpuBuffer::empty(&self.ctx, "step_q", d);
            let mut k_new = GpuBuffer::empty(&self.ctx, "step_k", d);
            let v_new = GpuBuffer::empty(&self.ctx, "step_v", d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &v_new, 1, d as u32, d as u32);
            if cfg.qk_norm {
                let q_n = GpuBuffer::empty(&self.ctx, "step_qn", d);
                let k_n = GpuBuffer::empty(&self.ctx, "step_kn", d);
                self.qknorm_fwd.dispatch_into(
                    &self.ctx, &mut enc, &q_new, &q_n, 1, n_head as u32, d_head as u32,
                );
                self.qknorm_fwd.dispatch_into(
                    &self.ctx, &mut enc, &k_new, &k_n, 1, n_head as u32, d_head as u32,
                );
                q_new = q_n;
                k_new = k_n;
            }
            if cfg.rope {
                self.rope.dispatch_into(&self.ctx, &mut enc, &q_new, 1, n_head as u32, d_head as u32, pos as u32, false);
                self.rope.dispatch_into(&self.ctx, &mut enc, &k_new, 1, n_head as u32, d_head as u32, pos as u32, false);
            }
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
                &ln2_mean, &ln2_inv, &ln2_xhat, 1, d as u32, 1e-5, cfg.rmsnorm,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln2_out, &blk.w_fc, &h_pre, 1, d_ff as u32, d as u32);
            self.gelu_fwd.dispatch_into(&self.ctx, &mut enc, &h_pre, &h_post, d_ff as u32, cfg.relu2);
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
            &f_mean, &f_inv, &f_xhat, 1, d as u32, 1e-5, cfg.rmsnorm,
        );
        let head_w = self.p.w_head.as_ref().unwrap_or(&self.p.w_tok);
        self.gemm_t.dispatch_into(
            &self.ctx, &mut enc, &h_final, head_w, &logits, 1, v_dim as u32, d as u32,
        );
        if cfg.softcap > 0.0 {
            self.softcap_fwd.dispatch_into(
                &self.ctx, &mut enc, &logits, v_dim as u32, cfg.softcap,
            );
        }
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
            t as u32, d as u32, v as u32, cfg.seq_len_max as u32, !cfg.rope,
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
            let mut q = GpuBuffer::empty(&self.ctx, "q", t * d);
            let mut k = GpuBuffer::empty(&self.ctx, "k", t * d);
            let val = GpuBuffer::empty(&self.ctx, "v", t * d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prefill block fwd"),
                });
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &x, &blk.ln1_gamma, &blk.ln1_beta, &ln1_out, &ln1_mean,
                &ln1_inv, &ln1_xhat, t as u32, d as u32, 1e-5, cfg.rmsnorm,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &val, t as u32, d as u32, d as u32);
            if cfg.qk_norm {
                let q_n = GpuBuffer::empty(&self.ctx, "q_n", t * d);
                let k_n = GpuBuffer::empty(&self.ctx, "k_n", t * d);
                self.qknorm_fwd.dispatch_into(
                    &self.ctx, &mut enc, &q, &q_n, t as u32, n_head as u32, d_head as u32,
                );
                self.qknorm_fwd.dispatch_into(
                    &self.ctx, &mut enc, &k, &k_n, t as u32, n_head as u32, d_head as u32,
                );
                q = q_n;
                k = k_n;
            }
            if cfg.rope {
                self.rope.dispatch_into(&self.ctx, &mut enc, &q, t as u32, n_head as u32, d_head as u32, 0, false);
                self.rope.dispatch_into(&self.ctx, &mut enc, &k, t as u32, n_head as u32, d_head as u32, 0, false);
            }

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
                &ln2_mean, &ln2_inv, &ln2_xhat, t as u32, d as u32, 1e-5, cfg.rmsnorm,
            );
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln2_out, &blk.w_fc, &h_pre, t as u32, d_ff as u32, d as u32);
            self.gelu_fwd.dispatch_into(&self.ctx, &mut enc, &h_pre, &h_post, (t * d_ff) as u32, cfg.relu2);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &h_post, &blk.w_proj, &ffn_out, t as u32, d as u32, d_ff as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &post_attn, &ffn_out, &z, (t * d) as u32);
            self.ctx.queue.submit(Some(enc.finish()));

            kv_per_block.push((k, val));
            x = z;
        }
        kv_per_block
    }
}

