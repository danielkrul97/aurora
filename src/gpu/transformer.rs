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
    GpuContext, GradNormSqReduceKernel, LayerNormBackwardKernel, LayerNormForwardKernel,
    OuterProductAccumKernel, RowReduceAddKernel, ScaleBufferKernel, TokenPosEmbedKernel,
    VectorAddKernel, ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig};
use crate::transformer::loss::cross_entropy_with_grad;
use crate::transformer::{DecoderTransformer, TransformerConfig};

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
    pub scaleb: ScaleBufferKernel,
    pub outer_accum: OuterProductAccumKernel,
    pub row_reduce: RowReduceAddKernel,
    pub gn_sq: GradNormSqReduceKernel,
    pub embed_scatter_reduce: EmbedScatterReduceKernel,
    pub single_embed: SingleTokenEmbedKernel,
    pub mha_kv: MhaKvCacheKernel,

    // Parameters + Adam moments.
    pub p: GpuParams,
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

        Self {
            gemm: GemmKernel::new(&ctx),
            gemm_t: GemmTransposeBKernel::new(&ctx),
            gelu_fwd: GeluKernel::new(&ctx),
            gelu_bwd: GeluBackwardKernel::new(&ctx),
            ln_fwd: LayerNormForwardKernel::new(&ctx),
            ln_bwd: LayerNormBackwardKernel::new(&ctx),
            mha_fwd: CausalMhaForwardKernel::new(&ctx),
            mha_bwd: CausalMhaBackwardKernel::new(&ctx),
            embed_fwd: TokenPosEmbedKernel::new(&ctx),
            embed_bwd: EmbedScatterAddKernel::new(&ctx),
            adamw: AdamWKernel::new(&ctx),
            vadd: VectorAddKernel::new(&ctx),
            zerob: ZeroBufferKernel::new(&ctx),
            scaleb: ScaleBufferKernel::new(&ctx),
            outer_accum: OuterProductAccumKernel::new(&ctx),
            row_reduce: RowReduceAddKernel::new(&ctx),
            gn_sq: GradNormSqReduceKernel::new(&ctx),
            embed_scatter_reduce: EmbedScatterReduceKernel::new(&ctx),
            single_embed: SingleTokenEmbedKernel::new(&ctx),
            mha_kv: MhaKvCacheKernel::new(&ctx),
            ctx,
            config: cfg,
            step: cpu.step,
            p,
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

    /// Zero all gradient buffers.
    pub fn zero_grad(&self) {
        let cfg = &self.config;
        let v = cfg.vocab;
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        let seq_max = cfg.seq_len_max;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero_grad") });
        self.zerob.dispatch_into(&self.ctx, &mut enc, &self.p.g_w_tok, (v * d) as u32);
        self.zerob.dispatch_into(&self.ctx, &mut enc, &self.p.g_w_pos, (seq_max * d) as u32);
        self.zerob.dispatch_into(&self.ctx, &mut enc, &self.p.g_final_gamma, d as u32);
        self.zerob.dispatch_into(&self.ctx, &mut enc, &self.p.g_final_beta, d as u32);
        for blk in &self.p.blocks {
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_ln1_gamma, d as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_ln1_beta, d as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_q, (d * d) as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_k, (d * d) as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_v, (d * d) as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_o, (d * d) as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_ln2_gamma, d as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_ln2_beta, d as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_fc, (d * d_ff) as u32);
            self.zerob.dispatch_into(&self.ctx, &mut enc, &blk.g_w_proj, (d * d_ff) as u32);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Compute global L2 norm using GPU reduce kernel. Each tensor
    /// produces 8 partial sums (one per workgroup); host sums the
    /// `N_TENSORS × 8` floats.
    pub fn grad_norm(&self) -> f32 {
        const N_GROUPS: u32 = 8;
        let cfg = &self.config;
        let mut buffers: Vec<(&GpuBuffer, u32)> = Vec::new();
        buffers.push((&self.p.g_w_tok, (cfg.vocab * cfg.d_model) as u32));
        buffers.push((&self.p.g_w_pos, (cfg.seq_len_max * cfg.d_model) as u32));
        buffers.push((&self.p.g_final_gamma, cfg.d_model as u32));
        buffers.push((&self.p.g_final_beta, cfg.d_model as u32));
        for blk in &self.p.blocks {
            buffers.push((&blk.g_ln1_gamma, cfg.d_model as u32));
            buffers.push((&blk.g_ln1_beta, cfg.d_model as u32));
            buffers.push((&blk.g_w_q, (cfg.d_model * cfg.d_model) as u32));
            buffers.push((&blk.g_w_k, (cfg.d_model * cfg.d_model) as u32));
            buffers.push((&blk.g_w_v, (cfg.d_model * cfg.d_model) as u32));
            buffers.push((&blk.g_w_o, (cfg.d_model * cfg.d_model) as u32));
            buffers.push((&blk.g_ln2_gamma, cfg.d_model as u32));
            buffers.push((&blk.g_ln2_beta, cfg.d_model as u32));
            buffers.push((&blk.g_w_fc, (cfg.d_model * cfg.d_ff) as u32));
            buffers.push((&blk.g_w_proj, (cfg.d_model * cfg.d_ff) as u32));
        }

        // One partials buffer per tensor; dispatch all in a single encoder.
        let partials: Vec<GpuBuffer> = buffers
            .iter()
            .map(|_| GpuBuffer::zeros(&self.ctx, "gn_partial", N_GROUPS as usize))
            .collect();
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grad_norm") });
        for (i, (buf, n)) in buffers.iter().enumerate() {
            self.gn_sq.dispatch_into(&self.ctx, &mut enc, buf, &partials[i], *n, N_GROUPS);
        }
        self.ctx.queue.submit(Some(enc.finish()));

        // Sum all partials on host (cheap: N_TENSORS × 8 floats).
        let mut sq = 0.0_f32;
        for part in &partials {
            let vals = part.download(&self.ctx);
            sq += vals.iter().sum::<f32>();
        }
        sq.sqrt()
    }

    /// Clip global gradient norm. Returns pre-clip norm for logging.
    pub fn clip_grads(&self, max_norm: f32) -> f32 {
        let cfg = self.config.clone();
        let n = self.grad_norm();
        if n <= max_norm || n == 0.0 {
            return n;
        }
        let factor = max_norm / n;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clip") });
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        self.scaleb.dispatch_into(&self.ctx, &mut enc, &self.p.g_w_tok, (cfg.vocab * d) as u32, factor);
        self.scaleb.dispatch_into(&self.ctx, &mut enc, &self.p.g_w_pos, (cfg.seq_len_max * d) as u32, factor);
        self.scaleb.dispatch_into(&self.ctx, &mut enc, &self.p.g_final_gamma, d as u32, factor);
        self.scaleb.dispatch_into(&self.ctx, &mut enc, &self.p.g_final_beta, d as u32, factor);
        for blk in &self.p.blocks {
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_ln1_gamma, d as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_ln1_beta, d as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_q, (d * d) as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_k, (d * d) as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_v, (d * d) as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_o, (d * d) as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_ln2_gamma, d as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_ln2_beta, d as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_fc, (d * d_ff) as u32, factor);
            self.scaleb.dispatch_into(&self.ctx, &mut enc, &blk.g_w_proj, (d * d_ff) as u32, factor);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        n
    }

    /// Full forward + backward + loss computation. Single-sequence
    /// version (B=1) to keep state minimal. Caller is expected to
    /// reset gradients via `zero_grad()` between minibatches.
    ///
    /// Returns the per-token mean cross-entropy loss.
    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> f32 {
        let cfg = self.config.clone();
        let t = tokens.len();
        let d = cfg.d_model;
        let v = cfg.vocab;
        let n_head = cfg.n_head;
        let d_head = d / n_head;
        let d_ff = cfg.d_ff;
        debug_assert_eq!(targets.len(), t);

        // ---- FORWARD ----
        let tok_buf = GpuBuffer::from_u32_slice(&self.ctx, "tok", tokens);
        let h0 = GpuBuffer::zeros(&self.ctx, "h0", t * d);

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd embed") });
        self.embed_fwd.dispatch_into(
            &self.ctx, &mut enc, &tok_buf, &self.p.w_tok, &self.p.w_pos, &h0,
            t as u32, d as u32, v as u32, cfg.seq_len_max as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // Per-block forward — cache GPU buffers needed for backward.
        let mut x = h0; // block input
        let mut block_caches: Vec<BlockForwardCache> = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            let blk = &self.p.blocks[li];
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd block"),
                });

            let ln1_out = GpuBuffer::zeros(&self.ctx, "ln1_out", t * d);
            let ln1_mean = GpuBuffer::zeros(&self.ctx, "ln1_mean", t);
            let ln1_inv = GpuBuffer::zeros(&self.ctx, "ln1_inv", t);
            let ln1_xhat = GpuBuffer::zeros(&self.ctx, "ln1_xhat", t * d);
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &x, &blk.ln1_gamma, &blk.ln1_beta, &ln1_out, &ln1_mean,
                &ln1_inv, &ln1_xhat, t as u32, d as u32, 1e-5,
            );

            let q = GpuBuffer::zeros(&self.ctx, "q", t * d);
            let k = GpuBuffer::zeros(&self.ctx, "k", t * d);
            let val = GpuBuffer::zeros(&self.ctx, "v", t * d);
            // Each weight is [d_out × d_in] row-major; linear forward
            // y[t, o] = sum_i x[t, i] * w[o, i] = gemm_t(x, w) with shapes
            // [t, d] × [d, d]^T → [t, d].
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k, t as u32, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &val, t as u32, d as u32, d as u32);

            let heads_out = GpuBuffer::zeros(&self.ctx, "heads", t * d);
            self.ctx.queue.submit(Some(enc.finish()));

            let mut head_probs: Vec<GpuBuffer> = Vec::with_capacity(n_head);
            for h in 0..n_head {
                let pb = GpuBuffer::zeros(&self.ctx, "probs_h", t * t);
                let mut e = self.ctx.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("mha fwd") },
                );
                self.mha_fwd.dispatch_into(
                    &self.ctx, &mut e, &q, &k, &val, &heads_out, &pb,
                    t as u32, n_head as u32, d_head as u32, h as u32,
                );
                self.ctx.queue.submit(Some(e.finish()));
                head_probs.push(pb);
            }

            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd block2"),
                });

            let mha_out = GpuBuffer::zeros(&self.ctx, "mha_out", t * d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &heads_out, &blk.w_o, &mha_out, t as u32, d as u32, d as u32);

            let post_attn = GpuBuffer::zeros(&self.ctx, "post_attn", t * d);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &x, &mha_out, &post_attn, (t * d) as u32);

            let ln2_out = GpuBuffer::zeros(&self.ctx, "ln2_out", t * d);
            let ln2_mean = GpuBuffer::zeros(&self.ctx, "ln2_mean", t);
            let ln2_inv = GpuBuffer::zeros(&self.ctx, "ln2_inv", t);
            let ln2_xhat = GpuBuffer::zeros(&self.ctx, "ln2_xhat", t * d);
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &post_attn, &blk.ln2_gamma, &blk.ln2_beta, &ln2_out,
                &ln2_mean, &ln2_inv, &ln2_xhat, t as u32, d as u32, 1e-5,
            );

            let h_pre = GpuBuffer::zeros(&self.ctx, "h_pre", t * d_ff);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln2_out, &blk.w_fc, &h_pre, t as u32, d_ff as u32, d as u32);

            let h_post = GpuBuffer::zeros(&self.ctx, "h_post", t * d_ff);
            self.gelu_fwd.dispatch_into(&self.ctx, &mut enc, &h_pre, &h_post, (t * d_ff) as u32);

            let ffn_out = GpuBuffer::zeros(&self.ctx, "ffn_out", t * d);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &h_post, &blk.w_proj, &ffn_out, t as u32, d as u32, d_ff as u32);

            let z = GpuBuffer::zeros(&self.ctx, "z", t * d);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &post_attn, &ffn_out, &z, (t * d) as u32);
            self.ctx.queue.submit(Some(enc.finish()));

            block_caches.push(BlockForwardCache {
                ln1_out,
                ln1_xhat,
                ln1_inv,
                q,
                k,
                v: val,
                heads_out,
                head_probs,
                ln2_out,
                ln2_xhat,
                ln2_inv,
                h_pre,
                h_post,
            });
            x = z;
        }

        // Final LN + tied LM head
        let h_final = GpuBuffer::zeros(&self.ctx, "h_final", t * d);
        let f_mean = GpuBuffer::zeros(&self.ctx, "f_mean", t);
        let f_inv = GpuBuffer::zeros(&self.ctx, "f_inv", t);
        let f_xhat = GpuBuffer::zeros(&self.ctx, "f_xhat", t * d);
        let logits_buf = GpuBuffer::zeros(&self.ctx, "logits", t * v);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd final") });
        self.ln_fwd.dispatch_into(
            &self.ctx, &mut enc, &x, &self.p.final_gamma, &self.p.final_beta, &h_final,
            &f_mean, &f_inv, &f_xhat, t as u32, d as u32, 1e-5,
        );
        // logits[T, V] = h_final[T, D] @ w_tok.T[D, V] = gemm_t(h_final, w_tok)
        self.gemm_t.dispatch_into(
            &self.ctx, &mut enc, &h_final, &self.p.w_tok, &logits_buf, t as u32, v as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        let logits = logits_buf.download(&self.ctx);

        // ---- LOSS + d_logits on host ----
        let mut total_loss = 0.0_f32;
        let mut d_logits = vec![0.0_f32; t * v];
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let (loss, grad) = cross_entropy_with_grad(row, targets[ti] as usize);
            total_loss += loss;
            for c in 0..v {
                d_logits[ti * v + c] = grad[c] / t as f32;
            }
        }
        let mean_loss = total_loss / t as f32;

        // ---- BACKWARD ----
        let d_logits_buf = GpuBuffer::from_slice(&self.ctx, "d_logits", &d_logits);

        // d_h_final[T, D] = d_logits[T, V] @ w_tok[V, D] = gemm(d_logits, w_tok)
        let d_h_final = GpuBuffer::zeros(&self.ctx, "d_h_final", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd head") });
        self.gemm.dispatch_into(
            &self.ctx, &mut enc, &d_logits_buf, &self.p.w_tok, &d_h_final, t as u32, d as u32, v as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // d_w_tok += d_logits.T @ h_final  (tied LM head backward).
        // Shape contract: x=h_final[t, d], d_y=d_logits[t, v], dest=g_w_tok[v, d].
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd w_tok tied head accum"),
            });
        self.outer_accum.dispatch_into(
            &self.ctx, &mut enc, &h_final, &d_logits_buf, &self.p.g_w_tok,
            v as u32, d as u32, t as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        let _ = d_logits;

        // Final LN backward: produces d_x_before_final_ln and per-row d_gamma/d_beta.
        let d_pre_final = GpuBuffer::zeros(&self.ctx, "d_pre_final", t * d);
        let dg_pr = GpuBuffer::zeros(&self.ctx, "dg_pr", t * d);
        let db_pr = GpuBuffer::zeros(&self.ctx, "db_pr", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd final ln") });
        self.ln_bwd.dispatch_into(
            &self.ctx, &mut enc, &d_h_final, &self.p.final_gamma, &f_xhat, &f_inv, &d_pre_final,
            &dg_pr, &db_pr, t as u32, d as u32,
        );
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd final ln reduce"),
            });
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &dg_pr, &self.p.g_final_gamma, t as u32, d as u32);
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &db_pr, &self.p.g_final_beta, t as u32, d as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // Block stack backward (in reverse).
        let mut d_h = d_pre_final;
        for li in (0..cfg.n_layer).rev() {
            d_h = self.block_backward(&block_caches[li], li, d_h, t, d, n_head, d_head, d_ff);
        }

        // Embedding backward — per-position deltas, then GPU scatter-reduce.
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd embed") });
        let d_w_tok_per_pos = GpuBuffer::zeros(&self.ctx, "dwt_pr", t * d);
        self.embed_bwd.dispatch_into(
            &self.ctx, &mut enc, &tok_buf, &d_h, &d_w_tok_per_pos, &self.p.g_w_pos,
            t as u32, d as u32, v as u32, cfg.seq_len_max as u32,
        );
        self.embed_scatter_reduce.dispatch_into(
            &self.ctx, &mut enc, &tok_buf, &d_w_tok_per_pos, &self.p.g_w_tok, t as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        mean_loss
    }

    fn block_backward(
        &self,
        cache: &BlockForwardCache,
        block_idx: usize,
        d_z: GpuBuffer,
        t: usize,
        d: usize,
        n_head: usize,
        d_head: usize,
        d_ff: usize,
    ) -> GpuBuffer {
        let blk = &self.p.blocks[block_idx];

        // d_post_attn (from FFN residual) = d_z directly counted, plus d_post_attn_from_ffn.
        // d_ffn_out = d_z, then FFN backward.
        // d_ln2_out = gemm(d_ffn_out, w_proj)  [shapes: T, d_ff] = [T, d] @ [d, d_ff]
        let d_h_post = GpuBuffer::zeros(&self.ctx, "d_h_post", t * d_ff);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd ffn proj") });
        self.gemm.dispatch_into(
            &self.ctx, &mut enc, &d_z, &blk.w_proj, &d_h_post, t as u32, d_ff as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // d_w_proj is shape [d, d_ff]: rows = output features, cols = input.
        // dy = d_z [t, d], x = h_post [t, d_ff].
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd w_proj accum"),
            });
        self.outer_accum.dispatch_into(
            &self.ctx, &mut enc, &cache.h_post, &d_z, &blk.g_w_proj,
            d as u32, d_ff as u32, t as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // d_h_pre = gelu_bwd(h_pre, d_h_post)
        let d_h_pre = GpuBuffer::zeros(&self.ctx, "d_h_pre", t * d_ff);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd gelu") });
        self.gelu_bwd.dispatch_into(
            &self.ctx, &mut enc, &cache.h_pre, &d_h_post, &d_h_pre, (t * d_ff) as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // d_ln2_out = gemm(d_h_pre, w_fc)
        let d_ln2_out = GpuBuffer::zeros(&self.ctx, "d_ln2", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd fc") });
        self.gemm.dispatch_into(
            &self.ctx, &mut enc, &d_h_pre, &blk.w_fc, &d_ln2_out, t as u32, d as u32, d_ff as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        // w_fc is [d_ff, d]: dy=d_h_pre [t, d_ff], x=ln2_out [t, d].
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd w_fc accum"),
            });
        self.outer_accum.dispatch_into(
            &self.ctx, &mut enc, &cache.ln2_out, &d_h_pre, &blk.g_w_fc,
            d_ff as u32, d as u32, t as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // d_post_attn_from_ffn = ln2_bwd(d_ln2_out)
        let d_post_attn_ffn = GpuBuffer::zeros(&self.ctx, "d_pa_ffn", t * d);
        let dg_pr = GpuBuffer::zeros(&self.ctx, "dg_pr2", t * d);
        let db_pr = GpuBuffer::zeros(&self.ctx, "db_pr2", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd ln2") });
        self.ln_bwd.dispatch_into(
            &self.ctx, &mut enc, &d_ln2_out, &blk.ln2_gamma, &cache.ln2_xhat, &cache.ln2_inv,
            &d_post_attn_ffn, &dg_pr, &db_pr, t as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd ln2 reduce") });
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &dg_pr, &blk.g_ln2_gamma, t as u32, d as u32);
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &db_pr, &blk.g_ln2_beta, t as u32, d as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // d_post_attn = d_z (residual) + d_post_attn_ffn
        let d_post_attn = GpuBuffer::zeros(&self.ctx, "d_post_attn", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd residual post") });
        self.vadd.dispatch_into(&self.ctx, &mut enc, &d_z, &d_post_attn_ffn, &d_post_attn, (t * d) as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // MHA branch: d_mha_out = d_post_attn (residual passthrough).
        // d_heads_out = gemm(d_mha_out, w_o)
        let d_heads_out = GpuBuffer::zeros(&self.ctx, "d_heads_out", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd wo") });
        self.gemm.dispatch_into(
            &self.ctx, &mut enc, &d_post_attn, &blk.w_o, &d_heads_out, t as u32, d as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd w_o accum") });
        self.outer_accum.dispatch_into(
            &self.ctx, &mut enc, &cache.heads_out, &d_post_attn, &blk.g_w_o,
            d as u32, d as u32, t as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        // MHA backward per head: accumulate dq, dk, dv across heads.
        // Sprint 60B: 3-pass tiled flash backward (parallelized).
        let dq = GpuBuffer::zeros(&self.ctx, "dq", t * d);
        let dk = GpuBuffer::zeros(&self.ctx, "dk", t * d);
        let dv = GpuBuffer::zeros(&self.ctx, "dv", t * d);
        let d_scores_scratch = GpuBuffer::zeros(&self.ctx, "d_scores", t * t);
        for h in 0..n_head {
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd mha") });
            self.mha_bwd.dispatch_into(
                &self.ctx, &mut enc, &cache.q, &cache.k, &cache.v, &cache.head_probs[h],
                &d_heads_out, &dq, &dk, &dv, &d_scores_scratch,
                t as u32, n_head as u32, d_head as u32, h as u32,
            );
            self.ctx.queue.submit(Some(enc.finish()));
        }

        let dx_q = GpuBuffer::zeros(&self.ctx, "dx_q", t * d);
        let dx_k = GpuBuffer::zeros(&self.ctx, "dx_k", t * d);
        let dx_v = GpuBuffer::zeros(&self.ctx, "dx_v", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd qkv proj") });
        self.gemm.dispatch_into(&self.ctx, &mut enc, &dq, &blk.w_q, &dx_q, t as u32, d as u32, d as u32);
        self.gemm.dispatch_into(&self.ctx, &mut enc, &dk, &blk.w_k, &dx_k, t as u32, d as u32, d as u32);
        self.gemm.dispatch_into(&self.ctx, &mut enc, &dv, &blk.w_v, &dx_v, t as u32, d as u32, d as u32);
        let d_ln1_qk = GpuBuffer::zeros(&self.ctx, "d_ln1_qk", t * d);
        self.vadd.dispatch_into(&self.ctx, &mut enc, &dx_q, &dx_k, &d_ln1_qk, (t * d) as u32);
        let d_ln1_out = GpuBuffer::zeros(&self.ctx, "d_ln1", t * d);
        self.vadd.dispatch_into(&self.ctx, &mut enc, &d_ln1_qk, &dx_v, &d_ln1_out, (t * d) as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // Accumulate dW_q, dW_k, dW_v: each is `dx[t, d_out].T @ ln1_out[t, d_in]`
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bwd qkv accum"),
            });
        self.outer_accum.dispatch_into(&self.ctx, &mut enc, &cache.ln1_out, &dq, &blk.g_w_q, d as u32, d as u32, t as u32);
        self.outer_accum.dispatch_into(&self.ctx, &mut enc, &cache.ln1_out, &dk, &blk.g_w_k, d as u32, d as u32, t as u32);
        self.outer_accum.dispatch_into(&self.ctx, &mut enc, &cache.ln1_out, &dv, &blk.g_w_v, d as u32, d as u32, t as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // ln1 backward: d_x_from_mha = ln1_bwd(d_ln1_out)
        let d_x_mha = GpuBuffer::zeros(&self.ctx, "d_x_mha", t * d);
        let dg_pr = GpuBuffer::zeros(&self.ctx, "dg_pr_ln1", t * d);
        let db_pr = GpuBuffer::zeros(&self.ctx, "db_pr_ln1", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd ln1") });
        self.ln_bwd.dispatch_into(
            &self.ctx, &mut enc, &d_ln1_out, &blk.ln1_gamma, &cache.ln1_xhat, &cache.ln1_inv,
            &d_x_mha, &dg_pr, &db_pr, t as u32, d as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd ln1 reduce") });
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &dg_pr, &blk.g_ln1_gamma, t as u32, d as u32);
        self.row_reduce.dispatch_into(&self.ctx, &mut enc, &db_pr, &blk.g_ln1_beta, t as u32, d as u32);
        self.ctx.queue.submit(Some(enc.finish()));

        // dx (block input) = d_post_attn + d_x_mha (residual ⊕ MHA branch)
        let dx = GpuBuffer::zeros(&self.ctx, "dx_blk", t * d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bwd dx blk") });
        self.vadd.dispatch_into(&self.ctx, &mut enc, &d_post_attn, &d_x_mha, &dx, (t * d) as u32);
        self.ctx.queue.submit(Some(enc.finish()));
        dx
    }

    /// Apply AdamW step to all parameter tensors.
    pub fn step_adamw(&mut self, config: &AdamWConfig) {
        self.step += 1;
        let (bc1, bc2) = bias_correction(self.step as usize, config.beta1, config.beta2);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora adamw step"),
            });
        let cfg = self.config.clone();
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        // Token embeddings (with WD)
        self.adamw.dispatch_into(
            &self.ctx, &mut enc, &self.p.w_tok, &self.p.g_w_tok,
            &self.p.m1_w_tok, &self.p.m2_w_tok,
            (cfg.vocab * d) as u32, true, config.lr, config.beta1, config.beta2,
            config.eps, bc1, bc2, config.weight_decay,
        );
        // Positional embeddings (no WD)
        self.adamw.dispatch_into(
            &self.ctx, &mut enc, &self.p.w_pos, &self.p.g_w_pos,
            &self.p.m1_w_pos, &self.p.m2_w_pos,
            (cfg.seq_len_max * d) as u32, false, config.lr, config.beta1, config.beta2,
            config.eps, bc1, bc2, config.weight_decay,
        );
        // Final LN
        self.adamw.dispatch_into(
            &self.ctx, &mut enc, &self.p.final_gamma, &self.p.g_final_gamma,
            &self.p.m1_final_gamma, &self.p.m2_final_gamma,
            d as u32, false, config.lr, config.beta1, config.beta2,
            config.eps, bc1, bc2, config.weight_decay,
        );
        self.adamw.dispatch_into(
            &self.ctx, &mut enc, &self.p.final_beta, &self.p.g_final_beta,
            &self.p.m1_final_beta, &self.p.m2_final_beta,
            d as u32, false, config.lr, config.beta1, config.beta2,
            config.eps, bc1, bc2, config.weight_decay,
        );
        // Per-block
        for blk in &mut self.p.blocks {
            macro_rules! step {
                ($p:expr, $g:expr, $m1:expr, $m2:expr, $n:expr, $wd:expr) => {
                    self.adamw.dispatch_into(
                        &self.ctx, &mut enc, &$p, &$g, &$m1, &$m2,
                        $n as u32, $wd, config.lr, config.beta1, config.beta2,
                        config.eps, bc1, bc2, config.weight_decay,
                    );
                };
            }
            step!(blk.ln1_gamma, blk.g_ln1_gamma, blk.m1.ln1_gamma, blk.m2.ln1_gamma, d, false);
            step!(blk.ln1_beta, blk.g_ln1_beta, blk.m1.ln1_beta, blk.m2.ln1_beta, d, false);
            step!(blk.w_q, blk.g_w_q, blk.m1.w_q, blk.m2.w_q, d * d, true);
            step!(blk.w_k, blk.g_w_k, blk.m1.w_k, blk.m2.w_k, d * d, true);
            step!(blk.w_v, blk.g_w_v, blk.m1.w_v, blk.m2.w_v, d * d, true);
            step!(blk.w_o, blk.g_w_o, blk.m1.w_o, blk.m2.w_o, d * d, true);
            step!(blk.ln2_gamma, blk.g_ln2_gamma, blk.m1.ln2_gamma, blk.m2.ln2_gamma, d, false);
            step!(blk.ln2_beta, blk.g_ln2_beta, blk.m1.ln2_beta, blk.m2.ln2_beta, d, false);
            step!(blk.w_fc, blk.g_w_fc, blk.m1.w_fc, blk.m2.w_fc, d * d_ff, true);
            step!(blk.w_proj, blk.g_w_proj, blk.m1.w_proj, blk.m2.w_proj, d * d_ff, true);
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

        // 1. Embed new token at position `pos`.
        let h_cur = GpuBuffer::zeros(&self.ctx, "step_h", d);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step embed"),
            });
        self.single_embed.dispatch_into(
            &self.ctx,
            &mut enc,
            &self.p.w_tok,
            &self.p.w_pos,
            &h_cur,
            d as u32,
            v_dim as u32,
            cfg.seq_len_max as u32,
            new_token,
            pos as u32,
        );
        self.ctx.queue.submit(Some(enc.finish()));

        let mha_scratch = GpuBuffer::zeros(&self.ctx, "step_scratch", cache.max_len);

        // 2. Per-block single-step forward.
        let mut h = h_cur;
        let t_total_after = pos + 1;
        for li in 0..cfg.n_layer {
            let blk = &self.p.blocks[li];
            let kv = &cache.blocks[li];

            // LN1
            let ln1_out = GpuBuffer::zeros(&self.ctx, "ln1_out", d);
            let ln1_mean = GpuBuffer::zeros(&self.ctx, "ln1_mean", 1);
            let ln1_inv = GpuBuffer::zeros(&self.ctx, "ln1_inv", 1);
            let ln1_xhat = GpuBuffer::zeros(&self.ctx, "ln1_xhat", d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("step ln1"),
                });
            self.ln_fwd.dispatch_into(
                &self.ctx, &mut enc, &h, &blk.ln1_gamma, &blk.ln1_beta, &ln1_out, &ln1_mean,
                &ln1_inv, &ln1_xhat, 1, d as u32, 1e-5,
            );
            self.ctx.queue.submit(Some(enc.finish()));

            // Q, K, V projections (T=1).
            let q_new = GpuBuffer::zeros(&self.ctx, "step_q", d);
            let k_new = GpuBuffer::zeros(&self.ctx, "step_k", d);
            let v_new = GpuBuffer::zeros(&self.ctx, "step_v", d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("step qkv"),
                });
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_q, &q_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_k, &k_new, 1, d as u32, d as u32);
            self.gemm_t.dispatch_into(&self.ctx, &mut enc, &ln1_out, &blk.w_v, &v_new, 1, d as u32, d as u32);
            // Append K_new and V_new to history at offset pos.
            let row_bytes = (d * std::mem::size_of::<f32>()) as u64;
            let dst_offset = (pos * d * std::mem::size_of::<f32>()) as u64;
            enc.copy_buffer_to_buffer(&k_new.inner, 0, &kv.k_history.inner, dst_offset, row_bytes);
            enc.copy_buffer_to_buffer(&v_new.inner, 0, &kv.v_history.inner, dst_offset, row_bytes);
            self.ctx.queue.submit(Some(enc.finish()));

            // MHA: single-query against full history, per head.
            let heads_out = GpuBuffer::zeros(&self.ctx, "step_heads", d);
            for head_idx in 0..n_head {
                let mut enc = self
                    .ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("step mha_kv"),
                    });
                self.mha_kv.dispatch_into(
                    &self.ctx,
                    &mut enc,
                    &q_new,
                    &kv.k_history,
                    &kv.v_history,
                    &heads_out,
                    &mha_scratch,
                    t_total_after as u32,
                    n_head as u32,
                    d_head as u32,
                    head_idx as u32,
                );
                self.ctx.queue.submit(Some(enc.finish()));
            }

            // Output proj + residual + FFN.
            let mha_out = GpuBuffer::zeros(&self.ctx, "step_mha_out", d);
            let post_attn = GpuBuffer::zeros(&self.ctx, "step_post_attn", d);
            let ln2_out = GpuBuffer::zeros(&self.ctx, "step_ln2_out", d);
            let ln2_mean = GpuBuffer::zeros(&self.ctx, "step_ln2_mean", 1);
            let ln2_inv = GpuBuffer::zeros(&self.ctx, "step_ln2_inv", 1);
            let ln2_xhat = GpuBuffer::zeros(&self.ctx, "step_ln2_xhat", d);
            let h_pre = GpuBuffer::zeros(&self.ctx, "step_h_pre", d_ff);
            let h_post = GpuBuffer::zeros(&self.ctx, "step_h_post", d_ff);
            let ffn_out = GpuBuffer::zeros(&self.ctx, "step_ffn_out", d);
            let z = GpuBuffer::zeros(&self.ctx, "step_z", d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("step block tail"),
                });
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
            self.ctx.queue.submit(Some(enc.finish()));

            h = z;
        }

        // 3. Final LN + tied LM head.
        let h_final = GpuBuffer::zeros(&self.ctx, "step_h_final", d);
        let f_mean = GpuBuffer::zeros(&self.ctx, "step_f_mean", 1);
        let f_inv = GpuBuffer::zeros(&self.ctx, "step_f_inv", 1);
        let f_xhat = GpuBuffer::zeros(&self.ctx, "step_f_xhat", d);
        let logits = GpuBuffer::zeros(&self.ctx, "step_logits", v_dim);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step final + head"),
            });
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
        let h0 = GpuBuffer::zeros(&self.ctx, "h0", t * d);

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

            let ln1_out = GpuBuffer::zeros(&self.ctx, "ln1_out", t * d);
            let ln1_mean = GpuBuffer::zeros(&self.ctx, "ln1_mean", t);
            let ln1_inv = GpuBuffer::zeros(&self.ctx, "ln1_inv", t);
            let ln1_xhat = GpuBuffer::zeros(&self.ctx, "ln1_xhat", t * d);
            let q = GpuBuffer::zeros(&self.ctx, "q", t * d);
            let k = GpuBuffer::zeros(&self.ctx, "k", t * d);
            let val = GpuBuffer::zeros(&self.ctx, "v", t * d);
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
            self.ctx.queue.submit(Some(enc.finish()));

            // Heads_out via per-head MHA forward.
            let heads_out = GpuBuffer::zeros(&self.ctx, "heads", t * d);
            for h in 0..n_head {
                let pb = GpuBuffer::zeros(&self.ctx, "probs_h", t * t);
                let mut e = self.ctx.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("prefill mha") },
                );
                self.mha_fwd.dispatch_into(
                    &self.ctx, &mut e, &q, &k, &val, &heads_out, &pb,
                    t as u32, n_head as u32, d_head as u32, h as u32,
                );
                self.ctx.queue.submit(Some(e.finish()));
            }

            // Output proj + residual + FFN to keep next-block input current.
            let mha_out = GpuBuffer::zeros(&self.ctx, "mha_out", t * d);
            let post_attn = GpuBuffer::zeros(&self.ctx, "post_attn", t * d);
            let ln2_out = GpuBuffer::zeros(&self.ctx, "ln2_out", t * d);
            let ln2_mean = GpuBuffer::zeros(&self.ctx, "ln2_mean", t);
            let ln2_inv = GpuBuffer::zeros(&self.ctx, "ln2_inv", t);
            let ln2_xhat = GpuBuffer::zeros(&self.ctx, "ln2_xhat", t * d);
            let h_pre = GpuBuffer::zeros(&self.ctx, "h_pre", t * d_ff);
            let h_post = GpuBuffer::zeros(&self.ctx, "h_post", t * d_ff);
            let ffn_out = GpuBuffer::zeros(&self.ctx, "ffn_out", t * d);
            let z = GpuBuffer::zeros(&self.ctx, "z", t * d);
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prefill block tail"),
                });
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

struct BlockForwardCache {
    ln1_out: GpuBuffer,
    ln1_xhat: GpuBuffer,
    ln1_inv: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    heads_out: GpuBuffer,
    head_probs: Vec<GpuBuffer>,
    ln2_out: GpuBuffer,
    ln2_xhat: GpuBuffer,
    ln2_inv: GpuBuffer,
    h_pre: GpuBuffer,
    h_post: GpuBuffer,
}

