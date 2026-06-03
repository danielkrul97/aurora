#![allow(clippy::too_many_arguments, clippy::similar_names, clippy::many_single_char_names, clippy::too_many_lines)]

//! Full on-device thermo-LM training: cached rollout forward + rollout
//! backward (BPTT) + AdamW step, assembled from the parity-verified
//! kernels. Mirrors `crate::thermo::model::ThermoLM` forward/backward
//! exactly, including the moment and/or spectral-DCT readout.
//!
//! The field is one chain across the window. Every sub-step caches its
//! input field + conv pre/post activations + diffusion Laplacian for the
//! reverse pass ("store full trajectory"; checkpointing deferred, §5).

use super::buffers::GpuBuffer;
use super::thermo_conv::{
    ThermoBiasGeluKernel, ThermoConv1x1Kernel, ThermoConv3x3DsKernel,
    ThermoConv3x3GeluCacheKernel, ThermoConvBwdHiddenKernel, ThermoIm2colDw1Kernel,
};
use super::thermo_step::{
    StepTransportBwdKernel, StepTransportFwdKernel, ThermoBiasRowAddKernel, ThermoScaleKernel,
};
use super::thermo_stencil::{
    ThermoAvgPoolBackwardKernel, ThermoAvgPoolKernel, ThermoDiffusionBwdDsKernel,
    ThermoDiffusionBwdParamKernel, ThermoDiffusionBwdPreKernel, ThermoDiffusionForwardKernel,
    ThermoFeatureAssembleKernel, ThermoFeatureDisassembleKernel, ThermoInjectBackwardBatchedKernel,
    ThermoInjectBatchedKernel, ThermoSpectralBackwardKernel, ThermoSpectralKernel,
};
use super::{
    AdamWKernel, GemmKernel, GemmTransposeBKernel, GpuContext, GradNormSqReduceKernel,
    MuonApplyKernel, MuonCombineBKernel, MuonFrobInvKernel, MuonLoadStoreKernel, MuonMomentumKernel,
    MuonStepXKernel, OuterProductAccumKernel, OuterProductSplitKKernel, RowReduceAddKernel,
    ScaleByBufferKernel, SoftmaxCrossEntropyKernel, VectorAdd3Kernel, VectorAddKernel,
    ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig, MuonConfig};
use crate::thermo::field::{free_channels_for, init_field, ThermoConfig};
use crate::thermo::model::{anneal_at, ThermoGrads, ThermoLM};

/// One trainable tensor on device: value + grad + AdamW moments.
struct Param {
    buf: GpuBuffer,
    grad: GpuBuffer,
    m1: GpuBuffer,
    m2: GpuBuffer,
    wd: bool,
}

impl Param {
    fn new(ctx: &GpuContext, label: &str, data: &[f32], wd: bool) -> Self {
        Self {
            buf: GpuBuffer::from_slice(ctx, label, data),
            grad: GpuBuffer::zeros(ctx, label, data.len()),
            m1: GpuBuffer::zeros(ctx, label, data.len()),
            m2: GpuBuffer::zeros(ctx, label, data.len()),
            wd,
        }
    }
}

pub struct ThermoGpu {
    ctx: GpuContext,
    cfg: ThermoConfig,
    seq_len: usize,
    /// On-device batch: B independent windows rolled out per step.
    batch: usize,
    step_count: u64,
    /// Use the fused 3-input `d_s` add (default) vs the two-`VectorAdd` chain.
    /// Runtime-toggleable so A/B benchmarks compare both under one GPU clock.
    pub fused_add: bool,
    /// Route the forward conv3×3 through im2col + tiled `GemmTransposeB`
    /// (+ bias/GELU epilogue) instead of the per-element fused kernel.
    /// Runtime-toggleable for in-process A/B benchmarks.
    pub conv_im2col: bool,
    /// Re-inject the token (anneal-scaled) before every sub-step (H31).
    /// Set by the trainer; mirrors `ThermoLM::inject_every`.
    pub inject_every: bool,

    conv3g: ThermoConv3x3GeluCacheKernel,
    bias_gelu: ThermoBiasGeluKernel,
    conv1: ThermoConv1x1Kernel,
    diff_fwd: ThermoDiffusionForwardKernel,
    transport_fwd: StepTransportFwdKernel,
    pool: ThermoAvgPoolKernel,
    gtb: GemmTransposeBKernel,
    vadd: VectorAddKernel,
    vadd3: VectorAdd3Kernel,
    inject: ThermoInjectBatchedKernel,
    bias_add: ThermoBiasRowAddKernel,
    sce: SoftmaxCrossEntropyKernel,
    scale: ThermoScaleKernel,
    diff_pre: ThermoDiffusionBwdPreKernel,
    diff_param: ThermoDiffusionBwdParamKernel,
    diff_ds: ThermoDiffusionBwdDsKernel,
    transport_bwd: StepTransportBwdKernel,
    gemm: GemmKernel,
    outer: OuterProductAccumKernel,
    splitk: OuterProductSplitKKernel,
    row: RowReduceAddKernel,
    conv_bwd_hidden: ThermoConvBwdHiddenKernel,
    conv_ds: ThermoConv3x3DsKernel,
    im2col_dw1: ThermoIm2colDw1Kernel,
    pool_bwd: ThermoAvgPoolBackwardKernel,
    inject_bwd: ThermoInjectBackwardBatchedKernel,
    spectral: ThermoSpectralKernel,
    spectral_bwd: ThermoSpectralBackwardKernel,
    assemble: ThermoFeatureAssembleKernel,
    disassemble: ThermoFeatureDisassembleKernel,
    zero: ZeroBufferKernel,
    adamw: AdamWKernel,

    // Muon (matrix params) — GPU port of `ThermoLM::step_muon_hybrid`.
    mu_momentum: MuonMomentumKernel,
    mu_frob_inv: MuonFrobInvKernel,
    mu_load_store: MuonLoadStoreKernel,
    mu_combine_b: MuonCombineBKernel,
    mu_step_x: MuonStepXKernel,
    mu_apply: MuonApplyKernel,
    gn_sq: GradNormSqReduceKernel,
    scaleb: ScaleByBufferKernel,
    /// Per-matrix Muon momentum buffers (W1, W2, readout proj).
    mu_m_w1: GpuBuffer,
    mu_m_w2: GpuBuffer,
    mu_m_proj: GpuBuffer,
    /// Shared Muon scratch (sized to the largest matrix). `update`/`x`/`bx`
    /// are `[m·n]`, `a`/`asq`/`b` are `[m·m]`; frob is the 8-partial reduce.
    mu_update: GpuBuffer,
    mu_x: GpuBuffer,
    mu_a: GpuBuffer,
    mu_asq: GpuBuffer,
    mu_b: GpuBuffer,
    mu_bx: GpuBuffer,
    mu_frob_partials: GpuBuffer,
    mu_frob_result: GpuBuffer,

    w1: Param,
    b1: Param,
    w2: Param,
    b2: Param,
    k0: Param,
    pa: Param,
    pb: Param,
    inj: Param,
    projw: Param,
    bias: Param,

    profile: GpuBuffer,
    kind: GpuBuffer,
    slot: GpuBuffer,
    cons_ch: GpuBuffer,
    free_ch: GpuBuffer,
    gate: GpuBuffer,

    /// Pre-inject scratch for inject-every (transport output before the next
    /// sub-step's re-injection), kept separate from `s_traj` to avoid binding
    /// the same buffer read+read_write in the inject kernel.
    preinj: GpuBuffer,
    s_traj: Vec<GpuBuffer>,
    hpre_traj: Vec<GpuBuffer>,
    hpost_traj: Vec<GpuBuffer>,
    lap_traj: Vec<GpuBuffer>,
    /// Cached readout features `[moments | spectral]` per timestep `[b*readout_dim]`.
    feat_traj: Vec<GpuBuffer>,
    fout_traj: Vec<GpuBuffer>,
    init_buf: GpuBuffer,

    /// CPU-built DCT basis `[modes²*cells]`, uploaded verbatim (spectral readout).
    basis: GpuBuffer,
    /// Forward spectral coeffs `[b*spectral_dim]` and the moments temp the
    /// spectral path pools into before assembly `[b*pooled_dim]`.
    spec: GpuBuffer,
    pooled_sc: GpuBuffer,
    /// Backward feature grad `[b*readout_dim]` and its spectral slice `[b*spectral_dim]`.
    d_feat: GpuBuffer,
    d_spec: GpuBuffer,

    raw: GpuBuffer,
    diff_delta: GpuBuffer,
    lraw: GpuBuffer,
    logit_row: GpuBuffer,
    logits: GpuBuffer,

    d_logits: GpuBuffer,
    loss_seq: GpuBuffer,
    dl_t: GpuBuffer,
    dfield_a: GpuBuffer,
    dfield_b: GpuBuffer,
    d_ro: GpuBuffer,
    d_pooled: GpuBuffer,
    ddelta: GpuBuffer,
    gbuf: GpuBuffer,
    ct: GpuBuffer,
    ds_diff: GpuBuffer,
    draw: GpuBuffer,
    /// Split-K partials for the `d_w2` reduction `[GROUPS·cout·chid]`.
    dw2_partials: GpuBuffer,
    /// Split-K partials for the `d_w1` outer-product reduction
    /// `[GROUPS·chid·c·27]` (im2col → GEMM, reuses the d_w2 path).
    dw1_partials: GpuBuffer,
    /// im2col patch matrix `[nb·cells·(c·27)]` feeding the `d_w1` GEMM.
    patch: GpuBuffer,
    dhpre: GpuBuffer,
    dsconv: GpuBuffer,
    tmp: GpuBuffer,

    tokens_buf: GpuBuffer,
    targets_buf: GpuBuffer,
}

impl ThermoGpu {
    #[must_use]
    pub fn new(ctx: &GpuContext, model: &ThermoLM, seq_len: usize, batch: usize) -> Self {
        let cfg = model.config.clone();
        assert!(cfg.readout_moments || cfg.readout_spectral, "readout must use moments and/or spectral");
        assert!(batch >= 1, "batch must be >= 1");
        let b = batch;
        let cells = cfg.cells();
        let c = cfg.c;
        let chid = cfg.conv_hidden;
        let cout = cfg.c_out();
        let pdim = cfg.pooled_dim();
        let sdim = cfg.spectral_dim();
        let rdim = cfg.readout_dim();
        let vocab = cfg.vocab;
        let nsub = seq_len * cfg.k_steps;

        let (kind, slot) = cfg.channel_decode();
        let cons_ch: Vec<u32> = cfg.conserved.iter().map(|&x| x as u32).collect();
        let free_ch: Vec<u32> = free_channels_for(&cfg).iter().map(|&x| x as u32).collect();

        let z = |n: usize| GpuBuffer::zeros(ctx, "scratch", n);
        let traj = |n: usize, len: usize| (0..len).map(|_| GpuBuffer::zeros(ctx, "traj", n)).collect::<Vec<_>>();

        // Muon matrix params: W1 (chid×c·9), W2 (cout×chid), proj (vocab×rdim).
        // Shared scratch sized to the largest: m·n (param len) and m·m (m=min dim).
        let (w1_len, w2_len, proj_len) = (chid * c * 27, cout * chid, vocab * rdim);
        let mn_max = w1_len.max(w2_len).max(proj_len);
        let mm = |r: usize, c: usize| { let m = r.min(c); m * m };
        let mm_max = mm(chid, c * 27).max(mm(cout, chid)).max(mm(vocab, rdim));

        Self {
            cfg: cfg.clone(),
            ctx: ctx.clone(),
            seq_len,
            batch,
            step_count: model.step_count,
            fused_add: true,
            // im2col + tiled GEMM forward wins once the conv dominates (3D,
            // 27-tap: +26.6% at d=4) but loses for the cheap 2D 9-tap conv
            // (−17.5% at d=1), so default to it only for d>1. Bit-exact to the
            // per-element kernel, so this is a pure speed switch.
            conv_im2col: cfg.d > 1,
            inject_every: false,
            conv3g: ThermoConv3x3GeluCacheKernel::new(ctx),
            bias_gelu: ThermoBiasGeluKernel::new(ctx),
            conv1: ThermoConv1x1Kernel::new(ctx),
            diff_fwd: ThermoDiffusionForwardKernel::new(ctx),
            transport_fwd: StepTransportFwdKernel::new(ctx),
            pool: ThermoAvgPoolKernel::new(ctx),
            gtb: GemmTransposeBKernel::new(ctx),
            vadd: VectorAddKernel::new(ctx),
            vadd3: VectorAdd3Kernel::new(ctx),
            inject: ThermoInjectBatchedKernel::new(ctx),
            bias_add: ThermoBiasRowAddKernel::new(ctx),
            sce: SoftmaxCrossEntropyKernel::new(ctx),
            scale: ThermoScaleKernel::new(ctx),
            diff_pre: ThermoDiffusionBwdPreKernel::new(ctx),
            diff_param: ThermoDiffusionBwdParamKernel::new(ctx),
            diff_ds: ThermoDiffusionBwdDsKernel::new(ctx),
            transport_bwd: StepTransportBwdKernel::new(ctx),
            gemm: GemmKernel::new(ctx),
            outer: OuterProductAccumKernel::new(ctx),
            splitk: OuterProductSplitKKernel::new(ctx),
            row: RowReduceAddKernel::new(ctx),
            conv_bwd_hidden: ThermoConvBwdHiddenKernel::new(ctx),
            conv_ds: ThermoConv3x3DsKernel::new(ctx),
            im2col_dw1: ThermoIm2colDw1Kernel::new(ctx),
            pool_bwd: ThermoAvgPoolBackwardKernel::new(ctx),
            inject_bwd: ThermoInjectBackwardBatchedKernel::new(ctx),
            spectral: ThermoSpectralKernel::new(ctx),
            spectral_bwd: ThermoSpectralBackwardKernel::new(ctx),
            assemble: ThermoFeatureAssembleKernel::new(ctx),
            disassemble: ThermoFeatureDisassembleKernel::new(ctx),
            zero: ZeroBufferKernel::new(ctx),
            adamw: AdamWKernel::new(ctx),
            mu_momentum: MuonMomentumKernel::new(ctx),
            mu_frob_inv: MuonFrobInvKernel::new(ctx),
            mu_load_store: MuonLoadStoreKernel::new(ctx),
            mu_combine_b: MuonCombineBKernel::new(ctx),
            mu_step_x: MuonStepXKernel::new(ctx),
            mu_apply: MuonApplyKernel::new(ctx),
            gn_sq: GradNormSqReduceKernel::new(ctx),
            scaleb: ScaleByBufferKernel::new(ctx),
            mu_m_w1: z(w1_len),
            mu_m_w2: z(w2_len),
            mu_m_proj: z(proj_len),
            mu_update: z(mn_max),
            mu_x: z(mn_max),
            mu_a: z(mm_max),
            mu_asq: z(mm_max),
            mu_b: z(mm_max),
            mu_bx: z(mn_max),
            mu_frob_partials: z(8),
            mu_frob_result: z(8),
            w1: Param::new(ctx, "w1", &model.step.conv.w1, true),
            b1: Param::new(ctx, "b1", &model.step.conv.b1, false),
            w2: Param::new(ctx, "w2", &model.step.conv.w2, true),
            b2: Param::new(ctx, "b2", &model.step.conv.b2, false),
            k0: Param::new(ctx, "k0", &model.step.diffusion.k0, false),
            pa: Param::new(ctx, "a", &model.step.diffusion.a, false),
            pb: Param::new(ctx, "b", &model.step.diffusion.b, false),
            inj: Param::new(ctx, "inj", &model.inject.inj, true),
            projw: Param::new(ctx, "projw", &model.readout.proj.w, true),
            bias: Param::new(ctx, "bias", &model.readout.bias, false),
            profile: GpuBuffer::from_slice(ctx, "profile", &model.inject.profile),
            kind: GpuBuffer::from_u32_slice(ctx, "kind", &kind),
            slot: GpuBuffer::from_u32_slice(ctx, "slot", &slot),
            cons_ch: GpuBuffer::from_u32_slice(ctx, "cons_ch", &cons_ch),
            free_ch: GpuBuffer::from_u32_slice(ctx, "free_ch", &free_ch),
            gate: z(3 * c),
            preinj: z(b * cells * c),
            s_traj: traj(b * cells * c, nsub),
            hpre_traj: traj(b * cells * chid, nsub),
            hpost_traj: traj(b * cells * chid, nsub),
            lap_traj: traj(b * cells * c, nsub),
            feat_traj: traj(b * rdim, seq_len),
            fout_traj: traj(b * cells * c, seq_len),
            init_buf: {
                // Warm tiled init, uploaded once (constant across steps; H8).
                let init = init_field(&cfg);
                let mut tiled = Vec::with_capacity(b * init.len());
                for _ in 0..b {
                    tiled.extend_from_slice(&init);
                }
                GpuBuffer::from_slice(ctx, "init", &tiled)
            },
            basis: GpuBuffer::from_slice(
                ctx,
                "basis",
                if model.readout.basis.is_empty() { &[0.0_f32] } else { &model.readout.basis },
            ),
            spec: z((b * sdim).max(1)),
            pooled_sc: z(b * pdim),
            d_feat: z(b * rdim),
            d_spec: z((b * sdim).max(1)),
            raw: z(b * cells * cout),
            diff_delta: z(b * cells * c),
            lraw: z(b * vocab),
            logit_row: z(b * vocab),
            logits: z(seq_len * b * vocab),
            d_logits: z(seq_len * b * vocab),
            loss_seq: z(seq_len * b),
            dl_t: z(b * vocab),
            dfield_a: z(b * cells * c),
            dfield_b: z(b * cells * c),
            d_ro: z(b * cells * c),
            d_pooled: z(b * pdim),
            ddelta: z(b * cells * c),
            gbuf: z(b * cells * c),
            ct: z(b * cells * c),
            ds_diff: z(b * cells * c),
            draw: z(b * cells * cout),
            dw2_partials: z(OuterProductSplitKKernel::GROUPS as usize * cout * chid),
            dw1_partials: z(OuterProductSplitKKernel::GROUPS as usize * chid * c * 27),
            patch: z(b * cells * c * 27),
            dhpre: z(b * cells * chid),
            dsconv: z(b * cells * c),
            tmp: z(b * cells * c),
            tokens_buf: GpuBuffer::from_u32_slice(ctx, "tok", &vec![0u32; seq_len * b]),
            targets_buf: GpuBuffer::from_u32_slice(ctx, "tgt", &vec![0u32; seq_len * b]),
        }
    }

    /// Rollout forward (teacher-forced), filling `self.logits` + caches.
    /// `tokens` is the flattened `[T, B]` window matrix (row = ti*B + b).
    pub fn forward(&self, tokens: &[u32]) {
        let cfg = &self.cfg;
        let b = self.batch;
        let t = self.seq_len;
        assert_eq!(tokens.len(), t * b, "tokens must be [T*B]");
        let (cells, c, chid, cout, nc) = (cfg.cells(), cfg.c, cfg.conv_hidden, cfg.c_out(), cfg.n_conserved());
        let (d, h, w, vocab, pdim, ks) = (cfg.d, cfg.h, cfg.w, cfg.vocab, cfg.pooled_dim(), cfg.k_steps);
        let (sdim, rdim, msq) = (cfg.spectral_dim(), cfg.readout_dim(), cfg.readout_modes * cfg.readout_modes * cfg.readout_modes);
        let dt = cfg.dt;
        let bu = b as u32;
        let du = d as u32;

        self.tokens_buf.upload_u32_prefix(&self.ctx, tokens);
        // `init_buf` holds the warm tiled init, uploaded once in `new()` — it
        // never changes (each window resets to it), so we don't rebuild and
        // re-upload it per step (H8: invariant work hoisted out of the loop).

        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo fwd") });

        for ti in 0..t {
            let carried = if ti == 0 { &self.init_buf } else { &self.fout_traj[ti - 1] };

            for k in 0..ks {
                let idx = ti * ks + k;
                let anneal = anneal_at(k, cfg);
                // Fill s_traj[idx] (the post-inject conv input). k==0 injects the
                // carried field; k>0 with inject_every injects `preinj` (the
                // previous sub-step's transport output). Without inject_every,
                // only k==0 injects (anneal 1.0) and k>0 reads transport output.
                if k == 0 {
                    let amp = if self.inject_every { anneal } else { 1.0 };
                    self.inject.dispatch_into(&self.ctx, &mut enc, carried, &self.profile, &self.inj.buf, &self.tokens_buf, &self.s_traj[idx], cells as u32, c as u32, bu, (ti * b) as u32, amp);
                } else if self.inject_every {
                    self.inject.dispatch_into(&self.ctx, &mut enc, &self.preinj, &self.profile, &self.inj.buf, &self.tokens_buf, &self.s_traj[idx], cells as u32, c as u32, bu, (ti * b) as u32, anneal);
                }
                let s_in = &self.s_traj[idx];
                if self.conv_im2col {
                    // im2col → tiled GemmTransposeB (hpre = patch @ w1ᵀ) → bias+GELU.
                    self.im2col_dw1.dispatch_b(&self.ctx, &mut enc, s_in, &self.patch, h as u32, w as u32, c as u32, bu, du);
                    self.gtb.dispatch_into(&self.ctx, &mut enc, &self.patch, &self.w1.buf, &self.hpre_traj[idx], (b * cells) as u32, chid as u32, (c * 27) as u32);
                    self.bias_gelu.dispatch_into(&self.ctx, &mut enc, &self.hpre_traj[idx], &self.b1.buf, &self.hpost_traj[idx], (b * cells) as u32, chid as u32);
                } else {
                    // fused conv3x3 + GELU: caches pre + post in one dispatch.
                    self.conv3g.dispatch_b(&self.ctx, &mut enc, s_in, &self.w1.buf, &self.b1.buf, &self.hpre_traj[idx], &self.hpost_traj[idx], h as u32, w as u32, c as u32, chid as u32, bu, du);
                }
                self.conv1.dispatch_into(&self.ctx, &mut enc, &self.hpost_traj[idx], &self.w2.buf, &self.b2.buf, &self.raw, (b * cells) as u32, chid as u32, cout as u32);
                if cfg.use_diffusion {
                    self.diff_fwd.dispatch_b(&self.ctx, &mut enc, s_in, &self.k0.buf, &self.pa.buf, &self.pb.buf, &self.diff_delta, &self.lap_traj[idx], h as u32, w as u32, c as u32, bu, du);
                }
                // Where transport writes: the last sub-step → fout; otherwise the
                // next sub-step's input slot, or `preinj` when inject_every (so
                // the next iteration can re-inject into s_traj[idx+1]).
                let out = if k == ks - 1 {
                    &self.fout_traj[ti]
                } else if self.inject_every {
                    &self.preinj
                } else {
                    &self.s_traj[idx + 1]
                };
                // fused extract + flux-divergence + assemble (no fx/fy/div buffers).
                self.transport_fwd.dispatch_b(&self.ctx, &mut enc, s_in, &self.raw, &self.diff_delta, &self.kind, &self.slot, out, h as u32, w as u32, c as u32, nc as u32, cout as u32, bu, dt, anneal, du);
            }

            // readout features → feat_traj[ti] = [moments | spectral]
            if cfg.readout_spectral {
                if cfg.readout_moments {
                    self.pool.dispatch_b(&self.ctx, &mut enc, &self.fout_traj[ti], &self.pooled_sc, h as u32, w as u32, c as u32, cfg.pool as u32, bu, du);
                }
                self.spectral.dispatch_b(&self.ctx, &mut enc, &self.fout_traj[ti], &self.basis, &self.spec, cells as u32, c as u32, msq as u32, bu);
                let mdim = if cfg.readout_moments { pdim } else { 0 };
                self.assemble.dispatch_b(&self.ctx, &mut enc, &self.pooled_sc, &self.spec, &self.feat_traj[ti], mdim as u32, sdim as u32, bu);
            } else {
                self.pool.dispatch_b(&self.ctx, &mut enc, &self.fout_traj[ti], &self.feat_traj[ti], h as u32, w as u32, c as u32, cfg.pool as u32, bu, du);
            }
            self.gtb.dispatch_into(&self.ctx, &mut enc, &self.feat_traj[ti], &self.projw.buf, &self.lraw, bu, vocab as u32, rdim as u32);
            self.bias_add.dispatch_into(&self.ctx, &mut enc, &self.lraw, &self.bias.buf, &self.logit_row, bu, vocab as u32);
            enc.copy_buffer_to_buffer(&self.logit_row.inner, 0, &self.logits.inner, (ti * b * vocab * 4) as u64, (b * vocab * 4) as u64);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Rollout backward (BPTT). Accumulates param grads; returns mean loss
    /// over the whole `B×T` batch. `tokens`/`targets` are flattened `[T, B]`.
    pub fn backward(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        self.backward_dispatch(tokens, targets);
        self.read_mean_loss()
    }

    /// Mean cross-entropy over the `B×T` batch from the last backward.
    /// Blocks on the GPU (buffer map) — call only when the value is needed
    /// (logging / eval), not every step.
    #[must_use]
    pub fn read_mean_loss(&self) -> f32 {
        let n = self.batch * self.seq_len;
        self.loss_seq.download(&self.ctx).iter().take(n).sum::<f32>() / n as f32
    }

    /// Validation NLL (nats) + scored-token count over stride-sampled,
    /// non-overlapping val windows — computed entirely on-device (the same
    /// batched `forward` + softmax-CE the training step uses), so eval no
    /// longer downloads params and re-runs the rollout on the CPU. The
    /// window striding mirrors the CPU `evaluate`, so the BPB matches.
    /// `max_windows == 0` scores every window.
    #[must_use]
    pub fn eval_nats(&self, val_toks: &[u32], max_windows: usize) -> (f64, usize) {
        let (seq, b, vocab) = (self.seq_len, self.batch, self.cfg.vocab);
        if val_toks.len() <= seq + 1 {
            return (0.0, 0);
        }
        let all: Vec<usize> = (0..)
            .map(|i| i * seq)
            .take_while(|&p| p + seq + 1 < val_toks.len())
            .collect();
        let windows: Vec<usize> = if max_windows == 0 || max_windows >= all.len() {
            all
        } else {
            (0..max_windows).map(|i| all[i * all.len() / max_windows]).collect()
        };

        let mut toks = vec![0u32; seq * b];
        let mut tgts = vec![0u32; seq * b];
        let mut total_nats = 0.0_f64;
        for chunk in windows.chunks(b) {
            let valid = chunk.len();
            for (bi, &p) in chunk.iter().enumerate() {
                for ti in 0..seq {
                    toks[ti * b + bi] = val_toks[p + ti];
                    tgts[ti * b + bi] = val_toks[p + ti + 1];
                }
            }
            // Pad the tail chunk's unused grids with window 0 so the rollout is
            // well-defined; those slots are excluded from the sum below.
            for bi in valid..b {
                for ti in 0..seq {
                    toks[ti * b + bi] = toks[ti * b];
                    tgts[ti * b + bi] = tgts[ti * b];
                }
            }

            self.forward(&toks);
            self.targets_buf.upload_u32_prefix(&self.ctx, &tgts);
            let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo eval") });
            // loss_scale 1.0 → loss_seq holds raw per-position NLL (d_logits
            // is written but unused). Same [T,B] drive as `backward_dispatch`.
            self.sce.dispatch_into(&self.ctx, &mut enc, &self.logits, &self.d_logits, &self.loss_seq, &self.targets_buf, vocab as u32, b as u32, seq as u32, 1.0, 0);
            self.ctx.queue.submit(Some(enc.finish()));

            let ls = self.loss_seq.download(&self.ctx);
            for ti in 0..seq {
                for bi in 0..valid {
                    total_nats += f64::from(ls[ti * b + bi]);
                }
            }
        }
        (total_nats, windows.len() * seq)
    }

    /// Records the BPTT pass into the queue without reading the loss back
    /// (no host sync). Use `read_mean_loss` afterwards when you need it.
    pub fn backward_dispatch(&self, tokens: &[u32], targets: &[u32]) {
        let cfg = &self.cfg;
        let b = self.batch;
        let t = self.seq_len;
        assert_eq!(tokens.len(), t * b);
        assert_eq!(targets.len(), t * b);
        let (cells, c, vocab, ks) = (cfg.cells(), cfg.c, cfg.vocab, cfg.k_steps);
        let bu = b as u32;

        self.tokens_buf.upload_u32_prefix(&self.ctx, tokens);
        self.targets_buf.upload_u32_prefix(&self.ctx, targets);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo bwd") });

        // Logits are laid out [T, B, vocab] (row = ti*B + b). Drive sce as
        // (batch=T, timesteps=B) so its bt_idx = ti*B + b matches, and scale
        // 1/(B*T) = mean over batch & positions (mirrors CPU mean + 1/B avg).
        self.sce.dispatch_into(&self.ctx, &mut enc, &self.logits, &self.d_logits, &self.loss_seq, &self.targets_buf, vocab as u32, bu, t as u32, 1.0 / (b * t) as f32, 0);

        // pack gate [k0|a|b] for the diffusion backward.
        enc.copy_buffer_to_buffer(&self.k0.buf.inner, 0, &self.gate.inner, 0, (c * 4) as u64);
        enc.copy_buffer_to_buffer(&self.pa.buf.inner, 0, &self.gate.inner, (c * 4) as u64, (c * 4) as u64);
        enc.copy_buffer_to_buffer(&self.pb.buf.inner, 0, &self.gate.inner, (2 * c * 4) as u64, (c * 4) as u64);

        // d_field ping-pong, start at zero.
        self.zero.dispatch_into(&self.ctx, &mut enc, &self.dfield_a, (b * cells * c) as u32);
        let df = [&self.dfield_a, &self.dfield_b];
        let mut cur = 0usize;

        for ti in (0..t).rev() {
            // d_logits block for the B grids at this timestep (contiguous rows).
            enc.copy_buffer_to_buffer(&self.d_logits.inner, (ti * b * vocab * 4) as u64, &self.dl_t.inner, 0, (b * vocab * 4) as u64);
            // readout backward → d_ro (d_s from readout)
            self.readout_backward(&mut enc, ti);
            // d_field += d_ro
            self.vadd.dispatch_into(&self.ctx, &mut enc, df[cur], &self.d_ro, df[1 - cur], (b * cells * c) as u32);
            cur = 1 - cur;
            // K step backwards (reverse). With inject_every, each sub-step had
            // its own (anneal-scaled) injection, so accumulate d_inj after every
            // step_backward (inject d_s is identity → df passes through).
            for k in (0..ks).rev() {
                let idx = ti * ks + k;
                self.step_backward(&mut enc, df[cur], df[1 - cur], idx, anneal_at(k, cfg));
                cur = 1 - cur;
                if self.inject_every {
                    self.inject_bwd.dispatch_into(&self.ctx, &mut enc, &self.profile, df[cur], &self.tokens_buf, &self.inj.grad, cells as u32, c as u32, bu, (ti * b) as u32, anneal_at(k, cfg));
                }
            }
            // inject backward once (d_field identity → carried) when not inject_every.
            if !self.inject_every {
                self.inject_bwd.dispatch_into(&self.ctx, &mut enc, &self.profile, df[cur], &self.tokens_buf, &self.inj.grad, cells as u32, c as u32, bu, (ti * b) as u32, 1.0);
            }
        }

        self.ctx.queue.submit(Some(enc.finish()));
    }

    fn readout_backward(&self, enc: &mut wgpu::CommandEncoder, ti: usize) {
        let cfg = &self.cfg;
        let b = self.batch as u32;
        let (vocab, rdim) = (cfg.vocab as u32, cfg.readout_dim() as u32);
        let (h, w, c, pool) = (cfg.h as u32, cfg.w as u32, cfg.c as u32, cfg.pool as u32);
        let du = cfg.d as u32;
        // d_bias += Σ_b dl[b] ; d_proj += Σ_b outer(feat[b], dl[b])
        self.row.dispatch_into(&self.ctx, enc, &self.dl_t, &self.bias.grad, b, vocab);
        self.outer.dispatch_into(&self.ctx, enc, &self.feat_traj[ti], &self.dl_t, &self.projw.grad, vocab, rdim, b);

        if cfg.readout_spectral {
            // d_feat = dl @ proj ; disassemble → moment/spectral grads → d_ro.
            self.gemm.dispatch_into(&self.ctx, enc, &self.dl_t, &self.projw.buf, &self.d_feat, b, rdim, vocab);
            let sdim = cfg.spectral_dim() as u32;
            let msq = (cfg.readout_modes * cfg.readout_modes * cfg.readout_modes) as u32;
            let cells = cfg.cells() as u32;
            let mdim = if cfg.readout_moments { cfg.pooled_dim() as u32 } else { 0 };
            self.disassemble.dispatch_b(&self.ctx, enc, &self.d_feat, &self.d_pooled, &self.d_spec, mdim, sdim, b);
            if cfg.readout_moments {
                // d_ro = avgpool_bwd(d_pooled) + spectral_bwd(d_spec)
                self.pool_bwd.dispatch_b(&self.ctx, enc, &self.d_pooled, &self.tmp, h, w, c, pool, b, du);
                self.spectral_bwd.dispatch_b(&self.ctx, enc, &self.d_spec, &self.basis, &self.ddelta, cells, c, msq, b);
                self.vadd.dispatch_into(&self.ctx, enc, &self.tmp, &self.ddelta, &self.d_ro, (self.batch * cfg.cells() * cfg.c) as u32);
            } else {
                self.spectral_bwd.dispatch_b(&self.ctx, enc, &self.d_spec, &self.basis, &self.d_ro, cells, c, msq, b);
            }
        } else {
            // moments-only: d_pooled = dl @ proj ; d_ro = avgpool_bwd(d_pooled)
            self.gemm.dispatch_into(&self.ctx, enc, &self.dl_t, &self.projw.buf, &self.d_pooled, b, rdim, vocab);
            self.pool_bwd.dispatch_b(&self.ctx, enc, &self.d_pooled, &self.d_ro, h, w, c, pool, b, du);
        }
    }

    fn step_backward(&self, enc: &mut wgpu::CommandEncoder, d_out: &GpuBuffer, d_s_out: &GpuBuffer, idx: usize, anneal: f32) {
        let cfg = &self.cfg;
        let b = self.batch;
        let (cells, c, chid, cout, nc) = (cfg.cells(), cfg.c, cfg.conv_hidden, cfg.c_out(), cfg.n_conserved());
        let (h, w) = (cfg.h as u32, cfg.w as u32);
        let du = cfg.d as u32;
        let dt = cfg.dt;
        let bu = b as u32;
        let bcells = (b * cells) as u32;

        // diffusion path
        if cfg.use_diffusion {
            self.scale.dispatch_into(&self.ctx, enc, d_out, &self.ddelta, (b * cells * c) as u32, dt * anneal);
            self.diff_pre.dispatch_b(&self.ctx, enc, &self.ddelta, &self.lap_traj[idx], &self.s_traj[idx], &self.gate, &self.gbuf, &self.ct, h, w, c as u32, bu, du);
            self.diff_param.dispatch_b(&self.ctx, enc, &self.ddelta, &self.lap_traj[idx], &self.s_traj[idx], &self.gate, &self.k0.grad, &self.pa.grad, &self.pb.grad, h, w, c as u32, bu, du);
            self.diff_ds.dispatch_b(&self.ctx, enc, &self.gbuf, &self.ct, &self.ds_diff, h, w, c as u32, bu, du);
        }

        // transport path: fused disassemble + flux-div adjoint + scatter.
        // Writes every d_raw slot, so no pre-zero is needed.
        self.transport_bwd.dispatch_b(&self.ctx, enc, d_out, &self.cons_ch, &self.free_ch, &self.draw, h, w, c as u32, nc as u32, cout as u32, bu, dt, du);
        // conv1x1 d_w2: tall-skinny (cout·chid tiny, bcells huge) → split-K
        // partials + row reduce, instead of the single-workgroup tiled GEMM.
        self.splitk.dispatch_into(&self.ctx, enc, &self.hpost_traj[idx], &self.draw, &self.dw2_partials, cout as u32, chid as u32, bcells);
        self.row.dispatch_into(&self.ctx, enc, &self.dw2_partials, &self.w2.grad, OuterProductSplitKKernel::groups_for(bcells), (cout * chid) as u32);
        self.row.dispatch_into(&self.ctx, enc, &self.draw, &self.b2.grad, bcells, cout as u32);
        // fused (d_raw·w2ᵀ) × GELU'(pre) → dhpre.
        self.conv_bwd_hidden.dispatch_into(&self.ctx, enc, &self.draw, &self.w2.buf, &self.hpre_traj[idx], &self.dhpre, bcells, chid as u32, cout as u32);
        self.conv_ds.dispatch_b(&self.ctx, enc, &self.dhpre, &self.w1.buf, &self.dsconv, h, w, c as u32, chid as u32, bu, du);
        // d_w1 via im2col → outer-product GEMM (dhpᵀ ⊗ patch): the patch
        // columns (ci*27+tap) land directly in the canonical w1 layout, and
        // the GEMM reuses the fast d_w2 split-K reduce (~1.3× vs conv split-K).
        self.im2col_dw1.dispatch_b(&self.ctx, enc, &self.s_traj[idx], &self.patch, h, w, c as u32, bu, du);
        self.splitk.dispatch_into(&self.ctx, enc, &self.patch, &self.dhpre, &self.dw1_partials, chid as u32, (c * 27) as u32, bcells);
        self.row.dispatch_into(&self.ctx, enc, &self.dw1_partials, &self.w1.grad, OuterProductSplitKKernel::groups_for(bcells), (chid * c * 27) as u32);
        self.row.dispatch_into(&self.ctx, enc, &self.dhpre, &self.b1.grad, bcells, chid as u32);

        // d_s = d_out + ds_diff + dsconv  (residual + diffusion + transport).
        // Fused 3-input add: one dispatch instead of two, no `tmp` round-trip.
        let n = (b * cells * c) as u32;
        if cfg.use_diffusion {
            if self.fused_add {
                self.vadd3.dispatch_into(&self.ctx, enc, d_out, &self.ds_diff, &self.dsconv, d_s_out, n);
            } else {
                self.vadd.dispatch_into(&self.ctx, enc, d_out, &self.ds_diff, &self.tmp, n);
                self.vadd.dispatch_into(&self.ctx, enc, &self.tmp, &self.dsconv, d_s_out, n);
            }
        } else {
            self.vadd.dispatch_into(&self.ctx, enc, d_out, &self.dsconv, d_s_out, n);
        }
    }

    fn params(&self) -> [&Param; 10] {
        [&self.w1, &self.b1, &self.w2, &self.b2, &self.k0, &self.pa, &self.pb, &self.inj, &self.projw, &self.bias]
    }

    pub fn zero_grad(&self) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo zero_grad") });
        for p in self.params() {
            self.zero.dispatch_into(&self.ctx, &mut enc, &p.grad, p.grad.len as u32);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn step_adamw(&mut self, cfg: &AdamWConfig) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, cfg.beta1, cfg.beta2);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo adamw") });
        for p in self.params() {
            self.adamw.dispatch_into(&self.ctx, &mut enc, &p.buf, &p.grad, &p.m1, &p.m2, p.buf.len as u32, p.wd, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, bc1, bc2, cfg.weight_decay);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// One matrix's Muon update recorded into `enc`: momentum → Frobenius
    /// bound → load (wide) → `ns_steps`×NS5 → store → param update. Mirrors
    /// `crate::optimizer::muon_apply`; reuses the parity-tested Muon kernels +
    /// the GEMMs. `rows`/`cols` are the matrix's native shape.
    fn muon_one(&self, enc: &mut wgpu::CommandEncoder, params: &GpuBuffer, grad: &GpuBuffer, mom: &GpuBuffer, rows: u32, cols: u32, cfg: &MuonConfig) {
        let ctx = &self.ctx;
        let n = rows * cols;
        let transposed = rows > cols;
        let (m, ndim) = if transposed { (cols, rows) } else { (rows, cols) };
        let (mn, mmm) = (m * ndim, m * m);
        fn pass(enc: &mut wgpu::CommandEncoder) -> wgpu::ComputePass<'_> {
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None })
        }

        // momentum + Nesterov → update
        let mom_bind = self.mu_momentum.build_bind(ctx, grad, mom, &self.mu_update);
        {
            let mut p = pass(enc);
            p.set_pipeline(&self.mu_momentum.pipeline);
            self.mu_momentum.dispatch_with_bind(&mut p, &mom_bind, n, cfg.nesterov, cfg.momentum);
        }
        // Frobenius²  partials → finalize 1/(‖·‖+eps) → scale update
        self.gn_sq.dispatch_with_offset(ctx, enc, &self.mu_update, &self.mu_frob_partials, n, 0, 8);
        let inv_bind = self.mu_frob_inv.build_bind(ctx, &self.mu_frob_partials, &self.mu_frob_result);
        {
            let mut p = pass(enc);
            p.set_pipeline(&self.mu_frob_inv.pipeline);
            self.mu_frob_inv.dispatch_with_bind(&mut p, &inv_bind, 8);
        }
        self.scaleb.dispatch_into(ctx, enc, &self.mu_update, &self.mu_frob_result, n, 1);
        // load update → x (wide orientation; transpose if rows > cols)
        let load_bind = self.mu_load_store.build_bind(ctx, &self.mu_update, &self.mu_x);
        {
            let mut p = pass(enc);
            p.set_pipeline(&self.mu_load_store.pipeline);
            self.mu_load_store.dispatch_with_bind(&mut p, &load_bind, rows, cols, transposed, false);
        }
        // NS5: A=XXᵀ, A²=A·A, B=bA+cA², BX=B·X, X=aX+BX
        let xxt = self.gtb.build_bind(ctx, &self.mu_x, &self.mu_x, &self.mu_a);
        let aa = self.gemm.build_bind(ctx, &self.mu_a, &self.mu_a, &self.mu_asq);
        let cb = self.mu_combine_b.build_bind(ctx, &self.mu_a, &self.mu_asq, &self.mu_b);
        let bx = self.gemm.build_bind(ctx, &self.mu_b, &self.mu_x, &self.mu_bx);
        let sx = self.mu_step_x.build_bind(ctx, &self.mu_x, &self.mu_bx);
        for _ in 0..cfg.ns_steps {
            { let mut p = pass(enc); p.set_pipeline(self.gtb.pipeline()); self.gtb.dispatch_with_bind(&mut p, &xxt, m, m, ndim); }
            { let mut p = pass(enc); p.set_pipeline(&self.gemm.pipeline); self.gemm.dispatch_with_bind(&mut p, &aa, m, m, m); }
            { let mut p = pass(enc); p.set_pipeline(&self.mu_combine_b.pipeline); self.mu_combine_b.dispatch_with_bind(&mut p, &cb, mmm); }
            { let mut p = pass(enc); p.set_pipeline(&self.gemm.pipeline); self.gemm.dispatch_with_bind(&mut p, &bx, m, ndim, m); }
            { let mut p = pass(enc); p.set_pipeline(&self.mu_step_x.pipeline); self.mu_step_x.dispatch_with_bind(&mut p, &sx, mn); }
        }
        // store x → update (transpose back), then param update
        let store_bind = self.mu_load_store.build_bind(ctx, &self.mu_x, &self.mu_update);
        {
            let mut p = pass(enc);
            p.set_pipeline(&self.mu_load_store.pipeline);
            self.mu_load_store.dispatch_with_bind(&mut p, &store_bind, rows, cols, transposed, true);
        }
        let scale = (ndim as f32 / m as f32).sqrt();
        let apply_bind = self.mu_apply.build_bind(ctx, params, &self.mu_update);
        {
            let mut p = pass(enc);
            p.set_pipeline(&self.mu_apply.pipeline);
            self.mu_apply.dispatch_with_bind(&mut p, &apply_bind, n, cfg.weight_decay > 0.0, cfg.lr, scale, cfg.weight_decay);
        }
    }

    /// Hybrid step: Muon for the matrix params (conv W1/W2, readout proj),
    /// AdamW for the 1-D gates/biases + injection table. GPU twin of
    /// `ThermoLM::step_muon_hybrid`.
    pub fn step_muon_hybrid(&mut self, adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo muon hybrid") });
        let adamw_subset: [(&Param, bool); 7] = [
            (&self.b1, false), (&self.b2, false), (&self.k0, false),
            (&self.pa, false), (&self.pb, false), (&self.inj, true), (&self.bias, false),
        ];
        for (p, wd) in adamw_subset {
            self.adamw.dispatch_into(&self.ctx, &mut enc, &p.buf, &p.grad, &p.m1, &p.m2, p.buf.len as u32, wd, adam_cfg.lr, adam_cfg.beta1, adam_cfg.beta2, adam_cfg.eps, bc1, bc2, adam_cfg.weight_decay);
        }
        let (c, chid, cout, vocab, rdim) =
            (self.cfg.c, self.cfg.conv_hidden, self.cfg.c_out(), self.cfg.vocab, self.cfg.readout_dim());
        self.muon_one(&mut enc, &self.w1.buf, &self.w1.grad, &self.mu_m_w1, chid as u32, (c * 27) as u32, muon_cfg);
        self.muon_one(&mut enc, &self.w2.buf, &self.w2.grad, &self.mu_m_w2, cout as u32, chid as u32, muon_cfg);
        self.muon_one(&mut enc, &self.projw.buf, &self.projw.grad, &self.mu_m_proj, vocab as u32, rdim as u32, muon_cfg);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// One on-device optimizer step using the Muon hybrid (see `train_step`).
    pub fn train_step_muon(&mut self, tokens: &[u32], targets: &[u32], adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig, want_loss: bool) -> Option<f32> {
        self.zero_grad();
        self.forward(tokens);
        self.backward_dispatch(tokens, targets);
        self.step_muon_hybrid(adam_cfg, muon_cfg);
        want_loss.then(|| self.read_mean_loss())
    }

    /// One on-device optimizer step over a `[T, B]` batch. Returns the mean
    /// loss only when `want_loss` (a host sync); pass `false` on non-logging
    /// steps to keep the pipeline async.
    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32], cfg: &AdamWConfig, want_loss: bool) -> Option<f32> {
        self.zero_grad();
        self.forward(tokens);
        self.backward_dispatch(tokens, targets);
        self.step_adamw(cfg);
        want_loss.then(|| self.read_mean_loss())
    }

    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Block until the GPU has finished all submitted work (profiling barrier).
    pub fn drain(&self) {
        let _ = self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    /// Download all param grads into a CPU `ThermoGrads` (for parity / debug).
    #[must_use]
    pub fn download_grads(&self) -> ThermoGrads {
        let mut g = ThermoGrads::zeros(&self.cfg);
        g.step.conv.d_w1 = self.w1.grad.download(&self.ctx);
        g.step.conv.d_b1 = self.b1.grad.download(&self.ctx);
        g.step.conv.d_w2 = self.w2.grad.download(&self.ctx);
        g.step.conv.d_b2 = self.b2.grad.download(&self.ctx);
        g.step.diffusion.d_k0 = self.k0.grad.download(&self.ctx);
        g.step.diffusion.d_a = self.pa.grad.download(&self.ctx);
        g.step.diffusion.d_b = self.pb.grad.download(&self.ctx);
        g.inject.d_inj = self.inj.grad.download(&self.ctx);
        g.readout.d_proj.d_w = self.projw.grad.download(&self.ctx);
        g.readout.d_bias = self.bias.grad.download(&self.ctx);
        g
    }

    /// Upload a CPU `ThermoGrads` into the device grad buffers (parity tests:
    /// drive the GPU optimizer step from identical grads).
    pub fn upload_grads(&self, g: &ThermoGrads) {
        self.w1.grad.upload(&self.ctx, &g.step.conv.d_w1);
        self.b1.grad.upload(&self.ctx, &g.step.conv.d_b1);
        self.w2.grad.upload(&self.ctx, &g.step.conv.d_w2);
        self.b2.grad.upload(&self.ctx, &g.step.conv.d_b2);
        self.k0.grad.upload(&self.ctx, &g.step.diffusion.d_k0);
        self.pa.grad.upload(&self.ctx, &g.step.diffusion.d_a);
        self.pb.grad.upload(&self.ctx, &g.step.diffusion.d_b);
        self.inj.grad.upload(&self.ctx, &g.inject.d_inj);
        self.projw.grad.upload(&self.ctx, &g.readout.d_proj.d_w);
        self.bias.grad.upload(&self.ctx, &g.readout.d_bias);
    }

    /// Snapshot device params back into a CPU model (viz / eval / checkpoint).
    #[must_use]
    pub fn to_cpu(&self) -> ThermoLM {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);
        let mut m = ThermoLM::new(self.cfg.clone(), &mut rng);
        m.step.conv.w1 = self.w1.buf.download(&self.ctx);
        m.step.conv.b1 = self.b1.buf.download(&self.ctx);
        m.step.conv.w2 = self.w2.buf.download(&self.ctx);
        m.step.conv.b2 = self.b2.buf.download(&self.ctx);
        m.step.diffusion.k0 = self.k0.buf.download(&self.ctx);
        m.step.diffusion.a = self.pa.buf.download(&self.ctx);
        m.step.diffusion.b = self.pb.buf.download(&self.ctx);
        m.inject.inj = self.inj.buf.download(&self.ctx);
        m.readout.proj.w = self.projw.buf.download(&self.ctx);
        m.readout.bias = self.bias.buf.download(&self.ctx);
        m.step_count = self.step_count;
        m
    }
}
