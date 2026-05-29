#![allow(clippy::too_many_arguments, clippy::similar_names, clippy::many_single_char_names, clippy::too_many_lines)]

//! Full on-device thermo-LM training: cached rollout forward + rollout
//! backward (BPTT) + AdamW step, assembled from the parity-verified
//! kernels. Mirrors `crate::thermo::model::ThermoLM` forward/backward
//! exactly. Moments-only readout (spectral GPU path is a later addition).
//!
//! The field is one chain across the window. Every sub-step caches its
//! input field + conv pre/post activations + diffusion Laplacian for the
//! reverse pass ("store full trajectory"; checkpointing deferred, §5).

use super::buffers::GpuBuffer;
use super::thermo_conv::{
    ThermoConv1x1Kernel, ThermoConv3x3Dw1Kernel, ThermoConv3x3DsKernel, ThermoConv3x3Kernel,
};
use super::thermo_step::{
    ExtractFluxKernel, StepAssembleKernel, ThermoDisassembleKernel, ThermoFluxScatterKernel,
    ThermoScaleKernel,
};
use super::thermo_stencil::{
    ThermoAvgPoolBackwardKernel, ThermoAvgPoolKernel, ThermoDiffusionBwdDsKernel,
    ThermoDiffusionBwdParamKernel, ThermoDiffusionBwdPreKernel, ThermoDiffusionForwardKernel,
    ThermoFluxDivBackwardKernel, ThermoFluxDivKernel, ThermoInjectBackwardKernel,
    ThermoInjectKernel,
};
use super::{
    AdamWKernel, GeluBackwardKernel, GeluKernel, GemmKernel, GemmTransposeBKernel, GpuContext,
    OuterProductAccumKernel, RowReduceAddKernel, SoftmaxCrossEntropyKernel, VectorAddKernel,
    ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig};
use crate::thermo::field::{init_field, ThermoConfig};
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
    step_count: u64,

    conv3: ThermoConv3x3Kernel,
    gelu: GeluKernel,
    conv1: ThermoConv1x1Kernel,
    extract: ExtractFluxKernel,
    flux: ThermoFluxDivKernel,
    diff_fwd: ThermoDiffusionForwardKernel,
    assemble: StepAssembleKernel,
    pool: ThermoAvgPoolKernel,
    gtb: GemmTransposeBKernel,
    vadd: VectorAddKernel,
    inject: ThermoInjectKernel,
    sce: SoftmaxCrossEntropyKernel,
    scale: ThermoScaleKernel,
    diff_pre: ThermoDiffusionBwdPreKernel,
    diff_param: ThermoDiffusionBwdParamKernel,
    diff_ds: ThermoDiffusionBwdDsKernel,
    disasm: ThermoDisassembleKernel,
    fluxb: ThermoFluxDivBackwardKernel,
    scatter: ThermoFluxScatterKernel,
    gemm: GemmKernel,
    outer: OuterProductAccumKernel,
    row: RowReduceAddKernel,
    gelu_bwd: GeluBackwardKernel,
    conv_ds: ThermoConv3x3DsKernel,
    conv_dw1: ThermoConv3x3Dw1Kernel,
    pool_bwd: ThermoAvgPoolBackwardKernel,
    inject_bwd: ThermoInjectBackwardKernel,
    zero: ZeroBufferKernel,
    adamw: AdamWKernel,

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
    gate: GpuBuffer,

    s_traj: Vec<GpuBuffer>,
    hpre_traj: Vec<GpuBuffer>,
    hpost_traj: Vec<GpuBuffer>,
    lap_traj: Vec<GpuBuffer>,
    pooled_traj: Vec<GpuBuffer>,
    fout_traj: Vec<GpuBuffer>,
    init_buf: GpuBuffer,

    raw: GpuBuffer,
    fx: GpuBuffer,
    fy: GpuBuffer,
    div: GpuBuffer,
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
    ddiv: GpuBuffer,
    draw: GpuBuffer,
    dfx: GpuBuffer,
    dfy: GpuBuffer,
    dhpost: GpuBuffer,
    dhpre: GpuBuffer,
    dsconv: GpuBuffer,
    tmp: GpuBuffer,

    targets_buf: GpuBuffer,
}

impl ThermoGpu {
    #[must_use]
    pub fn new(ctx: &GpuContext, model: &ThermoLM, seq_len: usize) -> Self {
        let cfg = model.config.clone();
        assert!(cfg.readout_moments && !cfg.readout_spectral, "GPU training = moments-only readout");
        let cells = cfg.cells();
        let c = cfg.c;
        let chid = cfg.conv_hidden;
        let cout = cfg.c_out();
        let nc = cfg.n_conserved();
        let pdim = cfg.pooled_dim();
        let vocab = cfg.vocab;
        let nsub = seq_len * cfg.k_steps;

        let (kind, slot) = cfg.channel_decode();

        let z = |n: usize| GpuBuffer::zeros(ctx, "scratch", n);
        let traj = |n: usize, len: usize| (0..len).map(|_| GpuBuffer::zeros(ctx, "traj", n)).collect::<Vec<_>>();

        Self {
            cfg: cfg.clone(),
            ctx: ctx.clone(),
            seq_len,
            step_count: model.step_count,
            conv3: ThermoConv3x3Kernel::new(ctx),
            gelu: GeluKernel::new(ctx),
            conv1: ThermoConv1x1Kernel::new(ctx),
            extract: ExtractFluxKernel::new(ctx),
            flux: ThermoFluxDivKernel::new(ctx),
            diff_fwd: ThermoDiffusionForwardKernel::new(ctx),
            assemble: StepAssembleKernel::new(ctx),
            pool: ThermoAvgPoolKernel::new(ctx),
            gtb: GemmTransposeBKernel::new(ctx),
            vadd: VectorAddKernel::new(ctx),
            inject: ThermoInjectKernel::new(ctx),
            sce: SoftmaxCrossEntropyKernel::new(ctx),
            scale: ThermoScaleKernel::new(ctx),
            diff_pre: ThermoDiffusionBwdPreKernel::new(ctx),
            diff_param: ThermoDiffusionBwdParamKernel::new(ctx),
            diff_ds: ThermoDiffusionBwdDsKernel::new(ctx),
            disasm: ThermoDisassembleKernel::new(ctx),
            fluxb: ThermoFluxDivBackwardKernel::new(ctx),
            scatter: ThermoFluxScatterKernel::new(ctx),
            gemm: GemmKernel::new(ctx),
            outer: OuterProductAccumKernel::new(ctx),
            row: RowReduceAddKernel::new(ctx),
            gelu_bwd: GeluBackwardKernel::new(ctx),
            conv_ds: ThermoConv3x3DsKernel::new(ctx),
            conv_dw1: ThermoConv3x3Dw1Kernel::new(ctx),
            pool_bwd: ThermoAvgPoolBackwardKernel::new(ctx),
            inject_bwd: ThermoInjectBackwardKernel::new(ctx),
            zero: ZeroBufferKernel::new(ctx),
            adamw: AdamWKernel::new(ctx),
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
            gate: z(3 * c),
            s_traj: traj(cells * c, nsub),
            hpre_traj: traj(cells * chid, nsub),
            hpost_traj: traj(cells * chid, nsub),
            lap_traj: traj(cells * c, nsub),
            pooled_traj: traj(pdim, seq_len),
            fout_traj: traj(cells * c, seq_len),
            init_buf: z(cells * c),
            raw: z(cells * cout),
            fx: z(cells * nc),
            fy: z(cells * nc),
            div: z(cells * nc),
            diff_delta: z(cells * c),
            lraw: z(vocab),
            logit_row: z(vocab),
            logits: z(seq_len * vocab),
            d_logits: z(seq_len * vocab),
            loss_seq: z(seq_len),
            dl_t: z(vocab),
            dfield_a: z(cells * c),
            dfield_b: z(cells * c),
            d_ro: z(cells * c),
            d_pooled: z(pdim),
            ddelta: z(cells * c),
            gbuf: z(cells * c),
            ct: z(cells * c),
            ds_diff: z(cells * c),
            ddiv: z(cells * nc),
            draw: z(cells * cout),
            dfx: z(cells * nc),
            dfy: z(cells * nc),
            dhpost: z(cells * chid),
            dhpre: z(cells * chid),
            dsconv: z(cells * c),
            tmp: z(cells * c),
            targets_buf: GpuBuffer::from_u32_slice(ctx, "tgt", &vec![0u32; seq_len]),
        }
    }

    /// Rollout forward (teacher-forced), filling `self.logits` + caches.
    pub fn forward(&self, tokens: &[u32]) {
        let cfg = &self.cfg;
        let t = tokens.len();
        let (cells, c, chid, cout, nc) = (cfg.cells(), cfg.c, cfg.conv_hidden, cfg.c_out(), cfg.n_conserved());
        let nfree = c - nc;
        let (h, w, vocab, pdim, ks) = (cfg.h, cfg.w, cfg.vocab, cfg.pooled_dim(), cfg.k_steps);
        let dt = cfg.dt;
        let diff_on = u32::from(cfg.use_diffusion);

        self.init_buf.upload(&self.ctx, &init_field(cfg));
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo fwd") });

        for ti in 0..t {
            let carried = if ti == 0 { &self.init_buf } else { &self.fout_traj[ti - 1] };
            self.inject.dispatch_into(&self.ctx, &mut enc, carried, &self.profile, &self.inj.buf, &self.s_traj[ti * ks], cells as u32, c as u32, tokens[ti], 1.0);

            for k in 0..ks {
                let idx = ti * ks + k;
                let anneal = anneal_at(k, cfg);
                let s_in = &self.s_traj[idx];
                self.conv3.dispatch_into(&self.ctx, &mut enc, s_in, &self.w1.buf, &self.b1.buf, &self.hpre_traj[idx], h as u32, w as u32, c as u32, chid as u32);
                self.gelu.dispatch_into(&self.ctx, &mut enc, &self.hpre_traj[idx], &self.hpost_traj[idx], (cells * chid) as u32);
                self.conv1.dispatch_into(&self.ctx, &mut enc, &self.hpost_traj[idx], &self.w2.buf, &self.b2.buf, &self.raw, cells as u32, chid as u32, cout as u32);
                self.extract.dispatch_into(&self.ctx, &mut enc, &self.raw, &self.fx, &self.fy, h as u32, w as u32, nc as u32, nfree as u32, cout as u32);
                self.flux.dispatch_into(&self.ctx, &mut enc, &self.fx, &self.fy, &self.div, h as u32, w as u32, nc as u32);
                if cfg.use_diffusion {
                    self.diff_fwd.dispatch_into(&self.ctx, &mut enc, s_in, &self.k0.buf, &self.pa.buf, &self.pb.buf, &self.diff_delta, &self.lap_traj[idx], h as u32, w as u32, c as u32);
                }
                let out = if k < ks - 1 { &self.s_traj[idx + 1] } else { &self.fout_traj[ti] };
                self.assemble.dispatch_into(&self.ctx, &mut enc, s_in, &self.div, &self.raw, &self.diff_delta, &self.kind, &self.slot, out, h as u32, w as u32, c as u32, nc as u32, cout as u32, diff_on, dt, anneal);
            }

            self.pool.dispatch_into(&self.ctx, &mut enc, &self.fout_traj[ti], &self.pooled_traj[ti], h as u32, w as u32, c as u32, cfg.pool as u32);
            self.gtb.dispatch_into(&self.ctx, &mut enc, &self.pooled_traj[ti], &self.projw.buf, &self.lraw, 1, vocab as u32, pdim as u32);
            self.vadd.dispatch_into(&self.ctx, &mut enc, &self.lraw, &self.bias.buf, &self.logit_row, vocab as u32);
            enc.copy_buffer_to_buffer(&self.logit_row.inner, 0, &self.logits.inner, (ti * vocab * 4) as u64, (vocab * 4) as u64);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Rollout backward (BPTT). Accumulates param grads; returns mean loss.
    pub fn backward(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let cfg = &self.cfg;
        let t = tokens.len();
        let (cells, c, vocab, ks) = (cfg.cells(), cfg.c, cfg.vocab, cfg.k_steps);

        self.targets_buf.upload_u32_prefix(&self.ctx, targets);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo bwd") });

        // d_logits + per-position loss (scaled by 1/T to match CPU mean).
        self.sce.dispatch_into(&self.ctx, &mut enc, &self.logits, &self.d_logits, &self.loss_seq, &self.targets_buf, vocab as u32, t as u32, 1, 1.0 / t as f32, 0);

        // pack gate [k0|a|b] for the diffusion backward.
        enc.copy_buffer_to_buffer(&self.k0.buf.inner, 0, &self.gate.inner, 0, (c * 4) as u64);
        enc.copy_buffer_to_buffer(&self.pa.buf.inner, 0, &self.gate.inner, (c * 4) as u64, (c * 4) as u64);
        enc.copy_buffer_to_buffer(&self.pb.buf.inner, 0, &self.gate.inner, (2 * c * 4) as u64, (c * 4) as u64);

        // d_field ping-pong, start at zero.
        self.zero.dispatch_into(&self.ctx, &mut enc, &self.dfield_a, (cells * c) as u32);
        let df = [&self.dfield_a, &self.dfield_b];
        let mut cur = 0usize;

        for ti in (0..t).rev() {
            // d_logits[ti] slice → dl_t
            enc.copy_buffer_to_buffer(&self.d_logits.inner, (ti * vocab * 4) as u64, &self.dl_t.inner, 0, (vocab * 4) as u64);
            // readout backward → d_ro (d_s from readout)
            self.readout_backward(&mut enc, ti);
            // d_field += d_ro
            self.vadd.dispatch_into(&self.ctx, &mut enc, df[cur], &self.d_ro, df[1 - cur], (cells * c) as u32);
            cur = 1 - cur;
            // K step backwards (reverse)
            for k in (0..ks).rev() {
                let idx = ti * ks + k;
                self.step_backward(&mut enc, df[cur], df[1 - cur], idx, anneal_at(k, cfg));
                cur = 1 - cur;
            }
            // inject backward (d_field identity → carried)
            self.inject_bwd.dispatch_into(&self.ctx, &mut enc, &self.profile, df[cur], &self.inj.grad, cells as u32, c as u32, tokens[ti], 1.0);
        }

        self.ctx.queue.submit(Some(enc.finish()));
        // loss_seq holds unscaled per-position CE; CPU returns the mean.
        self.loss_seq.download(&self.ctx).iter().take(t).sum::<f32>() / t as f32
    }

    fn readout_backward(&self, enc: &mut wgpu::CommandEncoder, ti: usize) {
        let cfg = &self.cfg;
        let (vocab, pdim) = (cfg.vocab as u32, cfg.pooled_dim() as u32);
        // d_bias += dl_t ; d_proj += outer(pooled, dl_t)
        self.row.dispatch_into(&self.ctx, enc, &self.dl_t, &self.bias.grad, 1, vocab);
        self.outer.dispatch_into(&self.ctx, enc, &self.pooled_traj[ti], &self.dl_t, &self.projw.grad, vocab, pdim, 1);
        // d_pooled = dl_t @ proj ; d_ro = avgpool_bwd(d_pooled)
        self.gemm.dispatch_into(&self.ctx, enc, &self.dl_t, &self.projw.buf, &self.d_pooled, 1, pdim, vocab);
        self.pool_bwd.dispatch_into(&self.ctx, enc, &self.d_pooled, &self.d_ro, cfg.h as u32, cfg.w as u32, cfg.c as u32, cfg.pool as u32);
    }

    fn step_backward(&self, enc: &mut wgpu::CommandEncoder, d_out: &GpuBuffer, d_s_out: &GpuBuffer, idx: usize, anneal: f32) {
        let cfg = &self.cfg;
        let (cells, c, chid, cout, nc) = (cfg.cells(), cfg.c, cfg.conv_hidden, cfg.c_out(), cfg.n_conserved());
        let nfree = c - nc;
        let (h, w) = (cfg.h as u32, cfg.w as u32);
        let dt = cfg.dt;

        // diffusion path
        if cfg.use_diffusion {
            self.scale.dispatch_into(&self.ctx, enc, d_out, &self.ddelta, (cells * c) as u32, dt * anneal);
            self.diff_pre.dispatch_into(&self.ctx, enc, &self.ddelta, &self.lap_traj[idx], &self.s_traj[idx], &self.gate, &self.gbuf, &self.ct, h, w, c as u32);
            self.diff_param.dispatch_into(&self.ctx, enc, &self.ddelta, &self.lap_traj[idx], &self.s_traj[idx], &self.gate, &self.k0.grad, &self.pa.grad, &self.pb.grad, h, w, c as u32);
            self.diff_ds.dispatch_into(&self.ctx, enc, &self.gbuf, &self.ct, &self.ds_diff, h, w, c as u32);
        }

        // transport path
        self.zero.dispatch_into(&self.ctx, enc, &self.ddiv, (cells * nc) as u32);
        self.zero.dispatch_into(&self.ctx, enc, &self.draw, (cells * cout) as u32);
        self.disasm.dispatch_into(&self.ctx, enc, d_out, &self.kind, &self.slot, &self.ddiv, &self.draw, h, w, c as u32, nc as u32, cout as u32, dt);
        self.fluxb.dispatch_into(&self.ctx, enc, &self.ddiv, &self.dfx, &self.dfy, h, w, nc as u32);
        self.scatter.dispatch_into(&self.ctx, enc, &self.dfx, &self.dfy, &self.draw, h, w, nc as u32, nfree as u32, cout as u32);
        self.gemm.dispatch_into(&self.ctx, enc, &self.draw, &self.w2.buf, &self.dhpost, cells as u32, chid as u32, cout as u32);
        self.outer.dispatch_into(&self.ctx, enc, &self.hpost_traj[idx], &self.draw, &self.w2.grad, cout as u32, chid as u32, cells as u32);
        self.row.dispatch_into(&self.ctx, enc, &self.draw, &self.b2.grad, cells as u32, cout as u32);
        self.gelu_bwd.dispatch_into(&self.ctx, enc, &self.hpre_traj[idx], &self.dhpost, &self.dhpre, (cells * chid) as u32);
        self.conv_ds.dispatch_into(&self.ctx, enc, &self.dhpre, &self.w1.buf, &self.dsconv, h, w, c as u32, chid as u32);
        self.conv_dw1.dispatch_into(&self.ctx, enc, &self.dhpre, &self.s_traj[idx], &self.w1.grad, h, w, c as u32, chid as u32);
        self.row.dispatch_into(&self.ctx, enc, &self.dhpre, &self.b1.grad, cells as u32, chid as u32);

        // d_s = d_out + ds_diff + dsconv  (residual + diffusion + transport)
        let n = (cells * c) as u32;
        if cfg.use_diffusion {
            self.vadd.dispatch_into(&self.ctx, enc, d_out, &self.ds_diff, &self.tmp, n);
            self.vadd.dispatch_into(&self.ctx, enc, &self.tmp, &self.dsconv, d_s_out, n);
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

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32], cfg: &AdamWConfig) -> f32 {
        self.zero_grad();
        self.forward(tokens);
        let loss = self.backward(tokens, targets);
        self.step_adamw(cfg);
        loss
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
