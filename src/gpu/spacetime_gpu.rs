#![allow(clippy::too_many_arguments, clippy::similar_names, clippy::many_single_char_names, clippy::too_many_lines)]

//! On-device training for the spacetime LM (`crate::thermo::spacetime_model::StLM`).
//!
//! Mirrors the CPU forward/backward bit-for-bit: embed-at-time → K residual
//! causal-dilated-conv steps (`field += dt·raw`) → time-local readout (GEMM) →
//! softmax-CE. The field is one `[B·L·c]` buffer (B windows batched); only K+1
//! trajectory snapshots + K hidden caches are kept (no per-timestep recurrence).
//! New kernels: causal conv1d (gelu-cache / d_s / d_w1), embed-inject, axpy.
//! Everything else reuses the parity-tested transformer/thermo kernels.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::thermo_conv::{ThermoConv1x1Kernel, ThermoConvBwdHiddenKernel};
use super::thermo_step::ThermoScaleKernel;
use super::{
    AdamWKernel, EmbedScatterReduceKernel, GemmKernel, GemmTransposeBKernel, GpuContext,
    GradNormSqReduceKernel, MuonApplyKernel, MuonCombineBKernel, MuonFrobInvKernel,
    MuonLoadStoreKernel, MuonMomentumKernel, MuonStepXKernel, OuterProductAccumKernel,
    RowReduceAddKernel, ScaleByBufferKernel, SoftmaxCrossEntropyKernel, ThermoBiasRowAddKernel,
    VectorAddKernel, ZeroBufferKernel,
};
use crate::optimizer::{bias_correction, AdamWConfig, MuonConfig};
use crate::thermo::spacetime_model::{StConfig, StGrads, StLM};

const WG: u32 = 64;

// ───── self-contained kernel-build helpers ──────────────────────────────────

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build(ctx: &GpuContext, label: &str, wgsl: &str, read_only: &[bool]) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let entries: Vec<_> = read_only.iter().enumerate().map(|(i, &ro)| storage_entry(i as u32, ro)).collect();
    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..48 }],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (Arc::new(pipeline), Arc::new(bgl))
}

fn dispatch_n(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bgl: &wgpu::BindGroupLayout, buffers: &[&GpuBuffer], push: &[u8], groups: u32) {
    let entries: Vec<_> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.inner.as_entire_binding() }).collect();
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: bgl, entries: &entries });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.set_push_constants(0, push);
    pass.dispatch_workgroups(groups, 1, 1);
}

// ───── new kernels ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvP { l: u32, c: u32, ch: u32, taps: u32, dil: u32, nb: u32, _p0: u32, _p1: u32 }

const ST_GELU_WGSL: &str = r#"
struct P { l:u32, c:u32, ch:u32, taps:u32, dil:u32, nb:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       s:     array<f32>;
@group(0) @binding(1) var<storage, read>       w1:    array<f32>;
@group(0) @binding(2) var<storage, read>       b1:    array<f32>;
@group(0) @binding(3) var<storage, read_write> hpre:  array<f32>;
@group(0) @binding(4) var<storage, read_write> hpost: array<f32>;
var<push_constant> p: P;
fn gelu(x: f32) -> f32 { let k = 0.7978845608028654; let inner = k*(x + 0.044715*x*x*x); return 0.5*x*(1.0+tanh(inner)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.l * p.ch) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let t = cell % p.l;
    let g = cell / p.l;
    var acc = b1[co];
    for (var tap: u32 = 0u; tap < p.taps; tap = tap + 1u) {
        let off = tap * p.dil;
        if (off > t) { continue; }
        let in_base = (g * p.l + (t - off)) * p.c;
        for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
            acc = acc + w1[(co * p.c + ci) * p.taps + tap] * s[in_base + ci];
        }
    }
    hpre[idx]  = acc;
    hpost[idx] = gelu(acc);
}
"#;

const ST_DS_WGSL: &str = r#"
struct P { l:u32, c:u32, ch:u32, taps:u32, dil:u32, nb:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;
@group(0) @binding(1) var<storage, read>       w1:  array<f32>;
@group(0) @binding(2) var<storage, read_write> ds:  array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.l * p.c) { return; }
    let ci = idx % p.c;
    let cell = idx / p.c;
    let yt = cell % p.l;
    let g = cell / p.l;
    var acc = 0.0;
    for (var tap: u32 = 0u; tap < p.taps; tap = tap + 1u) {
        let ot = yt + tap * p.dil;
        if (ot >= p.l) { continue; }
        let obase = (g * p.l + ot) * p.ch;
        for (var co: u32 = 0u; co < p.ch; co = co + 1u) {
            acc = acc + dhp[obase + co] * w1[(co * p.c + ci) * p.taps + tap];
        }
    }
    ds[idx] = acc;
}
"#;

const ST_DW1_WGSL: &str = r#"
struct P { l:u32, c:u32, ch:u32, taps:u32, dil:u32, nb:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;
@group(0) @binding(1) var<storage, read>       s:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dw1: array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let widx = gid.x;
    if (widx >= p.ch * p.c * p.taps) { return; }
    let tap = widx % p.taps;
    let r = widx / p.taps;
    let ci = r % p.c;
    let co = r / p.c;
    let off = tap * p.dil;
    var acc = 0.0;
    for (var g: u32 = 0u; g < p.nb; g = g + 1u) {
        for (var t: u32 = off; t < p.l; t = t + 1u) {
            let o  = (g * p.l + t) * p.ch + co;
            let ins = (g * p.l + (t - off)) * p.c + ci;
            acc = acc + dhp[o] * s[ins];
        }
    }
    dw1[widx] = dw1[widx] + acc;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InjectP { n: u32, c: u32, _p0: u32, _p1: u32 }

const ST_INJECT_WGSL: &str = r#"
struct P { n:u32, c:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       embed:  array<f32>;
@group(0) @binding(2) var<storage, read_write> field:  array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.n) { return; }
    let ci = idx % p.c;
    let cell = idx / p.c;
    let tok = tokens[cell];
    field[idx] = embed[tok * p.c + ci];
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyP { n: u32, f: f32, _p0: u32, _p1: u32 }

const ST_AXPY_WGSL: &str = r#"
struct P { n:u32, f:f32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] + p.f * x[i];
}
"#;

struct StConvGeluKernel { pipeline: Arc<wgpu::ComputePipeline>, bgl: Arc<wgpu::BindGroupLayout> }
struct StConvDsKernel { pipeline: Arc<wgpu::ComputePipeline>, bgl: Arc<wgpu::BindGroupLayout> }
struct StConvDw1Kernel { pipeline: Arc<wgpu::ComputePipeline>, bgl: Arc<wgpu::BindGroupLayout> }
struct StInjectKernel { pipeline: Arc<wgpu::ComputePipeline>, bgl: Arc<wgpu::BindGroupLayout> }
struct StAxpyKernel { pipeline: Arc<wgpu::ComputePipeline>, bgl: Arc<wgpu::BindGroupLayout> }

impl StConvGeluKernel {
    fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "st_conv_gelu", ST_GELU_WGSL, &[true, true, true, false, false]);
        Self { pipeline, bgl }
    }
    fn dispatch(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, s: &GpuBuffer, w1: &GpuBuffer, b1: &GpuBuffer, hpre: &GpuBuffer, hpost: &GpuBuffer, l: u32, c: u32, ch: u32, taps: u32, dil: u32, nb: u32) {
        let p = ConvP { l, c, ch, taps, dil, nb, _p0: 0, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[s, w1, b1, hpre, hpost], bytemuck::bytes_of(&p), (nb * l * ch).div_ceil(WG));
    }
}
impl StConvDsKernel {
    fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "st_conv_ds", ST_DS_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    fn dispatch(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, dhp: &GpuBuffer, w1: &GpuBuffer, ds: &GpuBuffer, l: u32, c: u32, ch: u32, taps: u32, dil: u32, nb: u32) {
        let p = ConvP { l, c, ch, taps, dil, nb, _p0: 0, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[dhp, w1, ds], bytemuck::bytes_of(&p), (nb * l * c).div_ceil(WG));
    }
}
impl StConvDw1Kernel {
    fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "st_conv_dw1", ST_DW1_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    fn dispatch(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, dhp: &GpuBuffer, s: &GpuBuffer, dw1: &GpuBuffer, l: u32, c: u32, ch: u32, taps: u32, dil: u32, nb: u32) {
        let p = ConvP { l, c, ch, taps, dil, nb, _p0: 0, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[dhp, s, dw1], bytemuck::bytes_of(&p), (ch * c * taps).div_ceil(WG));
    }
}
impl StInjectKernel {
    fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "st_inject", ST_INJECT_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    fn dispatch(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, tokens: &GpuBuffer, embed: &GpuBuffer, field: &GpuBuffer, n: u32, c: u32) {
        let p = InjectP { n, c, _p0: 0, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[tokens, embed, field], bytemuck::bytes_of(&p), n.div_ceil(WG));
    }
}
impl StAxpyKernel {
    fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "st_axpy", ST_AXPY_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    fn dispatch(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, a: &GpuBuffer, x: &GpuBuffer, y: &GpuBuffer, n: u32, f: f32) {
        let p = AxpyP { n, f, _p0: 0, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[a, x, y], bytemuck::bytes_of(&p), n.div_ceil(WG));
    }
}

// ───── on-device model ──────────────────────────────────────────────────────

struct Param { buf: GpuBuffer, grad: GpuBuffer, m1: GpuBuffer, m2: GpuBuffer, wd: bool }
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

pub struct StLmGpu {
    ctx: GpuContext,
    cfg: StConfig,
    taps: usize,
    seq_len: usize,
    batch: usize,
    step_count: u64,

    // new kernels
    st_gelu: StConvGeluKernel,
    st_ds: StConvDsKernel,
    st_dw1: StConvDw1Kernel,
    inject: StInjectKernel,
    axpy: StAxpyKernel,
    // reused kernels
    conv1: ThermoConv1x1Kernel,
    conv_bwd_hidden: ThermoConvBwdHiddenKernel,
    gtb: GemmTransposeBKernel,
    gemm: GemmKernel,
    bias_add: ThermoBiasRowAddKernel,
    sce: SoftmaxCrossEntropyKernel,
    outer: OuterProductAccumKernel,
    row: RowReduceAddKernel,
    embed_scatter: EmbedScatterReduceKernel,
    vadd: VectorAddKernel,
    scale: ThermoScaleKernel,
    zero: ZeroBufferKernel,
    adamw: AdamWKernel,
    // muon
    mu_momentum: MuonMomentumKernel,
    mu_frob_inv: MuonFrobInvKernel,
    mu_load_store: MuonLoadStoreKernel,
    mu_combine_b: MuonCombineBKernel,
    mu_step_x: MuonStepXKernel,
    mu_apply: MuonApplyKernel,
    gn_sq: GradNormSqReduceKernel,
    scaleb: ScaleByBufferKernel,
    mu_m_w1: GpuBuffer,
    mu_m_w2: GpuBuffer,
    mu_m_row: GpuBuffer,
    mu_update: GpuBuffer,
    mu_x: GpuBuffer,
    mu_a: GpuBuffer,
    mu_asq: GpuBuffer,
    mu_b: GpuBuffer,
    mu_bx: GpuBuffer,
    mu_frob_partials: GpuBuffer,
    mu_frob_result: GpuBuffer,

    // params
    w1: Param,
    b1: Param,
    w2: Param,
    b2: Param,
    embed: Param,
    ro_w: Param,
    ro_bias: Param,

    // trajectory + scratch
    s_traj: Vec<GpuBuffer>,    // K+1 fields [B*L*c]
    hpre_traj: Vec<GpuBuffer>, // K [B*L*ch]
    hpost_traj: Vec<GpuBuffer>,
    raw: GpuBuffer,
    lraw: GpuBuffer,
    logits: GpuBuffer,
    d_logits: GpuBuffer,
    loss_seq: GpuBuffer,
    dfield_a: GpuBuffer,
    dfield_b: GpuBuffer,
    d_raw: GpuBuffer,
    dhpre: GpuBuffer,
    dsconv: GpuBuffer,
    tokens_buf: GpuBuffer,
    targets_buf: GpuBuffer,
}

impl StLmGpu {
    #[must_use]
    pub fn new(ctx: &GpuContext, model: &StLM, seq_len: usize, batch: usize) -> Self {
        assert!(batch >= 1);
        let cfg = model.cfg.clone();
        let (c, ch, vocab, ks) = (cfg.c, cfg.conv_hidden, cfg.vocab, cfg.k_steps);
        let taps = model.conv.taps;
        let b = batch;
        let bl = b * seq_len;
        let z = |n: usize| GpuBuffer::zeros(ctx, "scratch", n);
        let traj = |n: usize, len: usize| (0..len).map(|_| GpuBuffer::zeros(ctx, "traj", n)).collect::<Vec<_>>();

        let (w1_len, w2_len, row_len) = (ch * c * taps, c * ch, vocab * c);
        let mn_max = w1_len.max(w2_len).max(row_len);
        let mm = |r: usize, c: usize| { let m = r.min(c); m * m };
        let mm_max = mm(ch, c * taps).max(mm(c, ch)).max(mm(vocab, c));

        Self {
            ctx: ctx.clone(),
            cfg: cfg.clone(),
            taps,
            seq_len,
            batch,
            step_count: model.step_count,
            st_gelu: StConvGeluKernel::new(ctx),
            st_ds: StConvDsKernel::new(ctx),
            st_dw1: StConvDw1Kernel::new(ctx),
            inject: StInjectKernel::new(ctx),
            axpy: StAxpyKernel::new(ctx),
            conv1: ThermoConv1x1Kernel::new(ctx),
            conv_bwd_hidden: ThermoConvBwdHiddenKernel::new(ctx),
            gtb: GemmTransposeBKernel::new(ctx),
            gemm: GemmKernel::new(ctx),
            bias_add: ThermoBiasRowAddKernel::new(ctx),
            sce: SoftmaxCrossEntropyKernel::new(ctx),
            outer: OuterProductAccumKernel::new(ctx),
            row: RowReduceAddKernel::new(ctx),
            embed_scatter: EmbedScatterReduceKernel::new(ctx),
            vadd: VectorAddKernel::new(ctx),
            scale: ThermoScaleKernel::new(ctx),
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
            mu_m_row: z(row_len),
            mu_update: z(mn_max),
            mu_x: z(mn_max),
            mu_a: z(mm_max),
            mu_asq: z(mm_max),
            mu_b: z(mm_max),
            mu_bx: z(mn_max),
            mu_frob_partials: z(8),
            mu_frob_result: z(8),
            w1: Param::new(ctx, "w1", &model.conv.w1, true),
            b1: Param::new(ctx, "b1", &model.conv.b1, false),
            w2: Param::new(ctx, "w2", &model.conv.w2, true),
            b2: Param::new(ctx, "b2", &model.conv.b2, false),
            embed: Param::new(ctx, "embed", &model.embed, true),
            ro_w: Param::new(ctx, "ro_w", &model.ro_w, true),
            ro_bias: Param::new(ctx, "ro_bias", &model.ro_bias, false),
            s_traj: traj(bl * c, ks + 1),
            hpre_traj: traj(bl * ch, ks),
            hpost_traj: traj(bl * ch, ks),
            raw: z(bl * c),
            lraw: z(bl * vocab),
            logits: z(bl * vocab),
            d_logits: z(bl * vocab),
            loss_seq: z(bl),
            dfield_a: z(bl * c),
            dfield_b: z(bl * c),
            d_raw: z(bl * c),
            dhpre: z(bl * ch),
            dsconv: z(bl * c),
            tokens_buf: GpuBuffer::from_u32_slice(ctx, "tok", &vec![0u32; bl]),
            targets_buf: GpuBuffer::from_u32_slice(ctx, "tgt", &vec![0u32; bl]),
        }
    }

    #[inline]
    fn dilation(&self, k: usize) -> u32 {
        self.cfg.dilation_base.checked_pow(k as u32).unwrap_or(usize::MAX).min(self.seq_len.max(1)).max(1) as u32
    }

    /// Forward over a flattened `[B*L]` token buffer (row = b*L + t). Fills
    /// `logits` and the K+1 / K caches.
    pub fn forward(&self, tokens: &[u32]) {
        let (c, ch, vocab, ks) = (self.cfg.c, self.cfg.conv_hidden, self.cfg.vocab, self.cfg.k_steps);
        let (b, l) = (self.batch, self.seq_len);
        let (bl, taps, dt) = (b * l, self.taps as u32, self.cfg.dt);
        assert_eq!(tokens.len(), bl);
        self.tokens_buf.upload_u32_prefix(&self.ctx, tokens);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st fwd") });

        // s_traj[0] = embed(tokens)
        self.inject.dispatch(&self.ctx, &mut enc, &self.tokens_buf, &self.embed.buf, &self.s_traj[0], (bl * c) as u32, c as u32);

        for k in 0..ks {
            let dil = self.dilation(k);
            self.st_gelu.dispatch(&self.ctx, &mut enc, &self.s_traj[k], &self.w1.buf, &self.b1.buf, &self.hpre_traj[k], &self.hpost_traj[k], l as u32, c as u32, ch as u32, taps, dil, b as u32);
            self.conv1.dispatch_into(&self.ctx, &mut enc, &self.hpost_traj[k], &self.w2.buf, &self.b2.buf, &self.raw, bl as u32, ch as u32, c as u32);
            // s_traj[k+1] = s_traj[k] + dt*raw
            self.axpy.dispatch(&self.ctx, &mut enc, &self.s_traj[k], &self.raw, &self.s_traj[k + 1], (bl * c) as u32, dt);
        }
        // readout: logits = s_traj[K] @ ro_wᵀ + bias  (time-local)
        self.gtb.dispatch_into(&self.ctx, &mut enc, &self.s_traj[ks], &self.ro_w.buf, &self.lraw, bl as u32, vocab as u32, c as u32);
        self.bias_add.dispatch_into(&self.ctx, &mut enc, &self.lraw, &self.ro_bias.buf, &self.logits, bl as u32, vocab as u32);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn backward_dispatch(&self, tokens: &[u32], targets: &[u32]) {
        let (c, ch, vocab, ks) = (self.cfg.c, self.cfg.conv_hidden, self.cfg.vocab, self.cfg.k_steps);
        let (b, l) = (self.batch, self.seq_len);
        let (bl, taps, dt) = (b * l, self.taps as u32, self.cfg.dt);
        self.tokens_buf.upload_u32_prefix(&self.ctx, tokens);
        self.targets_buf.upload_u32_prefix(&self.ctx, targets);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st bwd") });

        // softmax-CE over [B*L, vocab]: drive (n_timesteps=L, batch=B) so bt_idx = bi*L+t = row.
        let scale = 1.0 / bl as f32;
        self.sce.dispatch_into(&self.ctx, &mut enc, &self.logits, &self.d_logits, &self.loss_seq, &self.targets_buf, vocab as u32, l as u32, b as u32, scale, 0);

        // readout backward: d_ro_bias += Σ d_logits; d_ro_w += outer(field, d_logits); d_field = d_logits @ ro_w
        let field = &self.s_traj[ks];
        self.row.dispatch_into(&self.ctx, &mut enc, &self.d_logits, &self.ro_bias.grad, bl as u32, vocab as u32);
        self.outer.dispatch_into(&self.ctx, &mut enc, field, &self.d_logits, &self.ro_w.grad, vocab as u32, c as u32, bl as u32);
        self.gemm.dispatch_into(&self.ctx, &mut enc, &self.d_logits, &self.ro_w.buf, &self.dfield_a, bl as u32, c as u32, vocab as u32);

        let df = [&self.dfield_a, &self.dfield_b];
        let mut cur = 0usize;
        for k in (0..ks).rev() {
            let dil = self.dilation(k);
            // d_raw = dt * d_field
            self.scale.dispatch_into(&self.ctx, &mut enc, df[cur], &self.d_raw, (bl * c) as u32, dt);
            // conv1x1 bwd: d_w2 (output-tiled outer-product), d_b2, dhpre
            self.outer.dispatch_into(&self.ctx, &mut enc, &self.hpost_traj[k], &self.d_raw, &self.w2.grad, c as u32, ch as u32, bl as u32);
            self.row.dispatch_into(&self.ctx, &mut enc, &self.d_raw, &self.b2.grad, bl as u32, c as u32);
            self.conv_bwd_hidden.dispatch_into(&self.ctx, &mut enc, &self.d_raw, &self.w2.buf, &self.hpre_traj[k], &self.dhpre, bl as u32, ch as u32, c as u32);
            // causal conv bwd: d_s, d_w1, d_b1
            self.st_ds.dispatch(&self.ctx, &mut enc, &self.dhpre, &self.w1.buf, &self.dsconv, l as u32, c as u32, ch as u32, taps, dil, b as u32);
            self.st_dw1.dispatch(&self.ctx, &mut enc, &self.dhpre, &self.s_traj[k], &self.w1.grad, l as u32, c as u32, ch as u32, taps, dil, b as u32);
            self.row.dispatch_into(&self.ctx, &mut enc, &self.dhpre, &self.b1.grad, bl as u32, ch as u32);
            // d_field = d_field + d_s  (residual passthrough + conv path)
            self.vadd.dispatch_into(&self.ctx, &mut enc, df[cur], &self.dsconv, df[1 - cur], (bl * c) as u32);
            cur = 1 - cur;
        }
        // embed grad: d_embed[tok[cell]] += d_field[cell]
        self.embed_scatter.dispatch_into(&self.ctx, &mut enc, &self.tokens_buf, df[cur], &self.embed.grad, bl as u32, c as u32);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn zero_grad(&self) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st zero") });
        for p in [&self.w1, &self.b1, &self.w2, &self.b2, &self.embed, &self.ro_w, &self.ro_bias] {
            self.zero.dispatch_into(&self.ctx, &mut enc, &p.grad, p.grad.len as u32);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn read_mean_loss(&self) -> f32 {
        let n = self.batch * self.seq_len;
        self.loss_seq.download(&self.ctx).iter().take(n).sum::<f32>() / n as f32
    }

    fn muon_one(&self, enc: &mut wgpu::CommandEncoder, params: &GpuBuffer, grad: &GpuBuffer, mom: &GpuBuffer, rows: u32, cols: u32, cfg: &MuonConfig) {
        let ctx = &self.ctx;
        let n = rows * cols;
        let transposed = rows > cols;
        let (m, ndim) = if transposed { (cols, rows) } else { (rows, cols) };
        let (mn, mmm) = (m * ndim, m * m);
        let _ = mn;
        fn pass(enc: &mut wgpu::CommandEncoder) -> wgpu::ComputePass<'_> {
            enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None })
        }
        let mom_bind = self.mu_momentum.build_bind(ctx, grad, mom, &self.mu_update);
        { let mut p = pass(enc); p.set_pipeline(&self.mu_momentum.pipeline); self.mu_momentum.dispatch_with_bind(&mut p, &mom_bind, n, cfg.nesterov, cfg.momentum); }
        self.gn_sq.dispatch_with_offset(ctx, enc, &self.mu_update, &self.mu_frob_partials, n, 0, 8);
        let inv_bind = self.mu_frob_inv.build_bind(ctx, &self.mu_frob_partials, &self.mu_frob_result);
        { let mut p = pass(enc); p.set_pipeline(&self.mu_frob_inv.pipeline); self.mu_frob_inv.dispatch_with_bind(&mut p, &inv_bind, 8); }
        self.scaleb.dispatch_into(ctx, enc, &self.mu_update, &self.mu_frob_result, n, 1);
        let load_bind = self.mu_load_store.build_bind(ctx, &self.mu_update, &self.mu_x);
        { let mut p = pass(enc); p.set_pipeline(&self.mu_load_store.pipeline); self.mu_load_store.dispatch_with_bind(&mut p, &load_bind, rows, cols, transposed, false); }
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
            { let mut p = pass(enc); p.set_pipeline(&self.mu_step_x.pipeline); self.mu_step_x.dispatch_with_bind(&mut p, &sx, m * ndim); }
        }
        let store_bind = self.mu_load_store.build_bind(ctx, &self.mu_x, &self.mu_update);
        { let mut p = pass(enc); p.set_pipeline(&self.mu_load_store.pipeline); self.mu_load_store.dispatch_with_bind(&mut p, &store_bind, rows, cols, transposed, true); }
        let scale = (ndim as f32 / m as f32).sqrt();
        let apply_bind = self.mu_apply.build_bind(ctx, params, &self.mu_update);
        { let mut p = pass(enc); p.set_pipeline(&self.mu_apply.pipeline); self.mu_apply.dispatch_with_bind(&mut p, &apply_bind, n, cfg.weight_decay > 0.0, cfg.lr, scale, cfg.weight_decay); }
    }

    pub fn step_muon_hybrid(&mut self, adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig) {
        self.step_count += 1;
        let (bc1, bc2) = bias_correction(self.step_count as usize, adam_cfg.beta1, adam_cfg.beta2);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st muon") });
        for (p, wd) in [(&self.b1, false), (&self.b2, false), (&self.embed, true), (&self.ro_bias, false)] {
            self.adamw.dispatch_into(&self.ctx, &mut enc, &p.buf, &p.grad, &p.m1, &p.m2, p.buf.len as u32, wd, adam_cfg.lr, adam_cfg.beta1, adam_cfg.beta2, adam_cfg.eps, bc1, bc2, adam_cfg.weight_decay);
        }
        let (c, ch, vocab) = (self.cfg.c, self.cfg.conv_hidden, self.cfg.vocab);
        self.muon_one(&mut enc, &self.w1.buf, &self.w1.grad, &self.mu_m_w1, ch as u32, (c * self.taps) as u32, muon_cfg);
        self.muon_one(&mut enc, &self.w2.buf, &self.w2.grad, &self.mu_m_w2, c as u32, ch as u32, muon_cfg);
        self.muon_one(&mut enc, &self.ro_w.buf, &self.ro_w.grad, &self.mu_m_row, vocab as u32, c as u32, muon_cfg);
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn train_step_muon(&mut self, tokens: &[u32], targets: &[u32], adam_cfg: &AdamWConfig, muon_cfg: &MuonConfig, want_loss: bool) -> Option<f32> {
        self.zero_grad();
        self.forward(tokens);
        self.backward_dispatch(tokens, targets);
        self.step_muon_hybrid(adam_cfg, muon_cfg);
        want_loss.then(|| self.read_mean_loss())
    }

    /// On-device val NLL over stride-sampled non-overlapping windows.
    pub fn eval_nats(&self, val_toks: &[u32], max_windows: usize) -> (f64, usize) {
        let (seq, b, vocab) = (self.seq_len, self.batch, self.cfg.vocab);
        if val_toks.len() <= seq + 1 {
            return (0.0, 0);
        }
        let all: Vec<usize> = (0..).map(|i| i * seq).take_while(|&p| p + seq + 1 < val_toks.len()).collect();
        let windows: Vec<usize> = if max_windows == 0 || max_windows >= all.len() {
            all
        } else {
            (0..max_windows).map(|i| all[i * all.len() / max_windows]).collect()
        };
        let mut toks = vec![0u32; seq * b];
        let mut tgts = vec![0u32; seq * b];
        let mut total = 0.0_f64;
        for chunk in windows.chunks(b) {
            let valid = chunk.len();
            for (bi, &p) in chunk.iter().enumerate() {
                for ti in 0..seq {
                    toks[bi * seq + ti] = val_toks[p + ti];
                    tgts[bi * seq + ti] = val_toks[p + ti + 1];
                }
            }
            for bi in valid..b {
                for ti in 0..seq {
                    toks[bi * seq + ti] = toks[ti];
                    tgts[bi * seq + ti] = tgts[ti];
                }
            }
            self.forward(&toks);
            self.targets_buf.upload_u32_prefix(&self.ctx, &tgts);
            let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st eval") });
            self.sce.dispatch_into(&self.ctx, &mut enc, &self.logits, &self.d_logits, &self.loss_seq, &self.targets_buf, vocab as u32, seq as u32, b as u32, 1.0, 0);
            self.ctx.queue.submit(Some(enc.finish()));
            let ls = self.loss_seq.download(&self.ctx);
            for bi in 0..valid {
                for ti in 0..seq {
                    total += f64::from(ls[bi * seq + ti]);
                }
            }
        }
        (total, windows.len() * seq)
    }

    #[must_use]
    pub fn download_grads(&self) -> StGrads {
        let mut g = StGrads { conv: Default::default(), d_embed: Vec::new(), d_ro_w: Vec::new(), d_ro_bias: Vec::new() };
        g.conv.d_w1 = self.w1.grad.download(&self.ctx);
        g.conv.d_b1 = self.b1.grad.download(&self.ctx);
        g.conv.d_w2 = self.w2.grad.download(&self.ctx);
        g.conv.d_b2 = self.b2.grad.download(&self.ctx);
        g.d_embed = self.embed.grad.download(&self.ctx);
        g.d_ro_w = self.ro_w.grad.download(&self.ctx);
        g.d_ro_bias = self.ro_bias.grad.download(&self.ctx);
        g
    }

    #[must_use]
    pub fn to_cpu(&self) -> StLM {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);
        let mut m = StLM::new(self.cfg.clone(), &mut rng);
        m.conv.w1 = self.w1.buf.download(&self.ctx);
        m.conv.b1 = self.b1.buf.download(&self.ctx);
        m.conv.w2 = self.w2.buf.download(&self.ctx);
        m.conv.b2 = self.b2.buf.download(&self.ctx);
        m.embed = self.embed.buf.download(&self.ctx);
        m.ro_w = self.ro_w.buf.download(&self.ctx);
        m.ro_bias = self.ro_bias.buf.download(&self.ctx);
        m.step_count = self.step_count;
        m
    }

    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Download the `[B*L*vocab]` logits (parity / debug).
    #[must_use]
    pub fn download_logits(&self) -> Vec<f32> {
        self.logits.download(&self.ctx)
    }
}
