//! Canon causal depthwise-conv layer (GPU). Mirrors `canon_forward` /
//! `canon_backward` in `crate::transformer::block`. Operates on one
//! sequence `[T * d]` (the GPU runs one sequence per forward/backward),
//! so taps only look back within the sequence — no batch boundaries.
//!
//! Forward:  `out[t,c] = x[t,c] + Σ_{j<k, t≥j} w[c,j]·x[t-j,c]`
//! Backward splits into two race-free kernels (no f32 atomics):
//!   * dx (gather): `d_x[s,c] = d_out[s,c] + Σ_{j<k, s+j<T} w[c,j]·d_out[s+j,c]`
//!   * dw (per-tap reduce): `d_w[c,j] += Σ_{t≥j} d_out[t,c]·x[t-j,c]` (accumulates)

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const CANON_FWD_WGSL: &str = "
struct P { t: u32, d: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t * p.d) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    var acc = x[i];
    var j: u32 = 0u;
    loop {
        if (j >= p.k || j > ti) { break; }
        acc = acc + w[c * p.k + j] * x[(ti - j) * p.d + c];
        j = j + 1u;
    }
    out[i] = acc;
}
";

const CANON_DX_WGSL: &str = "
struct P { t: u32, d: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       d_out: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_x: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t * p.d) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    var acc = d_out[i];
    var j: u32 = 0u;
    loop {
        if (j >= p.k || ti + j >= p.t) { break; }
        acc = acc + w[c * p.k + j] * d_out[(ti + j) * p.d + c];
        j = j + 1u;
    }
    d_x[i] = acc;
}
";

const CANON_DW_WGSL: &str = "
struct P { t: u32, d: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       d_out: array<f32>;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_w: array<f32>;
var<push_constant> p: P;

// One thread per (c, j) tap; reduces over the sequence. Accumulates into
// d_w (the grad persists across the B-sequence minibatch loop).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.d * p.k) { return; }
    let c = idx / p.k;
    let j = idx % p.k;
    var acc = 0.0;
    var ti: u32 = j;
    loop {
        if (ti >= p.t) { break; }
        acc = acc + d_out[ti * p.d + c] * x[(ti - j) * p.d + c];
        ti = ti + 1u;
    }
    d_w[idx] = d_w[idx] + acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CanonParams {
    t: u32,
    d: u32,
    k: u32,
    _pad: u32,
}

fn build_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    (Arc::new(pipeline), Arc::new(bgl))
}

fn bind3(
    ctx: &GpuContext,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.inner.as_entire_binding() },
        ],
    })
}

/// Forward residual depthwise conv: `out = x + dwconv(x)`.
pub struct CanonForwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl CanonForwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora canon fwd",
            CANON_FWD_WGSL,
            &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer) -> wgpu::BindGroup {
        bind3(ctx, &self.bgl, "aurora canon fwd bind", x, w, out)
    }

    pub fn dispatch_with_bind(&self, pass: &mut wgpu::ComputePass<'_>, bind: &wgpu::BindGroup, t: u32, d: u32, k: u32) {
        let p = CanonParams { t, d, k, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}

/// Backward gather: `d_x = d_out + dwconv_transpose(d_out)`.
pub struct CanonDxKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl CanonDxKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora canon dx",
            CANON_DX_WGSL,
            &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(&self, ctx: &GpuContext, d_out: &GpuBuffer, w: &GpuBuffer, d_x: &GpuBuffer) -> wgpu::BindGroup {
        bind3(ctx, &self.bgl, "aurora canon dx bind", d_out, w, d_x)
    }

    pub fn dispatch_with_bind(&self, pass: &mut wgpu::ComputePass<'_>, bind: &wgpu::BindGroup, t: u32, d: u32, k: u32) {
        let p = CanonParams { t, d, k, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}

/// Backward per-tap reduction; accumulates into `d_w`.
pub struct CanonDwKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl CanonDwKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora canon dw",
            CANON_DW_WGSL,
            &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(&self, ctx: &GpuContext, d_out: &GpuBuffer, x: &GpuBuffer, d_w: &GpuBuffer) -> wgpu::BindGroup {
        bind3(ctx, &self.bgl, "aurora canon dw bind", d_out, x, d_w)
    }

    pub fn dispatch_with_bind(&self, pass: &mut wgpu::ComputePass<'_>, bind: &wgpu::BindGroup, t: u32, d: u32, k: u32) {
        let p = CanonParams { t, d, k, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((d * k).div_ceil(64), 1, 1);
    }
}
