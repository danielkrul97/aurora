//! Parameter-free QK-norm: RMS-normalize each per-head row of Q/K
//! (`[T * n_head * d_head]`). One workgroup per (position, head) row.
//! Mirrors `qk_norm_forward` / `qk_norm_backward` in `crate::attention`.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const QK_NORM_FORWARD_WGSL: &str = "
struct P { t: u32, n_head: u32, d_head: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;

const EPS: f32 = 1e-6;

var<workgroup> sh: array<f32, 64>;
var<workgroup> sh_r: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= p.t * p.n_head) { return; }
    let d = p.n_head * p.d_head;
    let ti = row / p.n_head;
    let h = row % p.n_head;
    let base = ti * d + h * p.d_head;

    var local_sq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.d_head) { break; }
        let v = x[base + i];
        local_sq = local_sq + v * v;
        i = i + 64u;
    }
    sh[tid] = local_sq;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh[tid] = sh[tid] + sh[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_r = inverseSqrt(sh[0] / f32(p.d_head) + EPS);
    }
    workgroupBarrier();

    i = tid;
    loop {
        if (i >= p.d_head) { break; }
        y[base + i] = x[base + i] * sh_r;
        i = i + 64u;
    }
}
";

// dy (grad w.r.t. normalized rows) is rewritten in place to the grad
// w.r.t. raw x: dx = r·(dy − y·mean(dy·y)) with y = x·r.
const QK_NORM_BACKWARD_WGSL: &str = "
struct P { t: u32, n_head: u32, d_head: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> dy: array<f32>;
var<push_constant> p: P;

const EPS: f32 = 1e-6;

var<workgroup> sh_sq: array<f32, 64>;
var<workgroup> sh_dot: array<f32, 64>;
var<workgroup> sh_r: f32;
var<workgroup> sh_s: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= p.t * p.n_head) { return; }
    let d = p.n_head * p.d_head;
    let ti = row / p.n_head;
    let h = row % p.n_head;
    let base = ti * d + h * p.d_head;

    var local_sq: f32 = 0.0;
    var local_dot: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.d_head) { break; }
        let v = x[base + i];
        local_sq = local_sq + v * v;
        local_dot = local_dot + dy[base + i] * v;
        i = i + 64u;
    }
    sh_sq[tid] = local_sq;
    sh_dot[tid] = local_dot;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sq[tid] = sh_sq[tid] + sh_sq[tid + stride];
            sh_dot[tid] = sh_dot[tid] + sh_dot[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        let inv_dh = 1.0 / f32(p.d_head);
        let r = inverseSqrt(sh_sq[0] * inv_dh + EPS);
        sh_r = r;
        sh_s = sh_dot[0] * r * inv_dh;
    }
    workgroupBarrier();

    i = tid;
    loop {
        if (i >= p.d_head) { break; }
        let y = x[base + i] * sh_r;
        dy[base + i] = sh_r * (dy[base + i] - y * sh_s);
        i = i + 64u;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QkNormParams {
    t: u32,
    n_head: u32,
    d_head: u32,
    _pad: u32,
}

fn build_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
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

pub struct QkNormKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl QkNormKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora qk_norm fwd",
            QK_NORM_FORWARD_WGSL,
            &[storage_entry(0, true), storage_entry(1, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, y: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora qk_norm fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
    ) {
        let p = QkNormParams { t, n_head, d_head, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t * n_head, 1, 1);
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        y: &GpuBuffer,
        t: u32,
        n_head: u32,
        d_head: u32,
    ) {
        let bind = self.build_bind(ctx, x, y);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora qk_norm fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        self.dispatch_with_bind(&mut pass, &bind, t, n_head, d_head);
    }
}

pub struct QkNormBackwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl QkNormBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora qk_norm bwd",
            QK_NORM_BACKWARD_WGSL,
            &[storage_entry(0, true), storage_entry(1, false)],
        );
        Self { pipeline, bgl }
    }

    /// `dy` is rewritten in place from grad-w.r.t.-normalized to
    /// grad-w.r.t.-raw; `x` is the cached pre-norm Q or K.
    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, dy: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora qk_norm bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
    ) {
        let p = QkNormParams { t, n_head, d_head, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t * n_head, 1, 1);
    }
}
