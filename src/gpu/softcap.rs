//! Logit softcap `cap·tanh(x/cap)`. Forward rewrites the logits buffer
//! in place; backward recovers the tanh from the capped logits
//! (`tanh = logit/cap`) and scales `d_logits` by `1 − tanh²` — no extra
//! activation cache needed.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const SOFTCAP_FORWARD_WGSL: &str = "
struct P { n: u32, cap: f32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = p.cap * tanh(y[i] / p.cap);
}
";

const SOFTCAP_BACKWARD_WGSL: &str = "
struct P { n: u32, cap: f32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       capped: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let th = capped[i] / p.cap;
    d[i] = d[i] * (1.0 - th * th);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftcapParams {
    n: u32,
    cap: f32,
    _p1: u32,
    _p2: u32,
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

pub struct SoftcapKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl SoftcapKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora softcap fwd",
            SOFTCAP_FORWARD_WGSL,
            &[storage_entry(0, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(&self, ctx: &GpuContext, y: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora softcap fwd bind"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: y.inner.as_entire_binding(),
            }],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        cap: f32,
    ) {
        let p = SoftcapParams { n, cap, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        y: &GpuBuffer,
        n: u32,
        cap: f32,
    ) {
        let bind = self.build_bind(ctx, y);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora softcap fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        self.dispatch_with_bind(&mut pass, &bind, n, cap);
    }
}

pub struct SoftcapBackwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl SoftcapBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora softcap bwd",
            SOFTCAP_BACKWARD_WGSL,
            &[storage_entry(0, true), storage_entry(1, false)],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        capped: &GpuBuffer,
        d: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora softcap bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: capped.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        cap: f32,
    ) {
        let p = SoftcapParams { n, cap, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
