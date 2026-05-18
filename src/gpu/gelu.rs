//! Element-wise GELU forward + backward.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const GELU_FORWARD_WGSL: &str = "
struct G { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
var<push_constant> p: G;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = x[i];
    let k: f32 = 0.7978845608028654; // sqrt(2/π)
    let inner = k * (v + 0.044715 * v * v * v);
    y[i] = 0.5 * v * (1.0 + tanh(inner));
}
";

const GELU_BACKWARD_WGSL: &str = "
struct G { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
var<push_constant> p: G;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = x[i];
    let k: f32 = 0.7978845608028654;
    let inner = k * (v + 0.044715 * v * v * v);
    let th = tanh(inner);
    let sech2 = 1.0 - th * th;
    let d_inner_dv = k * (1.0 + 3.0 * 0.044715 * v * v);
    let g = 0.5 * (1.0 + th) + 0.5 * v * sech2 * d_inner_dv;
    dx[i] = dy[i] * g;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GeluParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

pub struct GeluKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl GeluKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora gelu fwd"),
                source: wgpu::ShaderSource::Wgsl(GELU_FORWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora gelu fwd bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gelu fwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora gelu fwd"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline: Arc::new(pipeline),
            bgl: Arc::new(bgl),
        }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        y: &GpuBuffer,
        n: u32,
    ) {
        let p = GeluParams { n, _p0: 0, _p1: 0, _p2: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gelu fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora gelu fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let groups = n.div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, y: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gelu fwd bind"),
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
        n: u32,
    ) {
        let p = GeluParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

pub struct GeluBackwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl GeluBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora gelu bwd"),
                source: wgpu::ShaderSource::Wgsl(GELU_BACKWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora gelu bwd bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gelu bwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora gelu bwd"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline: Arc::new(pipeline),
            bgl: Arc::new(bgl),
        }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        dy: &GpuBuffer,
        dx: &GpuBuffer,
        n: u32,
    ) {
        let p = GeluParams { n, _p0: 0, _p1: 0, _p2: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gelu bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora gelu bwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let groups = n.div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        x: &GpuBuffer,
        dy: &GpuBuffer,
        dx: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gelu bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dx.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = GeluParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
