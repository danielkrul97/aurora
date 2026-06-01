//! Trivial element-wise kernels used for residual add and zero-grad.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const VECTOR_ADD_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] + b[i];
}
";

const ZERO_BUFFER_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = 0.0;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EltParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

pub struct VectorAddKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl VectorAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora vector_add"),
                source: wgpu::ShaderSource::Wgsl(VECTOR_ADD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora vector_add bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora vector_add pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora vector_add"),
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
        a: &GpuBuffer,
        b: &GpuBuffer,
        y: &GpuBuffer,
        n: u32,
    ) {
        let p = EltParams { n, _p0: 0, _p1: 0, _p2: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vector_add bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora vector_add pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        a: &GpuBuffer,
        b: &GpuBuffer,
        y: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vector_add bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = EltParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

const VECTOR_ADD3_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read>       c: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] + b[i] + c[i];
}
";

/// Fused 3-input elementwise add `y = a + b + c`. Replaces a two-`VectorAdd`
/// chain (saves one dispatch + one full-field round-trip per call).
pub struct VectorAdd3Kernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl VectorAdd3Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora vector_add3"),
            source: wgpu::ShaderSource::Wgsl(VECTOR_ADD3_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora vector_add3 bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora vector_add3 pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora vector_add3"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        y: &GpuBuffer,
        n: u32,
    ) {
        let p = EltParams { n, _p0: 0, _p1: 0, _p2: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vector_add3 bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora vector_add3 pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

pub struct ZeroBufferKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl ZeroBufferKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora zero_buffer"),
                source: wgpu::ShaderSource::Wgsl(ZERO_BUFFER_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora zero_buffer bgl"),
                entries: &[storage_entry(0, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora zero_buffer pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora zero_buffer"),
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
        n: u32,
    ) {
        let p = EltParams { n, _p0: 0, _p1: 0, _p2: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora zero_buffer bind"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() }],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora zero_buffer pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    /// Build a cached bind group for a specific buffer (call once at init).
    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora zero_buffer bind"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() }],
        })
    }

    /// Issue a zero dispatch on an existing pass with a pre-built bind
    /// group. Caller must `set_pipeline(&kernel.pipeline)` once before
    /// the first call.
    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = EltParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

