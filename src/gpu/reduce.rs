//! Reduction kernels for pure-GPU gradient accumulation. Replaces the
//! download → sum on CPU → re-upload roundtrips that dominated per-step
//! latency.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

// OuterProductAccumKernel — dW[i, j] += sum_t d_y[t, i] * x[t, j].
// dest [d_out, d_in], d_y [t, d_out], x [t, d_in]. One workgroup per
// (i, j) tile of 16×16; sum-over-t happens inside the workgroup with
// no atomics needed (each tile writes a disjoint slice of dest).

const OUTER_ACCUM_WGSL: &str = "
struct P {
    d_out: u32,
    d_in: u32,
    t: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       d_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> dest: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let i = wgid.y * 16u + lid.y;
    let j = wgid.x * 16u + lid.x;
    if (i >= p.d_out || j >= p.d_in) { return; }
    var acc: f32 = 0.0;
    for (var ti: u32 = 0u; ti < p.t; ti = ti + 1u) {
        acc = acc + d_y[ti * p.d_out + i] * x[ti * p.d_in + j];
    }
    dest[i * p.d_in + j] = dest[i * p.d_in + j] + acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterAccumParams {
    d_out: u32,
    d_in: u32,
    t: u32,
    _pad: u32,
}

pub struct OuterProductAccumKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl OuterProductAccumKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora outer_accum"),
                source: wgpu::ShaderSource::Wgsl(OUTER_ACCUM_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora outer_accum bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora outer_accum pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora outer_accum"),
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

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        d_y: &GpuBuffer,
        dest: &GpuBuffer,
        d_out: u32,
        d_in: u32,
        t: u32,
    ) {
        let p = OuterAccumParams { d_out, d_in, t, _pad: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora outer_accum bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d_y.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dest.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora outer_accum pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let gx = d_in.div_ceil(16);
        let gy = d_out.div_ceil(16);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

// RowReduceAddKernel — dest[c] += sum_t src[t, c].
// One workgroup per output column; sums over t with stride loop.

const ROW_REDUCE_ADD_WGSL: &str = "
struct P { t: u32, d: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dest: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.d) { return; }
    var acc: f32 = 0.0;
    for (var ti: u32 = 0u; ti < p.t; ti = ti + 1u) {
        acc = acc + src[ti * p.d + c];
    }
    dest[c] = dest[c] + acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RowReduceParams {
    t: u32,
    d: u32,
    _p0: u32,
    _p1: u32,
}

pub struct RowReduceAddKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl RowReduceAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora row_reduce_add"),
                source: wgpu::ShaderSource::Wgsl(ROW_REDUCE_ADD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora row_reduce_add bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora row_reduce_add pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora row_reduce_add"),
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
        src: &GpuBuffer,
        dest: &GpuBuffer,
        t: u32,
        d: u32,
    ) {
        let p = RowReduceParams { t, d, _p0: 0, _p1: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora row_reduce_add bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dest.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora row_reduce_add pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let groups = d.div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

// GradNormSqReduceKernel — sum of squares over a buffer.
// Workgroup-shared reduction within 256 threads; WGSL has no f32 atomic,
// so multi-workgroup runs write partials to a host-readable slice and
// the host finishes the reduction.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GradNormParams {
    n: u32,
    _pad: [u32; 3],
}

const GRAD_NORM_SQ_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
var<push_constant> p: P;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) ngroups: vec3<u32>,
) {
    let tid = lid.x;
    let stride = 256u * ngroups.x;
    var local: f32 = 0.0;
    var idx = wgid.x * 256u + tid;
    loop {
        if (idx >= p.n) { break; }
        let v = x[idx];
        local = local + v * v;
        idx = idx + stride;
    }
    sh[tid] = local;
    workgroupBarrier();
    var s: u32 = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) {
        partials[wgid.x] = sh[0];
    }
}
";

pub struct GradNormSqReduceKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl GradNormSqReduceKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora grad_norm_sq"),
                source: wgpu::ShaderSource::Wgsl(GRAD_NORM_SQ_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora grad_norm_sq bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora grad_norm_sq pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora grad_norm_sq"),
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

    /// Writes one f32 partial per workgroup into `partials`. Caller is
    /// expected to download `partials` and sum on host. With 32 groups
    /// covering buffers up to 8192 elements, the host sum is trivial.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        partials: &GpuBuffer,
        n: u32,
        n_groups: u32,
    ) {
        debug_assert!(partials.len >= n_groups as usize);
        let p = GradNormParams { n, _pad: [0; 3] };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora grad_norm_sq bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: partials.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora grad_norm_sq pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n_groups, 1, 1);
    }
}

// EmbedScatterReduceKernel — dest[tok[ti], c] += src[ti, c] over all ti.
// Serialized in a single workgroup because multiple positions with the
// same token would race on the same destination row, and WGSL has no
// f32 atomics. T is bounded (≤ seq_len), so 1 workgroup is fine.

const EMBED_SCATTER_REDUCE_WGSL: &str = "
struct P { t: u32, d: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       per_pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> dest: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(1)
fn main() {
    for (var ti: u32 = 0u; ti < p.t; ti = ti + 1u) {
        let tok = tokens[ti];
        for (var c: u32 = 0u; c < p.d; c = c + 1u) {
            dest[tok * p.d + c] = dest[tok * p.d + c] + per_pos[ti * p.d + c];
        }
    }
}
";

pub struct EmbedScatterReduceKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl EmbedScatterReduceKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora embed_scatter_reduce"),
                source: wgpu::ShaderSource::Wgsl(EMBED_SCATTER_REDUCE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora embed_scatter_reduce bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora embed_scatter_reduce pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora embed_scatter_reduce"),
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
        tokens: &GpuBuffer,
        per_pos: &GpuBuffer,
        dest: &GpuBuffer,
        t: u32,
        d: u32,
    ) {
        let p = RowReduceParams { t, d, _p0: 0, _p1: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed_scatter_reduce bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: per_pos.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dest.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora embed_scatter_reduce pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }
}
