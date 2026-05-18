//! Reduction kernels for pure-GPU gradient accumulation. Replaces the
//! download → sum on CPU → re-upload roundtrips that dominated per-step
//! latency.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

// OuterProductAccumKernel — dW[i, j] += sum_t d_y[t, i] * x[t, j].
// Tile geometry mirrors the matmul kernels: 64×64 output tile per
// workgroup, 16-wide T tile, 4×4 register tile per thread.

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

var<workgroup> ta: array<f32, 1024>;  // [WG_M=64, WG_K=16] — d_y^T view
var<workgroup> tb: array<f32, 1024>;  // [WG_K=16, WG_N=64] — x

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wgid.y * 64u + ty * 4u;
    let col0 = wgid.x * 64u + tx * 4u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let n_tiles = (p.t + 15u) / 16u;
    for (var t_idx: u32 = 0u; t_idx < n_tiles; t_idx = t_idx + 1u) {
        let k_off = t_idx * 16u;

        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let r = flat / 16u;
            let kk = flat % 16u;
            let g_row = wgid.y * 64u + r;
            let g_k = k_off + kk;
            if (g_row < p.d_out && g_k < p.t) {
                ta[r * 16u + kk] = d_y[g_k * p.d_out + g_row];
            } else {
                ta[r * 16u + kk] = 0.0;
            }
        }
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let kk = flat / 64u;
            let cc = flat % 64u;
            let g_k = k_off + kk;
            let g_col = wgid.x * 64u + cc;
            if (g_k < p.t && g_col < p.d_in) {
                tb[kk * 64u + cc] = x[g_k * p.d_in + g_col];
            } else {
                tb[kk * 64u + cc] = 0.0;
            }
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = ta[(ty * 4u + 0u) * 16u + kk];
            let a1 = ta[(ty * 4u + 1u) * 16u + kk];
            let a2 = ta[(ty * 4u + 2u) * 16u + kk];
            let a3 = ta[(ty * 4u + 3u) * 16u + kk];
            let b0 = tb[kk * 64u + tx * 4u + 0u];
            let b1 = tb[kk * 64u + tx * 4u + 1u];
            let b2 = tb[kk * 64u + tx * 4u + 2u];
            let b3 = tb[kk * 64u + tx * 4u + 3u];
            acc[0]  = acc[0]  + a0 * b0;
            acc[1]  = acc[1]  + a0 * b1;
            acc[2]  = acc[2]  + a0 * b2;
            acc[3]  = acc[3]  + a0 * b3;
            acc[4]  = acc[4]  + a1 * b0;
            acc[5]  = acc[5]  + a1 * b1;
            acc[6]  = acc[6]  + a1 * b2;
            acc[7]  = acc[7]  + a1 * b3;
            acc[8]  = acc[8]  + a2 * b0;
            acc[9]  = acc[9]  + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let r = row0 + i;
            let cc = col0 + j;
            if (r < p.d_out && cc < p.d_in) {
                dest[r * p.d_in + cc] = dest[r * p.d_in + cc] + acc[i * 4u + j];
            }
        }
    }
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
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
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
        let gx = d_in.div_ceil(64);
        let gy = d_out.div_ceil(64);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        x: &GpuBuffer,
        d_y: &GpuBuffer,
        dest: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora outer_accum bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d_y.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dest.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        d_out: u32,
        d_in: u32,
        t: u32,
    ) {
        let p = OuterAccumParams { d_out, d_in, t, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let gx = d_in.div_ceil(64);
        let gy = d_out.div_ceil(64);
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
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
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
// Workgroup-shared reduction within 256 threads. `offset` lets multiple
// tensors stage their partials side-by-side in one flat buffer, so a
// single finalize kernel can reduce all of them without per-tensor host
// downloads.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GradNormParams {
    n: u32,
    offset: u32,
    _pad: [u32; 2],
}

const GRAD_NORM_SQ_WGSL: &str = "
struct P { n: u32, offset: u32, _p0: u32, _p1: u32 };
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
        partials[p.offset + wgid.x] = sh[0];
    }
}
";

pub struct GradNormSqReduceKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
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

    /// Writes `n_groups` partial sums starting at `partials[offset]`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        partials: &GpuBuffer,
        n: u32,
        n_groups: u32,
    ) {
        self.dispatch_with_offset(ctx, encoder, x, partials, n, 0, n_groups);
    }

    pub fn dispatch_with_offset(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        partials: &GpuBuffer,
        n: u32,
        offset: u32,
        n_groups: u32,
    ) {
        debug_assert!(partials.len >= (offset + n_groups) as usize);
        let p = GradNormParams { n, offset, _pad: [0; 2] };
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

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        x: &GpuBuffer,
        partials: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora grad_norm_sq bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: partials.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        offset: u32,
        n_groups: u32,
    ) {
        let p = GradNormParams { n, offset, _pad: [0; 2] };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n_groups, 1, 1);
    }
}

// =====================================================================
// GradNormFinalizeKernel — sum flat partials → [norm, clip_factor]
// =====================================================================
//
// Reads `partials[0..n]`, sums them, writes:
//   result[0] = sqrt(sum)             — L2 norm of all grads
//   result[1] = min(1, max_norm/norm) — clip factor (1.0 if no clip)
//
// Single workgroup (256 threads, stride-loop over partials, workgroup
// shared-memory reduction). `n` is bounded by the total number of grad
// tensors × N_GROUPS_PER_TENSOR — for nano-GPT class ≤ 600.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FinalizeParams {
    n: u32,
    _p0: u32,
    max_norm: f32,
    eps: f32,
}

const GRAD_NORM_FINALIZE_WGSL: &str = "
struct P { n: u32, _p0: u32, max_norm: f32, eps: f32 };
@group(0) @binding(0) var<storage, read>       partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<push_constant> p: P;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var local: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.n) { break; }
        local = local + partials[i];
        i = i + 256u;
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
        let norm = sqrt(max(sh[0], 0.0));
        result[0] = norm;
        let f = p.max_norm / max(norm, p.eps);
        result[1] = min(1.0, f);
    }
}
";

pub struct GradNormFinalizeKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl GradNormFinalizeKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora gn_finalize"),
                source: wgpu::ShaderSource::Wgsl(GRAD_NORM_FINALIZE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora gn_finalize bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gn_finalize pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora gn_finalize"),
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

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        partials: &GpuBuffer,
        result: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gn_finalize bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: partials.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        max_norm: f32,
    ) {
        let p = FinalizeParams { n, _p0: 0, max_norm, eps: 1e-12 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Reduce `partials[0..n]` and write `[norm, clip_factor]` into `result`.
    /// `max_norm` = f32::INFINITY skips clipping (factor → 1.0).
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        partials: &GpuBuffer,
        result: &GpuBuffer,
        n: u32,
        max_norm: f32,
    ) {
        debug_assert!(result.len >= 2);
        let p = FinalizeParams { n, _p0: 0, max_norm, eps: 1e-12 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gn_finalize bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: partials.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora gn_finalize pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }
}

// =====================================================================
// ScaleByBufferKernel — x[i] *= factor_buf[factor_idx]
// =====================================================================
//
// In-place scale by a GPU-resident scalar so clip_grads never has to
// download the scale factor to the host.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleByBufParams {
    n: u32,
    factor_idx: u32,
    _p0: u32,
    _p1: u32,
}

const SCALE_BY_BUFFER_WGSL: &str = "
struct P { n: u32, factor_idx: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       factor: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] * factor[p.factor_idx];
}
";

pub struct ScaleByBufferKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl ScaleByBufferKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora scale_by_buffer"),
                source: wgpu::ShaderSource::Wgsl(SCALE_BY_BUFFER_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora scale_by_buffer bgl"),
                entries: &[storage_entry(0, false), storage_entry(1, true)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora scale_by_buffer pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora scale_by_buffer"),
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
        factor: &GpuBuffer,
        n: u32,
        factor_idx: u32,
    ) {
        let p = ScaleByBufParams { n, factor_idx, _p0: 0, _p1: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora scale_by_buffer bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: factor.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora scale_by_buffer pass"),
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
        x: &GpuBuffer,
        factor: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora scale_by_buffer bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: factor.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        factor_idx: u32,
    ) {
        let p = ScaleByBufParams { n, factor_idx, _p0: 0, _p1: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

// EmbedScatterReduceKernel — dest[tok[ti], c] += src[ti, c] over all ti.
// WGSL has u32 atomics but not f32 atomics; we get f32 atomic-add by
// CAS-looping on a bit-cast u32 view of the same buffer. dest is the
// shared `g_w_tok` storage — other kernels (OuterProductAccum, AdamW)
// read/write it as `array<f32>`; the bytes are identical.

const EMBED_SCATTER_REDUCE_WGSL: &str = "
struct P { t: u32, d: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       per_pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> dest: array<atomic<u32>>;
var<push_constant> p: P;

fn atomic_add_f32(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    loop {
        let old_u = atomicLoad(&dest[idx]);
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let r = atomicCompareExchangeWeak(&dest[idx], old_u, new_u);
        if (r.exchanged) { break; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    let total = p.t * p.d;
    if (lane >= total) { return; }
    let ti = lane / p.d;
    let c = lane % p.d;
    let tok = tokens[ti];
    let val = per_pos[ti * p.d + c];
    atomic_add_f32(tok * p.d + c, val);
}
";

pub struct EmbedScatterReduceKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
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
        let groups = (t * d).div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        tokens: &GpuBuffer,
        per_pos: &GpuBuffer,
        dest: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed_scatter_reduce bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: per_pos.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dest.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
    ) {
        let p = RowReduceParams { t, d, _p0: 0, _p1: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}
