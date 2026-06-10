//! LayerNorm / RMSNorm forward + backward (per-row reduction).
//! `mode == 1` selects RMSNorm: no mean subtraction, β unused (forward)
//! and ungradded (backward).

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const LN_FORWARD_WGSL: &str = "
struct P { t: u32, d: u32, eps: f32, mode: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> inv_std: array<f32>;
@group(0) @binding(6) var<storage, read_write> x_hat: array<f32>;
var<push_constant> p: P;

var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_sq: array<f32, 256>;
var<workgroup> sh_mean: f32;
var<workgroup> sh_inv: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= p.t) { return; }
    let base = row * p.d;

    var local_sum: f32 = 0.0;
    var local_sq: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.d) { break; }
        let v = x[base + i];
        local_sum = local_sum + v;
        local_sq = local_sq + v * v;
        i = i + 256u;
    }
    sh_sum[tid] = local_sum;
    sh_sq[tid] = local_sq;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
            sh_sq[tid] = sh_sq[tid] + sh_sq[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        let inv_d = 1.0 / f32(p.d);
        let mu = select(sh_sum[0] * inv_d, 0.0, p.mode == 1u);
        let var_ = sh_sq[0] * inv_d - mu * mu;
        let s = inverseSqrt(var_ + p.eps);
        sh_mean = mu;
        sh_inv = s;
        mean[row] = mu;
        inv_std[row] = s;
    }
    workgroupBarrier();

    i = tid;
    loop {
        if (i >= p.d) { break; }
        let h = (x[base + i] - sh_mean) * sh_inv;
        x_hat[base + i] = h;
        y[base + i] = h * gamma[i] + select(beta[i], 0.0, p.mode == 1u);
        i = i + 256u;
    }
}
";

// d_gamma / d_beta accumulate across T rows on the same address — use
// f32 atomic-add via CAS on a bit-cast u32 view. Eliminates the per-row
// [T × D] scratch + separate row_reduce dispatch that the previous design
// needed (~13 LN bwds per train step × 2 × T × D extra global writes).
//
// Caller must pre-zero d_gamma/d_beta (the model's `g_*_gamma`/`g_*_beta`
// buffers are zeroed by `zero_grad()` once per step before LN bwd runs).
const LN_BACKWARD_WGSL: &str = "
struct P { t: u32, d: u32, mode: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       dy: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       x_hat: array<f32>;
@group(0) @binding(3) var<storage, read>       inv_std: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_gamma: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> d_beta: array<atomic<u32>>;
var<push_constant> p: P;

var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;
var<workgroup> sh_sum_g: f32;
var<workgroup> sh_sum_gx: f32;

fn atomic_add_d_gamma(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    loop {
        let old_u = atomicLoad(&d_gamma[idx]);
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let r = atomicCompareExchangeWeak(&d_gamma[idx], old_u, new_u);
        if (r.exchanged) { break; }
    }
}

fn atomic_add_d_beta(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    loop {
        let old_u = atomicLoad(&d_beta[idx]);
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let r = atomicCompareExchangeWeak(&d_beta[idx], old_u, new_u);
        if (r.exchanged) { break; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= p.t) { return; }
    let base = row * p.d;
    let inv_d = 1.0 / f32(p.d);

    var local_a: f32 = 0.0;
    var local_b: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.d) { break; }
        let g = dy[base + i] * gamma[i];
        let xh = x_hat[base + i];
        local_a = local_a + g;
        local_b = local_b + g * xh;
        i = i + 256u;
    }
    sh_a[tid] = local_a;
    sh_b[tid] = local_b;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_a[tid] = sh_a[tid] + sh_a[tid + stride];
            sh_b[tid] = sh_b[tid] + sh_b[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_sum_g = sh_a[0];
        sh_sum_gx = sh_b[0];
    }
    workgroupBarrier();

    let s = inv_std[row];
    let mean_term = select(1.0, 0.0, p.mode == 1u);
    i = tid;
    loop {
        if (i >= p.d) { break; }
        let xh = x_hat[base + i];
        let dy_i = dy[base + i];
        let g = dy_i * gamma[i];
        dx[base + i] = s * (g - inv_d * (mean_term * sh_sum_g + xh * sh_sum_gx));
        atomic_add_d_gamma(i, dy_i * xh);
        if (p.mode == 0u) {
            atomic_add_d_beta(i, dy_i);
        }
        i = i + 256u;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LnForwardParams {
    t: u32,
    d: u32,
    eps: f32,
    mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LnBackwardParams {
    t: u32,
    d: u32,
    mode: u32,
    _pad1: u32,
}

pub struct LayerNormForwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl LayerNormForwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora ln fwd"),
                source: wgpu::ShaderSource::Wgsl(LN_FORWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora ln fwd bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora ln fwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora ln fwd"),
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
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        y: &GpuBuffer,
        mean: &GpuBuffer,
        inv_std: &GpuBuffer,
        x_hat: &GpuBuffer,
        t: u32,
        d: u32,
        eps: f32,
        rms: bool,
    ) {
        let p = LnForwardParams { t, d, eps, mode: u32::from(rms) };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ln fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: beta.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mean.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: inv_std.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: x_hat.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora ln fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        x: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        y: &GpuBuffer,
        mean: &GpuBuffer,
        inv_std: &GpuBuffer,
        x_hat: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ln fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: beta.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mean.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: inv_std.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: x_hat.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        eps: f32,
        rms: bool,
    ) {
        let p = LnForwardParams { t, d, eps, mode: u32::from(rms) };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, 1, 1);
    }
}

pub struct LayerNormBackwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl LayerNormBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora ln bwd"),
                source: wgpu::ShaderSource::Wgsl(LN_BACKWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora ln bwd bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora ln bwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora ln bwd"),
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

    /// Backward LayerNorm. `d_gamma` and `d_beta` are per-feature `[D]`
    /// buffers; the kernel atomically accumulates into them. Caller must
    /// pre-zero them (model grad buffers are zeroed by `zero_grad()` once
    /// per step).
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        dy: &GpuBuffer,
        gamma: &GpuBuffer,
        x_hat: &GpuBuffer,
        inv_std: &GpuBuffer,
        dx: &GpuBuffer,
        d_gamma: &GpuBuffer,
        d_beta: &GpuBuffer,
        t: u32,
        d: u32,
        rms: bool,
    ) {
        let p = LnBackwardParams { t, d, mode: u32::from(rms), _pad1: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ln bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x_hat.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: inv_std.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dx.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: d_gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: d_beta.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora ln bwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        dy: &GpuBuffer,
        gamma: &GpuBuffer,
        x_hat: &GpuBuffer,
        inv_std: &GpuBuffer,
        dx: &GpuBuffer,
        d_gamma: &GpuBuffer,
        d_beta: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ln bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x_hat.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: inv_std.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dx.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: d_gamma.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: d_beta.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        rms: bool,
    ) {
        let p = LnBackwardParams { t, d, mode: u32::from(rms), _pad1: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, 1, 1);
    }
}
