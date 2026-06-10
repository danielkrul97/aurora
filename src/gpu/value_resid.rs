//! Value residual: `v = λ[0]·v_raw + λ[1]·v0` with learnable per-layer
//! scalars read from a 2-float buffer. Backward rewrites d_v in place
//! (λ[0] chain), accumulates λ[1]·d_v into the cross-layer d_v0 buffer,
//! and reduces the two λ gradients via per-workgroup shared sums +
//! one atomic CAS add each.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const VMIX_FORWARD_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       v_raw: array<f32>;
@group(0) @binding(1) var<storage, read>       v0: array<f32>;
@group(0) @binding(2) var<storage, read>       lambda: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    v_out[i] = lambda[0] * v_raw[i] + lambda[1] * v0[i];
}
";

const VMIX_BACKWARD_WGSL: &str = "
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       v_raw: array<f32>;
@group(0) @binding(1) var<storage, read>       v0: array<f32>;
@group(0) @binding(2) var<storage, read>       lambda: array<f32>;
@group(0) @binding(3) var<storage, read_write> dv: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_v0_accum: array<f32>;
@group(0) @binding(5) var<storage, read_write> g_lambda: array<atomic<u32>>;
var<push_constant> p: P;

var<workgroup> sh_da: array<f32, 256>;
var<workgroup> sh_db: array<f32, 256>;

fn atomic_add_f32(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    loop {
        let old_u = atomicLoad(&g_lambda[idx]);
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let r = atomicCompareExchangeWeak(&g_lambda[idx], old_u, new_u);
        if (r.exchanged) { break; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let tid = lid.x;
    var da: f32 = 0.0;
    var db: f32 = 0.0;
    if (i < p.n) {
        let dv_old = dv[i];
        da = dv_old * v_raw[i];
        db = dv_old * v0[i];
        d_v0_accum[i] = d_v0_accum[i] + lambda[1] * dv_old;
        dv[i] = lambda[0] * dv_old;
    }
    sh_da[tid] = da;
    sh_db[tid] = db;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_da[tid] = sh_da[tid] + sh_da[tid + stride];
            sh_db[tid] = sh_db[tid] + sh_db[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        atomic_add_f32(0u, sh_da[0]);
        atomic_add_f32(1u, sh_db[0]);
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VMixParams {
    n: u32,
    _p0: u32,
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

pub struct VMixKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl VMixKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora vmix fwd",
            VMIX_FORWARD_WGSL,
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        );
        Self { pipeline, bgl }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        v_raw: &GpuBuffer,
        v0: &GpuBuffer,
        lambda: &GpuBuffer,
        v_out: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vmix fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v_raw.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lambda.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_out.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = VMixParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        v_raw: &GpuBuffer,
        v0: &GpuBuffer,
        lambda: &GpuBuffer,
        v_out: &GpuBuffer,
        n: u32,
    ) {
        let bind = self.build_bind(ctx, v_raw, v0, lambda, v_out);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora vmix fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        self.dispatch_with_bind(&mut pass, &bind, n);
    }
}

pub struct VMixBackwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl VMixBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(
            ctx,
            "aurora vmix bwd",
            VMIX_BACKWARD_WGSL,
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        );
        Self { pipeline, bgl }
    }

    /// Caller must pre-zero `g_lambda` (it is one of the model grad
    /// buffers, zeroed by `zero_grad()`), and zero `d_v0_accum` once
    /// per backward pass.
    #[allow(clippy::too_many_arguments)]
    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        v_raw: &GpuBuffer,
        v0: &GpuBuffer,
        lambda: &GpuBuffer,
        dv: &GpuBuffer,
        d_v0_accum: &GpuBuffer,
        g_lambda: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vmix bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v_raw.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lambda.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dv.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: d_v0_accum.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: g_lambda.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = VMixParams { n, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
