//! AdamW kernel + global gradient norm reduction for clipping.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const ADAMW_WGSL: &str = "
struct P {
    n: u32,
    apply_wd: u32,
    _p0: u32,
    _p1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    bc1: f32,
    bc2: f32,
    weight_decay: f32,
    _p2: f32,
    _p3: f32,
    _p4: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read>       grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m1: array<f32>;
@group(0) @binding(3) var<storage, read_write> m2: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = grads[i];
    m1[i] = p.beta1 * m1[i] + p.one_minus_b1 * g;
    m2[i] = p.beta2 * m2[i] + p.one_minus_b2 * g * g;
    let m1_hat = m1[i] / p.bc1;
    let m2_hat = m2[i] / p.bc2;
    var step = p.lr * m1_hat / (sqrt(m2_hat) + p.eps);
    if (p.apply_wd != 0u) {
        step = step + p.lr * p.weight_decay * params[i];
    }
    params[i] = params[i] - step;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWParams {
    n: u32,
    apply_wd: u32,
    _p0: u32,
    _p1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    bc1: f32,
    bc2: f32,
    weight_decay: f32,
    _p2: f32,
    _p3: f32,
    _p4: f32,
}

pub struct AdamWKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl AdamWKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora adamw"),
                source: wgpu::ShaderSource::Wgsl(ADAMW_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora adamw bgl"),
                entries: &[
                    storage_entry(0, false),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora adamw pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..64,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora adamw"),
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
        params: &GpuBuffer,
        grads: &GpuBuffer,
        m1: &GpuBuffer,
        m2: &GpuBuffer,
        n: u32,
        apply_wd: bool,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        weight_decay: f32,
    ) {
        let p = AdamWParams {
            n,
            apply_wd: u32::from(apply_wd),
            _p0: 0,
            _p1: 0,
            lr,
            beta1,
            beta2,
            eps,
            one_minus_b1: 1.0 - beta1,
            one_minus_b2: 1.0 - beta2,
            bc1,
            bc2,
            weight_decay,
            _p2: 0.0,
            _p3: 0.0,
            _p4: 0.0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora adamw bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grads.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: m2.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora adamw pass"),
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
        params: &GpuBuffer,
        grads: &GpuBuffer,
        m1: &GpuBuffer,
        m2: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora adamw bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grads.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: m2.inner.as_entire_binding() },
            ],
        })
    }

    /// Issue an AdamW step on an existing pass with a pre-built bind group.
    /// Caller must `set_pipeline(&kernel.pipeline)` once before the first call.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        apply_wd: bool,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        weight_decay: f32,
    ) {
        let p = AdamWParams {
            n,
            apply_wd: u32::from(apply_wd),
            _p0: 0,
            _p1: 0,
            lr,
            beta1,
            beta2,
            eps,
            one_minus_b1: 1.0 - beta1,
            one_minus_b2: 1.0 - beta2,
            bc1,
            bc2,
            weight_decay,
            _p2: 0.0,
            _p3: 0.0,
            _p4: 0.0,
        };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
