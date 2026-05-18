//! Vocab-agnostic softmax + cross-entropy kernel.
//!
//! Per-thread stride loop over the output dim removes the `O ≤ 256` cap;
//! works for any `n_output ≤ 65535` in practice. One dispatch handles
//! `B * T` softmaxes via a 2D workgroup grid.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::GpuContext;

const SOFTMAX_CE_WGSL: &str = "
struct SCE {
    n_output: u32,
    n_timesteps: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_seq: array<f32>;
@group(0) @binding(3) var<storage, read>       targets: array<u32>;
var<push_constant>             p: SCE;

var<workgroup> sh_buf: array<f32, 256>;
var<workgroup> sh_max: f32;
var<workgroup> sh_sum: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tid = lid.x;
    let b = wgid.x;
    let t = wgid.y;
    let bt_idx = b * p.n_timesteps + t;
    let base = bt_idx * p.n_output;
    let target_idx = targets[bt_idx];

    var local_max: f32 = -3.4028235e38;
    var idx: u32 = tid;
    loop {
        if (idx >= p.n_output) { break; }
        local_max = max(local_max, logits[base + idx]);
        idx = idx + 256u;
    }
    sh_buf[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_buf[tid] = max(sh_buf[tid], sh_buf[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_max = sh_buf[0];
    }
    workgroupBarrier();

    var local_sum: f32 = 0.0;
    idx = tid;
    loop {
        if (idx >= p.n_output) { break; }
        local_sum = local_sum + exp(logits[base + idx] - sh_max);
        idx = idx + 256u;
    }
    sh_buf[tid] = local_sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_buf[tid] = sh_buf[tid] + sh_buf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_sum = sh_buf[0];
    }
    workgroupBarrier();

    idx = tid;
    loop {
        if (idx >= p.n_output) { break; }
        let prob = exp(logits[base + idx] - sh_max) / sh_sum;
        var d = prob;
        if (idx == target_idx) {
            d = prob - 1.0;
        }
        d_logits[base + idx] = d;
        idx = idx + 256u;
    }
    if (tid == 0u) {
        let p_target = max(exp(logits[base + target_idx] - sh_max) / sh_sum, 1e-12);
        loss_seq[bt_idx] = -log(p_target);
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxCeParams {
    n_output: u32,
    n_timesteps: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct SoftmaxCrossEntropyKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl SoftmaxCrossEntropyKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora softmax_ce wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOFTMAX_CE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora softmax_ce bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, true),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora softmax_ce pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora softmax_ce pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bgl),
        }
    }

    /// One dispatch handles `B * T` softmaxes via a 2D workgroup grid.
    /// Per-thread stride loop covers any `n_output ≤ 65535` in practice.
    ///
    /// # Panics
    /// Panics if `n_output == 0`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        logits: &GpuBuffer,
        d_logits: &GpuBuffer,
        loss_seq: &GpuBuffer,
        targets: &GpuBuffer,
        n_output: u32,
        n_timesteps: u32,
        batch_size: u32,
    ) {
        assert!(n_output > 0, "n_output must be > 0");
        debug_assert!(logits.len >= (batch_size * n_timesteps * n_output) as usize);
        debug_assert!(d_logits.len >= (batch_size * n_timesteps * n_output) as usize);
        debug_assert!(loss_seq.len >= (batch_size * n_timesteps) as usize);
        debug_assert!(targets.len >= (batch_size * n_timesteps) as usize);

        let p = SoftmaxCeParams {
            n_output,
            n_timesteps,
            _pad0: 0,
            _pad1: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora softmax_ce bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_logits.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: loss_seq.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: targets.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora softmax_ce pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            pass.dispatch_workgroups(batch_size, n_timesteps, 1);
        }
    }
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
