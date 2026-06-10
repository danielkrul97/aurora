//! Rotary position embedding applied in place to Q/K activations.
//!
//! Thread per feature pair, RoFormer-style interleaved layout — must
//! match `crate::attention::rope_rotate` exactly (gpu_parity asserts).
//! `inverse` applies the transposed rotation: the backward map for dQ/dK.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const ROPE_WGSL: &str = "
struct P { t: u32, n_head: u32, d_head: u32, pos_offset: u32, inverse: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.d_head / 2u;
    let pairs_per_row = p.n_head * half;
    let i = gid.x;
    if (i >= p.t * pairs_per_row) { return; }
    let ti = i / pairs_per_row;
    let rem = i % pairs_per_row;
    let h = rem / half;
    let pi = rem % half;
    let off = ti * p.n_head * p.d_head + h * p.d_head + 2u * pi;
    let freq = pow(10000.0, -2.0 * f32(pi) / f32(p.d_head));
    var theta = f32(p.pos_offset + ti) * freq;
    if (p.inverse != 0u) { theta = -theta; }
    let c = cos(theta);
    let s = sin(theta);
    let x0 = x[off];
    let x1 = x[off + 1u];
    x[off] = x0 * c - x1 * s;
    x[off + 1u] = x0 * s + x1 * c;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    t: u32,
    n_head: u32,
    d_head: u32,
    pos_offset: u32,
    inverse: u32,
}

pub struct RopeKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl RopeKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora rope"),
            source: wgpu::ShaderSource::Wgsl(ROPE_WGSL.into()),
        });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora rope bgl"),
                entries: &[storage_entry(0, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora rope pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..20,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora rope"),
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

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora rope bind"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: x.inner.as_entire_binding(),
            }],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
        pos_offset: u32,
        inverse: bool,
    ) {
        let p = RopeParams {
            t,
            n_head,
            d_head,
            pos_offset,
            inverse: u32::from(inverse),
        };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let pairs = t * n_head * (d_head / 2);
        pass.dispatch_workgroups(pairs.div_ceil(256), 1, 1);
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        t: u32,
        n_head: u32,
        d_head: u32,
        pos_offset: u32,
        inverse: bool,
    ) {
        let bind = self.build_bind(ctx, x);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora rope pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        self.dispatch_with_bind(&mut pass, &bind, t, n_head, d_head, pos_offset, inverse);
    }
}
