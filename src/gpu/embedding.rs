//! Token + positional embedding lookup, plus scatter-add for backward.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const EMBED_FORWARD_WGSL: &str = "
struct P { t: u32, d: u32, vocab: u32, seq_max: u32, tok_offset: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       w_tok: array<f32>;
@group(0) @binding(2) var<storage, read>       w_pos: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.t * p.d;
    if (i >= total) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    let tok = tokens[p.tok_offset + ti];
    out[i] = w_tok[tok * p.d + c] + w_pos[ti * p.d + c];
}
";

const EMBED_BACKWARD_WGSL: &str = "
struct P { t: u32, d: u32, vocab: u32, seq_max: u32, tok_offset: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_w_tok_per_pos: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_w_pos: array<f32>;
var<push_constant> p: P;

// d_w_tok_per_pos shape: [T, D] — caller does final scatter-reduce.
// d_w_pos shape: [seq_max, D] — direct write since each (ti, c) is unique.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.t * p.d;
    if (i >= total) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    let dyi = dy[ti * p.d + c];
    d_w_tok_per_pos[ti * p.d + c] = dyi;
    d_w_pos[ti * p.d + c] = d_w_pos[ti * p.d + c] + dyi;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbedParams {
    t: u32,
    d: u32,
    vocab: u32,
    seq_max: u32,
    /// Read tokens starting at this flat index — lets one `B·T` token
    /// buffer feed per-sequence embed dispatches without re-uploading.
    tok_offset: u32,
}

pub struct TokenPosEmbedKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl TokenPosEmbedKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora embed fwd"),
                source: wgpu::ShaderSource::Wgsl(EMBED_FORWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora embed fwd bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora embed fwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..20,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora embed fwd"),
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
        w_tok: &GpuBuffer,
        w_pos: &GpuBuffer,
        out: &GpuBuffer,
        t: u32,
        d: u32,
        vocab: u32,
        seq_max: u32,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_tok.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_pos.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora embed fwd pass"),
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
        w_tok: &GpuBuffer,
        w_pos: &GpuBuffer,
        out: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_tok.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_pos.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        vocab: u32,
        seq_max: u32,
        tok_offset: u32,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}

/// Scatter-add gradient back into per-position buffer + positional table.
/// The token-side scatter-reduce (collapse `[T, D]` → `[vocab, D]`) is
/// done on the host because GPU atomics are inconvenient here and T is
/// tiny.
pub struct EmbedScatterAddKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl EmbedScatterAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora embed bwd"),
                source: wgpu::ShaderSource::Wgsl(EMBED_BACKWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora embed bwd bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora embed bwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..20,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora embed bwd"),
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
        dy: &GpuBuffer,
        d_w_tok_per_pos: &GpuBuffer,
        d_w_pos: &GpuBuffer,
        t: u32,
        d: u32,
        vocab: u32,
        seq_max: u32,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: d_w_tok_per_pos.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 3, resource: d_w_pos.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora embed bwd pass"),
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
        dy: &GpuBuffer,
        d_w_tok_per_pos: &GpuBuffer,
        d_w_pos: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora embed bwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: d_w_tok_per_pos.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 3, resource: d_w_pos.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        vocab: u32,
        seq_max: u32,
        tok_offset: u32,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}
