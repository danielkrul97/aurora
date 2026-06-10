//! Token + positional embedding lookup, plus scatter-add for backward.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const EMBED_FORWARD_WGSL: &str = "
struct P { t: u32, d: u32, vocab: u32, seq_max: u32, tok_offset: u32, use_pos: u32 };
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
    var v = w_tok[tok * p.d + c];
    if (p.use_pos != 0u) {
        v = v + w_pos[ti * p.d + c];
    }
    out[i] = v;
}
";

const EMBED_BACKWARD_WGSL: &str = "
struct P { t: u32, d: u32, vocab: u32, seq_max: u32, tok_offset: u32, use_pos: u32 };
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
    if (p.use_pos != 0u) {
        d_w_pos[ti * p.d + c] = d_w_pos[ti * p.d + c] + dyi;
    }
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
    /// 0 when RoPE replaces the learned positional table.
    use_pos: u32,
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
                    range: 0..24,
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
        use_pos: bool,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset: 0, use_pos: u32::from(use_pos) };
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
        use_pos: bool,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset, use_pos: u32::from(use_pos) };
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
                    range: 0..24,
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
        use_pos: bool,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset: 0, use_pos: u32::from(use_pos) };
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
        use_pos: bool,
    ) {
        let p = EmbedParams { t, d, vocab, seq_max, tok_offset, use_pos: u32::from(use_pos) };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}

// ───── Hashed bigram (n-gram) embedding ─────────────────────────────────
//
// Mirrors `ngram_hash` / `embed_forward`'s n-gram branch in
// `crate::transformer::embedding` — the u32 mixing must stay bit-identical.

const NGRAM_ADD_WGSL: &str = "
struct P { t: u32, d: u32, rows: u32, tok_offset: u32, prev0: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       w_ngram: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<push_constant> p: P;

fn ngram_hash(prev: u32, cur: u32, rows: u32) -> u32 {
    var h = prev * 0x9E3779B1u ^ cur * 0x85EBCA6Bu;
    h ^= h >> 16u;
    h = h * 0x7FEB352Du;
    h ^= h >> 15u;
    h = h * 0x846CA68Bu;
    h ^= h >> 16u;
    return h % rows;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t * p.d) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    var prev: u32 = p.prev0;
    if (ti > 0u) {
        prev = tokens[p.tok_offset + ti - 1u];
    }
    let idx = ngram_hash(prev, tokens[p.tok_offset + ti], p.rows);
    out[i] = out[i] + w_ngram[idx * p.d + c];
}
";

const NGRAM_SCATTER_WGSL: &str = "
struct P { t: u32, d: u32, rows: u32, tok_offset: u32, prev0: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_w_ngram: array<atomic<u32>>;
var<push_constant> p: P;

fn ngram_hash(prev: u32, cur: u32, rows: u32) -> u32 {
    var h = prev * 0x9E3779B1u ^ cur * 0x85EBCA6Bu;
    h ^= h >> 16u;
    h = h * 0x7FEB352Du;
    h ^= h >> 15u;
    h = h * 0x846CA68Bu;
    h ^= h >> 16u;
    return h % rows;
}

fn atomic_add_f32(idx: u32, val: f32) {
    if (val == 0.0) { return; }
    loop {
        let old_u = atomicLoad(&g_w_ngram[idx]);
        let new_u = bitcast<u32>(bitcast<f32>(old_u) + val);
        let r = atomicCompareExchangeWeak(&g_w_ngram[idx], old_u, new_u);
        if (r.exchanged) { break; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t * p.d) { return; }
    let ti = i / p.d;
    let c = i % p.d;
    var prev: u32 = p.prev0;
    if (ti > 0u) {
        prev = tokens[p.tok_offset + ti - 1u];
    }
    let idx = ngram_hash(prev, tokens[p.tok_offset + ti], p.rows);
    atomic_add_f32(idx * p.d + c, dy[ti * p.d + c]);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NgramParams {
    t: u32,
    d: u32,
    rows: u32,
    tok_offset: u32,
    prev0: u32,
    _p: u32,
}

fn ngram_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
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
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..24,
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

pub struct NgramEmbedAddKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl NgramEmbedAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = ngram_pipeline(ctx, "aurora ngram add", NGRAM_ADD_WGSL);
        Self { pipeline, bgl }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        tokens: &GpuBuffer,
        w_ngram: &GpuBuffer,
        out: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ngram add bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_ngram.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        rows: u32,
        tok_offset: u32,
        prev0: u32,
    ) {
        let p = NgramParams { t, d, rows, tok_offset, prev0, _p: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        tokens: &GpuBuffer,
        w_ngram: &GpuBuffer,
        out: &GpuBuffer,
        t: u32,
        d: u32,
        rows: u32,
        tok_offset: u32,
        prev0: u32,
    ) {
        let bind = self.build_bind(ctx, tokens, w_ngram, out);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora ngram add pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        self.dispatch_with_bind(&mut pass, &bind, t, d, rows, tok_offset, prev0);
    }
}

pub struct NgramScatterAddKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl NgramScatterAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = ngram_pipeline(ctx, "aurora ngram scatter", NGRAM_SCATTER_WGSL);
        Self { pipeline, bgl }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        tokens: &GpuBuffer,
        dy: &GpuBuffer,
        g_w_ngram: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora ngram scatter bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dy.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_w_ngram.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        d: u32,
        rows: u32,
        tok_offset: u32,
        prev0: u32,
    ) {
        let p = NgramParams { t, d, rows, tok_offset, prev0, _p: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((t * d).div_ceil(256), 1, 1);
    }
}
