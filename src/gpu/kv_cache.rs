//! GPU KV cache + single-step kernels for fast autoregressive sampling.
//!
//! Mirrors `crate::transformer::kv_cache` on the GPU side. The cache
//! pre-allocates K/V history buffers of size `[max_len × d_model]` per
//! block; `prefill_kv` populates them from the full forward of a seed
//! sequence; `generate_step_kv` appends one row + runs one
//! single-position forward.
//!
//! New kernels:
//! - `SingleTokenEmbedKernel`: `h[c] = w_tok[tok·d + c] + w_pos[pos·d + c]`
//! - `MhaKvCacheKernel`: single-query attention against full history.
//!   Workgroup per head; sequential inside (current_len ≤ 1024 typical).

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::elementwise::ZeroBufferKernel;
use super::softmax_ce::storage_entry;
use super::GpuContext;

pub struct KvBlockGpu {
    pub k_history: GpuBuffer,
    pub v_history: GpuBuffer,
}

pub struct KvCacheGpu {
    pub blocks: Vec<KvBlockGpu>,
    pub current_len: usize,
    pub max_len: usize,
    pub d_model: usize,
    /// Owned by the cache so `new`/`reset` never touch host memory for
    /// buffer initialization. Cached pipeline is cheap to keep around
    /// (Arc<ComputePipeline>).
    zero_kernel: ZeroBufferKernel,
}

impl KvCacheGpu {
    /// Allocate K/V history buffers as uninit GPU storage, then zero
    /// them via a single GPU command encoder dispatch. Avoids the
    /// host->GPU upload of the previous implementation.
    #[must_use]
    pub fn new(ctx: &GpuContext, n_layer: usize, d_model: usize, max_len: usize) -> Self {
        let cap = max_len * d_model;
        let zero_kernel = ZeroBufferKernel::new(ctx);
        let blocks: Vec<KvBlockGpu> = (0..n_layer)
            .map(|_| KvBlockGpu {
                k_history: GpuBuffer::empty(ctx, "kv_k", cap),
                v_history: GpuBuffer::empty(ctx, "kv_v", cap),
            })
            .collect();
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv_cache init zero"),
            });
        for blk in &blocks {
            zero_kernel.dispatch_into(ctx, &mut enc, &blk.k_history, cap as u32);
            zero_kernel.dispatch_into(ctx, &mut enc, &blk.v_history, cap as u32);
        }
        ctx.queue.submit(Some(enc.finish()));
        Self {
            blocks,
            current_len: 0,
            max_len,
            d_model,
            zero_kernel,
        }
    }

    /// Zero all history buffers (pure GPU dispatch — no host upload).
    /// Strictly speaking the contents past `current_len` are never read,
    /// so a zero is not required for correctness; we do it anyway to
    /// guard against future bugs reading stale data, and because the
    /// kernel dispatch cost is negligible compared to the old
    /// host-upload path.
    pub fn reset(&mut self, ctx: &GpuContext) {
        let cap = self.max_len * self.d_model;
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv_cache reset"),
            });
        for blk in &self.blocks {
            self.zero_kernel
                .dispatch_into(ctx, &mut enc, &blk.k_history, cap as u32);
            self.zero_kernel
                .dispatch_into(ctx, &mut enc, &blk.v_history, cap as u32);
        }
        ctx.queue.submit(Some(enc.finish()));
        self.current_len = 0;
    }
}

// SingleTokenEmbedKernel — embed lookup at a specific position.

const SINGLE_EMBED_WGSL: &str = "
struct P {
    d: u32,
    vocab: u32,
    seq_max: u32,
    tok: u32,
    pos: u32,
    use_pos: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       w_tok: array<f32>;
@group(0) @binding(1) var<storage, read>       w_pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.d) { return; }
    var v = w_tok[p.tok * p.d + c];
    if (p.use_pos != 0u) {
        v = v + w_pos[p.pos * p.d + c];
    }
    out[c] = v;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SingleEmbedParams {
    d: u32,
    vocab: u32,
    seq_max: u32,
    tok: u32,
    pos: u32,
    use_pos: u32,
    _p1: u32,
    _p2: u32,
}

pub struct SingleTokenEmbedKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl SingleTokenEmbedKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora single_embed"),
                source: wgpu::ShaderSource::Wgsl(SINGLE_EMBED_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora single_embed bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora single_embed pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora single_embed"),
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
        w_tok: &GpuBuffer,
        w_pos: &GpuBuffer,
        out: &GpuBuffer,
        d: u32,
        vocab: u32,
        seq_max: u32,
        tok: u32,
        pos: u32,
        use_pos: bool,
    ) {
        let p = SingleEmbedParams {
            d,
            vocab,
            seq_max,
            tok,
            pos,
            use_pos: u32::from(use_pos),
            _p1: 0,
            _p2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora single_embed bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_tok.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_pos.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora single_embed pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(d.div_ceil(256), 1, 1);
    }
}

// MhaKvCacheKernel — single-query attention against full K/V history.
//
// Inputs:
//   q_new      : [d_model]
//   k_history  : [max_len * d_model] (only first t_total rows valid)
//   v_history  : [max_len * d_model]
// Output:
//   out        : [d_model] (filled for all heads in one dispatch)
//
// Workgroup per head (wg.y = head_idx), 64 threads stride-loop over j.
// Scratch is `[n_head * max_len]` so heads don't race on the score row.
// Phases 1-3: cooperative max → exp/sum → V-output, identical structure
// to the training-path forward MHA but for a single query position.

const MHA_KV_CACHE_WGSL: &str = "
struct P {
    t_total: u32,
    d_head: u32,
    n_head: u32,
    max_len: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       q_new: array<f32>;
@group(0) @binding(1) var<storage, read>       k_history: array<f32>;
@group(0) @binding(2) var<storage, read>       v_history: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<f32>;
var<push_constant> p: P;

var<workgroup> sh_max: array<f32, 64>;
var<workgroup> sh_sum: array<f32, 64>;
var<workgroup> sh_row_max: f32;
var<workgroup> sh_sum_e: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head_idx = wgid.y;
    let tid = lid.x;
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let scratch_base = head_idx * p.max_len;

    // Phase 1: parallel score compute + local max reduction.
    var local_max: f32 = -3.4028235e38;
    var j: u32 = tid;
    loop {
        if (j >= p.t_total) { break; }
        var s: f32 = 0.0;
        for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
            s = s + q_new[h_off + c] * k_history[j * d_model + h_off + c];
        }
        s = s * p.scale;
        scratch[scratch_base + j] = s;
        if (s > local_max) { local_max = s; }
        j = j + 64u;
    }
    sh_max[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_max[tid] = max(sh_max[tid], sh_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { sh_row_max = sh_max[0]; }
    workgroupBarrier();
    let row_max = sh_row_max;

    // Phase 2: parallel exp + local sum reduction.
    var local_sum: f32 = 0.0;
    j = tid;
    loop {
        if (j >= p.t_total) { break; }
        let e = exp(scratch[scratch_base + j] - row_max);
        scratch[scratch_base + j] = e;
        local_sum = local_sum + e;
        j = j + 64u;
    }
    sh_sum[tid] = local_sum;
    workgroupBarrier();

    stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { sh_sum_e = sh_sum[0]; }
    workgroupBarrier();
    let inv = select(0.0, 1.0 / sh_sum_e, sh_sum_e > 0.0);

    // Phase 3: parallel output per feature (one thread per c).
    // Stride-loop over c to handle d_head > 64 (uncommon but supported).
    var c: u32 = tid;
    loop {
        if (c >= p.d_head) { break; }
        var acc: f32 = 0.0;
        for (var jj: u32 = 0u; jj < p.t_total; jj = jj + 1u) {
            acc = acc + scratch[scratch_base + jj] * inv * v_history[jj * d_model + h_off + c];
        }
        out[h_off + c] = acc;
        c = c + 64u;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MhaKvParams {
    t_total: u32,
    d_head: u32,
    n_head: u32,
    max_len: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

pub struct MhaKvCacheKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl MhaKvCacheKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora mha_kv"),
                source: wgpu::ShaderSource::Wgsl(MHA_KV_CACHE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora mha_kv bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    storage_entry(4, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora mha_kv pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora mha_kv"),
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

    /// One dispatch fires all `n_head` heads. `scratch` must be sized
    /// `n_head * max_len`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        q_new: &GpuBuffer,
        k_history: &GpuBuffer,
        v_history: &GpuBuffer,
        out: &GpuBuffer,
        scratch: &GpuBuffer,
        t_total: u32,
        n_head: u32,
        d_head: u32,
        max_len: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = MhaKvParams {
            t_total,
            d_head,
            n_head,
            max_len,
            scale,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha_kv bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_new.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_history.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_history.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scratch.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora mha_kv pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, n_head, 1);
    }
}
