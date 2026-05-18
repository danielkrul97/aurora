//! GPU causal MHA. Forward + backward. Single dispatch per head per
//! direction; the `head` dim is the workgroup `y`, `batch_pos` is `x`.
//!
//! Layout: Q/K/V/output are all `[T × D]` packed (per-sequence). Heads
//! are split implicitly via head_idx + d_head stride; multi-head is
//! n_head sequential dispatches (each writes its slice of the output).

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const ATTN_FORWARD_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    head_idx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> probs: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.t) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = p.head_idx * p.d_head;

    var row_max: f32 = -3.4028235e38;
    // first pass: compute s[j] and row max
    for (var j: u32 = 0u; j < p.t; j = j + 1u) {
        if (j > i) {
            // causal: store -inf so the second pass sees it as 0
            probs[i * p.t + j] = -3.4028235e38;
            continue;
        }
        var s: f32 = 0.0;
        for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
            s = s + q[i * d_model + h_off + c] * k[j * d_model + h_off + c];
        }
        s = s * p.scale;
        probs[i * p.t + j] = s;
        if (s > row_max) { row_max = s; }
    }
    // second pass: exp + sum
    var sum_e: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let e = exp(probs[i * p.t + j] - row_max);
        probs[i * p.t + j] = e;
        sum_e = sum_e + e;
    }
    let inv = select(0.0, 1.0 / sum_e, sum_e > 0.0);
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        probs[i * p.t + j] = probs[i * p.t + j] * inv;
    }
    // third pass: probs @ V
    for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j <= i; j = j + 1u) {
            acc = acc + probs[i * p.t + j] * v[j * d_model + h_off + c];
        }
        out[i * d_model + h_off + c] = acc;
    }
}
";

// Tiled 3-pass MHA backward — the per-pass split is what keeps everything
// race-free without f32 atomics:
// Pass 1 (per (i, j) tile): `d_scores[i, j]` into scratch — each (i, j) is unique.
// Pass 2 (per i): `dq[i] += sum_j d_scores[i, j] * k[j]` — each workgroup owns one i.
// Pass 3 (per j): `dk[j] += sum_i d_scores[i, j] * q[i]` and
//                 `dv[j] += sum_i probs[i, j] * d_out[i]` — each workgroup owns one j.
// Host allocates the [t × t] `d_scores` scratch.

const ATTN_BWD_D_SCORES_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    head_idx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       v: array<f32>;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       d_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_scores: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let i = wgid.y * 16u + lid.y;
    let j = wgid.x * 16u + lid.x;
    if (i >= p.t || j >= p.t) { return; }
    if (j > i) {
        d_scores[i * p.t + j] = 0.0;
        return;
    }
    let d_model = p.d_head * p.n_head;
    let h_off = p.head_idx * p.d_head;

    // d_probs[i, j] = sum_c d_out[i, c] * v[j, c]
    var dp: f32 = 0.0;
    for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
        dp = dp + d_out[i * d_model + h_off + c] * v[j * d_model + h_off + c];
    }
    // row_dot[i] = sum_k probs[i, k] * d_probs[i, k]
    // We recompute on the fly — small cost vs scratch row.
    var row_dot: f32 = 0.0;
    for (var k: u32 = 0u; k <= i; k = k + 1u) {
        var dp_k: f32 = 0.0;
        for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
            dp_k = dp_k + d_out[i * d_model + h_off + c] * v[k * d_model + h_off + c];
        }
        row_dot = row_dot + probs[i * p.t + k] * dp_k;
    }
    d_scores[i * p.t + j] = probs[i * p.t + j] * (dp - row_dot) * p.scale;
}
";

const ATTN_BWD_DQ_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    head_idx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       k: array<f32>;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> dq: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let i = wgid.x;
    if (i >= p.t) { return; }
    let c = lid.x;
    if (c >= p.d_head) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = p.head_idx * p.d_head;

    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        acc = acc + d_scores[i * p.t + j] * k[j * d_model + h_off + c];
    }
    dq[i * d_model + h_off + c] = dq[i * d_model + h_off + c] + acc;
}
";

const ATTN_BWD_DK_DV_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    head_idx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       d_out: array<f32>;
@group(0) @binding(3) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> dk: array<f32>;
@group(0) @binding(5) var<storage, read_write> dv: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let j = wgid.x;
    if (j >= p.t) { return; }
    let c = lid.x;
    if (c >= p.d_head) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = p.head_idx * p.d_head;

    var dk_acc: f32 = 0.0;
    var dv_acc: f32 = 0.0;
    for (var i: u32 = j; i < p.t; i = i + 1u) {
        dk_acc = dk_acc + d_scores[i * p.t + j] * q[i * d_model + h_off + c];
        dv_acc = dv_acc + probs[i * p.t + j] * d_out[i * d_model + h_off + c];
    }
    dk[j * d_model + h_off + c] = dk[j * d_model + h_off + c] + dk_acc;
    dv[j * d_model + h_off + c] = dv[j * d_model + h_off + c] + dv_acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnParams {
    t: u32,
    d_head: u32,
    n_head: u32,
    head_idx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

pub struct CausalMhaForwardKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl CausalMhaForwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora mha fwd"),
                source: wgpu::ShaderSource::Wgsl(ATTN_FORWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora mha fwd bgl"),
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
                label: Some("aurora mha fwd pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora mha fwd"),
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
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        out: &GpuBuffer,
        probs: &GpuBuffer,
        t: u32,
        n_head: u32,
        d_head: u32,
        head_idx: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t,
            d_head,
            n_head,
            head_idx,
            scale,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: probs.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora mha fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, 1, 1);
    }
}

pub struct CausalMhaBackwardKernel {
    pipeline_scores: Arc<wgpu::ComputePipeline>,
    bgl_scores: Arc<wgpu::BindGroupLayout>,
    pipeline_dq: Arc<wgpu::ComputePipeline>,
    bgl_dq: Arc<wgpu::BindGroupLayout>,
    pipeline_dkv: Arc<wgpu::ComputePipeline>,
    bgl_dkv: Arc<wgpu::BindGroupLayout>,
}

fn build_pipeline(
    ctx: &GpuContext,
    wgsl: &str,
    label: &str,
    bindings: &[(u32, bool)],
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let entries: Vec<wgpu::BindGroupLayoutEntry> =
        bindings.iter().map(|(b, ro)| storage_entry(*b, *ro)).collect();
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..32,
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

impl CausalMhaBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline_scores, bgl_scores) = build_pipeline(
            ctx,
            ATTN_BWD_D_SCORES_WGSL,
            "aurora mha bwd scores",
            &[(0, true), (1, true), (2, true), (3, false)],
        );
        let (pipeline_dq, bgl_dq) = build_pipeline(
            ctx,
            ATTN_BWD_DQ_WGSL,
            "aurora mha bwd dq",
            &[(0, true), (1, true), (2, false)],
        );
        let (pipeline_dkv, bgl_dkv) = build_pipeline(
            ctx,
            ATTN_BWD_DK_DV_WGSL,
            "aurora mha bwd dkv",
            &[
                (0, true),
                (1, true),
                (2, true),
                (3, true),
                (4, false),
                (5, false),
            ],
        );
        Self {
            pipeline_scores,
            bgl_scores,
            pipeline_dq,
            bgl_dq,
            pipeline_dkv,
            bgl_dkv,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        probs: &GpuBuffer,
        d_out: &GpuBuffer,
        dq: &GpuBuffer,
        dk: &GpuBuffer,
        dv: &GpuBuffer,
        d_scores_scratch: &GpuBuffer,
        t: u32,
        n_head: u32,
        d_head: u32,
        head_idx: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t,
            d_head,
            n_head,
            head_idx,
            scale,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let pc = bytemuck::bytes_of(&p);

        // Pass 1: compute d_scores[i, j] → scratch
        let bind1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd scores bind"),
            layout: &self.bgl_scores,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probs.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_out.inner.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: d_scores_scratch.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd scores pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scores);
            pass.set_bind_group(0, &bind1, &[]);
            pass.set_push_constants(0, pc);
            let groups = t.div_ceil(16);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        // Pass 2: dq accumulation per i
        let bind2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dq bind"),
            layout: &self.bgl_dq,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: k.inner.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_scores_scratch.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 2, resource: dq.inner.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd dq pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dq);
            pass.set_bind_group(0, &bind2, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, 1, 1);
        }

        // Pass 3: dk + dv accumulation per j
        let bind3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dkv bind"),
            layout: &self.bgl_dkv,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probs.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_out.inner.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: d_scores_scratch.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 4, resource: dk.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dv.inner.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd dkv pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dkv);
            pass.set_bind_group(0, &bind3, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, 1, 1);
        }
    }
}
