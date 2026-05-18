//! GPU causal MHA. Forward + backward. All `n_head` heads run in a
//! single dispatch — `head_idx` is encoded as `workgroup_id.y` (fwd, dq,
//! dkv, row_dot) or `.z` (d_scores tile), so the host issues one
//! dispatch per phase regardless of head count.
//!
//! Layout: Q/K/V/output are all `[T × D]` packed (per-sequence). Heads
//! are split implicitly via head_idx + d_head stride. Probs and bwd
//! scratch are `[n_head, T, T]` shared across all heads in one buffer.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

// One workgroup (64 threads) per (query position i, head). Threads
// stride-loop over the j axis cooperatively for the 4 phases. The
// per-query probs row is staged in workgroup-shared memory to avoid
// 3 round-trips through global memory — the final softmax output is
// still mirrored to global for backward.
const ATTN_FORWARD_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    _pad0: u32,
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

var<workgroup> sh_max: array<f32, 128>;
var<workgroup> sh_sum: array<f32, 128>;
var<workgroup> sh_row_max: f32;
var<workgroup> sh_inv: f32;
// Per-query scores live in shared so phase 2 + phase 4 read locally.
// Bound 1024 — supports seq_len up to 1024 (project caps at 512).
var<workgroup> sh_probs: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wgid.x;
    let head_idx = wgid.y;
    let tid = lid.x;
    if (i >= p.t) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let probs_head_base = head_idx * p.t * p.t;
    let row_base = probs_head_base + i * p.t;
    let q_base = i * d_model + h_off;

    // Phase 1: scores live in shared. Causal: j > i → 0 in shared.
    var local_max: f32 = -3.4028235e38;
    var j: u32 = tid;
    loop {
        if (j >= p.t) { break; }
        if (j > i) {
            sh_probs[j] = 0.0;
        } else {
            var s: f32 = 0.0;
            for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
                s = s + q[q_base + c] * k[j * d_model + h_off + c];
            }
            s = s * p.scale;
            sh_probs[j] = s;
            if (s > local_max) { local_max = s; }
        }
        j = j + 128u;
    }
    sh_max[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = 64u;
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

    // Phase 2: exp + per-thread sum, write back into shared (only j ≤ i).
    var local_sum: f32 = 0.0;
    j = tid;
    loop {
        if (j >= p.t) { break; }
        if (j <= i) {
            let e = exp(sh_probs[j] - row_max);
            sh_probs[j] = e;
            local_sum = local_sum + e;
        }
        j = j + 128u;
    }
    sh_sum[tid] = local_sum;
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_inv = select(0.0, 1.0 / sh_sum[0], sh_sum[0] > 0.0);
    }
    workgroupBarrier();
    let inv = sh_inv;

    // Phase 3: normalise + mirror to global for backward (single write).
    j = tid;
    loop {
        if (j >= p.t) { break; }
        if (j <= i) {
            let v_norm = sh_probs[j] * inv;
            sh_probs[j] = v_norm;
            probs[row_base + j] = v_norm;
        } else {
            probs[row_base + j] = 0.0;
        }
        j = j + 128u;
    }
    workgroupBarrier();

    // Phase 4: out[i, c] = Σ_{j ≤ i} probs[i, j] · v[j, c]. Read probs
    // from shared — no global re-read.
    var c: u32 = tid;
    loop {
        if (c >= p.d_head) { break; }
        var acc: f32 = 0.0;
        for (var jj: u32 = 0u; jj <= i; jj = jj + 1u) {
            acc = acc + sh_probs[jj] * v[jj * d_model + h_off + c];
        }
        out[i * d_model + h_off + c] = acc;
        c = c + 128u;
    }
}
";

// Fused backward pass A: row_dot + d_scores in one kernel. One
// workgroup per (i, head); threads stride over j. Phase A computes
// dp[j] = Σ_c d_out[i,c]·v[j,c] for j ≤ i, stages it in shared, and
// reduces row_dot[i] = Σ probs · dp. Phase B re-reads sh_dp + sh_probs
// to write d_scores[i, j] = probs · (dp − row_dot) · scale.
//
// Replaces the previous 2-pass (row_dot_wgsl + d_scores_wgsl)
// pipeline — saves one dispatch + one global trip through dp / row_dot.
const ATTN_BWD_DSCORES_FUSED_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    _pad0: u32,
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

var<workgroup> sh_red: array<f32, 256>;
var<workgroup> sh_row_dot: f32;
// dp staging — same bound as sh_probs in fwd.
var<workgroup> sh_dp: array<f32, 2048>;
// probs row staging — re-used by both phases.
var<workgroup> sh_probs: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wgid.x;
    let head_idx = wgid.y;
    let tid = lid.x;
    if (i >= p.t) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let probs_head_base = head_idx * p.t * p.t;
    let row_base = probs_head_base + i * p.t;
    let dout_base = i * d_model + h_off;

    // Phase A: stage probs + dp into shared, accumulate row_dot partial.
    var local: f32 = 0.0;
    var j: u32 = tid;
    loop {
        if (j >= p.t) { break; }
        let pij = probs[row_base + j];
        sh_probs[j] = pij;
        if (j <= i) {
            var dp: f32 = 0.0;
            for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
                dp = dp + d_out[dout_base + c] * v[j * d_model + h_off + c];
            }
            sh_dp[j] = dp;
            local = local + pij * dp;
        } else {
            sh_dp[j] = 0.0;
        }
        j = j + 256u;
    }
    sh_red[tid] = local;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_red[tid] = sh_red[tid] + sh_red[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { sh_row_dot = sh_red[0]; }
    workgroupBarrier();
    let rd = sh_row_dot;

    // Phase B: d_scores[i, j] = probs[i,j] · (dp − row_dot) · scale.
    j = tid;
    loop {
        if (j >= p.t) { break; }
        if (j > i) {
            d_scores[row_base + j] = 0.0;
        } else {
            d_scores[row_base + j] = sh_probs[j] * (sh_dp[j] - rd) * p.scale;
        }
        j = j + 256u;
    }
}
";

const ATTN_BWD_DQ_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    _pad0: u32,
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
    let head_idx = wgid.y;
    if (i >= p.t) { return; }
    let c = lid.x;
    if (c >= p.d_head) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let row_base = head_idx * p.t * p.t + i * p.t;

    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        acc = acc + d_scores[row_base + j] * k[j * d_model + h_off + c];
    }
    dq[i * d_model + h_off + c] = dq[i * d_model + h_off + c] + acc;
}
";

const ATTN_BWD_DK_DV_WGSL: &str = "
struct P {
    t: u32,
    d_head: u32,
    n_head: u32,
    _pad0: u32,
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
    let head_idx = wgid.y;
    if (j >= p.t) { return; }
    let c = lid.x;
    if (c >= p.d_head) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let probs_head_base = head_idx * p.t * p.t;

    var dk_acc: f32 = 0.0;
    var dv_acc: f32 = 0.0;
    for (var i: u32 = j; i < p.t; i = i + 1u) {
        dk_acc = dk_acc + d_scores[probs_head_base + i * p.t + j] * q[i * d_model + h_off + c];
        dv_acc = dv_acc + probs[probs_head_base + i * p.t + j] * d_out[i * d_model + h_off + c];
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
    _pad0: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

pub struct CausalMhaForwardKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
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

    /// One dispatch fires all `n_head` heads in parallel.
    /// `probs` layout: `[n_head, t, t]` row-major.
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
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t,
            d_head,
            n_head,
            _pad0: 0,
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
        pass.dispatch_workgroups(t, n_head, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        out: &GpuBuffer,
        probs: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha fwd bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: probs.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t, d_head, n_head, _pad0: 0, scale, _p0: 0, _p1: 0, _p2: 0,
        };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, n_head, 1);
    }
}

pub struct CausalMhaBackwardKernel {
    pub(crate) pipeline_dscores: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dscores: Arc<wgpu::BindGroupLayout>,
    pub(crate) pipeline_dq: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dq: Arc<wgpu::BindGroupLayout>,
    pub(crate) pipeline_dkv: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dkv: Arc<wgpu::BindGroupLayout>,
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
        let (pipeline_dscores, bgl_dscores) = build_pipeline(
            ctx,
            ATTN_BWD_DSCORES_FUSED_WGSL,
            "aurora mha bwd dscores fused",
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
            pipeline_dscores,
            bgl_dscores,
            pipeline_dq,
            bgl_dq,
            pipeline_dkv,
            bgl_dkv,
        }
    }

    /// Backward MHA for all `n_head` heads in 3 dispatches total.
    /// `probs` and `d_scores_scratch` must be `[n_head, t, t]`.
    /// `dq`, `dk`, `dv` are accumulated; caller must zero them before
    /// this is invoked.
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
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t,
            d_head,
            n_head,
            _pad0: 0,
            scale,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        let pc = bytemuck::bytes_of(&p);

        // Fused pass: row_dot + d_scores in one kernel, one wg per (i, head).
        let bind0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dscores bind"),
            layout: &self.bgl_dscores,
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
                label: Some("aurora mha bwd dscores pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dscores);
            pass.set_bind_group(0, &bind0, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }

        // dq accumulation per (i, head)
        let bind1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            pass.set_bind_group(0, &bind1, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }

        // dk + dv accumulation per (j, head)
        let bind2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            pass.set_bind_group(0, &bind2, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }
    }

    pub fn build_bind_dscores(
        &self,
        ctx: &GpuContext,
        v: &GpuBuffer,
        probs: &GpuBuffer,
        d_out: &GpuBuffer,
        d_scores: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dscores bind"),
            layout: &self.bgl_dscores,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: v.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probs.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: d_scores.inner.as_entire_binding() },
            ],
        })
    }

    pub fn build_bind_dq(
        &self,
        ctx: &GpuContext,
        k: &GpuBuffer,
        d_scores: &GpuBuffer,
        dq: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dq bind"),
            layout: &self.bgl_dq,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: k.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: d_scores.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dq.inner.as_entire_binding() },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_bind_dkv(
        &self,
        ctx: &GpuContext,
        q: &GpuBuffer,
        probs: &GpuBuffer,
        d_out: &GpuBuffer,
        d_scores: &GpuBuffer,
        dk: &GpuBuffer,
        dv: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora mha bwd dkv bind"),
            layout: &self.bgl_dkv,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probs.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: d_scores.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dk.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dv.inner.as_entire_binding() },
            ],
        })
    }

    /// All three backward passes (dscores, dq, dkv) dispatched into the
    /// given encoder using pre-built bind groups. Each is in its own
    /// compute pass because they have data dependencies through
    /// `d_scores`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_binds(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_dscores: &wgpu::BindGroup,
        bind_dq: &wgpu::BindGroup,
        bind_dkv: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = AttnParams {
            t, d_head, n_head, _pad0: 0, scale, _p0: 0, _p1: 0, _p2: 0,
        };
        let pc = bytemuck::bytes_of(&p);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd dscores pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dscores);
            pass.set_bind_group(0, bind_dscores, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd dq pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dq);
            pass.set_bind_group(0, bind_dq, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora mha bwd dkv pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dkv);
            pass.set_bind_group(0, bind_dkv, &[]);
            pass.set_push_constants(0, pc);
            pass.dispatch_workgroups(t, n_head, 1);
        }
    }
}
