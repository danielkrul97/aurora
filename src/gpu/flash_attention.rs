//! Flash-style causal attention: O(T·d) memory instead of O(T²).
//!
//! Forward runs an online softmax over key tiles and stores only the
//! per-row stats `(m_i, l_i)`; the `[n_head, T, T]` probs buffer is
//! never materialized. Backward recomputes probabilities blockwise from
//! the stats (FlashAttention-style), at the cost of re-doing the QKᵀ
//! scores — ~2× attention FLOPs for ~T²-free memory, which is what
//! unlocks seq_len ≥ 1024 on a 4 GB card.
//!
//! Math mirrors the cached-probs kernels exactly:
//!   P_ij = exp(scale·q_i·k_j − m_i) / l_i        (0 outside the mask)
//!   D_i  = Σ_c dO_ic · O_ic  (= Σ_j P_ij·dP_ij, the softmax row-dot)
//!   dS_ij = P_ij · (dP_ij − D_i),  dP_ij = dO_i · v_j
//!   dq_i = scale·Σ_j dS_ij·k_j,  dk_j = scale·Σ_i dS_ij·q_i,
//!   dv_j = Σ_i P_ij·dO_i
//!
//! Sliding-window masking composes: visible(j, i) ⇔ j ≤ i ∧ (wnd = 0 ∨
//! j + wnd > i). All kernels take the window via push constants.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const TILE: u32 = 128;

const FLASH_FWD_WGSL: &str = "
struct P { t: u32, d_head: u32, n_head: u32, wnd: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
// Per-row softmax stats: ml[(head*t + i)*2] = m_i, +1 = l_i.
@group(0) @binding(4) var<storage, read_write> ml: array<f32>;
var<push_constant> p: P;

var<workgroup> sh_s: array<f32, 128>;
var<workgroup> sh_red: array<f32, 128>;
var<workgroup> sh_m: f32;
var<workgroup> sh_rescale: f32;

fn visible(j: u32, i: u32, wnd: u32) -> bool {
    return j <= i && (wnd == 0u || j + wnd > i);
}

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
    let q_base = i * d_model + h_off;

    var j_start: u32 = 0u;
    if (p.wnd > 0u && i + 1u > p.wnd) { j_start = i + 1u - p.wnd; }

    var m: f32 = -3.4028235e38;
    var l: f32 = 0.0;
    var acc: f32 = 0.0; // thread tid owns output channel c = tid (< d_head)

    var jt: u32 = j_start;
    loop {
        if (jt > i) { break; }

        // Phase A: one score per thread for this tile.
        let j = jt + tid;
        var s: f32 = -3.4028235e38;
        if (j <= i && visible(j, i, p.wnd)) {
            var dot_qk: f32 = 0.0;
            for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
                dot_qk = dot_qk + q[q_base + c] * k[j * d_model + h_off + c];
            }
            s = dot_qk * p.scale;
        }
        sh_s[tid] = s;
        sh_red[tid] = s;
        workgroupBarrier();

        // Tile max.
        var stride: u32 = 64u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { sh_red[tid] = max(sh_red[tid], sh_red[tid + stride]); }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (tid == 0u) {
            let new_m = max(m, sh_red[0]);
            sh_m = new_m;
            sh_rescale = exp(m - new_m);
        }
        workgroupBarrier();
        let new_m = sh_m;
        let rescale = sh_rescale;

        // Phase B: exponentiate into shared, tile sum.
        var pj: f32 = 0.0;
        if (sh_s[tid] > -3.0e38) { pj = exp(sh_s[tid] - new_m); }
        sh_s[tid] = pj;
        sh_red[tid] = pj;
        workgroupBarrier();
        stride = 64u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { sh_red[tid] = sh_red[tid] + sh_red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        l = l * rescale + sh_red[0];
        m = new_m;

        // Phase C: accumulate output channels (thread = channel).
        if (tid < p.d_head) {
            acc = acc * rescale;
            let tile_n = min(TILE_U, i - jt + 1u);
            for (var jj: u32 = 0u; jj < tile_n; jj = jj + 1u) {
                let pv = sh_s[jj];
                if (pv != 0.0) {
                    acc = acc + pv * v[(jt + jj) * d_model + h_off + tid];
                }
            }
        }
        workgroupBarrier();

        jt = jt + 128u;
    }

    if (tid < p.d_head) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        out[i * d_model + h_off + tid] = acc * inv;
    }
    if (tid == 0u) {
        ml[(head_idx * p.t + i) * 2u] = m;
        ml[(head_idx * p.t + i) * 2u + 1u] = l;
    }
}
";

// D_i = Σ_c dO_ic·O_ic — one workgroup per (i, head), reduce over d_head.
const FLASH_DPRE_WGSL: &str = "
struct P { t: u32, d_head: u32, n_head: u32, wnd: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       d_out: array<f32>;
@group(0) @binding(1) var<storage, read>       out: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_row: array<f32>;
var<push_constant> p: P;

var<workgroup> sh: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wgid.x;
    let head_idx = wgid.y;
    let tid = lid.x;
    if (i >= p.t) { return; }
    let d_model = p.d_head * p.n_head;
    let base = i * d_model + head_idx * p.d_head;

    var local_d: f32 = 0.0;
    var c: u32 = tid;
    loop {
        if (c >= p.d_head) { break; }
        local_d = local_d + d_out[base + c] * out[base + c];
        c = c + 64u;
    }
    sh[tid] = local_d;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh[tid] = sh[tid] + sh[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { d_row[head_idx * p.t + i] = sh[0]; }
}
";

const FLASH_DQ_WGSL: &str = "
struct P { t: u32, d_head: u32, n_head: u32, wnd: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       d_out: array<f32>;
@group(0) @binding(4) var<storage, read>       ml: array<f32>;
@group(0) @binding(5) var<storage, read>       d_row: array<f32>;
@group(0) @binding(6) var<storage, read_write> dq: array<f32>;
var<push_constant> p: P;

var<workgroup> sh_ds: array<f32, 128>;

fn visible(j: u32, i: u32, wnd: u32) -> bool {
    return j <= i && (wnd == 0u || j + wnd > i);
}

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
    let q_base = i * d_model + h_off;
    let m_i = ml[(head_idx * p.t + i) * 2u];
    let l_i = ml[(head_idx * p.t + i) * 2u + 1u];
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    let d_i = d_row[head_idx * p.t + i];

    var j_start: u32 = 0u;
    if (p.wnd > 0u && i + 1u > p.wnd) { j_start = i + 1u - p.wnd; }

    var acc: f32 = 0.0; // thread tid owns dq channel tid

    var jt: u32 = j_start;
    loop {
        if (jt > i) { break; }

        let j = jt + tid;
        var ds: f32 = 0.0;
        if (j <= i && visible(j, i, p.wnd)) {
            var dot_qk: f32 = 0.0;
            var dp: f32 = 0.0;
            for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
                dot_qk = dot_qk + q[q_base + c] * k[j * d_model + h_off + c];
                dp = dp + d_out[q_base + c] * v[j * d_model + h_off + c];
            }
            let prob = exp(dot_qk * p.scale - m_i) * inv_l;
            ds = prob * (dp - d_i) * p.scale;
        }
        sh_ds[tid] = ds;
        workgroupBarrier();

        if (tid < p.d_head) {
            let tile_n = min(TILE_U, i - jt + 1u);
            for (var jj: u32 = 0u; jj < tile_n; jj = jj + 1u) {
                let dsv = sh_ds[jj];
                if (dsv != 0.0) {
                    acc = acc + dsv * k[(jt + jj) * d_model + h_off + tid];
                }
            }
        }
        workgroupBarrier();

        jt = jt + 128u;
    }

    if (tid < p.d_head) {
        dq[q_base + tid] = acc;
    }
}
";

const FLASH_DKV_WGSL: &str = "
struct P { t: u32, d_head: u32, n_head: u32, wnd: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       d_out: array<f32>;
@group(0) @binding(4) var<storage, read>       ml: array<f32>;
@group(0) @binding(5) var<storage, read>       d_row: array<f32>;
@group(0) @binding(6) var<storage, read_write> dk: array<f32>;
@group(0) @binding(7) var<storage, read_write> dv: array<f32>;
var<push_constant> p: P;

var<workgroup> sh_p: array<f32, 128>;
var<workgroup> sh_ds: array<f32, 128>;

fn visible(j: u32, i: u32, wnd: u32) -> bool {
    return j <= i && (wnd == 0u || j + wnd > i);
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j = wgid.x;
    let head_idx = wgid.y;
    let tid = lid.x;
    if (j >= p.t) { return; }
    let d_model = p.d_head * p.n_head;
    let h_off = head_idx * p.d_head;
    let k_base = j * d_model + h_off;

    // Queries that can see j: i ∈ [j, j+wnd) ∩ [0, t).
    var i_end: u32 = p.t;
    if (p.wnd > 0u && j + p.wnd < p.t) { i_end = j + p.wnd; }

    var acc_k: f32 = 0.0;
    var acc_v: f32 = 0.0;

    var it: u32 = j;
    loop {
        if (it >= i_end) { break; }

        let i = it + tid;
        var prob: f32 = 0.0;
        var ds: f32 = 0.0;
        if (i < i_end && visible(j, i, p.wnd)) {
            let q_base = i * d_model + h_off;
            var dot_qk: f32 = 0.0;
            var dp: f32 = 0.0;
            for (var c: u32 = 0u; c < p.d_head; c = c + 1u) {
                dot_qk = dot_qk + q[q_base + c] * k[k_base + c];
                dp = dp + d_out[q_base + c] * v[k_base + c];
            }
            let m_i = ml[(head_idx * p.t + i) * 2u];
            let l_i = ml[(head_idx * p.t + i) * 2u + 1u];
            let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
            prob = exp(dot_qk * p.scale - m_i) * inv_l;
            ds = prob * (dp - d_row[head_idx * p.t + i]) * p.scale;
        }
        sh_p[tid] = prob;
        sh_ds[tid] = ds;
        workgroupBarrier();

        if (tid < p.d_head) {
            let tile_n = min(TILE_U, i_end - it);
            for (var ii: u32 = 0u; ii < tile_n; ii = ii + 1u) {
                let src = (it + ii) * d_model + h_off + tid;
                if (sh_p[ii] != 0.0) {
                    acc_v = acc_v + sh_p[ii] * d_out[src];
                    acc_k = acc_k + sh_ds[ii] * q[src];
                }
            }
        }
        workgroupBarrier();

        it = it + 128u;
    }

    if (tid < p.d_head) {
        dk[k_base + tid] = acc_k;
        dv[k_base + tid] = acc_v;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashParams {
    t: u32,
    d_head: u32,
    n_head: u32,
    wnd: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

fn build_pipeline(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
    n_binds_read: usize,
    n_binds_rw: usize,
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let src = wgsl.replace("TILE_U", &format!("{TILE}u"));
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let mut entries = Vec::new();
    for b in 0..n_binds_read {
        entries.push(storage_entry(b as u32, true));
    }
    for b in 0..n_binds_rw {
        entries.push(storage_entry((n_binds_read + b) as u32, false));
    }
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

fn bind(
    ctx: &GpuContext,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    bufs: &[&GpuBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.inner.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: bgl,
        entries: &entries,
    })
}

pub struct FlashAttnForwardKernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl FlashAttnForwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build_pipeline(ctx, "aurora flash fwd", FLASH_FWD_WGSL, 3, 2);
        Self { pipeline, bgl }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        out: &GpuBuffer,
        ml: &GpuBuffer,
    ) -> wgpu::BindGroup {
        bind(ctx, "aurora flash fwd bind", &self.bgl, &[q, k, v, out, ml])
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
        window: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = FlashParams { t, d_head, n_head, wnd: window, scale, _p0: 0, _p1: 0, _p2: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(t, n_head, 1);
    }
}

pub struct FlashAttnBackwardKernel {
    pub(crate) pipe_dpre: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dpre: Arc<wgpu::BindGroupLayout>,
    pub(crate) pipe_dq: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dq: Arc<wgpu::BindGroupLayout>,
    pub(crate) pipe_dkv: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl_dkv: Arc<wgpu::BindGroupLayout>,
}

impl FlashAttnBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipe_dpre, bgl_dpre) = build_pipeline(ctx, "aurora flash dpre", FLASH_DPRE_WGSL, 2, 1);
        let (pipe_dq, bgl_dq) = build_pipeline(ctx, "aurora flash dq", FLASH_DQ_WGSL, 6, 1);
        let (pipe_dkv, bgl_dkv) = build_pipeline(ctx, "aurora flash dkv", FLASH_DKV_WGSL, 6, 2);
        Self { pipe_dpre, bgl_dpre, pipe_dq, bgl_dq, pipe_dkv, bgl_dkv }
    }

    pub fn build_bind_dpre(
        &self,
        ctx: &GpuContext,
        d_out: &GpuBuffer,
        out: &GpuBuffer,
        d_row: &GpuBuffer,
    ) -> wgpu::BindGroup {
        bind(ctx, "aurora flash dpre bind", &self.bgl_dpre, &[d_out, out, d_row])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_bind_dq(
        &self,
        ctx: &GpuContext,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        d_out: &GpuBuffer,
        ml: &GpuBuffer,
        d_row: &GpuBuffer,
        dq: &GpuBuffer,
    ) -> wgpu::BindGroup {
        bind(ctx, "aurora flash dq bind", &self.bgl_dq, &[q, k, v, d_out, ml, d_row, dq])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_bind_dkv(
        &self,
        ctx: &GpuContext,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        d_out: &GpuBuffer,
        ml: &GpuBuffer,
        d_row: &GpuBuffer,
        dk: &GpuBuffer,
        dv: &GpuBuffer,
    ) -> wgpu::BindGroup {
        bind(ctx, "aurora flash dkv bind", &self.bgl_dkv, &[q, k, v, d_out, ml, d_row, dk, dv])
    }

    /// Three sequential passes: D row-dots → dq → dk/dv. Writes dq/dk/dv
    /// directly (no accumulation — callers must not pre-seed them).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_binds(
        &self,
        enc: &mut wgpu::CommandEncoder,
        bind_dpre: &wgpu::BindGroup,
        bind_dq: &wgpu::BindGroup,
        bind_dkv: &wgpu::BindGroup,
        t: u32,
        n_head: u32,
        d_head: u32,
        window: u32,
    ) {
        let scale = (d_head as f32).sqrt().recip();
        let p = FlashParams { t, d_head, n_head, wnd: window, scale, _p0: 0, _p1: 0, _p2: 0 };
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flash dpre"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_dpre);
            pass.set_bind_group(0, bind_dpre, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            pass.dispatch_workgroups(t, n_head, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flash dq"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_dq);
            pass.set_bind_group(0, bind_dq, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            pass.dispatch_workgroups(t, n_head, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flash dkv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_dkv);
            pass.set_bind_group(0, bind_dkv, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            pass.dispatch_workgroups(t, n_head, 1);
        }
    }
}
