//! GPU kernels for the thermodynamic-LM local operator
//! `N_θ = Conv1x1(GELU(Conv3x3(s)))` (2D). Mirrors the CPU
//! `crate::thermo::conv` forward; parity-tested in
//! `tests/thermo_gpu_parity.rs`. One thread per output element.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const WG: u32 = 64;

// ───── conv3x3 + GELU ───────────────────────────────────────────────────

const CONV3X3_GELU_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:  array<f32>;   // [h*w*c]
@group(0) @binding(1) var<storage, read>       w1: array<f32>;   // [ch*c*9]
@group(0) @binding(2) var<storage, read>       b1: array<f32>;   // [ch]
@group(0) @binding(3) var<storage, read_write> hp: array<f32>;   // [h*w*ch] (post-GELU)
var<push_constant> p: P;

fn gelu(x: f32) -> f32 {
    let k = 0.7978845608028654;          // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.h * p.w * p.ch;
    if (idx >= total) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let i = cell / p.w;
    let j = cell % p.w;
    var acc = b1[co];
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let ii = i32(i) + i32(ki) - 1;
        if (ii < 0 || ii >= i32(p.h)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let jj = i32(j) + i32(kj) - 1;
            if (jj < 0 || jj >= i32(p.w)) { continue; }
            let in_base = (u32(ii) * p.w + u32(jj)) * p.c;
            for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
                let widx = ((co * p.c + ci) * 3u + ki) * 3u + kj;
                acc = acc + w1[widx] * s[in_base + ci];
            }
        }
    }
    hp[idx] = gelu(acc);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv3Params {
    h: u32,
    w: u32,
    c: u32,
    ch: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

pub struct ThermoConv3x3GeluKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3GeluKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3_gelu", CONV3X3_GELU_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        w1: &GpuBuffer,
        b1: &GpuBuffer,
        hp: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
    ) {
        let p = Conv3Params { h, w, c, ch, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermo_conv3x3_gelu bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: s.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: hp.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("thermo_conv3x3_gelu pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((h * w * ch).div_ceil(WG), 1, 1);
    }
}

// ───── conv3x3 (no GELU; outputs pre-activation for the bwd cache) ──────

const CONV3X3_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:  array<f32>;
@group(0) @binding(1) var<storage, read>       w1: array<f32>;
@group(0) @binding(2) var<storage, read>       b1: array<f32>;
@group(0) @binding(3) var<storage, read_write> hp: array<f32>;   // pre-GELU
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.ch) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let i = cell / p.w;
    let j = cell % p.w;
    var acc = b1[co];
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let ii = i32(i) + i32(ki) - 1;
        if (ii < 0 || ii >= i32(p.h)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let jj = i32(j) + i32(kj) - 1;
            if (jj < 0 || jj >= i32(p.w)) { continue; }
            let in_base = (u32(ii) * p.w + u32(jj)) * p.c;
            for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
                let widx = ((co * p.c + ci) * 3u + ki) * 3u + kj;
                acc = acc + w1[widx] * s[in_base + ci];
            }
        }
    }
    hp[idx] = acc;
}
";

pub struct ThermoConv3x3Kernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3", CONV3X3_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        w1: &GpuBuffer,
        b1: &GpuBuffer,
        hp: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
    ) {
        let p = Conv3Params { h, w, c, ch, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, w1, b1, hp], bytemuck::bytes_of(&p), (h * w * ch).div_ceil(WG));
    }
}

// ───── conv1x1 (per-cell linear) ────────────────────────────────────────

const CONV1X1_WGSL: &str = "
struct P { cells:u32, ch:u32, cout:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32, _p4:u32 };
@group(0) @binding(0) var<storage, read>       hp:  array<f32>;  // [cells*ch]
@group(0) @binding(1) var<storage, read>       w2:  array<f32>;  // [cout*ch]
@group(0) @binding(2) var<storage, read>       b2:  array<f32>;  // [cout]
@group(0) @binding(3) var<storage, read_write> raw: array<f32>;  // [cells*cout]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.cells * p.cout;
    if (idx >= total) { return; }
    let co = idx % p.cout;
    let cell = idx / p.cout;
    let hbase = cell * p.ch;
    let wbase = co * p.ch;
    var acc = b2[co];
    for (var ci: u32 = 0u; ci < p.ch; ci = ci + 1u) {
        acc = acc + w2[wbase + ci] * hp[hbase + ci];
    }
    raw[idx] = acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv1Params {
    cells: u32,
    ch: u32,
    cout: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
}

pub struct ThermoConv1x1Kernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv1x1Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv1x1", CONV1X1_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        hp: &GpuBuffer,
        w2: &GpuBuffer,
        b2: &GpuBuffer,
        raw: &GpuBuffer,
        cells: u32,
        ch: u32,
        cout: u32,
    ) {
        let p = Conv1Params { cells, ch, cout, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0 };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermo_conv1x1 bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: hp.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w2.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b2.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: raw.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("thermo_conv1x1 pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((cells * cout).div_ceil(WG), 1, 1);
    }
}

// ───── conv3x3 backward: d_s (gather, atomics-free) ─────────────────────

const CONV3X3_DS_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;  // [h*w*ch] (d_h_pre)
@group(0) @binding(1) var<storage, read>       w1:  array<f32>;  // [ch*c*9]
@group(0) @binding(2) var<storage, read_write> ds:  array<f32>;  // [h*w*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let ci = idx % p.c;
    let cell = idx / p.c;
    let yi = cell / p.w;
    let yj = cell % p.w;
    var acc = 0.0;
    // s[yi,yj,ci] feeds h_pre[i,j,co] with i = yi-ki+1, j = yj-kj+1.
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let i = i32(yi) - i32(ki) + 1;
        if (i < 0 || i >= i32(p.h)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let j = i32(yj) - i32(kj) + 1;
            if (j < 0 || j >= i32(p.w)) { continue; }
            let obase = (u32(i) * p.w + u32(j)) * p.ch;
            for (var co: u32 = 0u; co < p.ch; co = co + 1u) {
                let widx = ((co * p.c + ci) * 3u + ki) * 3u + kj;
                acc = acc + dhp[obase + co] * w1[widx];
            }
        }
    }
    ds[idx] = acc;
}
";

pub struct ThermoConv3x3DsKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3DsKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3_ds", CONV3X3_DS_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        dhp: &GpuBuffer,
        w1: &GpuBuffer,
        ds: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
    ) {
        let p = Conv3Params { h, w, c, ch, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[dhp, w1, ds], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

// ───── conv3x3 backward: d_w1 (one thread per weight) ───────────────────

const CONV3X3_DW1_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;  // [h*w*ch]
@group(0) @binding(1) var<storage, read>       s:   array<f32>;  // [h*w*c]
@group(0) @binding(2) var<storage, read_write> dw1: array<f32>;  // [ch*c*9]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let widx = gid.x;
    if (widx >= p.ch * p.c * 9u) { return; }
    let kj = widx % 3u;
    let r1 = widx / 3u;
    let ki = r1 % 3u;
    let r2 = r1 / 3u;
    let ci = r2 % p.c;
    let co = r2 / p.c;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.h; i = i + 1u) {
        let ii = i32(i) + i32(ki) - 1;
        if (ii < 0 || ii >= i32(p.h)) { continue; }
        for (var j: u32 = 0u; j < p.w; j = j + 1u) {
            let jj = i32(j) + i32(kj) - 1;
            if (jj < 0 || jj >= i32(p.w)) { continue; }
            let o = (i * p.w + j) * p.ch + co;
            let in_idx = (u32(ii) * p.w + u32(jj)) * p.c + ci;
            acc = acc + dhp[o] * s[in_idx];
        }
    }
    dw1[widx] = dw1[widx] + acc;  // accumulate (rollout sums over T×K steps)
}
";

pub struct ThermoConv3x3Dw1Kernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3Dw1Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3_dw1", CONV3X3_DW1_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        dhp: &GpuBuffer,
        s: &GpuBuffer,
        dw1: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
    ) {
        let p = Conv3Params { h, w, c, ch, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[dhp, s, dw1], bytemuck::bytes_of(&p), (ch * c * 9).div_ceil(WG));
    }
}

// ───── shared pipeline builder ──────────────────────────────────────────

/// Dispatch a kernel binding `buffers` in order with raw push-constant
/// bytes and a 1D workgroup count.
pub(crate) fn dispatch_n(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    buffers: &[&GpuBuffer],
    push: &[u8],
    groups: u32,
) {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.inner.as_entire_binding(),
        })
        .collect();
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("thermo dispatch_n bind"),
        layout: bgl,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("thermo dispatch_n pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.set_push_constants(0, push);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Build a compute pipeline. `read_only[i]` sets binding `i`'s access
/// (false = read_write). 32-byte push-constant range.
pub(crate) fn build(
    ctx: &GpuContext,
    label: &str,
    wgsl: &str,
    read_only: &[bool],
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(b, &ro)| storage_entry(b as u32, ro))
        .collect();
    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..32,
        }],
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (Arc::new(pipeline), Arc::new(bgl))
}
