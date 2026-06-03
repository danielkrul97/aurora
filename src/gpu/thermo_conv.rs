//! GPU kernels for the thermodynamic-LM local operator
//! `N_θ = Conv1x1(GELU(Conv3x3x3(s)))` (3D; `D=1` = the 2D case). Mirrors
//! the CPU `crate::thermo::conv` forward; parity-tested in
//! `tests/thermo_gpu_parity.rs`. One thread per output element. Kernel
//! offsets (ki,kj,kl) map to the (depth,height,width) axes, exactly as the
//! CPU pack; `tap = (ki*3+kj)*3+kl`, `widx = (co*c+ci)*27 + tap`.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const WG: u32 = 64;

// ───── conv3x3 + GELU ───────────────────────────────────────────────────

const CONV3X3_GELU_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:  array<f32>;   // [nb*d*h*w*c]
@group(0) @binding(1) var<storage, read>       w1: array<f32>;   // [ch*c*27]
@group(0) @binding(2) var<storage, read>       b1: array<f32>;   // [ch]
@group(0) @binding(3) var<storage, read_write> hp: array<f32>;   // [nb*d*h*w*ch] (post-GELU)
var<push_constant> p: P;

fn gelu(x: f32) -> f32 {
    let k = 0.7978845608028654;          // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.nb * p.d * p.h * p.w * p.ch;
    if (idx >= total) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let j = cell % p.w;
    let r1 = cell / p.w;
    let i = r1 % p.h;
    let r2 = r1 / p.h;
    let z = r2 % p.d;            // depth within grid
    let g = r2 / p.d;            // batch grid index
    var acc = b1[co];
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let zz = i32(z) + i32(ki) - 1;
        if (zz < 0 || zz >= i32(p.d)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let ii = i32(i) + i32(kj) - 1;
            if (ii < 0 || ii >= i32(p.h)) { continue; }
            for (var kl: u32 = 0u; kl < 3u; kl = kl + 1u) {
                let jj = i32(j) + i32(kl) - 1;
                if (jj < 0 || jj >= i32(p.w)) { continue; }
                let in_base = (((g * p.d + u32(zz)) * p.h + u32(ii)) * p.w + u32(jj)) * p.c;
                let tap = (ki * 3u + kj) * 3u + kl;
                for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
                    let widx = (co * p.c + ci) * 27u + tap;
                    acc = acc + w1[widx] * s[in_base + ci];
                }
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
    nb: u32,
    d: u32,
    _p2: u32,
    _p3: u32,
}

impl Conv3Params {
    fn new(h: u32, w: u32, c: u32, ch: u32, nb: u32, d: u32) -> Self {
        Self { h, w, c, ch, nb, d, _p2: 0, _p3: 0 }
    }
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
        d: u32,
    ) {
        self.dispatch_b(ctx, encoder, s, w1, b1, hp, h, w, c, ch, 1, d);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, w1, b1, hp], bytemuck::bytes_of(&p), (nb * d * h * w * ch).div_ceil(WG));
    }
}

// ───── conv3x3 (no GELU; outputs pre-activation for the bwd cache) ──────

const CONV3X3_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:  array<f32>;
@group(0) @binding(1) var<storage, read>       w1: array<f32>;
@group(0) @binding(2) var<storage, read>       b1: array<f32>;
@group(0) @binding(3) var<storage, read_write> hp: array<f32>;   // pre-GELU
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.d * p.h * p.w * p.ch) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let j = cell % p.w;
    let r1 = cell / p.w;
    let i = r1 % p.h;
    let r2 = r1 / p.h;
    let z = r2 % p.d;
    let g = r2 / p.d;
    var acc = b1[co];
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let zz = i32(z) + i32(ki) - 1;
        if (zz < 0 || zz >= i32(p.d)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let ii = i32(i) + i32(kj) - 1;
            if (ii < 0 || ii >= i32(p.h)) { continue; }
            for (var kl: u32 = 0u; kl < 3u; kl = kl + 1u) {
                let jj = i32(j) + i32(kl) - 1;
                if (jj < 0 || jj >= i32(p.w)) { continue; }
                let in_base = (((g * p.d + u32(zz)) * p.h + u32(ii)) * p.w + u32(jj)) * p.c;
                let tap = (ki * 3u + kj) * 3u + kl;
                for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
                    let widx = (co * p.c + ci) * 27u + tap;
                    acc = acc + w1[widx] * s[in_base + ci];
                }
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
        d: u32,
    ) {
        self.dispatch_b(ctx, encoder, s, w1, b1, hp, h, w, c, ch, 1, d);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, w1, b1, hp], bytemuck::bytes_of(&p), (nb * d * h * w * ch).div_ceil(WG));
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
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;  // [nb*d*h*w*ch] (d_h_pre)
@group(0) @binding(1) var<storage, read>       w1:  array<f32>;  // [ch*c*27]
@group(0) @binding(2) var<storage, read_write> ds:  array<f32>;  // [nb*d*h*w*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.d * p.h * p.w * p.c) { return; }
    let ci = idx % p.c;
    let cell = idx / p.c;
    let yj = cell % p.w;
    let r1 = cell / p.w;
    let yi = r1 % p.h;
    let r2 = r1 / p.h;
    let yz = r2 % p.d;
    let g = r2 / p.d;
    var acc = 0.0;
    // s[yz,yi,yj,ci] feeds h_pre[z,i,j,co] with z=yz-ki+1, i=yi-kj+1, j=yj-kl+1.
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let zc = i32(yz) - i32(ki) + 1;
        if (zc < 0 || zc >= i32(p.d)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let i = i32(yi) - i32(kj) + 1;
            if (i < 0 || i >= i32(p.h)) { continue; }
            for (var kl: u32 = 0u; kl < 3u; kl = kl + 1u) {
                let j = i32(yj) - i32(kl) + 1;
                if (j < 0 || j >= i32(p.w)) { continue; }
                let obase = (((g * p.d + u32(zc)) * p.h + u32(i)) * p.w + u32(j)) * p.ch;
                let tap = (ki * 3u + kj) * 3u + kl;
                for (var co: u32 = 0u; co < p.ch; co = co + 1u) {
                    let widx = (co * p.c + ci) * 27u + tap;
                    acc = acc + dhp[obase + co] * w1[widx];
                }
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
        d: u32,
    ) {
        self.dispatch_b(ctx, encoder, dhp, w1, ds, h, w, c, ch, 1, d);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[dhp, w1, ds], bytemuck::bytes_of(&p), (nb * d * h * w * c).div_ceil(WG));
    }
}

// ───── conv3x3 backward: d_w1 (one thread per weight) ───────────────────

const CONV3X3_DW1_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       dhp: array<f32>;  // [nb*d*h*w*ch]
@group(0) @binding(1) var<storage, read>       s:   array<f32>;  // [nb*d*h*w*c]
@group(0) @binding(2) var<storage, read_write> dw1: array<f32>;  // [ch*c*27]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let widx = gid.x;
    if (widx >= p.ch * p.c * 27u) { return; }
    let kl = widx % 3u;
    let r1 = widx / 3u;
    let kj = r1 % 3u;
    let r2 = r1 / 3u;
    let ki = r2 % 3u;
    let r3 = r2 / 3u;
    let ci = r3 % p.c;
    let co = r3 / p.c;
    let cells = p.d * p.h * p.w;
    var acc = 0.0;
    for (var g: u32 = 0u; g < p.nb; g = g + 1u) {
        for (var zc: u32 = 0u; zc < p.d; zc = zc + 1u) {
            let zz = i32(zc) + i32(ki) - 1;
            if (zz < 0 || zz >= i32(p.d)) { continue; }
            for (var i: u32 = 0u; i < p.h; i = i + 1u) {
                let ii = i32(i) + i32(kj) - 1;
                if (ii < 0 || ii >= i32(p.h)) { continue; }
                for (var j: u32 = 0u; j < p.w; j = j + 1u) {
                    let jj = i32(j) + i32(kl) - 1;
                    if (jj < 0 || jj >= i32(p.w)) { continue; }
                    let o = (g * cells + (zc * p.h + i) * p.w + j) * p.ch + co;
                    let in_idx = (g * cells + (u32(zz) * p.h + u32(ii)) * p.w + u32(jj)) * p.c + ci;
                    acc = acc + dhp[o] * s[in_idx];
                }
            }
        }
    }
    dw1[widx] = dw1[widx] + acc;  // accumulate (rollout sums over batch × T×K steps)
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
        d: u32,
    ) {
        self.dispatch_b(ctx, encoder, dhp, s, dw1, h, w, c, ch, 1, d);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[dhp, s, dw1], bytemuck::bytes_of(&p), (ch * c * 27).div_ceil(WG));
    }
}

// ───── conv3x3 backward: d_w1 split-K (one group per batch grid) ────────
//
// The plain `d_w1` runs `ch·c·9` threads (≈1728), each serially summing all
// `nb·h·w` cells — far too few threads to fill the device (the #2 thermo
// backward cost). Here each batch grid `g` is its own K-slice: `nb·ch·c·9`
// threads write per-grid partials, then a `RowReduceAdd` over `nb` sums them
// into `d_w1`. Same math, `nb×` the parallelism.

const CONV3X3_DW1_SPLITK_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       dhp:      array<f32>;  // [nb*d*h*w*ch]
@group(0) @binding(1) var<storage, read>       s:        array<f32>;  // [nb*d*h*w*c]
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;  // [nb, ch*c*27]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nw = p.ch * p.c * 27u;
    let idx = gid.x;
    if (idx >= p.nb * nw) { return; }
    let g = idx / nw;             // batch grid = K-slice
    let widx = idx % nw;
    let kl = widx % 3u;
    let r1 = widx / 3u;
    let kj = r1 % 3u;
    let r2 = r1 / 3u;
    let ki = r2 % 3u;
    let r3 = r2 / 3u;
    let ci = r3 % p.c;
    let co = r3 / p.c;
    let cells = p.d * p.h * p.w;
    var acc = 0.0;
    for (var zc: u32 = 0u; zc < p.d; zc = zc + 1u) {
        let zz = i32(zc) + i32(ki) - 1;
        if (zz < 0 || zz >= i32(p.d)) { continue; }
        for (var i: u32 = 0u; i < p.h; i = i + 1u) {
            let ii = i32(i) + i32(kj) - 1;
            if (ii < 0 || ii >= i32(p.h)) { continue; }
            for (var j: u32 = 0u; j < p.w; j = j + 1u) {
                let jj = i32(j) + i32(kl) - 1;
                if (jj < 0 || jj >= i32(p.w)) { continue; }
                let o = (g * cells + (zc * p.h + i) * p.w + j) * p.ch + co;
                let in_idx = (g * cells + (u32(zz) * p.h + u32(ii)) * p.w + u32(jj)) * p.c + ci;
                acc = acc + dhp[o] * s[in_idx];
            }
        }
    }
    partials[idx] = acc;
}
";

pub struct ThermoConv3x3Dw1SplitKKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3Dw1SplitKKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3_dw1_splitk", CONV3X3_DW1_SPLITK_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }

    /// Writes `[nb, ch*c*9]` per-grid partials into `partials` (sized
    /// `nb*ch*c*9`). Follow with `RowReduceAdd` (`t = nb`, `d = ch*c*9`) to
    /// accumulate into `d_w1`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        dhp: &GpuBuffer,
        s: &GpuBuffer,
        partials: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[dhp, s, partials], bytemuck::bytes_of(&p), (nb * ch * c * 27).div_ceil(WG));
    }
}

// ───── im2col for the d_w1 backward (patch matrix → reuse the GEMM path) ─
//
// Materializes `patch[t, ci*27+tap] = s[neighbour(cell,tap), ci]` (0 outside
// the grid), `t = g*cells + cell`, shape `[nb*cells, c*27]`. Then
// `d_w1[co, ci*27+tap] = Σ_t dhp[t,co]·patch[t,·]` is an outer-product
// reduction — the SAME op as `d_w2`, run via `OuterProductSplitK` +
// `RowReduceAdd` (which clocks ~40× faster than the conv-style split-K
// d_w1, whose scattered s-reads over a serial cell loop were the cost).
// The patch column order `ci*27+tap` makes the GEMM output land directly in
// the canonical `w1[(co*c+ci)*27+tap]` layout.

const IM2COL_DW1_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:   array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read_write> pat: array<f32>;  // [nb*cells*(c*27)]
var<push_constant> p: P;

// One thread per (t, ci); the 27 taps loop inside (contiguous writes), so the
// grid stays well under the 65535 workgroup-per-dim limit.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cells = p.d * p.h * p.w;
    if (idx >= p.nb * cells * p.c) { return; }
    let ci = idx % p.c;
    let t = idx / p.c;
    let cell = t % cells;
    let g = t / cells;
    let j = cell % p.w;
    let r1 = cell / p.w;
    let i = r1 % p.h;
    let z = r1 / p.h;
    let base = (t * p.c + ci) * 27u;
    for (var tap: u32 = 0u; tap < 27u; tap = tap + 1u) {
        let ki = tap / 9u;
        let kj = (tap / 3u) % 3u;
        let kl = tap % 3u;
        let zz = i32(z) + i32(ki) - 1;
        let ii = i32(i) + i32(kj) - 1;
        let jj = i32(j) + i32(kl) - 1;
        var v = 0.0;
        if (zz >= 0 && zz < i32(p.d) && ii >= 0 && ii < i32(p.h) && jj >= 0 && jj < i32(p.w)) {
            let nbr = (u32(zz) * p.h + u32(ii)) * p.w + u32(jj);
            v = s[(g * cells + nbr) * p.c + ci];
        }
        pat[base + tap] = v;
    }
}
";

pub struct ThermoIm2colDw1Kernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoIm2colDw1Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_im2col_dw1", IM2COL_DW1_WGSL, &[true, false]);
        Self { pipeline, bgl }
    }

    /// Build the `[nb*cells, c*27]` patch matrix from `s`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        patch: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, 0, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, patch], bytemuck::bytes_of(&p), (nb * d * h * w * c).div_ceil(WG));
    }
}

// ───── fused conv3x3 + GELU writing BOTH pre/post (forward cache) ───────
//
// Folds the separate `conv3x3` (pre-act cache) and `gelu` dispatches into
// one: each thread computes the conv accumulation once and writes the
// pre-activation (for the bwd gelu') and the post-GELU (for conv1x1).

const CONV3X3_GELU2_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, ch:u32, nb:u32, d:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>       s:     array<f32>;  // [nb*d*h*w*c]
@group(0) @binding(1) var<storage, read>       w1:    array<f32>;  // [ch*c*27]
@group(0) @binding(2) var<storage, read>       b1:    array<f32>;  // [ch]
@group(0) @binding(3) var<storage, read_write> hpre:  array<f32>;  // [nb*d*h*w*ch]
@group(0) @binding(4) var<storage, read_write> hpost: array<f32>;  // [nb*d*h*w*ch]
var<push_constant> p: P;

fn gelu(x: f32) -> f32 {
    let k = 0.7978845608028654;
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.d * p.h * p.w * p.ch) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let j = cell % p.w;
    let r1 = cell / p.w;
    let i = r1 % p.h;
    let r2 = r1 / p.h;
    let z = r2 % p.d;
    let g = r2 / p.d;
    var acc = b1[co];
    for (var ki: u32 = 0u; ki < 3u; ki = ki + 1u) {
        let zz = i32(z) + i32(ki) - 1;
        if (zz < 0 || zz >= i32(p.d)) { continue; }
        for (var kj: u32 = 0u; kj < 3u; kj = kj + 1u) {
            let ii = i32(i) + i32(kj) - 1;
            if (ii < 0 || ii >= i32(p.h)) { continue; }
            for (var kl: u32 = 0u; kl < 3u; kl = kl + 1u) {
                let jj = i32(j) + i32(kl) - 1;
                if (jj < 0 || jj >= i32(p.w)) { continue; }
                let in_base = (((g * p.d + u32(zz)) * p.h + u32(ii)) * p.w + u32(jj)) * p.c;
                let tap = (ki * 3u + kj) * 3u + kl;
                for (var ci: u32 = 0u; ci < p.c; ci = ci + 1u) {
                    let widx = (co * p.c + ci) * 27u + tap;
                    acc = acc + w1[widx] * s[in_base + ci];
                }
            }
        }
    }
    hpre[idx] = acc;
    hpost[idx] = gelu(acc);
}
";

pub struct ThermoConv3x3GeluCacheKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConv3x3GeluCacheKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv3x3_gelu2", CONV3X3_GELU2_WGSL, &[true, true, true, false, false]);
        Self { pipeline, bgl }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        w1: &GpuBuffer,
        b1: &GpuBuffer,
        hpre: &GpuBuffer,
        hpost: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        ch: u32,
        nb: u32,
        d: u32,
    ) {
        let p = Conv3Params::new(h, w, c, ch, nb, d);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, w1, b1, hpre, hpost], bytemuck::bytes_of(&p), (nb * d * h * w * ch).div_ceil(WG));
    }
}

// ───── fused conv1x1 backward: dhpost = d_raw·w2ᵀ, then ×GELU' → dhpre ────
//
// Folds the conv1x1-backward matmul and the GELU backward into one
// per-(cell,channel) thread. `cells` is the stacked B*cells count (flat).

const CONV_BWD_HIDDEN_WGSL: &str = "
struct P { cells:u32, ch:u32, cout:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32, _p4:u32 };
@group(0) @binding(0) var<storage, read>       draw:  array<f32>;  // [cells*cout]
@group(0) @binding(1) var<storage, read>       w2:    array<f32>;  // [cout*ch]
@group(0) @binding(2) var<storage, read>       hpre:  array<f32>;  // [cells*ch]
@group(0) @binding(3) var<storage, read_write> dhpre: array<f32>;  // [cells*ch]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.cells * p.ch) { return; }
    let co = idx % p.ch;
    let cell = idx / p.ch;
    let dbase = cell * p.cout;
    var acc = 0.0;                              // dhpost = Σ_cout d_raw·w2
    for (var cc: u32 = 0u; cc < p.cout; cc = cc + 1u) {
        acc = acc + draw[dbase + cc] * w2[cc * p.ch + co];
    }
    let v = hpre[idx];                          // GELU'(pre) · dhpost
    let k = 0.7978845608028654;
    let inner = k * (v + 0.044715 * v * v * v);
    let th = tanh(inner);
    let sech2 = 1.0 - th * th;
    let d_inner = k * (1.0 + 3.0 * 0.044715 * v * v);
    let gp = 0.5 * (1.0 + th) + 0.5 * v * sech2 * d_inner;
    dhpre[idx] = acc * gp;
}
";

pub struct ThermoConvBwdHiddenKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoConvBwdHiddenKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_conv_bwd_hidden", CONV_BWD_HIDDEN_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        draw: &GpuBuffer,
        w2: &GpuBuffer,
        hpre: &GpuBuffer,
        dhpre: &GpuBuffer,
        cells: u32,
        ch: u32,
        cout: u32,
    ) {
        let p = Conv1Params { cells, ch, cout, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[draw, w2, hpre, dhpre], bytemuck::bytes_of(&p), (cells * ch).div_ceil(WG));
    }
}

// ───── bias-add + GELU epilogue (for the im2col + GEMM forward conv) ─────
//
// The im2col + `GemmTransposeB` path computes `hpre = patch @ w1ᵀ` without the
// bias; this folds `+b1` (broadcast over cells) and the GELU into one pass,
// writing the bias-corrected hpre (for the bwd) and hpost (for conv1x1).

const BIAS_GELU_WGSL: &str = "
struct P { cells:u32, ch:u32, _p0:u32, _p1:u32, _p2:u32, _p3:u32, _p4:u32, _p5:u32 };
@group(0) @binding(0) var<storage, read_write> hpre:  array<f32>;  // in: no-bias, out: +bias
@group(0) @binding(1) var<storage, read>       b1:    array<f32>;  // [ch]
@group(0) @binding(2) var<storage, read_write> hpost: array<f32>;  // [cells*ch]
var<push_constant> p: P;

fn gelu(x: f32) -> f32 {
    let k = 0.7978845608028654;
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.cells * p.ch) { return; }
    let co = idx % p.ch;
    let v = hpre[idx] + b1[co];
    hpre[idx] = v;
    hpost[idx] = gelu(v);
}
";

pub struct ThermoBiasGeluKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoBiasGeluKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_bias_gelu", BIAS_GELU_WGSL, &[false, true, false]);
        Self { pipeline, bgl }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        hpre: &GpuBuffer,
        b1: &GpuBuffer,
        hpost: &GpuBuffer,
        cells: u32,
        ch: u32,
    ) {
        let p = Conv1Params { cells, ch, cout: 0, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[hpre, b1, hpost], bytemuck::bytes_of(&p), (cells * ch).div_ceil(WG));
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
            // 48 bytes (12 words): most thermo kernels use ≤32, but the fused
            // 3D transport-forward params need 9 words. The device limit is
            // ≥128 (see GpuContext), so the slack is free.
            range: 0..48,
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
