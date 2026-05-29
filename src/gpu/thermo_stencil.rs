//! GPU stencil kernels for the thermo step: flux-form divergence (the
//! conservation transport, forward + backward) and temperature-gated
//! diffusion (forward). Mirror `crate::thermo::flux` /
//! `crate::thermo::diffusion`; parity-tested in `tests/thermo_gpu_parity.rs`.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::thermo_conv::{build, dispatch_n};
use super::GpuContext;

const WG: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StencilParams {
    h: u32,
    w: u32,
    /// conserved-channel count (flux) or total channels (diffusion).
    c: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
}

impl StencilParams {
    fn new(h: u32, w: u32, c: u32) -> Self {
        Self { h, w, c, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0 }
    }
}

// ───── flux divergence (forward) ────────────────────────────────────────

const FLUX_FWD_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       fx:  array<f32>;  // [cells*nc]
@group(0) @binding(1) var<storage, read>       fy:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dv:  array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    let i = cell / p.w;
    let j = cell % p.w;
    var in_left = 0.0;
    if (j > 0u) { in_left = fx[(i * p.w + (j - 1u)) * p.nc + m]; }
    var in_down = 0.0;
    if (i > 0u) { in_down = fy[((i - 1u) * p.w + j) * p.nc + m]; }
    var fx_self = 0.0;
    if (j + 1u < p.w) { fx_self = fx[idx]; }
    var fy_self = 0.0;
    if (i + 1u < p.h) { fy_self = fy[idx]; }
    dv[idx] = (in_left - fx_self) + (in_down - fy_self);
}
";

// ───── flux divergence (backward / adjoint) ─────────────────────────────

const FLUX_BWD_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;  // d_div [cells*nc]
@group(0) @binding(1) var<storage, read_write> dfx:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dfy:  array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    let i = cell / p.w;
    let j = cell % p.w;
    var v = 0.0;
    if (j + 1u < p.w) { v = -dd[idx] + dd[(i * p.w + (j + 1u)) * p.nc + m]; }
    dfx[idx] = v;
    var u = 0.0;
    if (i + 1u < p.h) { u = -dd[idx] + dd[((i + 1u) * p.w + j) * p.nc + m]; }
    dfy[idx] = u;
}
";

// ───── temperature-gated diffusion (forward) ────────────────────────────

const DIFF_FWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       s:     array<f32>;  // [cells*c]
@group(0) @binding(1) var<storage, read>       k0:    array<f32>;  // [c]
@group(0) @binding(2) var<storage, read>       a:     array<f32>;
@group(0) @binding(3) var<storage, read>       b:     array<f32>;
@group(0) @binding(4) var<storage, read_write> delta: array<f32>;  // [cells*c]
@group(0) @binding(5) var<storage, read_write> lap:   array<f32>;  // [cells*c]
var<push_constant> p: P;

const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let cc = idx % p.c;
    let cell = idx / p.c;
    let i = cell / p.w;
    let j = cell % p.w;
    let center = s[idx];
    var lv = 0.0;
    if (i > 0u)        { lv = lv + s[((i - 1u) * p.w + j) * p.c + cc] - center; }
    if (i + 1u < p.h)  { lv = lv + s[((i + 1u) * p.w + j) * p.c + cc] - center; }
    if (j > 0u)        { lv = lv + s[(i * p.w + (j - 1u)) * p.c + cc] - center; }
    if (j + 1u < p.w)  { lv = lv + s[(i * p.w + (j + 1u)) * p.c + cc] - center; }
    lap[idx] = lv;
    let t = s[cell * p.c + CH_T];
    let z = a[cc] + b[cc] * t;
    let sig = 1.0 / (1.0 + exp(-z));
    delta[idx] = k0[cc] * sig * lv;
}
";

// ───── diffusion backward ───────────────────────────────────────────────
//
// `[k0,a,b]` are packed into one `gate` buffer (`gate[c*g + ch]`) to keep
// each kernel within the 8-storage-buffer limit. Three kernels:
//   PRE  → gbuf = κ·d_delta, cT = d_delta·lap·(k0·σ'·b)  (CH_T coupling term)
//   PARAM (one thread/channel, reduce over cells) → d_k0, d_a, d_b
//   DS   → d_s = ∇²(gbuf) + (channel==T ? Σ_ch cT : 0)

const DIFF_BWD_PRE_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;  // d_delta [cells*c]
@group(0) @binding(1) var<storage, read>       lap:  array<f32>;
@group(0) @binding(2) var<storage, read>       s:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;  // [3*c] = k0|a|b
@group(0) @binding(4) var<storage, read_write> gbuf: array<f32>;
@group(0) @binding(5) var<storage, read_write> cT:   array<f32>;
var<push_constant> p: P;
const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let cc = idx % p.c;
    let cell = idx / p.c;
    let k0 = gate[cc];
    let a = gate[p.c + cc];
    let b = gate[2u * p.c + cc];
    let t = s[cell * p.c + CH_T];
    let z = a + b * t;
    let sig = 1.0 / (1.0 + exp(-z));
    let sigp = sig * (1.0 - sig);
    let ddv = dd[idx];
    let lapv = lap[idx];
    gbuf[idx] = k0 * sig * ddv;
    cT[idx] = ddv * lapv * (k0 * sigp * b);
}
";

const DIFF_BWD_PARAM_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;
@group(0) @binding(1) var<storage, read>       lap:  array<f32>;
@group(0) @binding(2) var<storage, read>       s:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dk0:  array<f32>;
@group(0) @binding(5) var<storage, read_write> da:   array<f32>;
@group(0) @binding(6) var<storage, read_write> db:   array<f32>;
var<push_constant> p: P;
const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= p.c) { return; }
    let cells = p.h * p.w;
    let k0 = gate[ch];
    let a = gate[p.c + ch];
    let b = gate[2u * p.c + ch];
    var sk0 = 0.0;
    var sa = 0.0;
    var sb = 0.0;
    for (var cell: u32 = 0u; cell < cells; cell = cell + 1u) {
        let t = s[cell * p.c + CH_T];
        let z = a + b * t;
        let sig = 1.0 / (1.0 + exp(-z));
        let sigp = sig * (1.0 - sig);
        let dl = dd[cell * p.c + ch] * lap[cell * p.c + ch];
        sk0 = sk0 + dl * sig;
        sa = sa + dl * k0 * sigp;
        sb = sb + dl * k0 * sigp * t;
    }
    dk0[ch] = dk0[ch] + sk0;
    da[ch] = da[ch] + sa;
    db[ch] = db[ch] + sb;
}
";

const DIFF_BWD_DS_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       gbuf: array<f32>;
@group(0) @binding(1) var<storage, read>       cT:   array<f32>;
@group(0) @binding(2) var<storage, read_write> ds:   array<f32>;
var<push_constant> p: P;
const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let cc = idx % p.c;
    let cell = idx / p.c;
    let i = cell / p.w;
    let j = cell % p.w;
    let center = gbuf[idx];
    var lg = 0.0;
    if (i > 0u)       { lg = lg + gbuf[((i - 1u) * p.w + j) * p.c + cc] - center; }
    if (i + 1u < p.h) { lg = lg + gbuf[((i + 1u) * p.w + j) * p.c + cc] - center; }
    if (j > 0u)       { lg = lg + gbuf[(i * p.w + (j - 1u)) * p.c + cc] - center; }
    if (j + 1u < p.w) { lg = lg + gbuf[(i * p.w + (j + 1u)) * p.c + cc] - center; }
    var v = lg;
    if (cc == CH_T) {
        for (var ch2: u32 = 0u; ch2 < p.c; ch2 = ch2 + 1u) {
            v = v + cT[cell * p.c + ch2];
        }
    }
    ds[idx] = v;
}
";

// ───── token injection (scatter encoder) ───────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InjectParams {
    cells: u32,
    c: u32,
    token: u32,
    anneal: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

const INJECT_FWD_WGSL: &str = "
struct P { cells:u32, c:u32, token:u32, anneal:f32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       s:       array<f32>;  // [cells*c]
@group(0) @binding(1) var<storage, read>       profile: array<f32>;  // [cells]
@group(0) @binding(2) var<storage, read>       inj:     array<f32>;  // [vocab*c]
@group(0) @binding(3) var<storage, read_write> out:     array<f32>;  // [cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.cells * p.c) { return; }
    let ch = idx % p.c;
    let cell = idx / p.c;
    out[idx] = s[idx] + p.anneal * profile[cell] * inj[p.token * p.c + ch];
}
";

const INJECT_BWD_WGSL: &str = "
struct P { cells:u32, c:u32, token:u32, anneal:f32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       profile: array<f32>;  // [cells]
@group(0) @binding(1) var<storage, read>       d_out:   array<f32>;  // [cells*c]
@group(0) @binding(2) var<storage, read_write> d_inj:   array<f32>;  // [vocab*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= p.c) { return; }
    var acc = 0.0;
    for (var cell: u32 = 0u; cell < p.cells; cell = cell + 1u) {
        acc = acc + profile[cell] * d_out[cell * p.c + ch];
    }
    let widx = p.token * p.c + ch;
    d_inj[widx] = d_inj[widx] + p.anneal * acc;
}
";

// ───── readout: average pooling (moment readout, §3.1) ─────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PoolParams {
    h: u32,
    w: u32,
    c: u32,
    pool: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

const AVGPOOL_FWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, pool:u32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       s:      array<f32>;  // [cells*c]
@group(0) @binding(1) var<storage, read_write> pooled: array<f32>;  // [ph*pw*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ph = p.h / p.pool;
    let pw = p.w / p.pool;
    let idx = gid.x;
    if (idx >= ph * pw * p.c) { return; }
    let ch = idx % p.c;
    let pcell = idx / p.c;
    let bi = pcell / pw;
    let bj = pcell % pw;
    let inv = 1.0 / f32(p.pool * p.pool);
    var acc = 0.0;
    for (var di: u32 = 0u; di < p.pool; di = di + 1u) {
        for (var dj: u32 = 0u; dj < p.pool; dj = dj + 1u) {
            let cell = (bi * p.pool + di) * p.w + (bj * p.pool + dj);
            acc = acc + s[cell * p.c + ch];
        }
    }
    pooled[idx] = acc * inv;
}
";

const AVGPOOL_BWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, pool:u32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       d_pooled: array<f32>;  // [ph*pw*c]
@group(0) @binding(1) var<storage, read_write> d_s:      array<f32>;  // [cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let ch = idx % p.c;
    let cell = idx / p.c;
    let i = cell / p.w;
    let j = cell % p.w;
    let pw = p.w / p.pool;
    let bi = i / p.pool;
    let bj = j / p.pool;
    let inv = 1.0 / f32(p.pool * p.pool);
    d_s[idx] = d_pooled[(bi * pw + bj) * p.c + ch] * inv;
}
";

macro_rules! stencil_kernel {
    ($name:ident, $wgsl:ident, $ro:expr) => {
        pub struct $name {
            pipeline: Arc<wgpu::ComputePipeline>,
            bgl: Arc<wgpu::BindGroupLayout>,
        }
        impl $name {
            #[must_use]
            pub fn new(ctx: &GpuContext) -> Self {
                let (pipeline, bgl) = build(ctx, stringify!($name), $wgsl, $ro);
                Self { pipeline, bgl }
            }
        }
    };
}

stencil_kernel!(ThermoFluxDivKernel, FLUX_FWD_WGSL, &[true, true, false]);
stencil_kernel!(ThermoFluxDivBackwardKernel, FLUX_BWD_WGSL, &[true, false, false]);
stencil_kernel!(ThermoDiffusionForwardKernel, DIFF_FWD_WGSL, &[true, true, true, true, false, false]);
stencil_kernel!(ThermoDiffusionBwdPreKernel, DIFF_BWD_PRE_WGSL, &[true, true, true, true, false, false]);
stencil_kernel!(ThermoDiffusionBwdParamKernel, DIFF_BWD_PARAM_WGSL, &[true, true, true, true, false, false, false]);
stencil_kernel!(ThermoDiffusionBwdDsKernel, DIFF_BWD_DS_WGSL, &[true, true, false]);
stencil_kernel!(ThermoInjectKernel, INJECT_FWD_WGSL, &[true, true, true, false]);
stencil_kernel!(ThermoInjectBackwardKernel, INJECT_BWD_WGSL, &[true, true, false]);
stencil_kernel!(ThermoAvgPoolKernel, AVGPOOL_FWD_WGSL, &[true, false]);
stencil_kernel!(ThermoAvgPoolBackwardKernel, AVGPOOL_BWD_WGSL, &[true, false]);

impl ThermoFluxDivKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        fx: &GpuBuffer,
        fy: &GpuBuffer,
        div: &GpuBuffer,
        h: u32,
        w: u32,
        nc: u32,
    ) {
        let p = StencilParams::new(h, w, nc);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[fx, fy, div], bytemuck::bytes_of(&p), (h * w * nc).div_ceil(WG));
    }
}

impl ThermoFluxDivBackwardKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_div: &GpuBuffer,
        d_fx: &GpuBuffer,
        d_fy: &GpuBuffer,
        h: u32,
        w: u32,
        nc: u32,
    ) {
        let p = StencilParams::new(h, w, nc);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_div, d_fx, d_fy], bytemuck::bytes_of(&p), (h * w * nc).div_ceil(WG));
    }
}

impl ThermoDiffusionForwardKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        k0: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        delta: &GpuBuffer,
        lap: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
    ) {
        let p = StencilParams::new(h, w, c);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, k0, a, b, delta, lap], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

impl ThermoDiffusionBwdPreKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_delta: &GpuBuffer,
        lap: &GpuBuffer,
        s: &GpuBuffer,
        gate: &GpuBuffer,
        gbuf: &GpuBuffer,
        c_t: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
    ) {
        let p = StencilParams::new(h, w, c);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_delta, lap, s, gate, gbuf, c_t], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

impl ThermoDiffusionBwdParamKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_delta: &GpuBuffer,
        lap: &GpuBuffer,
        s: &GpuBuffer,
        gate: &GpuBuffer,
        d_k0: &GpuBuffer,
        d_a: &GpuBuffer,
        d_b: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
    ) {
        let p = StencilParams::new(h, w, c);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_delta, lap, s, gate, d_k0, d_a, d_b], bytemuck::bytes_of(&p), c.div_ceil(WG));
    }
}

impl ThermoDiffusionBwdDsKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        gbuf: &GpuBuffer,
        c_t: &GpuBuffer,
        d_s: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
    ) {
        let p = StencilParams::new(h, w, c);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[gbuf, c_t, d_s], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

impl ThermoInjectKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        profile: &GpuBuffer,
        inj: &GpuBuffer,
        out: &GpuBuffer,
        cells: u32,
        c: u32,
        token: u32,
        anneal: f32,
    ) {
        let p = InjectParams { cells, c, token, anneal, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, profile, inj, out], bytemuck::bytes_of(&p), (cells * c).div_ceil(WG));
    }
}

impl ThermoInjectBackwardKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        profile: &GpuBuffer,
        d_out: &GpuBuffer,
        d_inj: &GpuBuffer,
        cells: u32,
        c: u32,
        token: u32,
        anneal: f32,
    ) {
        let p = InjectParams { cells, c, token, anneal, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[profile, d_out, d_inj], bytemuck::bytes_of(&p), c.div_ceil(WG));
    }
}

impl ThermoAvgPoolKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        pooled: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        pool: u32,
    ) {
        let p = PoolParams { h, w, c, pool, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        let groups = ((h / pool) * (w / pool) * c).div_ceil(WG);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, pooled], bytemuck::bytes_of(&p), groups);
    }
}

impl ThermoAvgPoolBackwardKernel {
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_pooled: &GpuBuffer,
        d_s: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        pool: u32,
    ) {
        let p = PoolParams { h, w, c, pool, _p0: 0, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_pooled, d_s], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}
