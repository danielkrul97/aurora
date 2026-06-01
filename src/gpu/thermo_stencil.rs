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
    /// batch grids stacked in the buffer (1 = unbatched).
    nb: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
}

impl StencilParams {
    fn new_b(h: u32, w: u32, c: u32, nb: u32) -> Self {
        Self { h, w, c, nb, _p1: 0, _p2: 0, _p3: 0, _p4: 0 }
    }
}

// ───── flux divergence (forward) ────────────────────────────────────────

const FLUX_FWD_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       fx:  array<f32>;  // [nb*cells*nc]
@group(0) @binding(1) var<storage, read>       fy:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dv:  array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let gbase = g * p.h * p.w;
    var in_left = 0.0;
    if (j > 0u) { in_left = fx[(gbase + i * p.w + (j - 1u)) * p.nc + m]; }
    var in_down = 0.0;
    if (i > 0u) { in_down = fy[(gbase + (i - 1u) * p.w + j) * p.nc + m]; }
    var fx_self = 0.0;
    if (j + 1u < p.w) { fx_self = fx[idx]; }
    var fy_self = 0.0;
    if (i + 1u < p.h) { fy_self = fy[idx]; }
    dv[idx] = (in_left - fx_self) + (in_down - fy_self);
}
";

// ───── flux divergence (backward / adjoint) ─────────────────────────────

const FLUX_BWD_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;  // d_div [nb*cells*nc]
@group(0) @binding(1) var<storage, read_write> dfx:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dfy:  array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let gbase = g * p.h * p.w;
    var v = 0.0;
    if (j + 1u < p.w) { v = -dd[idx] + dd[(gbase + i * p.w + (j + 1u)) * p.nc + m]; }
    dfx[idx] = v;
    var u = 0.0;
    if (i + 1u < p.h) { u = -dd[idx] + dd[(gbase + (i + 1u) * p.w + j) * p.nc + m]; }
    dfy[idx] = u;
}
";

// ───── temperature-gated diffusion (forward) ────────────────────────────

const DIFF_FWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       s:     array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read>       k0:    array<f32>;  // [c]
@group(0) @binding(2) var<storage, read>       a:     array<f32>;
@group(0) @binding(3) var<storage, read>       b:     array<f32>;
@group(0) @binding(4) var<storage, read_write> delta: array<f32>;  // [nb*cells*c]
@group(0) @binding(5) var<storage, read_write> lap:   array<f32>;  // [nb*cells*c]
var<push_constant> p: P;

const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.c) { return; }
    let cc = idx % p.c;
    let cell = idx / p.c;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let gbase = g * p.h * p.w;
    let center = s[idx];
    var lv = 0.0;
    if (i > 0u)        { lv = lv + s[(gbase + (i - 1u) * p.w + j) * p.c + cc] - center; }
    if (i + 1u < p.h)  { lv = lv + s[(gbase + (i + 1u) * p.w + j) * p.c + cc] - center; }
    if (j > 0u)        { lv = lv + s[(gbase + i * p.w + (j - 1u)) * p.c + cc] - center; }
    if (j + 1u < p.w)  { lv = lv + s[(gbase + i * p.w + (j + 1u)) * p.c + cc] - center; }
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
struct P { h:u32, w:u32, c:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;  // d_delta [nb*cells*c]
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
    if (idx >= p.nb * p.h * p.w * p.c) { return; }
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
struct P { h:u32, w:u32, c:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       dd:   array<f32>;
@group(0) @binding(1) var<storage, read>       lap:  array<f32>;
@group(0) @binding(2) var<storage, read>       s:    array<f32>;
@group(0) @binding(3) var<storage, read>       gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dk0:  array<f32>;
@group(0) @binding(5) var<storage, read_write> da:   array<f32>;
@group(0) @binding(6) var<storage, read_write> db:   array<f32>;
var<push_constant> p: P;
const CH_T: u32 = 4u;
// One workgroup per channel; 256 threads cooperatively reduce the nb*h*w
// batch×grid dim (shared-mem tree, 3 accumulators k0/a/b). The naive version
// ran c (=8) threads each over all ~25k cells — a top backward cost.
var<workgroup> sk: array<f32, 256>;
var<workgroup> sav: array<f32, 256>;
var<workgroup> sbv: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ch = wid.x;
    let tid = lid.x;
    let cells = p.nb * p.h * p.w;
    var sk0 = 0.0;
    var sa = 0.0;
    var sb = 0.0;
    if (ch < p.c) {
        let k0 = gate[ch];
        let a = gate[p.c + ch];
        let b = gate[2u * p.c + ch];
        var cell = tid;
        loop {
            if (cell >= cells) { break; }
            let t = s[cell * p.c + CH_T];
            let z = a + b * t;
            let sig = 1.0 / (1.0 + exp(-z));
            let sigp = sig * (1.0 - sig);
            let dl = dd[cell * p.c + ch] * lap[cell * p.c + ch];
            sk0 = sk0 + dl * sig;
            sa = sa + dl * k0 * sigp;
            sb = sb + dl * k0 * sigp * t;
            cell = cell + 256u;
        }
    }
    sk[tid] = sk0;
    sav[tid] = sa;
    sbv[tid] = sb;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sk[tid] = sk[tid] + sk[tid + stride];
            sav[tid] = sav[tid] + sav[tid + stride];
            sbv[tid] = sbv[tid] + sbv[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (tid == 0u && ch < p.c) {
        dk0[ch] = dk0[ch] + sk[0];
        da[ch] = da[ch] + sav[0];
        db[ch] = db[ch] + sbv[0];
    }
}
";

const DIFF_BWD_DS_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nb:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       gbuf: array<f32>;
@group(0) @binding(1) var<storage, read>       cT:   array<f32>;
@group(0) @binding(2) var<storage, read_write> ds:   array<f32>;
var<push_constant> p: P;
const CH_T: u32 = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.c) { return; }
    let cc = idx % p.c;
    let cell = idx / p.c;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let gbase = g * p.h * p.w;
    let center = gbuf[idx];
    var lg = 0.0;
    if (i > 0u)       { lg = lg + gbuf[(gbase + (i - 1u) * p.w + j) * p.c + cc] - center; }
    if (i + 1u < p.h) { lg = lg + gbuf[(gbase + (i + 1u) * p.w + j) * p.c + cc] - center; }
    if (j > 0u)       { lg = lg + gbuf[(gbase + i * p.w + (j - 1u)) * p.c + cc] - center; }
    if (j + 1u < p.w) { lg = lg + gbuf[(gbase + i * p.w + (j + 1u)) * p.c + cc] - center; }
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

// ───── batched token injection (per-grid token from a tokens buffer) ────
//
// Trainer path: `nb` grids stacked in `out`, each injecting its own token
// `tokens[tok_base + g]` (so one rollout step covers the whole batch).

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InjectBatchedParams {
    cells: u32,
    c: u32,
    nb: u32,
    tok_base: u32,
    anneal: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

const INJECT_FWD_B_WGSL: &str = "
struct P { cells:u32, c:u32, nb:u32, tok_base:u32, anneal:f32, _0:u32, _1:u32, _2:u32 };
@group(0) @binding(0) var<storage, read>       s:       array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read>       profile: array<f32>;  // [cells]
@group(0) @binding(2) var<storage, read>       inj:     array<f32>;  // [vocab*c]
@group(0) @binding(3) var<storage, read>       tokens:  array<u32>;  // [>= tok_base+nb]
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;  // [nb*cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellc = p.cells * p.c;
    if (idx >= p.nb * cellc) { return; }
    let g = idx / cellc;
    let local = idx % cellc;
    let ch = local % p.c;
    let cell = local / p.c;
    let tok = tokens[p.tok_base + g];
    out[idx] = s[idx] + p.anneal * profile[cell] * inj[tok * p.c + ch];
}
";

const INJECT_BWD_B_WGSL: &str = "
struct P { cells:u32, c:u32, nb:u32, tok_base:u32, anneal:f32, _0:u32, _1:u32, _2:u32 };
@group(0) @binding(0) var<storage, read>       profile: array<f32>;  // [cells]
@group(0) @binding(1) var<storage, read>       d_out:   array<f32>;  // [nb*cells*c]
@group(0) @binding(2) var<storage, read>       tokens:  array<u32>;
@group(0) @binding(3) var<storage, read_write> d_inj:   array<f32>;  // [vocab*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= p.c) { return; }
    let cellc = p.cells * p.c;
    // One thread owns column `ch` across all token rows → no cross-thread
    // race even when grids share a token (same thread accumulates both).
    for (var g: u32 = 0u; g < p.nb; g = g + 1u) {
        let tok = tokens[p.tok_base + g];
        let base = g * cellc;
        var acc = 0.0;
        for (var cell: u32 = 0u; cell < p.cells; cell = cell + 1u) {
            acc = acc + profile[cell] * d_out[base + cell * p.c + ch];
        }
        let widx = tok * p.c + ch;
        d_inj[widx] = d_inj[widx] + p.anneal * acc;
    }
}
";

pub struct ThermoInjectBatchedKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}
pub struct ThermoInjectBackwardBatchedKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoInjectBatchedKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_inject_fwd_b", INJECT_FWD_B_WGSL, &[true, true, true, true, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        profile: &GpuBuffer,
        inj: &GpuBuffer,
        tokens: &GpuBuffer,
        out: &GpuBuffer,
        cells: u32,
        c: u32,
        nb: u32,
        tok_base: u32,
        anneal: f32,
    ) {
        let p = InjectBatchedParams { cells, c, nb, tok_base, anneal, _p0: 0, _p1: 0, _p2: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, profile, inj, tokens, out], bytemuck::bytes_of(&p), (nb * cells * c).div_ceil(WG));
    }
}

impl ThermoInjectBackwardBatchedKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_inject_bwd_b", INJECT_BWD_B_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        profile: &GpuBuffer,
        d_out: &GpuBuffer,
        tokens: &GpuBuffer,
        d_inj: &GpuBuffer,
        cells: u32,
        c: u32,
        nb: u32,
        tok_base: u32,
        anneal: f32,
    ) {
        let p = InjectBatchedParams { cells, c, nb, tok_base, anneal, _p0: 0, _p1: 0, _p2: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[profile, d_out, tokens, d_inj], bytemuck::bytes_of(&p), c.div_ceil(WG));
    }
}

// ───── readout: average pooling (moment readout, §3.1) ─────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PoolParams {
    h: u32,
    w: u32,
    c: u32,
    pool: u32,
    nb: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

const AVGPOOL_FWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, pool:u32, nb:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       s:      array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read_write> pooled: array<f32>;  // [nb*ph*pw*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ph = p.h / p.pool;
    let pw = p.w / p.pool;
    let idx = gid.x;
    if (idx >= p.nb * ph * pw * p.c) { return; }
    let ch = idx % p.c;
    let pcell = idx / p.c;
    let g = pcell / (ph * pw);
    let lpcell = pcell % (ph * pw);
    let bi = lpcell / pw;
    let bj = lpcell % pw;
    let gbase = g * p.h * p.w;
    let inv = 1.0 / f32(p.pool * p.pool);
    var acc = 0.0;
    for (var di: u32 = 0u; di < p.pool; di = di + 1u) {
        for (var dj: u32 = 0u; dj < p.pool; dj = dj + 1u) {
            let cell = gbase + (bi * p.pool + di) * p.w + (bj * p.pool + dj);
            acc = acc + s[cell * p.c + ch];
        }
    }
    pooled[idx] = acc * inv;
}
";

const AVGPOOL_BWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, pool:u32, nb:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       d_pooled: array<f32>;  // [nb*ph*pw*c]
@group(0) @binding(1) var<storage, read_write> d_s:      array<f32>;  // [nb*cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.c) { return; }
    let ch = idx % p.c;
    let cell = idx / p.c;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let ph = p.h / p.pool;
    let pw = p.w / p.pool;
    let bi = i / p.pool;
    let bj = j / p.pool;
    let inv = 1.0 / f32(p.pool * p.pool);
    d_s[idx] = d_pooled[(g * ph * pw + bi * pw + bj) * p.c + ch] * inv;
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
        self.dispatch_b(ctx, encoder, fx, fy, div, h, w, nc, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        fx: &GpuBuffer,
        fy: &GpuBuffer,
        div: &GpuBuffer,
        h: u32,
        w: u32,
        nc: u32,
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, nc, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[fx, fy, div], bytemuck::bytes_of(&p), (nb * h * w * nc).div_ceil(WG));
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
        self.dispatch_b(ctx, encoder, d_div, d_fx, d_fy, h, w, nc, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_div: &GpuBuffer,
        d_fx: &GpuBuffer,
        d_fy: &GpuBuffer,
        h: u32,
        w: u32,
        nc: u32,
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, nc, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_div, d_fx, d_fy], bytemuck::bytes_of(&p), (nb * h * w * nc).div_ceil(WG));
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
        self.dispatch_b(ctx, encoder, s, k0, a, b, delta, lap, h, w, c, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, c, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, k0, a, b, delta, lap], bytemuck::bytes_of(&p), (nb * h * w * c).div_ceil(WG));
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
        self.dispatch_b(ctx, encoder, d_delta, lap, s, gate, gbuf, c_t, h, w, c, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, c, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_delta, lap, s, gate, gbuf, c_t], bytemuck::bytes_of(&p), (nb * h * w * c).div_ceil(WG));
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
        self.dispatch_b(ctx, encoder, d_delta, lap, s, gate, d_k0, d_a, d_b, h, w, c, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
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
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, c, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_delta, lap, s, gate, d_k0, d_a, d_b], bytemuck::bytes_of(&p), c); // one workgroup per channel
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
        self.dispatch_b(ctx, encoder, gbuf, c_t, d_s, h, w, c, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        gbuf: &GpuBuffer,
        c_t: &GpuBuffer,
        d_s: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        nb: u32,
    ) {
        let p = StencilParams::new_b(h, w, c, nb);
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[gbuf, c_t, d_s], bytemuck::bytes_of(&p), (nb * h * w * c).div_ceil(WG));
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
        self.dispatch_b(ctx, encoder, s, pooled, h, w, c, pool, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        pooled: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        pool: u32,
        nb: u32,
    ) {
        let p = PoolParams { h, w, c, pool, nb, _p1: 0, _p2: 0, _p3: 0 };
        let groups = (nb * (h / pool) * (w / pool) * c).div_ceil(WG);
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
        self.dispatch_b(ctx, encoder, d_pooled, d_s, h, w, c, pool, 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_pooled: &GpuBuffer,
        d_s: &GpuBuffer,
        h: u32,
        w: u32,
        c: u32,
        pool: u32,
        nb: u32,
    ) {
        let p = PoolParams { h, w, c, pool, nb, _p1: 0, _p2: 0, _p3: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_pooled, d_s], bytemuck::bytes_of(&p), (nb * h * w * c).div_ceil(WG));
    }
}

// ───── readout: spectral DCT modes (§3.2) + feature assembly ───────────
//
// Mirrors `crate::thermo::readout::spectral_coeffs` and the spectral branch
// of `readout_backward`. The DCT basis is the CPU-built `ReadoutWeights.basis`
// uploaded verbatim (so the GPU never recomputes it — no basis drift). The
// spectral coeff layout is channel-major `[ch*m + mode]`, matching the CPU
// feature order `[moments | spectral]`; assemble/disassemble interleave the
// two contiguous feature blocks into the `[nb, readout_dim]` matrix the proj
// matmul consumes. No atomics: every output element is owned by one thread.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectralParams {
    cells: u32,
    c: u32,
    /// modes² (total spectral modes per channel).
    m: u32,
    nb: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FeatureParams {
    mdim: u32,
    sdim: u32,
    nb: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
    _p5: u32,
}

const SPECTRAL_FWD_WGSL: &str = "
struct P { cells:u32, c:u32, m:u32, nb:u32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       s:     array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read>       basis: array<f32>;  // [m*cells]
@group(0) @binding(2) var<storage, read_write> spec:  array<f32>;  // [nb*c*m] channel-major
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cm = p.c * p.m;
    if (idx >= p.nb * cm) { return; }
    let g = idx / cm;
    let local = idx % cm;
    let ch = local / p.m;
    let mode = local % p.m;
    let sbase = g * p.cells * p.c;
    let bbase = mode * p.cells;
    var acc = 0.0;
    for (var cell: u32 = 0u; cell < p.cells; cell = cell + 1u) {
        acc = acc + basis[bbase + cell] * s[sbase + cell * p.c + ch];
    }
    spec[idx] = acc;
}
";

const SPECTRAL_BWD_WGSL: &str = "
struct P { cells:u32, c:u32, m:u32, nb:u32, _0:u32, _1:u32, _2:u32, _3:u32 };
@group(0) @binding(0) var<storage, read>       d_spec: array<f32>;  // [nb*c*m]
@group(0) @binding(1) var<storage, read>       basis:  array<f32>;  // [m*cells]
@group(0) @binding(2) var<storage, read_write> d_s:    array<f32>;  // [nb*cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cellc = p.cells * p.c;
    if (idx >= p.nb * cellc) { return; }
    let g = idx / cellc;
    let local = idx % cellc;
    let cell = local / p.c;
    let ch = local % p.c;
    let dbase = g * p.c * p.m + ch * p.m;
    var acc = 0.0;
    for (var mode: u32 = 0u; mode < p.m; mode = mode + 1u) {
        acc = acc + basis[mode * p.cells + cell] * d_spec[dbase + mode];
    }
    d_s[idx] = acc;
}
";

const FEATURE_ASSEMBLE_WGSL: &str = "
struct P { mdim:u32, sdim:u32, nb:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       moments: array<f32>;  // [nb*mdim]
@group(0) @binding(1) var<storage, read>       spec:    array<f32>;  // [nb*sdim]
@group(0) @binding(2) var<storage, read_write> feat:    array<f32>;  // [nb*(mdim+sdim)]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rdim = p.mdim + p.sdim;
    let idx = gid.x;
    if (idx >= p.nb * rdim) { return; }
    let g = idx / rdim;
    let f = idx % rdim;
    if (f < p.mdim) {
        feat[idx] = moments[g * p.mdim + f];
    } else {
        feat[idx] = spec[g * p.sdim + (f - p.mdim)];
    }
}
";

const FEATURE_DISASSEMBLE_WGSL: &str = "
struct P { mdim:u32, sdim:u32, nb:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32 };
@group(0) @binding(0) var<storage, read>       d_feat: array<f32>;  // [nb*(mdim+sdim)]
@group(0) @binding(1) var<storage, read_write> d_mom:  array<f32>;  // [nb*mdim]
@group(0) @binding(2) var<storage, read_write> d_spec: array<f32>;  // [nb*sdim]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rdim = p.mdim + p.sdim;
    let idx = gid.x;
    if (idx >= p.nb * rdim) { return; }
    let g = idx / rdim;
    let f = idx % rdim;
    if (f < p.mdim) {
        d_mom[g * p.mdim + f] = d_feat[idx];
    } else {
        d_spec[g * p.sdim + (f - p.mdim)] = d_feat[idx];
    }
}
";

stencil_kernel!(ThermoSpectralKernel, SPECTRAL_FWD_WGSL, &[true, true, false]);
stencil_kernel!(ThermoSpectralBackwardKernel, SPECTRAL_BWD_WGSL, &[true, true, false]);
stencil_kernel!(ThermoFeatureAssembleKernel, FEATURE_ASSEMBLE_WGSL, &[true, true, false]);
stencil_kernel!(ThermoFeatureDisassembleKernel, FEATURE_DISASSEMBLE_WGSL, &[true, false, false]);

impl ThermoSpectralKernel {
    /// `spec[g, ch*m+mode] = Σ_cell basis[mode, cell]·s[g, cell, ch]` over `nb` grids.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        s: &GpuBuffer,
        basis: &GpuBuffer,
        spec: &GpuBuffer,
        cells: u32,
        c: u32,
        m: u32,
        nb: u32,
    ) {
        let p = SpectralParams { cells, c, m, nb, _p1: 0, _p2: 0, _p3: 0, _p4: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[s, basis, spec], bytemuck::bytes_of(&p), (nb * c * m).div_ceil(WG));
    }
}

impl ThermoSpectralBackwardKernel {
    /// Adjoint of the basis projection: `d_s[g, cell, ch] = Σ_mode basis[mode, cell]·d_spec[g, ch*m+mode]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_spec: &GpuBuffer,
        basis: &GpuBuffer,
        d_s: &GpuBuffer,
        cells: u32,
        c: u32,
        m: u32,
        nb: u32,
    ) {
        let p = SpectralParams { cells, c, m, nb, _p1: 0, _p2: 0, _p3: 0, _p4: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_spec, basis, d_s], bytemuck::bytes_of(&p), (nb * cells * c).div_ceil(WG));
    }
}

impl ThermoFeatureAssembleKernel {
    /// Interleave `[moments(mdim) | spec(sdim)]` per grid into `feat[nb, mdim+sdim]`.
    /// `mdim == 0` selects the spectral-only readout (moments buffer untouched).
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        moments: &GpuBuffer,
        spec: &GpuBuffer,
        feat: &GpuBuffer,
        mdim: u32,
        sdim: u32,
        nb: u32,
    ) {
        let p = FeatureParams { mdim, sdim, nb, _p1: 0, _p2: 0, _p3: 0, _p4: 0, _p5: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[moments, spec, feat], bytemuck::bytes_of(&p), (nb * (mdim + sdim)).div_ceil(WG));
    }
}

impl ThermoFeatureDisassembleKernel {
    /// Split `d_feat[nb, mdim+sdim]` back into `d_mom(mdim)` and `d_spec(sdim)`.
    pub fn dispatch_b(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_feat: &GpuBuffer,
        d_mom: &GpuBuffer,
        d_spec: &GpuBuffer,
        mdim: u32,
        sdim: u32,
        nb: u32,
    ) {
        let p = FeatureParams { mdim, sdim, nb, _p1: 0, _p2: 0, _p3: 0, _p4: 0, _p5: 0 };
        dispatch_n(ctx, encoder, &self.pipeline, &self.bgl, &[d_feat, d_mom, d_spec], bytemuck::bytes_of(&p), (nb * (mdim + sdim)).div_ceil(WG));
    }
}
