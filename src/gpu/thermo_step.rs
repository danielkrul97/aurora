//! On-device rollout **forward** for the thermodynamic LM: chains the
//! verified per-op kernels (`inject → K×step → readout`) over the T×K
//! unroll, keeping the field on-device (ping-pong buffers). This is the
//! orchestration layer; the backward pass + optimizer + `--gpu` follow.
//!
//! Two new glue kernels live here: `ExtractFlux` (pull the conserved-
//! channel flux slots out of the conv output) and `StepAssemble`
//! (`out = s + Δt·transport + Δt·anneal·diffusion`). Everything else
//! reuses kernels already parity-tested in `tests/thermo_gpu_parity.rs`.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::thermo_conv::{build, dispatch_n, ThermoConv1x1Kernel, ThermoConv3x3GeluKernel};
use super::thermo_stencil::{
    ThermoAvgPoolKernel, ThermoDiffusionForwardKernel, ThermoFeatureAssembleKernel,
    ThermoFluxDivKernel, ThermoInjectKernel, ThermoSpectralKernel,
};
use super::{GemmTransposeBKernel, GpuContext, VectorAddKernel};
use crate::thermo::field::{init_field, ThermoConfig};
use crate::thermo::model::{anneal_at, ThermoLM};

const WG: u32 = 64;

// ───── glue kernels ─────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ExtractParams {
    h: u32,
    w: u32,
    nc: u32,
    nfree: u32,
    cout: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

const EXTRACT_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, nfree:u32, cout:u32, _0:u32, _1:u32, _2:u32 };
@group(0) @binding(0) var<storage, read>       raw: array<f32>;  // [cells*cout]
@group(0) @binding(1) var<storage, read_write> fx:  array<f32>;  // [cells*nc]
@group(0) @binding(2) var<storage, read_write> fy:  array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    fx[idx] = raw[cell * p.cout + p.nfree + m];
    fy[idx] = raw[cell * p.cout + p.nfree + p.nc + m];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AssembleParams {
    h: u32,
    w: u32,
    c: u32,
    nc: u32,
    cout: u32,
    diff_on: u32,
    dt: f32,
    anneal: f32,
}

const ASSEMBLE_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nc:u32, cout:u32, diff_on:u32, dt:f32, anneal:f32 };
@group(0) @binding(0) var<storage, read>       s:    array<f32>;  // [cells*c]
@group(0) @binding(1) var<storage, read>       dv:   array<f32>;  // [cells*nc] divergence
@group(0) @binding(2) var<storage, read>       raw:  array<f32>;  // [cells*cout]
@group(0) @binding(3) var<storage, read>       diff: array<f32>;  // [cells*c]
@group(0) @binding(4) var<storage, read>       kind: array<u32>;  // [c] 1=conserved 0=free
@group(0) @binding(5) var<storage, read>       slot: array<u32>;  // [c] m or free-slot
@group(0) @binding(6) var<storage, read_write> out:  array<f32>;  // [cells*c]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.h * p.w * p.c) { return; }
    let ch = g % p.c;
    let cell = g / p.c;
    var v = s[g];
    if (p.diff_on == 1u) { v = v + p.dt * p.anneal * diff[g]; }
    if (kind[ch] == 1u) { v = v + p.dt * dv[cell * p.nc + slot[ch]]; }
    else                { v = v + p.dt * raw[cell * p.cout + slot[ch]]; }
    out[g] = v;
}
";

pub struct ExtractFluxKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}
pub struct StepAssembleKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ExtractFluxKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_extract", EXTRACT_WGSL, &[true, false, false]);
        Self { pipeline, bgl }
    }
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, raw: &GpuBuffer, fx: &GpuBuffer, fy: &GpuBuffer, h: u32, w: u32, nc: u32, nfree: u32, cout: u32) {
        let p = ExtractParams { h, w, nc, nfree, cout, _p0: 0, _p1: 0, _p2: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[raw, fx, fy], bytemuck::bytes_of(&p), (h * w * nc).div_ceil(WG));
    }
}

impl StepAssembleKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_assemble", ASSEMBLE_WGSL, &[true, true, true, true, true, true, false]);
        Self { pipeline, bgl }
    }
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, s: &GpuBuffer, div: &GpuBuffer, raw: &GpuBuffer, diff: &GpuBuffer, kind: &GpuBuffer, slot: &GpuBuffer, out: &GpuBuffer, h: u32, w: u32, c: u32, nc: u32, cout: u32, diff_on: u32, dt: f32, anneal: f32) {
        let p = AssembleParams { h, w, c, nc, cout, diff_on, dt, anneal };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[s, div, raw, diff, kind, slot, out], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

// ───── backward glue kernels ────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    n: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
    _p5: u32,
}

/// `y = scale · x` (scalar in push constant).
const SCALE_WGSL: &str = "
struct P { n:u32, scale:f32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32, _5:u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = p.scale * x[i];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DisassembleParams {
    h: u32,
    w: u32,
    c: u32,
    nc: u32,
    cout: u32,
    _p0: u32,
    dt: f32,
    _p1: u32,
}

/// Adjoint of `StepAssemble`'s transport term: scatter `Δt·d_out` into
/// `d_div` (conserved channels) and the free-delta slots of `d_raw`.
/// `d_raw` must be pre-zeroed (the flux slots are filled by `FluxScatter`).
const DISASSEMBLE_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nc:u32, cout:u32, _0:u32, dt:f32, _1:u32 };
@group(0) @binding(0) var<storage, read>       d_out: array<f32>;  // [cells*c]
@group(0) @binding(1) var<storage, read>       kind:  array<u32>;  // [c]
@group(0) @binding(2) var<storage, read>       slot:  array<u32>;  // [c]
@group(0) @binding(3) var<storage, read_write> d_div: array<f32>;  // [cells*nc]
@group(0) @binding(4) var<storage, read_write> d_raw: array<f32>;  // [cells*cout]
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= p.h * p.w * p.c) { return; }
    let ch = g % p.c;
    let cell = g / p.c;
    let v = p.dt * d_out[g];
    if (kind[ch] == 1u) { d_div[cell * p.nc + slot[ch]] = v; }
    else                { d_raw[cell * p.cout + slot[ch]] = v; }
}
";

/// Adjoint of `ExtractFlux`: write `d_fx`/`d_fy` into `d_raw`'s flux slots.
const FLUX_SCATTER_WGSL: &str = "
struct P { h:u32, w:u32, nc:u32, nfree:u32, cout:u32, _0:u32, _1:u32, _2:u32 };
@group(0) @binding(0) var<storage, read>       d_fx:  array<f32>;  // [cells*nc]
@group(0) @binding(1) var<storage, read>       d_fy:  array<f32>;
@group(0) @binding(2) var<storage, read_write> d_raw: array<f32>;  // [cells*cout]
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.h * p.w * p.nc) { return; }
    let m = idx % p.nc;
    let cell = idx / p.nc;
    d_raw[cell * p.cout + p.nfree + m] = d_fx[idx];
    d_raw[cell * p.cout + p.nfree + p.nc + m] = d_fy[idx];
}
";

/// `out[r*n + j] = a[r*n + j] + bias[j]` — bias broadcast over `rows`
/// (batched readout: B logit rows share one bias vector).
const BIAS_ROW_ADD_WGSL: &str = "
struct P { rows:u32, n:u32, _0:u32, _1:u32, _2:u32, _3:u32, _4:u32, _5:u32 };
@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;
var<push_constant> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.rows * p.n) { return; }
    out[idx] = a[idx] + bias[idx % p.n];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasRowAddParams {
    rows: u32,
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    _p4: u32,
    _p5: u32,
}

pub struct ThermoBiasRowAddKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoBiasRowAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_bias_row_add", BIAS_ROW_ADD_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, a: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer, rows: u32, n: u32) {
        let p = BiasRowAddParams { rows, n, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0, _p5: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[a, bias, out], bytemuck::bytes_of(&p), (rows * n).div_ceil(WG));
    }
}

pub struct ThermoScaleKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}
pub struct ThermoDisassembleKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}
pub struct ThermoFluxScatterKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl ThermoScaleKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_scale", SCALE_WGSL, &[true, false]);
        Self { pipeline, bgl }
    }
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, x: &GpuBuffer, y: &GpuBuffer, n: u32, scale: f32) {
        let p = ScaleParams { n, scale, _p0: 0, _p1: 0, _p2: 0, _p3: 0, _p4: 0, _p5: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[x, y], bytemuck::bytes_of(&p), n.div_ceil(WG));
    }
}

impl ThermoDisassembleKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_disassemble", DISASSEMBLE_WGSL, &[true, true, true, false, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, d_out: &GpuBuffer, kind: &GpuBuffer, slot: &GpuBuffer, d_div: &GpuBuffer, d_raw: &GpuBuffer, h: u32, w: u32, c: u32, nc: u32, cout: u32, dt: f32) {
        let p = DisassembleParams { h, w, c, nc, cout, _p0: 0, dt, _p1: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[d_out, kind, slot, d_div, d_raw], bytemuck::bytes_of(&p), (h * w * c).div_ceil(WG));
    }
}

impl ThermoFluxScatterKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_flux_scatter", FLUX_SCATTER_WGSL, &[true, true, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, d_fx: &GpuBuffer, d_fy: &GpuBuffer, d_raw: &GpuBuffer, h: u32, w: u32, nc: u32, nfree: u32, cout: u32) {
        let p = ExtractParams { h, w, nc, nfree, cout, _p0: 0, _p1: 0, _p2: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[d_fx, d_fy, d_raw], bytemuck::bytes_of(&p), (h * w * nc).div_ceil(WG));
    }
}

// ───── fused transport forward: extract + flux-div + assemble ───────────
//
// One thread per field element. Conserved channels compute the flux
// divergence directly from `raw`'s flux slots at this cell + the left/down
// neighbours (no fx/fy/div buffers); free channels read their delta slot;
// every channel adds the residual and the (anneal-scaled) diffusion delta.
// `diff` must be zero when diffusion is off (anneal·0 = 0).

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransportFwdParams {
    h: u32,
    w: u32,
    c: u32,
    nc: u32,
    cout: u32,
    nb: u32,
    dt: f32,
    anneal: f32,
}

const TRANSPORT_FWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nc:u32, cout:u32, nb:u32, dt:f32, anneal:f32 };
@group(0) @binding(0) var<storage, read>       s:    array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read>       raw:  array<f32>;  // [nb*cells*cout]
@group(0) @binding(2) var<storage, read>       diff: array<f32>;  // [nb*cells*c]
@group(0) @binding(3) var<storage, read>       kind: array<u32>;  // [c]
@group(0) @binding(4) var<storage, read>       slot: array<u32>;  // [c]
@group(0) @binding(5) var<storage, read_write> out:  array<f32>;  // [nb*cells*c]
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
    let nfree = p.c - p.nc;
    let cbase = cell * p.cout;
    let gbase = g * p.h * p.w;
    var v = s[idx] + p.dt * p.anneal * diff[idx];
    if (kind[ch] == 1u) {
        let m = slot[ch];
        var in_left = 0.0;
        if (j > 0u) { in_left = raw[(gbase + i * p.w + (j - 1u)) * p.cout + nfree + m]; }
        var fx_self = 0.0;
        if (j + 1u < p.w) { fx_self = raw[cbase + nfree + m]; }
        var in_down = 0.0;
        if (i > 0u) { in_down = raw[(gbase + (i - 1u) * p.w + j) * p.cout + nfree + p.nc + m]; }
        var fy_self = 0.0;
        if (i + 1u < p.h) { fy_self = raw[cbase + nfree + p.nc + m]; }
        v = v + p.dt * ((in_left - fx_self) + (in_down - fy_self));
    } else {
        v = v + p.dt * raw[cbase + slot[ch]];
    }
    out[idx] = v;
}
";

// ───── fused transport backward: disassemble + flux-div adjoint + scatter ─
//
// Adjoint of `TransportFwd`. One thread per `d_raw` slot. Writes every
// `cout` slot (free deltas, then fx/fy flux components), so `d_raw` needs
// no pre-zero. Reads only `d_out` (cell + right/bottom neighbours).

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransportBwdParams {
    h: u32,
    w: u32,
    c: u32,
    nc: u32,
    cout: u32,
    nb: u32,
    dt: f32,
    _pad: u32,
}

const TRANSPORT_BWD_WGSL: &str = "
struct P { h:u32, w:u32, c:u32, nc:u32, cout:u32, nb:u32, dt:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read>       d_out:   array<f32>;  // [nb*cells*c]
@group(0) @binding(1) var<storage, read>       cons_ch: array<u32>;  // [nc]
@group(0) @binding(2) var<storage, read>       free_ch: array<u32>;  // [nfree]
@group(0) @binding(3) var<storage, read_write> draw:    array<f32>;  // [nb*cells*cout]
var<push_constant> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.nb * p.h * p.w * p.cout) { return; }
    let co = idx % p.cout;
    let cell = idx / p.cout;
    let row = cell / p.w;
    let g = row / p.h;
    let i = row % p.h;
    let j = cell % p.w;
    let nfree = p.c - p.nc;
    let cbase = cell * p.c;
    let gbase = g * p.h * p.w;
    var val = 0.0;
    if (co < nfree) {
        let ch = free_ch[co];
        val = p.dt * d_out[cbase + ch];
    } else if (co < nfree + p.nc) {
        let m = co - nfree;
        let ch = cons_ch[m];
        if (j + 1u < p.w) {
            let self_v = p.dt * d_out[cbase + ch];
            let right = p.dt * d_out[(gbase + i * p.w + (j + 1u)) * p.c + ch];
            val = right - self_v;
        }
    } else {
        let m = co - nfree - p.nc;
        let ch = cons_ch[m];
        if (i + 1u < p.h) {
            let self_v = p.dt * d_out[cbase + ch];
            let bottom = p.dt * d_out[(gbase + (i + 1u) * p.w + j) * p.c + ch];
            val = bottom - self_v;
        }
    }
    draw[idx] = val;
}
";

pub struct StepTransportFwdKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}
pub struct StepTransportBwdKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
}

impl StepTransportFwdKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_transport_fwd", TRANSPORT_FWD_WGSL, &[true, true, true, true, true, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, s: &GpuBuffer, raw: &GpuBuffer, diff: &GpuBuffer, kind: &GpuBuffer, slot: &GpuBuffer, out: &GpuBuffer, h: u32, w: u32, c: u32, nc: u32, cout: u32, nb: u32, dt: f32, anneal: f32) {
        let p = TransportFwdParams { h, w, c, nc, cout, nb, dt, anneal };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[s, raw, diff, kind, slot, out], bytemuck::bytes_of(&p), (nb * h * w * c).div_ceil(WG));
    }
}

impl StepTransportBwdKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let (pipeline, bgl) = build(ctx, "thermo_transport_bwd", TRANSPORT_BWD_WGSL, &[true, true, true, false]);
        Self { pipeline, bgl }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_b(&self, ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, d_out: &GpuBuffer, cons_ch: &GpuBuffer, free_ch: &GpuBuffer, draw: &GpuBuffer, h: u32, w: u32, c: u32, nc: u32, cout: u32, nb: u32, dt: f32) {
        let p = TransportBwdParams { h, w, c, nc, cout, nb, dt, _pad: 0 };
        dispatch_n(ctx, enc, &self.pipeline, &self.bgl, &[d_out, cons_ch, free_ch, draw], bytemuck::bytes_of(&p), (nb * h * w * cout).div_ceil(WG));
    }
}

// ───── orchestrator (forward only) ──────────────────────────────────────

pub struct ThermoGpuForward {
    ctx: GpuContext,
    cfg: ThermoConfig,
    // kernels
    conv3: ThermoConv3x3GeluKernel,
    conv1: ThermoConv1x1Kernel,
    extract: ExtractFluxKernel,
    flux: ThermoFluxDivKernel,
    diff: ThermoDiffusionForwardKernel,
    assemble: StepAssembleKernel,
    pool: ThermoAvgPoolKernel,
    spectral: ThermoSpectralKernel,
    feat_assemble: ThermoFeatureAssembleKernel,
    gtb: GemmTransposeBKernel,
    vadd: VectorAddKernel,
    inject: ThermoInjectKernel,
    // params (device)
    w1: GpuBuffer,
    b1: GpuBuffer,
    w2: GpuBuffer,
    b2: GpuBuffer,
    k0: GpuBuffer,
    a: GpuBuffer,
    b: GpuBuffer,
    inj: GpuBuffer,
    profile: GpuBuffer,
    projw: GpuBuffer,
    bias: GpuBuffer,
    kind: GpuBuffer,
    slot: GpuBuffer,
    // scratch (device)
    field_a: GpuBuffer,
    field_b: GpuBuffer,
    h_post: GpuBuffer,
    raw: GpuBuffer,
    fx: GpuBuffer,
    fy: GpuBuffer,
    div: GpuBuffer,
    diff_delta: GpuBuffer,
    lap: GpuBuffer,
    pooled: GpuBuffer,
    basis: GpuBuffer,
    spec: GpuBuffer,
    feat: GpuBuffer,
    lraw: GpuBuffer,
    logits_t: GpuBuffer,
}

impl ThermoGpuForward {
    /// Moment and/or spectral-DCT readout (mirrors `ThermoLM::forward`).
    #[must_use]
    pub fn new(ctx: &GpuContext, model: &ThermoLM) -> Self {
        let cfg = model.config.clone();
        assert!(cfg.readout_moments || cfg.readout_spectral, "readout must use moments and/or spectral");
        let cells = cfg.cells();
        let c = cfg.c;
        let ch = cfg.conv_hidden;
        let cout = cfg.c_out();
        let nc = cfg.n_conserved();
        let sdim = cfg.spectral_dim();
        let rdim = cfg.readout_dim();

        let (kind, slot) = cfg.channel_decode();

        Self {
            cfg: cfg.clone(),
            ctx: ctx.clone(),
            conv3: ThermoConv3x3GeluKernel::new(ctx),
            conv1: ThermoConv1x1Kernel::new(ctx),
            extract: ExtractFluxKernel::new(ctx),
            flux: ThermoFluxDivKernel::new(ctx),
            diff: ThermoDiffusionForwardKernel::new(ctx),
            assemble: StepAssembleKernel::new(ctx),
            pool: ThermoAvgPoolKernel::new(ctx),
            spectral: ThermoSpectralKernel::new(ctx),
            feat_assemble: ThermoFeatureAssembleKernel::new(ctx),
            gtb: GemmTransposeBKernel::new(ctx),
            vadd: VectorAddKernel::new(ctx),
            inject: ThermoInjectKernel::new(ctx),
            w1: GpuBuffer::from_slice(ctx, "w1", &model.step.conv.w1),
            b1: GpuBuffer::from_slice(ctx, "b1", &model.step.conv.b1),
            w2: GpuBuffer::from_slice(ctx, "w2", &model.step.conv.w2),
            b2: GpuBuffer::from_slice(ctx, "b2", &model.step.conv.b2),
            k0: GpuBuffer::from_slice(ctx, "k0", &model.step.diffusion.k0),
            a: GpuBuffer::from_slice(ctx, "a", &model.step.diffusion.a),
            b: GpuBuffer::from_slice(ctx, "b", &model.step.diffusion.b),
            inj: GpuBuffer::from_slice(ctx, "inj", &model.inject.inj),
            profile: GpuBuffer::from_slice(ctx, "profile", &model.inject.profile),
            projw: GpuBuffer::from_slice(ctx, "projw", &model.readout.proj.w),
            bias: GpuBuffer::from_slice(ctx, "bias", &model.readout.bias),
            kind: GpuBuffer::from_u32_slice(ctx, "kind", &kind),
            slot: GpuBuffer::from_u32_slice(ctx, "slot", &slot),
            field_a: GpuBuffer::zeros(ctx, "field_a", cells * c),
            field_b: GpuBuffer::zeros(ctx, "field_b", cells * c),
            h_post: GpuBuffer::zeros(ctx, "h_post", cells * ch),
            raw: GpuBuffer::zeros(ctx, "raw", cells * cout),
            fx: GpuBuffer::zeros(ctx, "fx", cells * nc),
            fy: GpuBuffer::zeros(ctx, "fy", cells * nc),
            div: GpuBuffer::zeros(ctx, "div", cells * nc),
            diff_delta: GpuBuffer::zeros(ctx, "diff_delta", cells * c),
            lap: GpuBuffer::zeros(ctx, "lap", cells * c),
            pooled: GpuBuffer::zeros(ctx, "pooled", cfg.pooled_dim()),
            basis: GpuBuffer::from_slice(
                ctx,
                "basis",
                if model.readout.basis.is_empty() { &[0.0_f32] } else { &model.readout.basis },
            ),
            spec: GpuBuffer::zeros(ctx, "spec", sdim.max(1)),
            feat: GpuBuffer::zeros(ctx, "feat", rdim),
            lraw: GpuBuffer::zeros(ctx, "lraw", cfg.vocab),
            logits_t: GpuBuffer::zeros(ctx, "logits_t", cfg.vocab),
        }
    }

    /// Run the rollout; returns `logits[T*vocab]`.
    #[must_use]
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let t = tokens.len();
        let (cells, c, chid, cout, nc) =
            (cfg.cells(), cfg.c, cfg.conv_hidden, cfg.c_out(), cfg.n_conserved());
        let (h, w) = (cfg.h, cfg.w);
        let vocab = cfg.vocab;
        let logits = GpuBuffer::zeros(&self.ctx, "logits", t * vocab);

        // init field (warm) into field_a.
        self.field_a.upload(&self.ctx, &init_field(cfg));

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thermo gpu fwd") });

        let fields = [&self.field_a, &self.field_b];
        let mut cur = 0usize; // index of the live field

        let diff_on = u32::from(cfg.use_diffusion);
        for (ti, &tok) in tokens.iter().enumerate() {
            // inject (anneal 1.0 in training): fields[cur] → fields[1-cur]
            self.inject.dispatch_into(&self.ctx, &mut enc, fields[cur], &self.profile, &self.inj, fields[1 - cur], cells as u32, c as u32, tok, 1.0);
            cur = 1 - cur;

            for k in 0..cfg.k_steps {
                let anneal = anneal_at(k, cfg);
                let src = fields[cur];
                let dst = fields[1 - cur];
                self.conv3.dispatch_into(&self.ctx, &mut enc, src, &self.w1, &self.b1, &self.h_post, h as u32, w as u32, c as u32, chid as u32);
                self.conv1.dispatch_into(&self.ctx, &mut enc, &self.h_post, &self.w2, &self.b2, &self.raw, cells as u32, chid as u32, cout as u32);
                self.extract_flux(&mut enc, h, w, nc, cfg.c - nc, cout);
                self.flux.dispatch_into(&self.ctx, &mut enc, &self.fx, &self.fy, &self.div, h as u32, w as u32, nc as u32);
                if cfg.use_diffusion {
                    self.diff.dispatch_into(&self.ctx, &mut enc, src, &self.k0, &self.a, &self.b, &self.diff_delta, &self.lap, h as u32, w as u32, c as u32);
                }
                self.assemble_step(&mut enc, src, dst, h, w, c, nc, cout, diff_on, cfg.dt, anneal);
                cur = 1 - cur;
            }

            // readout: moments (avgpool) and/or spectral (DCT) → feat → gemmᵀ → +bias
            let rdim = cfg.readout_dim();
            if cfg.readout_spectral {
                if cfg.readout_moments {
                    self.pool.dispatch_into(&self.ctx, &mut enc, fields[cur], &self.pooled, h as u32, w as u32, c as u32, cfg.pool as u32);
                }
                let msq = (cfg.readout_modes * cfg.readout_modes) as u32;
                self.spectral.dispatch_b(&self.ctx, &mut enc, fields[cur], &self.basis, &self.spec, cells as u32, c as u32, msq, 1);
                let mdim = if cfg.readout_moments { cfg.pooled_dim() } else { 0 };
                self.feat_assemble.dispatch_b(&self.ctx, &mut enc, &self.pooled, &self.spec, &self.feat, mdim as u32, cfg.spectral_dim() as u32, 1);
                self.gtb.dispatch_into(&self.ctx, &mut enc, &self.feat, &self.projw, &self.lraw, 1, vocab as u32, rdim as u32);
            } else {
                self.pool.dispatch_into(&self.ctx, &mut enc, fields[cur], &self.pooled, h as u32, w as u32, c as u32, cfg.pool as u32);
                self.gtb.dispatch_into(&self.ctx, &mut enc, &self.pooled, &self.projw, &self.lraw, 1, vocab as u32, rdim as u32);
            }
            self.vadd.dispatch_into(&self.ctx, &mut enc, &self.lraw, &self.bias, &self.logits_t, vocab as u32);
            enc.copy_buffer_to_buffer(&self.logits_t.inner, 0, &logits.inner, (ti * vocab * 4) as u64, (vocab * 4) as u64);
        }

        self.ctx.queue.submit(Some(enc.finish()));
        logits.download(&self.ctx)
    }

    fn extract_flux(&self, enc: &mut wgpu::CommandEncoder, h: usize, w: usize, nc: usize, nfree: usize, cout: usize) {
        self.extract.dispatch_into(&self.ctx, enc, &self.raw, &self.fx, &self.fy, h as u32, w as u32, nc as u32, nfree as u32, cout as u32);
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_step(&self, enc: &mut wgpu::CommandEncoder, s: &GpuBuffer, out: &GpuBuffer, h: usize, w: usize, c: usize, nc: usize, cout: usize, diff_on: u32, dt: f32, anneal: f32) {
        self.assemble.dispatch_into(&self.ctx, enc, s, &self.div, &self.raw, &self.diff_delta, &self.kind, &self.slot, out, h as u32, w as u32, c as u32, nc as u32, cout as u32, diff_on, dt, anneal);
    }
}
