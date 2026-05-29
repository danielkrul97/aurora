//! Muon optimizer kernels for GPU. Per-matrix Newton–Schulz5
//! orthogonalization of momentum-modified gradients. Used by
//! `DecoderTransformerGpu::step_muon_hybrid` together with AdamW for
//! the 1-D / embedding slice.
//!
//! All kernels follow the same patterns as `src/gpu/optimizer.rs` and
//! `src/gpu/reduce.rs`: f32 storage buffers, push-constants for scalar
//! arguments, `build_bind` + `dispatch_with_bind` for cached re-use
//! inside the step orchestrator.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

// ───── 1. Momentum + Nesterov: update = g + β·m_new, where m_new = β·m + g ─

const MUON_MOMENTUM_WGSL: &str = "
struct P { n: u32, nesterov: u32, beta: f32, _p: f32 };
@group(0) @binding(0) var<storage, read>       grads: array<f32>;
@group(0) @binding(1) var<storage, read_write> m: array<f32>;
@group(0) @binding(2) var<storage, read_write> update: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = grads[i];
    let mn = p.beta * m[i] + g;
    m[i] = mn;
    if (p.nesterov != 0u) {
        update[i] = g + p.beta * mn;
    } else {
        update[i] = mn;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MomentumParams {
    n: u32,
    nesterov: u32,
    beta: f32,
    _p: f32,
}

pub struct MuonMomentumKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonMomentumKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_momentum"),
            source: wgpu::ShaderSource::Wgsl(MUON_MOMENTUM_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_momentum bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, false), storage_entry(2, false)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_momentum pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_momentum"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        grads: &GpuBuffer,
        m: &GpuBuffer,
        update: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_momentum bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grads.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: m.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: update.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        nesterov: bool,
        beta: f32,
    ) {
        let p = MomentumParams { n, nesterov: u32::from(nesterov), beta, _p: 0.0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

// ───── 2. Frobenius-bound inverse: result[1] = 1 / (sqrt(sum_partials) + eps) ─
//
// Single-workgroup reduce of `partials[0..n]`. Mirrors GradNormFinalize
// but writes 1/(norm+eps) into result[1] instead of a clip factor —
// downstream scale-by-buffer uses index 1 to multiply the matrix.

const MUON_FROB_INV_WGSL: &str = "
struct P { n: u32, _p0: u32, eps: f32, _p1: f32 };
@group(0) @binding(0) var<storage, read>       partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
var<push_constant> p: P;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var local: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= p.n) { break; }
        local = local + partials[i];
        i = i + 256u;
    }
    sh[tid] = local;
    workgroupBarrier();
    var s: u32 = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) {
        let norm = sqrt(max(sh[0], 0.0));
        result[0] = norm;
        result[1] = 1.0 / (norm + p.eps);
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FrobInvParams {
    n: u32,
    _p0: u32,
    eps: f32,
    _p1: f32,
}

pub struct MuonFrobInvKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonFrobInvKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_frob_inv"),
            source: wgpu::ShaderSource::Wgsl(MUON_FROB_INV_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_frob_inv bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_frob_inv pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_frob_inv"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        partials: &GpuBuffer,
        result: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_frob_inv bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: partials.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(&self, pass: &mut wgpu::ComputePass<'_>, bind: &wgpu::BindGroup, n: u32) {
        let p = FrobInvParams { n, _p0: 0, eps: 1e-7, _p1: 0.0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(1, 1, 1);
    }
}

// ───── 3. Transpose-aware load/store between [rows × cols] and [m × n] ─
//
// When rows ≤ cols (matrix already wide), this is a straight copy.
// When rows > cols, we transpose into wide orientation [cols × rows].
// `direction = 0` writes src[r,c] → dst[r,c] or dst[c,r]; `direction = 1`
// is the inverse (dst[r,c] = src[r,c] or src[c,r]). Same kernel, sense
// flipped by the push constant.

const MUON_LOAD_STORE_WGSL: &str = "
struct P { rows: u32, cols: u32, transposed: u32, store: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.rows * p.cols;
    if (i >= n) { return; }
    let r = i / p.cols;
    let c = i % p.cols;
    // Source layout is always [rows × cols]; destination is either
    // [rows × cols] (transposed == 0) or [cols × rows] (transposed == 1).
    if (p.store == 0u) {
        // load: read native, write wide orientation
        if (p.transposed != 0u) {
            dst[c * p.rows + r] = src[r * p.cols + c];
        } else {
            dst[r * p.cols + c] = src[r * p.cols + c];
        }
    } else {
        // store: read wide orientation, write native
        if (p.transposed != 0u) {
            dst[r * p.cols + c] = src[c * p.rows + r];
        } else {
            dst[r * p.cols + c] = src[r * p.cols + c];
        }
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LoadStoreParams {
    rows: u32,
    cols: u32,
    transposed: u32,
    store: u32,
}

pub struct MuonLoadStoreKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonLoadStoreKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_load_store"),
            source: wgpu::ShaderSource::Wgsl(MUON_LOAD_STORE_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_load_store bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_load_store pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_load_store"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(&self, ctx: &GpuContext, src: &GpuBuffer, dst: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_load_store bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        rows: u32,
        cols: u32,
        transposed: bool,
        store: bool,
    ) {
        let p = LoadStoreParams {
            rows,
            cols,
            transposed: u32::from(transposed),
            store: u32::from(store),
        };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups((rows * cols).div_ceil(256), 1, 1);
    }
}

// ───── 4. NS quintic combine: B = b·A + c·A_sq (elementwise [m × m]) ─

const MUON_COMBINE_B_WGSL: &str = "
struct P { n: u32, _p: u32, b_coef: f32, c_coef: f32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       a_sq: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    b[i] = p.b_coef * a[i] + p.c_coef * a_sq[i];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineBParams {
    n: u32,
    _p: u32,
    b_coef: f32,
    c_coef: f32,
}

pub struct MuonCombineBKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonCombineBKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_combine_b"),
            source: wgpu::ShaderSource::Wgsl(MUON_COMBINE_B_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_combine_b bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_combine_b pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_combine_b"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        a: &GpuBuffer,
        a_sq: &GpuBuffer,
        b: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_combine_b bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_sq.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = CombineBParams { n, _p: 0, b_coef: -4.7750, c_coef: 2.0315 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

// ───── 5. NS quintic step: X = a·X + B·X (in-place; BX already computed) ─

const MUON_STEP_X_WGSL: &str = "
struct P { n: u32, _p: u32, a_coef: f32, _q: f32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       bx: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = p.a_coef * x[i] + bx[i];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StepXParams {
    n: u32,
    _p: u32,
    a_coef: f32,
    _q: f32,
}

pub struct MuonStepXKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonStepXKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_step_x"),
            source: wgpu::ShaderSource::Wgsl(MUON_STEP_X_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_step_x bgl"),
            entries: &[storage_entry(0, false), storage_entry(1, true)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_step_x pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_step_x"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, bx: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_step_x bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bx.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = StepXParams { n, _p: 0, a_coef: 3.4445, _q: 0.0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

// ───── 6. Final param update: params -= lr·scale·update (+ wd) ─

const MUON_APPLY_WGSL: &str = "
struct P { n: u32, apply_wd: u32, lr_scale: f32, lr_wd: f32 };
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read>       update: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    var step = p.lr_scale * update[i];
    if (p.apply_wd != 0u) {
        step = step + p.lr_wd * params[i];
    }
    params[i] = params[i] - step;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ApplyParams {
    n: u32,
    apply_wd: u32,
    lr_scale: f32,
    lr_wd: f32,
}

pub struct MuonApplyKernel {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl MuonApplyKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora muon_apply"),
            source: wgpu::ShaderSource::Wgsl(MUON_APPLY_WGSL.into()),
        });
        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora muon_apply bgl"),
            entries: &[storage_entry(0, false), storage_entry(1, true)],
        });
        let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora muon_apply pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aurora muon_apply"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline: Arc::new(pipeline), bgl: Arc::new(bgl) }
    }

    pub fn build_bind(&self, ctx: &GpuContext, params: &GpuBuffer, update: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora muon_apply bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: update.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        apply_wd: bool,
        lr: f32,
        scale: f32,
        weight_decay: f32,
    ) {
        let p = ApplyParams {
            n,
            apply_wd: u32::from(apply_wd),
            lr_scale: lr * scale,
            lr_wd: lr * weight_decay,
        };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
