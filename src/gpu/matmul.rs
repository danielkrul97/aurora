//! Dense GEMM kernels: forward `C = A @ B` and `C = A @ Bᵀ`.
//!
//! Tile-based (16×16 workgroup). Both A and B are row-major. The
//! transpose variant lets backward share the same kernel family
//! (`d_x = d_y @ Wᵀ` in linear layers).

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

// Tile geometry — 64×64 workgroup output tile, 16-wide K tile, 4×4
// register tile per thread (256 threads/workgroup = 16×16). Each thread
// computes 16 output elements out of its register tile, hiding shared-
// memory latency behind FMA throughput. ~5–10× the per-thread arithmetic
// of the previous 1-element-per-thread layout.
const WG_M: u32 = 64;
const WG_N: u32 = 64;

const GEMM_WGSL: &str = "
struct G {
    m: u32, n: u32, k: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
var<push_constant> p: G;

var<workgroup> ta: array<f32, 1024>;  // [WG_M=64, WG_K=16]
var<workgroup> tb: array<f32, 1024>;  // [WG_K=16, WG_N=64]

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wgid.y * 64u + ty * 4u;
    let col0 = wgid.x * 64u + tx * 4u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let n_tiles = (p.k + 15u) / 16u;
    for (var t_idx: u32 = 0u; t_idx < n_tiles; t_idx = t_idx + 1u) {
        let k_off = t_idx * 16u;

        // Cooperatively load 64×16 A-tile (1024 floats, 4 per thread).
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let r = flat / 16u;
            let kk = flat % 16u;
            let g_row = wgid.y * 64u + r;
            let g_col = k_off + kk;
            if (g_row < p.m && g_col < p.k) {
                ta[r * 16u + kk] = a[g_row * p.k + g_col];
            } else {
                ta[r * 16u + kk] = 0.0;
            }
        }
        // Cooperatively load 16×64 B-tile.
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let kk = flat / 64u;
            let cc = flat % 64u;
            let g_row = k_off + kk;
            let g_col = wgid.x * 64u + cc;
            if (g_row < p.k && g_col < p.n) {
                tb[kk * 64u + cc] = b[g_row * p.n + g_col];
            } else {
                tb[kk * 64u + cc] = 0.0;
            }
        }
        workgroupBarrier();

        // 4×4 register-tile multiply-accumulate.
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = ta[(ty * 4u + 0u) * 16u + kk];
            let a1 = ta[(ty * 4u + 1u) * 16u + kk];
            let a2 = ta[(ty * 4u + 2u) * 16u + kk];
            let a3 = ta[(ty * 4u + 3u) * 16u + kk];
            let b0 = tb[kk * 64u + tx * 4u + 0u];
            let b1 = tb[kk * 64u + tx * 4u + 1u];
            let b2 = tb[kk * 64u + tx * 4u + 2u];
            let b3 = tb[kk * 64u + tx * 4u + 3u];
            acc[0]  = acc[0]  + a0 * b0;
            acc[1]  = acc[1]  + a0 * b1;
            acc[2]  = acc[2]  + a0 * b2;
            acc[3]  = acc[3]  + a0 * b3;
            acc[4]  = acc[4]  + a1 * b0;
            acc[5]  = acc[5]  + a1 * b1;
            acc[6]  = acc[6]  + a1 * b2;
            acc[7]  = acc[7]  + a1 * b3;
            acc[8]  = acc[8]  + a2 * b0;
            acc[9]  = acc[9]  + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let r = row0 + i;
            let cc = col0 + j;
            if (r < p.m && cc < p.n) {
                c[r * p.n + cc] = acc[i * 4u + j];
            }
        }
    }
}
";

const GEMM_TRANSPOSE_B_WGSL: &str = "
struct G {
    m: u32, n: u32, k: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
var<push_constant> p: G;

var<workgroup> ta: array<f32, 1024>;
var<workgroup> tb: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wgid.y * 64u + ty * 4u;
    let col0 = wgid.x * 64u + tx * 4u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let n_tiles = (p.k + 15u) / 16u;
    for (var t_idx: u32 = 0u; t_idx < n_tiles; t_idx = t_idx + 1u) {
        let k_off = t_idx * 16u;

        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let r = flat / 16u;
            let kk = flat % 16u;
            let g_row = wgid.y * 64u + r;
            let g_col = k_off + kk;
            if (g_row < p.m && g_col < p.k) {
                ta[r * 16u + kk] = a[g_row * p.k + g_col];
            } else {
                ta[r * 16u + kk] = 0.0;
            }
        }
        // B is [n, k] row-major; B^T[k, n] requires reading b[col * k + kk].
        for (var l: u32 = 0u; l < 4u; l = l + 1u) {
            let flat = tid * 4u + l;
            let kk = flat / 64u;
            let cc = flat % 64u;
            let g_row = k_off + kk;
            let g_col = wgid.x * 64u + cc;
            if (g_row < p.k && g_col < p.n) {
                tb[kk * 64u + cc] = b[g_col * p.k + g_row];
            } else {
                tb[kk * 64u + cc] = 0.0;
            }
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = ta[(ty * 4u + 0u) * 16u + kk];
            let a1 = ta[(ty * 4u + 1u) * 16u + kk];
            let a2 = ta[(ty * 4u + 2u) * 16u + kk];
            let a3 = ta[(ty * 4u + 3u) * 16u + kk];
            let b0 = tb[kk * 64u + tx * 4u + 0u];
            let b1 = tb[kk * 64u + tx * 4u + 1u];
            let b2 = tb[kk * 64u + tx * 4u + 2u];
            let b3 = tb[kk * 64u + tx * 4u + 3u];
            acc[0]  = acc[0]  + a0 * b0;
            acc[1]  = acc[1]  + a0 * b1;
            acc[2]  = acc[2]  + a0 * b2;
            acc[3]  = acc[3]  + a0 * b3;
            acc[4]  = acc[4]  + a1 * b0;
            acc[5]  = acc[5]  + a1 * b1;
            acc[6]  = acc[6]  + a1 * b2;
            acc[7]  = acc[7]  + a1 * b3;
            acc[8]  = acc[8]  + a2 * b0;
            acc[9]  = acc[9]  + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let r = row0 + i;
            let cc = col0 + j;
            if (r < p.m && cc < p.n) {
                c[r * p.n + cc] = acc[i * 4u + j];
            }
        }
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

pub struct GemmKernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl GemmKernel {
    fn build(ctx: &GpuContext, wgsl: &str, label: &str) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
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
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bgl),
        }
    }

    /// Forward `C = A @ B`. A is [m × k], B is [k × n], C is [m × n].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        Self::build(ctx, GEMM_WGSL, "aurora gemm")
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let p = GemmParams {
            m,
            n,
            k,
            _pad: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gemm bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.inner.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora gemm pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let groups_x = n.div_ceil(WG_N);
        let groups_y = m.div_ceil(WG_M);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gemm bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let p = GemmParams { m, n, k, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        let groups_x = n.div_ceil(WG_N);
        let groups_y = m.div_ceil(WG_M);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
}

/// `C = A @ Bᵀ`. A is [m × k], B is [n × k] (row-major), C is [m × n].
pub struct GemmTransposeBKernel {
    inner: GemmKernel,
}

impl GemmTransposeBKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            inner: GemmKernel::build(ctx, GEMM_TRANSPOSE_B_WGSL, "aurora gemm_transpose_b"),
        }
    }

    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) {
        self.inner.dispatch_into(ctx, encoder, a, b, c, m, n, k);
    }

    pub fn build_bind(
        &self,
        ctx: &GpuContext,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
    ) -> wgpu::BindGroup {
        self.inner.build_bind(ctx, a, b, c)
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        m: u32,
        n: u32,
        k: u32,
    ) {
        self.inner.dispatch_with_bind(pass, bind, m, n, k);
    }

    pub(crate) fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.inner.pipeline
    }

    /// Public pipeline accessor for tests.
    #[must_use]
    pub fn pipeline_pub(&self) -> &wgpu::ComputePipeline {
        &self.inner.pipeline
    }
}

/// CPU reference for `C = A @ B` where A:[m×k], B:[k×n], C:[m×n].
#[must_use]
pub fn cpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// CPU reference for `C = A @ Bᵀ` where A:[m×k], B:[n×k], C:[m×n].
#[must_use]
pub fn cpu_gemm_transpose_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ───── Half-precision weight (B operand) variants ────────────────────────
//
// Same tile geometry; only the B global loads change to f16 (converted
// to f32 at tile-load time, so the inner FMA loop and the f32
// accumulation are bit-identical in structure). This halves the
// dominant DRAM traffic — the weight matrix — which is what the thermo
// roofline lesson says actually moves the needle. A full f16-compute
// (vec2<f16> FMA) variant is a follow-up once this is measured.

fn f16_b_source(wgsl: &str) -> String {
    let s = wgsl
        .replace(
            "@group(0) @binding(1) var<storage, read>       b: array<f32>;",
            "@group(0) @binding(1) var<storage, read>       b: array<f16>;",
        )
        .replace(
            "tb[kk * 64u + cc] = b[g_row * p.n + g_col];",
            "tb[kk * 64u + cc] = f32(b[g_row * p.n + g_col]);",
        )
        .replace(
            "tb[kk * 64u + cc] = b[g_col * p.k + g_row];",
            "tb[kk * 64u + cc] = f32(b[g_col * p.k + g_row]);",
        );
    format!("enable f16;\n{s}")
}

impl GemmKernel {
    /// `C = A @ B` with B stored as f16. Requires `SHADER_F16`.
    #[must_use]
    pub fn new_f16b(ctx: &GpuContext) -> Self {
        Self::build(ctx, &f16_b_source(GEMM_WGSL), "aurora gemm f16b")
    }
}

impl GemmTransposeBKernel {
    /// `C = A @ Bᵀ` with B stored as f16. Requires `SHADER_F16`.
    #[must_use]
    pub fn new_f16b(ctx: &GpuContext) -> Self {
        Self {
            inner: GemmKernel::build(
                ctx,
                &f16_b_source(GEMM_TRANSPOSE_B_WGSL),
                "aurora gemm_transpose_b f16b",
            ),
        }
    }
}

// On-device f32 → f16 cast for refreshing weight shadows after each
// optimizer step.
//
// NOTE: the hardware `f16(x)` conversion truncates (round-toward-zero)
// on NVIDIA/Naga — ~half of all values differ from round-to-nearest by
// one f16 ULP. Harmless for training: the f32 masters are re-cast every
// step (no accumulation) and the ~1e-4 relative shrink is far below
// weight-decay scale. Tests therefore compare with an RMS-normalized
// absolute tolerance, not bitwise against an RNE reference.
const CAST_F16_WGSL: &str = "
enable f16;
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f16>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = f16(x[i]);
}
";

pub struct CastF16Kernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl CastF16Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora cast f16"),
                source: wgpu::ShaderSource::Wgsl(CAST_F16_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora cast f16 bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora cast f16 pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora cast f16"),
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

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, y: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora cast f16 bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = GemmParams { m: n, n: 0, k: 0, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}

// f16 → f32 readback cast — debugging/tests only (shadows are otherwise
// write-only from the host's perspective).
const CAST_F32_WGSL: &str = "
enable f16;
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f16>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
var<push_constant> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = f32(x[i]);
}
";

pub struct CastF32Kernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bgl: Arc<wgpu::BindGroupLayout>,
}

impl CastF32Kernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora cast f32"),
                source: wgpu::ShaderSource::Wgsl(CAST_F32_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora cast f32 bgl"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora cast f32 pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora cast f32"),
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

    pub fn build_bind(&self, ctx: &GpuContext, x: &GpuBuffer, y: &GpuBuffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora cast f32 bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.inner.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch_with_bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
    ) {
        let p = GemmParams { m: n, n: 0, k: 0, _pad: 0 };
        pass.set_bind_group(0, bind, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&p));
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }
}
