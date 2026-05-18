//! Dense GEMM kernels: forward `C = A @ B` and `C = A @ Bᵀ`.
//!
//! Tile-based (16×16 workgroup). Both A and B are row-major. The
//! transpose variant lets backward share the same kernel family
//! (`d_x = d_y @ Wᵀ` in linear layers).

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::softmax_ce::storage_entry;
use super::GpuContext;

const TILE: u32 = 16;

const GEMM_WGSL: &str = "
struct G {
    m: u32, n: u32, k: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
var<push_constant> p: G;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.y;
    let lc = lid.x;
    let gr = wgid.y * 16u + lr;
    let gc = wgid.x * 16u + lc;
    var acc: f32 = 0.0;
    let n_tiles = (p.k + 15u) / 16u;
    for (var ti: u32 = 0u; ti < n_tiles; ti = ti + 1u) {
        let ka = ti * 16u + lc;
        let kb = ti * 16u + lr;
        if (gr < p.m && ka < p.k) {
            ta[lr * 16u + lc] = a[gr * p.k + ka];
        } else {
            ta[lr * 16u + lc] = 0.0;
        }
        if (kb < p.k && gc < p.n) {
            tb[lr * 16u + lc] = b[kb * p.n + gc];
        } else {
            tb[lr * 16u + lc] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            acc = acc + ta[lr * 16u + kk] * tb[kk * 16u + lc];
        }
        workgroupBarrier();
    }
    if (gr < p.m && gc < p.n) {
        c[gr * p.n + gc] = acc;
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

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lr = lid.y;
    let lc = lid.x;
    let gr = wgid.y * 16u + lr;
    let gc = wgid.x * 16u + lc;
    var acc: f32 = 0.0;
    let n_tiles = (p.k + 15u) / 16u;
    for (var ti: u32 = 0u; ti < n_tiles; ti = ti + 1u) {
        let ka = ti * 16u + lc;
        let kb = ti * 16u + lr;
        if (gr < p.m && ka < p.k) {
            ta[lr * 16u + lc] = a[gr * p.k + ka];
        } else {
            ta[lr * 16u + lc] = 0.0;
        }
        // B is [n, k] row-major; we need B^T[k, n] → reading b[gc * k + kb]
        if (gc < p.n && kb < p.k) {
            tb[lr * 16u + lc] = b[gc * p.k + kb];
        } else {
            tb[lr * 16u + lc] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            acc = acc + ta[lr * 16u + kk] * tb[kk * 16u + lc];
        }
        workgroupBarrier();
    }
    if (gr < p.m && gc < p.n) {
        c[gr * p.n + gc] = acc;
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
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
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
        let groups_x = n.div_ceil(TILE);
        let groups_y = m.div_ceil(TILE);
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
