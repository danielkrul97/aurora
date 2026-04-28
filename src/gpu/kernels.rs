//! WGSL compute kernels + Rust dispatch helpers.
//!
//! Forward path kernels (Sprint 14):
//! - `sparse_matvec_onehot`: column read for one-hot input.
//! - `dense_matvec`: generic `out = W @ v` (used for `W_rec`, `W_gate_rec`,
//!   `W_out` since the spike vector is dense f32 even when spike values
//!   are 0/1 — GPU prefers dense over branchy sparse).
//! - `gated_cell_step`: fused element-wise per-timestep update — gate
//!   sigmoid, membrane integration with adaptive decay, spike threshold,
//!   subtraction reset.
//!
//! Backward + Adam kernels (Sprint 15):
//! - `dense_matvec_transpose`: `out = W^T @ v` with optional accumulate
//!   (used for backward through `W_out`, `W_rec`, `W_gate_rec`).
//! - `gated_cell_step_backward`: fused per-element backward for the
//!   gated LIF cell — surrogate-gradient spike, gate sigmoid grad,
//!   membrane carryover.
//! - `outer_product_accum`: `M[r,c] += a[r] * b[c]` for weight-matrix
//!   gradient accumulation.
//! - `column_accum`: `W[:, byte_idx] += d[:]` for sparse one-hot input
//!   weight gradients (`W_in`, `W_gate_in`).
//! - `bias_accum`: `b[i] += d[i]` for bias gradients.
//! - `adam_step`: vectorized Adam update over a parameter buffer.
//! - `scale_buffer`: `b[i] *= factor` for `clip_grads` rescale.
//! - `zero_buffer`: `b[i] = 0` for `zero_grad`.
//!
//! Performance kernels (Sprint 16):
//! - `fused_gate_pre`: `gate_pre = b_gate + gate_in_contrib + gate_rec_contrib`
//!   elementwise add, eliminates the per-timestep CPU roundtrip in
//!   forward.
//! - `softmax_cross_entropy`: workgroup-shared softmax reduction
//!   producing `d_logits = softmax - one_hot(target)` and per-timestep
//!   loss in a single dispatch per timestep, eliminating the per-chunk
//!   CPU download/upload of logits.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::GpuContext;

const SPARSE_MATVEC_ONEHOT_WGSL: &str = r"
struct Meta {
    n_hidden: u32,
    n_input: u32,
    byte_idx: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<push_constant> params: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.n_hidden) {
        return;
    }
    out[h] = w[h * params.n_input + params.byte_idx];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    n_hidden: u32,
    n_input: u32,
    byte_idx: u32,
    _pad: u32,
}

pub struct SparseMatvecOneHotKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl SparseMatvecOneHotKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora sparse_matvec_onehot wgsl"),
                source: wgpu::ShaderSource::Wgsl(SPARSE_MATVEC_ONEHOT_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora sparse_matvec_onehot bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora sparse_matvec_onehot pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora sparse_matvec_onehot pipeline"),
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

    /// Dispatch one matvec: `out[h] = w[h * n_input + byte_idx]`.
    /// `w` shape `[n_hidden][n_input]`, `out` shape `[n_hidden]`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        w: &GpuBuffer,
        out: &GpuBuffer,
        n_hidden: u32,
        n_input: u32,
        byte_idx: u32,
    ) {
        debug_assert_eq!(w.len, (n_hidden * n_input) as usize);
        debug_assert_eq!(out.len, n_hidden as usize);

        let p = Meta {
            n_hidden,
            n_input,
            byte_idx,
            _pad: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora sparse_matvec_onehot bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora sparse_matvec_onehot pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const DENSE_MATVEC_WGSL: &str = "
struct DM {
    rows: u32,
    cols: u32,
    bias_mode: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<push_constant> dm: DM;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= dm.rows) {
        return;
    }
    var acc: f32 = 0.0;
    if (dm.bias_mode == 1u) {
        acc = bias[r];
    }
    let row_offset = r * dm.cols;
    for (var c: u32 = 0u; c < dm.cols; c = c + 1u) {
        acc = acc + w[row_offset + c] * v[c];
    }
    out[r] = acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMatvecParams {
    rows: u32,
    cols: u32,
    bias_mode: u32,
    _pad: u32,
}

pub struct DenseMatvecKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl DenseMatvecKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora dense_matvec wgsl"),
                source: wgpu::ShaderSource::Wgsl(DENSE_MATVEC_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora dense_matvec bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora dense_matvec pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora dense_matvec pipeline"),
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

    /// `out[r] = (bias_mode ? bias[r] : 0) + sum_c w[r,c] * v[c]`.
    /// Pass a zero-length placeholder buffer for `bias` if `add_bias=false`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        w: &GpuBuffer,
        v: &GpuBuffer,
        bias: &GpuBuffer,
        out: &GpuBuffer,
        rows: u32,
        cols: u32,
        add_bias: bool,
    ) {
        debug_assert_eq!(w.len, (rows * cols) as usize);
        debug_assert_eq!(v.len, cols as usize);
        debug_assert_eq!(out.len, rows as usize);

        let p = DenseMatvecParams {
            rows,
            cols,
            bias_mode: u32::from(add_bias),
            _pad: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora dense_matvec bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora dense_matvec pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = rows.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

/// Outputs packed into a single buffer of length `4 * n_hidden`:
/// `[gate (N), h_v_pre (N), h_spike (N), h_v_after (N)]`. Reduces
/// binding count to 7 storage + 1 uniform = under the 8-storage limit
/// of `Limits::default()`.
const GATED_CELL_STEP_WGSL: &str = "
struct CS {
    n_hidden: u32,
    threshold: f32,
};

@group(0) @binding(0) var<storage, read>       gate_pre: array<f32>;
@group(0) @binding(1) var<storage, read>       h_in: array<f32>;
@group(0) @binding(2) var<storage, read>       h_rec: array<f32>;
@group(0) @binding(3) var<storage, read>       h_v_after_prev: array<f32>;
@group(0) @binding(4) var<storage, read>       b_h: array<f32>;
@group(0) @binding(5) var<storage, read_write> packed_out: array<f32>;
var<push_constant>             cs: CS;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= cs.n_hidden) {
        return;
    }
    let g = sigmoid(gate_pre[h]);
    let v_pre = h_v_after_prev[h] * g + h_in[h] + h_rec[h] + b_h[h];
    let s = select(0.0, 1.0, v_pre >= cs.threshold);
    let n = cs.n_hidden;
    packed_out[h]         = g;
    packed_out[n + h]     = v_pre;
    packed_out[2u * n + h] = s;
    packed_out[3u * n + h] = v_pre - cs.threshold * s;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatedCellParams {
    n_hidden: u32,
    threshold: f32,
}

pub struct GatedCellStepKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl GatedCellStepKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora gated_cell_step wgsl"),
                source: wgpu::ShaderSource::Wgsl(GATED_CELL_STEP_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora gated_cell_step bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gated_cell_step pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora gated_cell_step pipeline"),
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

    /// `packed_out` is `4 * n_hidden` floats: `[gate, h_v_pre, h_spike, h_v_after]`
    /// concatenated.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        gate_pre: &GpuBuffer,
        h_in: &GpuBuffer,
        h_rec: &GpuBuffer,
        h_v_after_prev: &GpuBuffer,
        b_h: &GpuBuffer,
        packed_out: &GpuBuffer,
        n_hidden: u32,
        threshold: f32,
    ) {
        let p = GatedCellParams {
            n_hidden,
            threshold,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gated_cell_step bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gate_pre.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_in.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: h_rec.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_v_after_prev.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_h.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: packed_out.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora gated_cell_step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

// =====================================================================
// Sprint 15: backward + Adam kernels.
// =====================================================================

const DENSE_MATVEC_TRANSPOSE_WGSL: &str = "
struct DMT {
    rows: u32,
    cols: u32,
    accumulate: u32,
    v_offset: u32,
};

@group(0) @binding(0) var<storage, read>       w: array<f32>;
@group(0) @binding(1) var<storage, read>       v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<push_constant>             p: DMT;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.cols) {
        return;
    }
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < p.rows; r = r + 1u) {
        acc = acc + w[r * p.cols + c] * v[p.v_offset + r];
    }
    if (p.accumulate == 1u) {
        out[c] = out[c] + acc;
    } else {
        out[c] = acc;
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMatvecTransposeParams {
    rows: u32,
    cols: u32,
    accumulate: u32,
    v_offset: u32,
}

pub struct DenseMatvecTransposeKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl DenseMatvecTransposeKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora dense_matvec_transpose wgsl"),
                source: wgpu::ShaderSource::Wgsl(DENSE_MATVEC_TRANSPOSE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora dense_matvec_transpose bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora dense_matvec_transpose pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora dense_matvec_transpose pipeline"),
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

    /// `out[c] = (acc ? out[c] : 0) + sum_r w[r,c] * v[v_offset + r]`.
    /// `w` shape `[rows, cols]`, `v` is at least `v_offset+rows` long
    /// (Sprint 18: lets v point into a packed `[T*rows]` buffer).
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        w: &GpuBuffer,
        v: &GpuBuffer,
        out: &GpuBuffer,
        rows: u32,
        cols: u32,
        accumulate: bool,
        v_offset: u32,
    ) {
        debug_assert_eq!(w.len, (rows * cols) as usize);
        debug_assert!(v.len >= (v_offset + rows) as usize);
        debug_assert_eq!(out.len, cols as usize);

        let p = DenseMatvecTransposeParams {
            rows,
            cols,
            accumulate: u32::from(accumulate),
            v_offset,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora dense_matvec_transpose bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora dense_matvec_transpose pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = cols.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const GATED_CELL_BACKWARD_WGSL: &str = "
struct GCB {
    n_hidden: u32,
    threshold: f32,
    alpha: f32,
    h_v_pre_offset: u32,
    h_v_after_prev_offset: u32,
    gate_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       d_h_spike_out: array<f32>;
@group(0) @binding(1) var<storage, read>       d_h_spike_rec: array<f32>;
@group(0) @binding(2) var<storage, read>       d_h_v_after_carry_in: array<f32>;
@group(0) @binding(3) var<storage, read>       h_v_pre: array<f32>;
@group(0) @binding(4) var<storage, read>       h_v_after_prev: array<f32>;
@group(0) @binding(5) var<storage, read>       gate: array<f32>;
@group(0) @binding(6) var<storage, read_write> packed_out: array<f32>;
var<push_constant>             p: GCB;

fn surrogate(v_minus_theta: f32, alpha: f32) -> f32 {
    let denom = 1.0 + alpha * abs(v_minus_theta);
    return 1.0 / (denom * denom);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.n_hidden) { return; }
    let n = p.n_hidden;
    let d_spike = d_h_spike_out[h] + d_h_spike_rec[h];
    let s_grad = surrogate(h_v_pre[p.h_v_pre_offset + h] - p.threshold, p.alpha);
    let d_v_pre = d_spike * s_grad + d_h_v_after_carry_in[h];
    let g = gate[p.gate_offset + h];
    let d_gate_h = d_v_pre * h_v_after_prev[p.h_v_after_prev_offset + h];
    let d_gate_pre = d_gate_h * g * (1.0 - g);
    let d_v_after_out = d_v_pre * g;
    packed_out[h]         = d_v_pre;
    packed_out[n + h]     = d_gate_pre;
    packed_out[2u*n + h]  = d_v_after_out;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatedCellBackwardParams {
    n_hidden: u32,
    threshold: f32,
    alpha: f32,
    h_v_pre_offset: u32,
    h_v_after_prev_offset: u32,
    gate_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct GatedCellBackwardKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl GatedCellBackwardKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora gated_cell_backward wgsl"),
                source: wgpu::ShaderSource::Wgsl(GATED_CELL_BACKWARD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora gated_cell_backward bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gated_cell_backward pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora gated_cell_backward pipeline"),
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

    /// `packed_out` is `3 * n_hidden` floats:
    /// `[d_h_v_pre, d_gate_pre, d_h_v_after_carry_out]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d_h_spike_out: &GpuBuffer,
        d_h_spike_rec: &GpuBuffer,
        d_h_v_after_carry_in: &GpuBuffer,
        h_v_pre: &GpuBuffer,
        h_v_after_prev: &GpuBuffer,
        gate: &GpuBuffer,
        packed_out: &GpuBuffer,
        n_hidden: u32,
        threshold: f32,
        alpha: f32,
        h_v_pre_offset: u32,
        h_v_after_prev_offset: u32,
        gate_offset: u32,
    ) {
        let p = GatedCellBackwardParams {
            n_hidden,
            threshold,
            alpha,
            h_v_pre_offset,
            h_v_after_prev_offset,
            gate_offset,
            _pad0: 0,
            _pad1: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora gated_cell_backward bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: d_h_spike_out.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_h_spike_rec.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: d_h_v_after_carry_in.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_v_pre.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: h_v_after_prev.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gate.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: packed_out.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora gated_cell_backward pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const OUTER_PRODUCT_ACCUM_WGSL: &str = "
struct OPA {
    rows: u32,
    cols: u32,
    a_offset: u32,
    b_offset: u32,
};

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
var<push_constant>             p: OPA;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r >= p.rows || c >= p.cols) { return; }
    let idx = r * p.cols + c;
    m[idx] = m[idx] + a[p.a_offset + r] * b[p.b_offset + c];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterProductAccumParams {
    rows: u32,
    cols: u32,
    a_offset: u32,
    b_offset: u32,
}

pub struct OuterProductAccumKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl OuterProductAccumKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora outer_product_accum wgsl"),
                source: wgpu::ShaderSource::Wgsl(OUTER_PRODUCT_ACCUM_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora outer_product_accum bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora outer_product_accum pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora outer_product_accum pipeline"),
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

    /// `m[r,c] += a[a_offset + r] * b[b_offset + c]`. `m` shape
    /// `[rows, cols]`. Offsets allow `a` and `b` to point into
    /// packed `[T*rows]` / `[T*cols]` buffers (Sprint 18).
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        m: &GpuBuffer,
        rows: u32,
        cols: u32,
        a_offset: u32,
        b_offset: u32,
    ) {
        debug_assert!(a.len >= (a_offset + rows) as usize);
        debug_assert!(b.len >= (b_offset + cols) as usize);
        debug_assert_eq!(m.len, (rows * cols) as usize);

        let p = OuterProductAccumParams {
            rows,
            cols,
            a_offset,
            b_offset,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora outer_product_accum bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora outer_product_accum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let wgx = rows.div_ceil(8);
            let wgy = cols.div_ceil(8);
            pass.dispatch_workgroups(wgx, wgy, 1);
        }
    }
}

const COLUMN_ACCUM_WGSL: &str = "
struct CA {
    rows: u32,
    n_input: u32,
    byte_idx: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       d: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
var<push_constant>             p: CA;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= p.rows) { return; }
    let idx = r * p.n_input + p.byte_idx;
    w[idx] = w[idx] + d[r];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColumnAccumParams {
    rows: u32,
    n_input: u32,
    byte_idx: u32,
    _pad: u32,
}

pub struct ColumnAccumKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl ColumnAccumKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora column_accum wgsl"),
                source: wgpu::ShaderSource::Wgsl(COLUMN_ACCUM_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora column_accum bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora column_accum pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora column_accum pipeline"),
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

    /// `w[r * n_input + byte_idx] += d[r]`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d: &GpuBuffer,
        w: &GpuBuffer,
        rows: u32,
        n_input: u32,
        byte_idx: u32,
    ) {
        debug_assert_eq!(d.len, rows as usize);
        debug_assert_eq!(w.len, (rows * n_input) as usize);

        let p = ColumnAccumParams {
            rows,
            n_input,
            byte_idx,
            _pad: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora column_accum bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: d.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora column_accum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = rows.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const VECTOR_ADD_WGSL: &str = "
struct VA { len: u32, d_offset: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read>       d: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
var<push_constant>             p: VA;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    b[i] = b[i] + d[p.d_offset + i];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorAddParams {
    len: u32,
    d_offset: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct VectorAddKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl VectorAddKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora vector_add wgsl"),
                source: wgpu::ShaderSource::Wgsl(VECTOR_ADD_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora vector_add bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora vector_add pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora vector_add pipeline"),
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

    /// `b[i] += d[d_offset + i]`. Used for bias-grad accumulation
    /// (`b_h`, `b_o`, `b_gate`). `d_offset` lets `d` point into a
    /// packed `[T*len]` buffer (Sprint 18); pass 0 for scratch input.
    pub fn dispatch_into(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, d: &GpuBuffer, b: &GpuBuffer, len: u32, d_offset: u32) {
        debug_assert!(d.len >= (d_offset + len) as usize);
        debug_assert_eq!(b.len, len as usize);

        let p = VectorAddParams {
            len,
            d_offset,
            _pad1: 0,
            _pad2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora vector_add bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: d.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora vector_add pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = len.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const ADAM_STEP_WGSL: &str = "
struct AS {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       grads: array<f32>;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> m1: array<f32>;
@group(0) @binding(3) var<storage, read_write> m2: array<f32>;
var<push_constant>             p: AS;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let g = grads[i];
    let m1_new = p.beta1 * m1[i] + (1.0 - p.beta1) * g;
    let m2_new = p.beta2 * m2[i] + (1.0 - p.beta2) * g * g;
    m1[i] = m1_new;
    m2[i] = m2_new;
    let m1_hat = m1_new / p.bc1;
    let m2_hat = m2_new / p.bc2;
    params[i] = params[i] - p.lr * m1_hat / (sqrt(m2_hat) + p.eps);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamStepParams {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    _pad: u32,
}

pub struct AdamStepKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl AdamStepKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora adam_step wgsl"),
                source: wgpu::ShaderSource::Wgsl(ADAM_STEP_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora adam_step bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora adam_step pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora adam_step pipeline"),
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

    /// One Adam step over a parameter buffer: updates `m1`, `m2`,
    /// `params` in place. `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t` —
    /// caller computes them.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        grads: &GpuBuffer,
        params: &GpuBuffer,
        m1: &GpuBuffer,
        m2: &GpuBuffer,
        len: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
    ) {
        debug_assert_eq!(grads.len, len as usize);
        debug_assert_eq!(params.len, len as usize);
        debug_assert_eq!(m1.len, len as usize);
        debug_assert_eq!(m2.len, len as usize);

        let p = AdamStepParams {
            len,
            lr,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            _pad: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora adam_step bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grads.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m1.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: m2.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora adam_step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = len.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const SCALE_BUFFER_WGSL: &str = "
struct SB { len: u32, factor: f32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read_write> b: array<f32>;
var<push_constant>             p: SB;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    b[i] = b[i] * p.factor;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleBufferParams {
    len: u32,
    factor: f32,
    _pad0: u32,
    _pad1: u32,
}

pub struct ScaleBufferKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl ScaleBufferKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora scale_buffer wgsl"),
                source: wgpu::ShaderSource::Wgsl(SCALE_BUFFER_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora scale_buffer bgl"),
                entries: &[storage_entry(0, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora scale_buffer pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora scale_buffer pipeline"),
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

    pub fn dispatch_into(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, b: &GpuBuffer, factor: f32) {
        let p = ScaleBufferParams {
            len: b.len as u32,
            factor,
            _pad0: 0,
            _pad1: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora scale_buffer bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora scale_buffer pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = (b.len as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

const ZERO_BUFFER_WGSL: &str = "
struct ZB { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read_write> b: array<f32>;
var<push_constant>             p: ZB;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    b[i] = 0.0;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ZeroBufferParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct ZeroBufferKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl ZeroBufferKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora zero_buffer wgsl"),
                source: wgpu::ShaderSource::Wgsl(ZERO_BUFFER_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora zero_buffer bgl"),
                entries: &[storage_entry(0, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora zero_buffer pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora zero_buffer pipeline"),
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

    pub fn dispatch_into(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, b: &GpuBuffer) {
        let p = ZeroBufferParams {
            len: b.len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora zero_buffer bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora zero_buffer pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = (b.len as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

// =====================================================================
// Sprint 16: GPU-side perf kernels (eliminate CPU roundtrips).
// =====================================================================

const FUSED_GATE_PRE_WGSL: &str = "
struct GP { n_hidden: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       b_gate: array<f32>;
@group(0) @binding(1) var<storage, read>       gate_in_contrib: array<f32>;
@group(0) @binding(2) var<storage, read>       gate_rec_contrib: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate_pre: array<f32>;
var<push_constant>             p: GP;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.n_hidden) { return; }
    gate_pre[h] = b_gate[h] + gate_in_contrib[h] + gate_rec_contrib[h];
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGatePreParams {
    n_hidden: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct FusedGatePreKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl FusedGatePreKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora fused_gate_pre wgsl"),
                source: wgpu::ShaderSource::Wgsl(FUSED_GATE_PRE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora fused_gate_pre bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora fused_gate_pre pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora fused_gate_pre pipeline"),
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

    /// `gate_pre[h] = b_gate[h] + gate_in_contrib[h] + gate_rec_contrib[h]`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        b_gate: &GpuBuffer,
        gate_in_contrib: &GpuBuffer,
        gate_rec_contrib: &GpuBuffer,
        gate_pre: &GpuBuffer,
        n_hidden: u32,
    ) {
        debug_assert_eq!(b_gate.len, n_hidden as usize);
        debug_assert_eq!(gate_in_contrib.len, n_hidden as usize);
        debug_assert_eq!(gate_rec_contrib.len, n_hidden as usize);
        debug_assert_eq!(gate_pre.len, n_hidden as usize);

        let p = FusedGatePreParams {
            n_hidden,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora fused_gate_pre bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_gate.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_in_contrib.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gate_rec_contrib.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gate_pre.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora fused_gate_pre pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

// Sprint 19: dispatched with `(B, T, 1)` workgroups. Each workgroup
// handles one (sample b, timestep t) softmax + d_logits + loss in
// parallel. Targets come from a storage buffer (`targets[b*T + t]`).
//
// Constraint: O <= 256 (one thread per output). For VOCAB=256 byte LM
// this is exact; the dispatch panics if O > 256.
const SOFTMAX_CE_WGSL: &str = "
struct SCE {
    n_output: u32,
    n_timesteps: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_seq: array<f32>;
@group(0) @binding(3) var<storage, read>       targets: array<u32>;
var<push_constant>             p: SCE;

var<workgroup> sh_buf: array<f32, 256>;
var<workgroup> sh_max: f32;
var<workgroup> sh_sum: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tid = lid.x;
    let b = wgid.x;
    let t = wgid.y;
    let bt_idx = b * p.n_timesteps + t;
    let base = bt_idx * p.n_output;
    let target_idx = targets[bt_idx];

    var v: f32 = -3.4028235e38;
    if (tid < p.n_output) {
        v = logits[base + tid];
        sh_buf[tid] = v;
    } else {
        sh_buf[tid] = -3.4028235e38;
    }
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_buf[tid] = max(sh_buf[tid], sh_buf[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_max = sh_buf[0];
    }
    workgroupBarrier();

    var e: f32 = 0.0;
    if (tid < p.n_output) {
        e = exp(v - sh_max);
        sh_buf[tid] = e;
    } else {
        sh_buf[tid] = 0.0;
    }
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_buf[tid] = sh_buf[tid] + sh_buf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        sh_sum = sh_buf[0];
    }
    workgroupBarrier();

    if (tid < p.n_output) {
        let prob = e / sh_sum;
        var d = prob;
        if (tid == target_idx) {
            d = prob - 1.0;
        }
        d_logits[base + tid] = d;
    }
    if (tid == 0u) {
        let p_target = max(exp(logits[base + target_idx] - sh_max) / sh_sum, 1e-12);
        loss_seq[bt_idx] = -log(p_target);
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxCeParams {
    n_output: u32,
    n_timesteps: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct SoftmaxCrossEntropyKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl SoftmaxCrossEntropyKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora softmax_ce wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOFTMAX_CE_WGSL.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora softmax_ce bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, true),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora softmax_ce pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..32 }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora softmax_ce pipeline"),
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

    /// Sprint 19: one dispatch handles `B * T` softmaxes via a 2D grid
    /// of workgroups (`workgroup_id.x = b`, `workgroup_id.y = t`). Each
    /// workgroup is 256 threads, one per output. `targets` is a storage
    /// buffer of `u32` of length `>= B * T` (one byte index per
    /// (sample, timestep)).
    ///
    /// # Panics
    /// Panics if `n_output > 256`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        logits: &GpuBuffer,
        d_logits: &GpuBuffer,
        loss_seq: &GpuBuffer,
        targets: &GpuBuffer,
        n_output: u32,
        n_timesteps: u32,
        batch_size: u32,
    ) {
        assert!(
            n_output <= 256,
            "softmax_ce kernel hardcoded to workgroup_size(256); n_output={n_output} > 256",
        );
        debug_assert!(logits.len >= (batch_size * n_timesteps * n_output) as usize);
        debug_assert!(d_logits.len >= (batch_size * n_timesteps * n_output) as usize);
        debug_assert!(loss_seq.len >= (batch_size * n_timesteps) as usize);
        debug_assert!(targets.len >= (batch_size * n_timesteps) as usize);

        let p = SoftmaxCeParams {
            n_output,
            n_timesteps,
            _pad0: 0,
            _pad1: 0,
        };
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora softmax_ce bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_logits.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: loss_seq.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: targets.inner.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora softmax_ce pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&p));
            pass.dispatch_workgroups(batch_size, n_timesteps, 1);
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

