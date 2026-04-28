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

// Sprint 20: 2D dispatch `(cols/64, B, 1)`. Each thread `(c, b)`
// computes `out[b * out_batch_stride + c] = sum_r W[r,c]
//   * v[b * v_batch_stride + v_offset + r]`. `accumulate` lets
// successive calls add to the same `out` slot (used for the
// `W_rec.T @ d_h_v_pre + W_gate_rec.T @ d_gate_pre` fan-in).
const DENSE_MATVEC_TRANSPOSE_WGSL: &str = "
struct DMT {
    rows: u32,
    cols: u32,
    accumulate: u32,
    v_offset: u32,
    v_batch_stride: u32,
    out_batch_stride: u32,
    batch_size: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       w: array<f32>;
@group(0) @binding(1) var<storage, read>       v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<push_constant>             p: DMT;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let c = gid.x;
    let b = wgid.y;
    if (c >= p.cols || b >= p.batch_size) {
        return;
    }
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < p.rows; r = r + 1u) {
        acc = acc + w[r * p.cols + c] * v[b * p.v_batch_stride + p.v_offset + r];
    }
    let out_idx = b * p.out_batch_stride + c;
    if (p.accumulate == 1u) {
        out[out_idx] = out[out_idx] + acc;
    } else {
        out[out_idx] = acc;
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
    v_batch_stride: u32,
    out_batch_stride: u32,
    batch_size: u32,
    _pad: u32,
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

    /// Sprint 20 batch-fused `out = W^T @ v` where each sample's
    /// `out[b, c] = sum_r W[r, c] * v[b * v_batch_stride + v_offset
    ///                                  + r]`. Dispatched as
    /// `(cols/64, batch_size, 1)` workgroups.
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
        v_batch_stride: u32,
        out_batch_stride: u32,
        batch_size: u32,
    ) {
        debug_assert_eq!(w.len, (rows * cols) as usize);
        let max_b = batch_size.saturating_sub(1);
        debug_assert!(v.len >= (max_b * v_batch_stride + v_offset + rows) as usize);
        debug_assert!(out.len >= (max_b * out_batch_stride + cols) as usize);

        let p = DenseMatvecTransposeParams {
            rows,
            cols,
            accumulate: u32::from(accumulate),
            v_offset,
            v_batch_stride,
            out_batch_stride,
            batch_size,
            _pad: 0,
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
            let workgroups_x = cols.div_ceil(64);
            pass.dispatch_workgroups(workgroups_x, batch_size.max(1), 1);
        }
    }
}

// Sprint 20 batched: 2D dispatch `(h_chunks, B, 1)`. Each thread
// `(h, b)` reads its sample's slice of the packed forward cache and
// scratch buffers, writes to `packed_out` at `[slot][b][h]` layout
// where slot_stride = B * N (caller passes `slot_stride`).
//
// Strides: `cache_stride_no_init = T*N` (h_v_pre, gate),
// `cache_stride_init = (T+1)*N` (h_v_after_prev). Per-sample base for
// scratch buffers (d_h_spike_out, d_h_spike_rec, d_h_v_after_carry,
// d_h_v_after_prev/pre/gate) is `b * N`.
const GATED_CELL_BACKWARD_WGSL: &str = "
struct GCB {
    n_hidden: u32,
    threshold: f32,
    alpha: f32,
    h_v_pre_offset: u32,
    h_v_after_prev_offset: u32,
    gate_offset: u32,
    cache_stride_no_init: u32,
    cache_stride_init: u32,
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
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let h = gid.x;
    let b = wgid.y;
    if (h >= p.n_hidden) { return; }
    let n = p.n_hidden;
    let scratch_base = b * n;
    let slot_stride = nwg.y * n;  // = batch_size * n
    let h_v_pre_idx = b * p.cache_stride_no_init + p.h_v_pre_offset + h;
    let h_v_after_prev_idx = b * p.cache_stride_init + p.h_v_after_prev_offset + h;
    let gate_idx = b * p.cache_stride_no_init + p.gate_offset + h;

    let d_spike = d_h_spike_out[scratch_base + h] + d_h_spike_rec[scratch_base + h];
    let s_grad = surrogate(h_v_pre[h_v_pre_idx] - p.threshold, p.alpha);
    let d_v_pre = d_spike * s_grad + d_h_v_after_carry_in[scratch_base + h];
    let g = gate[gate_idx];
    let d_gate_h = d_v_pre * h_v_after_prev[h_v_after_prev_idx];
    let d_gate_pre = d_gate_h * g * (1.0 - g);
    let d_v_after_out = d_v_pre * g;
    packed_out[0u * slot_stride + scratch_base + h] = d_v_pre;
    packed_out[1u * slot_stride + scratch_base + h] = d_gate_pre;
    packed_out[2u * slot_stride + scratch_base + h] = d_v_after_out;
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
    cache_stride_no_init: u32,
    cache_stride_init: u32,
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

    /// Sprint 20 batch-fused fused per-element backward. Dispatched
    /// `(h_chunks, batch_size, 1)`. `packed_out` is sized `[3 * B *
    /// n_hidden]` in slot-major layout: slot 0 = d_h_v_pre, slot 1 =
    /// d_gate_pre, slot 2 = d_h_v_after_carry_out, each containing
    /// per-sample blocks of size `n_hidden`.
    ///
    /// `cache_stride_no_init` is per-sample stride into `h_v_pre` /
    /// `gate` (= `T * N`), `cache_stride_init` is per-sample stride
    /// into `h_v_after_prev` (= `(T+1) * N`).
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
        cache_stride_no_init: u32,
        cache_stride_init: u32,
        batch_size: u32,
    ) {
        let p = GatedCellBackwardParams {
            n_hidden,
            threshold,
            alpha,
            h_v_pre_offset,
            h_v_after_prev_offset,
            gate_offset,
            cache_stride_no_init,
            cache_stride_init,
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
            let workgroups_x = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups_x, batch_size.max(1), 1);
        }
    }
}

// Sprint 20: each thread `(r, c)` reduces over batch dim in-thread.
// `M[r, c] += sum_bi a[bi * a_batch_stride + a_offset + r]
//                  * b[bi * b_batch_stride + p.b_offset + c]`.
//
// Sprint 22: skip the inner add when `b[bi, c] == 0`. The right
// operand in all three backward outer-product calls is h_spike or
// h_spike_prev (binary 0/1, ~5% nonzero in steady-state SNNs), so
// the check saves both a load of `a` and the FMA most of the time.
// The check is universally beneficial in our backward usage; for
// cases where `b` is dense the check is essentially free.
const OUTER_PRODUCT_ACCUM_WGSL: &str = "
struct OPA {
    rows: u32,
    cols: u32,
    a_offset: u32,
    b_offset: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    batch_size: u32,
    _pad: u32,
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
    var acc: f32 = 0.0;
    for (var bi: u32 = 0u; bi < p.batch_size; bi = bi + 1u) {
        let bv = b[bi * p.b_batch_stride + p.b_offset + c];
        if (bv != 0.0) {
            acc = acc + a[bi * p.a_batch_stride + p.a_offset + r] * bv;
        }
    }
    m[idx] = m[idx] + acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterProductAccumParams {
    rows: u32,
    cols: u32,
    a_offset: u32,
    b_offset: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    batch_size: u32,
    _pad: u32,
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

    /// Sprint 20 batch-fused outer-product accumulation:
    /// `m[r, c] += sum_bi a[bi * a_batch_stride + a_offset + r]
    ///                  * b[bi * b_batch_stride + b_offset + c]`.
    /// Each thread reduces over the batch dim (no atomics needed).
    /// For single-sample callers, pass `(batch_size=1, *_batch_stride=0)`.
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
        a_batch_stride: u32,
        b_batch_stride: u32,
        batch_size: u32,
    ) {
        let max_bi = batch_size.saturating_sub(1);
        debug_assert!(a.len >= (max_bi * a_batch_stride + a_offset + rows) as usize);
        debug_assert!(b.len >= (max_bi * b_batch_stride + b_offset + cols) as usize);
        debug_assert_eq!(m.len, (rows * cols) as usize);

        let p = OuterProductAccumParams {
            rows,
            cols,
            a_offset,
            b_offset,
            a_batch_stride,
            b_batch_stride,
            batch_size,
            _pad: 0,
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

// Sprint 20: each thread `r` loops over the batch dim. For each
// sample `bi`, the byte at this timestep is read from `byte_buf` at
// `bi * t_capacity + t_idx`; weight column at that byte gets
// `d[bi * d_batch_stride + r]` added. Different `r` threads write to
// different addresses; within a thread, the loop is sequential so
// duplicate bytes accumulate correctly.
const COLUMN_ACCUM_WGSL: &str = "
struct CA {
    rows: u32,
    n_input: u32,
    t_capacity: u32,
    t_idx: u32,
    d_batch_stride: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       d: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read>       byte_buf: array<u32>;
var<push_constant>             p: CA;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= p.rows) { return; }
    for (var bi: u32 = 0u; bi < p.batch_size; bi = bi + 1u) {
        let byte_idx = byte_buf[bi * p.t_capacity + p.t_idx];
        let idx = r * p.n_input + byte_idx;
        w[idx] = w[idx] + d[bi * p.d_batch_stride + r];
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColumnAccumParams {
    rows: u32,
    n_input: u32,
    t_capacity: u32,
    t_idx: u32,
    d_batch_stride: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
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
                    storage_entry(2, true),
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

    /// Sprint 20 batch-fused column accumulator:
    /// for each sample `bi` and each row `r`, adds
    /// `d[bi * d_batch_stride + r]` into `w[r * n_input +
    /// byte_buf[bi * t_capacity + t_idx]]`. Different rows hit
    /// different rows of `w`; within a thread the batch loop is
    /// sequential so duplicate bytes accumulate.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d: &GpuBuffer,
        w: &GpuBuffer,
        byte_buf: &GpuBuffer,
        rows: u32,
        n_input: u32,
        t_capacity: u32,
        t_idx: u32,
        d_batch_stride: u32,
        batch_size: u32,
    ) {
        debug_assert!(d.len >= (batch_size.saturating_sub(1) * d_batch_stride + rows) as usize);
        debug_assert_eq!(w.len, (rows * n_input) as usize);

        let p = ColumnAccumParams {
            rows,
            n_input,
            t_capacity,
            t_idx,
            d_batch_stride,
            batch_size,
            _pad0: 0,
            _pad1: 0,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: byte_buf.inner.as_entire_binding(),
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

// Sprint 20: each thread `i` reduces over the batch dim by looping
// `b in 0..batch_size`. `d_batch_stride` is the per-sample stride into
// `d` (e.g. `T*O` for d_logits, `N` for scratch buffers).
// `b[i] += sum_b d[b * d_batch_stride + d_offset + i]`.
const VECTOR_ADD_WGSL: &str = "
struct VA {
    len: u32,
    d_offset: u32,
    d_batch_stride: u32,
    batch_size: u32,
};

@group(0) @binding(0) var<storage, read>       d: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
var<push_constant>             p: VA;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    var acc: f32 = 0.0;
    for (var bi: u32 = 0u; bi < p.batch_size; bi = bi + 1u) {
        acc = acc + d[bi * p.d_batch_stride + p.d_offset + i];
    }
    b[i] = b[i] + acc;
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorAddParams {
    len: u32,
    d_offset: u32,
    d_batch_stride: u32,
    batch_size: u32,
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

    /// `b[i] += sum_bi d[bi * d_batch_stride + d_offset + i]` for
    /// `bi in 0..batch_size`. Sprint 20 batch-fused bias-grad accum:
    /// each thread reduces across the batch dim in-place (no atomics).
    /// For single-sample `(batch_size=1, d_batch_stride=0)` reduces to
    /// the original `b[i] += d[d_offset + i]`.
    pub fn dispatch_into(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        d: &GpuBuffer,
        b: &GpuBuffer,
        len: u32,
        d_offset: u32,
        d_batch_stride: u32,
        batch_size: u32,
    ) {
        debug_assert!(d.len >= (d_offset + len) as usize);
        debug_assert_eq!(b.len, len as usize);

        let p = VectorAddParams {
            len,
            d_offset,
            d_batch_stride,
            batch_size,
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
// Sprint 26: per-thread stride loop over output dim removes the
// hard `O <= 256` cap. Each thread handles `ceil(O / 256)` outputs
// across a stride of 256, then 256-way reductions over `sh_buf`
// proceed as before. Works for any `n_output ≤ 65535` in practice
// (limited by `MAX_OUTPUT` workgroup-shared scratch).
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

    // Phase 1: each thread folds its assigned outputs (stride 256)
    // into a local max, then 256-way reduce in shared memory.
    var local_max: f32 = -3.4028235e38;
    var idx: u32 = tid;
    loop {
        if (idx >= p.n_output) { break; }
        local_max = max(local_max, logits[base + idx]);
        idx = idx + 256u;
    }
    sh_buf[tid] = local_max;
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

    // Phase 2: stride-loop sum over exp(logit - max).
    var local_sum: f32 = 0.0;
    idx = tid;
    loop {
        if (idx >= p.n_output) { break; }
        local_sum = local_sum + exp(logits[base + idx] - sh_max);
        idx = idx + 256u;
    }
    sh_buf[tid] = local_sum;
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

    // Phase 3: write d_logits = softmax - one_hot(target). Stride loop
    // again so each thread covers ceil(n_output/256) outputs.
    idx = tid;
    loop {
        if (idx >= p.n_output) { break; }
        let prob = exp(logits[base + idx] - sh_max) / sh_sum;
        var d = prob;
        if (idx == target_idx) {
            d = prob - 1.0;
        }
        d_logits[base + idx] = d;
        idx = idx + 256u;
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
    /// workgroup is 256 threads. Sprint 26 lifted the `n_output ≤ 256`
    /// cap via per-thread stride loops, so this works for any
    /// vocabulary size (BPE-tokenized models can use n_output up to
    /// 65535 in practice).
    ///
    /// # Panics
    /// Panics if `n_output == 0`.
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
        assert!(n_output > 0, "n_output must be > 0");
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

