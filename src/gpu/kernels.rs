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
//! Backward kernels remain Sprint 15+ work.

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
@group(0) @binding(2) var<uniform> params: Meta;

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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                push_constant_ranges: &[],
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
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        w: &GpuBuffer,
        out: &GpuBuffer,
        n_hidden: u32,
        n_input: u32,
        byte_idx: u32,
    ) {
        debug_assert_eq!(w.len, (n_hidden * n_input) as usize);
        debug_assert_eq!(out.len, n_hidden as usize);

        let params = Meta {
            n_hidden,
            n_input,
            byte_idx,
            _pad: 0,
        };
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora sparse_matvec_onehot params"),
            size: std::mem::size_of::<Meta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora sparse_matvec_onehot enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora sparse_matvec_onehot pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
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
@group(0) @binding(4) var<uniform> dm: DM;

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
                    uniform_entry(4),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora dense_matvec pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
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
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
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

        let dm = DenseMatvecParams {
            rows,
            cols,
            bias_mode: u32::from(add_bias),
            _pad: 0,
        };
        let dm_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora dense_matvec params"),
            size: std::mem::size_of::<DenseMatvecParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&dm_buf, 0, bytemuck::bytes_of(&dm));

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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dm_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora dense_matvec enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora dense_matvec pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let workgroups = rows.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
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
@group(0) @binding(6) var<uniform>             cs: CS;

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
                    uniform_entry(6),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora gated_cell_step pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
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
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        gate_pre: &GpuBuffer,
        h_in: &GpuBuffer,
        h_rec: &GpuBuffer,
        h_v_after_prev: &GpuBuffer,
        b_h: &GpuBuffer,
        packed_out: &GpuBuffer,
        n_hidden: u32,
        threshold: f32,
    ) {
        let cs = GatedCellParams {
            n_hidden,
            threshold,
        };
        let cs_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora gated_cell_step params"),
            size: std::mem::size_of::<GatedCellParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&cs_buf, 0, bytemuck::bytes_of(&cs));

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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cs_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora gated_cell_step enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aurora gated_cell_step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let workgroups = n_hidden.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
