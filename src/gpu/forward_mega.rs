//! Sprint 18: forward megakernel.
//!
//! Single WGSL compute shader that runs the entire `GatedRecurrentSnn`
//! forward pass for one chunk in **one** dispatch. Replaces the per-
//! timestep dispatch loop (Sprint 14-17 had ~7 dispatches × T = ~448
//! per chunk) with a single-shot launch that loops over T timesteps
//! in-shader, persisting hidden state in workgroup-shared memory.
//!
//! The kernel writes per-timestep state directly to packed cache
//! buffers: `h_v_pre[T*N]`, `h_v_after[T*N]`, `h_spike[T*N]`,
//! `gate[T*N]`, `logits[T*O]`. Backward path reads from those same
//! packed buffers via offset push constants on the smaller backward
//! kernels (Sprint 18 also refactors the backward path).
//!
//! Bind groups (created once per `GatedRecurrentSnnGpu` instance and
//! cached, since all referenced buffers are stable):
//! - Group 0: 8 read-only weight buffers (W_in, W_rec, W_out,
//!   W_gate_in, W_gate_rec, b_h, b_o, b_gate).
//! - Group 1: 8 per-chunk i/o buffers (input_bytes, initial_h_v_after,
//!   initial_h_spike, h_v_pre_packed, h_v_after_packed, h_spike_packed,
//!   gate_packed, logits_packed).
//!
//! Push constants: `(n_input, n_hidden, n_output, n_timesteps,
//! threshold)` — 20 bytes, fits in our 32-byte allotment.
//!
//! Constraint: `n_hidden <= 1024` (workgroup-shared array fixed size).
//! Workgroup size 64; each thread strides over `n_hidden / 64` neurons.
//! For the byte-LM case (`n_input = n_output = 256`, hidden 128-512)
//! all parameters fit comfortably.

use std::sync::Arc;

use super::GpuContext;

const FORWARD_MEGA_WGSL: &str = "
struct ForwardParams {
    n_input: u32,
    n_hidden: u32,
    n_output: u32,
    n_timesteps: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> w_in: array<f32>;
@group(0) @binding(1) var<storage, read> w_rec: array<f32>;
@group(0) @binding(2) var<storage, read> w_out: array<f32>;
@group(0) @binding(3) var<storage, read> w_gate_in: array<f32>;
@group(0) @binding(4) var<storage, read> w_gate_rec: array<f32>;
@group(0) @binding(5) var<storage, read> b_h: array<f32>;
@group(0) @binding(6) var<storage, read> b_o: array<f32>;
@group(0) @binding(7) var<storage, read> b_gate: array<f32>;

@group(1) @binding(0) var<storage, read> input_bytes: array<u32>;
@group(1) @binding(1) var<storage, read> initial_h_v_after: array<f32>;
@group(1) @binding(2) var<storage, read> initial_h_spike: array<f32>;
@group(1) @binding(3) var<storage, read_write> h_v_pre_packed: array<f32>;
@group(1) @binding(4) var<storage, read_write> h_v_after_packed: array<f32>;
@group(1) @binding(5) var<storage, read_write> h_spike_packed: array<f32>;
@group(1) @binding(6) var<storage, read_write> gate_packed: array<f32>;
@group(1) @binding(7) var<storage, read_write> logits_packed: array<f32>;

var<push_constant> p: ForwardParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_h_spike: array<f32, MAX_HIDDEN>;
var<workgroup> sh_h_v_after: array<f32, MAX_HIDDEN>;
// Sprint 21: workgroup-shared compaction of `sh_h_spike` so the
// W_rec / W_gate_rec / W_out matvecs only read weight columns for
// neurons that actually fired (~5% in steady-state SNNs). Atomic
// counter is reset before each compaction; per-thread stride loop
// scans `sh_h_spike` and appends each non-zero index.
var<workgroup> sh_active_indices: array<u32, MAX_HIDDEN>;
var<workgroup> sh_n_active: atomic<u32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn build_active_indices(tid: u32, n: u32) {
    if (tid == 0u) {
        atomicStore(&sh_n_active, 0u);
    }
    workgroupBarrier();
    var c: u32 = tid;
    loop {
        if (c >= n) { break; }
        if (sh_h_spike[c] != 0.0) {
            let idx = atomicAdd(&sh_n_active, 1u);
            sh_active_indices[idx] = c;
        }
        c = c + 64u;
    }
    workgroupBarrier();
}

// Sprint 19: dispatched with `(B, 1, 1)` workgroups; each workgroup
// handles ONE batch sample identified by `workgroup_id.x`. Workgroup-
// shared state stays per-workgroup so different samples don't interfere.
//
// Buffer layout (b = sample, t = timestep, h = neuron):
// - `h_spike_packed` / `h_v_after_packed`: `b * (T+1) * N + slot * N + h`
//   where slot 0 = initial state, slot t+1 = post-step at t.
// - `h_v_pre_packed` / `gate_packed`: `b * T * N + t * N + h`.
// - `logits_packed`: `b * T * O + t * O + h`.
// - `input_bytes` / `loss_seq`: `b * T + t`.
// - `initial_h_v_after` / `initial_h_spike`: `b * N + h`.
@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tid = lid.x;
    let b = wgid.x;
    let n = p.n_hidden;
    let i_dim = p.n_input;
    let o_dim = p.n_output;
    let t_steps = p.n_timesteps;
    let theta = p.threshold;

    let init_off = b * n;
    let input_off = b * t_steps;
    let cache_off = b * (t_steps + 1u) * n;        // h_spike, h_v_after
    let cache_off_no_init = b * t_steps * n;        // h_v_pre, gate
    let logits_off = b * t_steps * o_dim;

    var hi: u32 = tid;
    loop {
        if (hi >= n) { break; }
        let init_s = initial_h_spike[init_off + hi];
        let init_v = initial_h_v_after[init_off + hi];
        h_spike_packed[cache_off + hi] = init_s;
        h_v_after_packed[cache_off + hi] = init_v;
        sh_h_spike[hi] = init_s;
        sh_h_v_after[hi] = init_v;
        hi = hi + 64u;
    }
    workgroupBarrier();

    // Initial compaction: active_indices reflects sh_h_spike (= initial
    // state, mostly zero). Used by t=0's Phase 1 matvecs.
    build_active_indices(tid, n);

    for (var t: u32 = 0u; t < t_steps; t = t + 1u) {
        let byte = input_bytes[input_off + t];
        let next_slot = (t + 1u) * n;

        let n_act = atomicLoad(&sh_n_active);
        var h2: u32 = tid;
        loop {
            if (h2 >= n) { break; }
            var h_rec: f32 = 0.0;
            var gate_rec: f32 = 0.0;
            // Sprint 21 sparse: iterate only over neurons that fired
            // last timestep. Spike value is 1.0 by construction (binary
            // select(0,1, ...) in last iter), so the multiplication
            // factor is dropped — read W column directly.
            for (var k: u32 = 0u; k < n_act; k = k + 1u) {
                let c = sh_active_indices[k];
                h_rec = h_rec + w_rec[h2 * n + c];
                gate_rec = gate_rec + w_gate_rec[h2 * n + c];
            }
            let h_in_val = w_in[h2 * i_dim + byte];
            let gate_in = w_gate_in[h2 * i_dim + byte];
            let gate_pre = b_gate[h2] + gate_in + gate_rec;
            let g = sigmoid(gate_pre);
            let h_v_pre = sh_h_v_after[h2] * g + h_in_val + h_rec + b_h[h2];
            let s = select(0.0, 1.0, h_v_pre >= theta);
            let h_v_after_t = h_v_pre - theta * s;

            h_v_pre_packed[cache_off_no_init + t * n + h2] = h_v_pre;
            gate_packed[cache_off_no_init + t * n + h2] = g;
            h_spike_packed[cache_off + next_slot + h2] = s;
            h_v_after_packed[cache_off + next_slot + h2] = h_v_after_t;

            h2 = h2 + 64u;
        }
        workgroupBarrier();

        h2 = tid;
        loop {
            if (h2 >= n) { break; }
            sh_h_spike[h2] = h_spike_packed[cache_off + next_slot + h2];
            sh_h_v_after[h2] = h_v_after_packed[cache_off + next_slot + h2];
            h2 = h2 + 64u;
        }
        workgroupBarrier();

        // Compaction over h_spike[t] (just-loaded sh_h_spike). Used by
        // Phase 3 W_out matvec AND by next iteration's Phase 1.
        build_active_indices(tid, n);

        let n_act_out = atomicLoad(&sh_n_active);
        var oi: u32 = tid;
        loop {
            if (oi >= o_dim) { break; }
            var logit: f32 = b_o[oi];
            for (var k: u32 = 0u; k < n_act_out; k = k + 1u) {
                let c = sh_active_indices[k];
                logit = logit + w_out[oi * n + c];
            }
            logits_packed[logits_off + t * o_dim + oi] = logit;
            oi = oi + 64u;
        }
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForwardMegaParams {
    pub n_input: u32,
    pub n_hidden: u32,
    pub n_output: u32,
    pub n_timesteps: u32,
    pub threshold: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub struct ForwardMegaKernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub bgl_weights: Arc<wgpu::BindGroupLayout>,
    pub bgl_io: Arc<wgpu::BindGroupLayout>,
}

impl ForwardMegaKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora forward_mega wgsl"),
                source: wgpu::ShaderSource::Wgsl(FORWARD_MEGA_WGSL.into()),
            });
        let bgl_weights = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora forward_mega bgl weights"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                ],
            });
        let bgl_io = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora forward_mega bgl io"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    storage_entry(7, false),
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora forward_mega pl"),
                bind_group_layouts: &[&bgl_weights, &bgl_io],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora forward_mega pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline: Arc::new(pipeline),
            bgl_weights: Arc::new(bgl_weights),
            bgl_io: Arc::new(bgl_io),
        }
    }

    /// Records a single dispatch into `encoder`. Each of the `batch_size`
    /// workgroups (64 threads) handles one full sample chunk in parallel
    /// (Sprint 19). Single-sample callers pass `batch_size = 1`.
    pub fn dispatch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_weights: &wgpu::BindGroup,
        bind_io: &wgpu::BindGroup,
        params: &ForwardMegaParams,
        batch_size: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora forward_mega pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_weights, &[]);
        pass.set_bind_group(1, bind_io, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(batch_size, 1, 1);
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

/// Widen a byte slice to a `Vec<u32>`.
///
/// Each input byte becomes one `u32` (high bits zero). WGSL storage
/// buffers only support 32-bit element types without an extension,
/// so the byte stream is widened here before upload.
#[must_use]
pub fn pack_bytes_u32(input_bytes: &[u8]) -> Vec<u32> {
    input_bytes.iter().map(|&b| u32::from(b)).collect()
}
