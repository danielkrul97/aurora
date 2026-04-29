//! Sprint 25: 2-layer forward megakernel.
//!
//! Single-shot WGSL kernel runs forward for both layers across all
//! `T` timesteps in one dispatch. Each workgroup handles one batch
//! sample (`workgroup_id.x = b`). Workgroup-shared memory holds
//! per-layer h_spike + h_v_after across timesteps; `active_indices`
//! per layer is rebuilt after each layer's Phase 2.
//!
//! Constraints: 2 layers hardcoded; uniform `n_hidden` across layers;
//! `n_hidden <= 1024` (workgroup memory budget at MAX_HIDDEN=1024:
//! 4 × 1024 × 4 = 16 KB state + 2 × 1024 × 4 = 8 KB active_indices =
//! 24 KB. Workgroup is 256 threads.

use std::sync::Arc;

use super::GpuContext;

const FORWARD_MEGA_2LAYER_WGSL: &str = "
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

// Layer 0 weights (input is byte → sparse one-hot column read).
@group(0) @binding(0) var<storage, read> w_in_0: array<f32>;     // [H, n_input]
@group(0) @binding(1) var<storage, read> w_rec_0: array<f32>;    // [H, H]
@group(0) @binding(2) var<storage, read> w_gate_in_0: array<f32>;
@group(0) @binding(3) var<storage, read> w_gate_rec_0: array<f32>;
@group(0) @binding(4) var<storage, read> b_h_0: array<f32>;
@group(0) @binding(5) var<storage, read> b_gate_0: array<f32>;

// Layer 1 weights + top-level output projection.
@group(1) @binding(0) var<storage, read> w_in_1: array<f32>;     // [H, H] (h2h)
@group(1) @binding(1) var<storage, read> w_rec_1: array<f32>;
@group(1) @binding(2) var<storage, read> w_gate_in_1: array<f32>;
@group(1) @binding(3) var<storage, read> w_gate_rec_1: array<f32>;
@group(1) @binding(4) var<storage, read> b_h_1: array<f32>;
@group(1) @binding(5) var<storage, read> b_gate_1: array<f32>;
@group(1) @binding(6) var<storage, read> w_out: array<f32>;
@group(1) @binding(7) var<storage, read> b_o: array<f32>;

// Per-chunk i/o buffers.
@group(2) @binding(0) var<storage, read> input_bytes: array<u32>;
@group(2) @binding(1) var<storage, read> initial_h_v_after: array<f32>;  // [B, L, H]
@group(2) @binding(2) var<storage, read> initial_h_spike: array<f32>;
@group(2) @binding(3) var<storage, read_write> h_v_pre_packed: array<f32>;     // [B, L, T, H]
@group(2) @binding(4) var<storage, read_write> h_v_after_packed: array<f32>;   // [B, L, T+1, H]
@group(2) @binding(5) var<storage, read_write> h_spike_packed: array<f32>;     // [B, L, T+1, H]
@group(2) @binding(6) var<storage, read_write> gate_packed: array<f32>;        // [B, L, T, H]
@group(2) @binding(7) var<storage, read_write> logits_packed: array<f32>;      // [B, T, O]

var<push_constant> p: ForwardParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_l0_h_spike: array<f32, MAX_HIDDEN>;
var<workgroup> sh_l0_h_v_after: array<f32, MAX_HIDDEN>;
var<workgroup> sh_l1_h_spike: array<f32, MAX_HIDDEN>;
var<workgroup> sh_l1_h_v_after: array<f32, MAX_HIDDEN>;
var<workgroup> sh_active_l0: array<u32, MAX_HIDDEN>;
var<workgroup> sh_n_active_l0: atomic<u32>;
var<workgroup> sh_active_l1: array<u32, MAX_HIDDEN>;
var<workgroup> sh_n_active_l1: atomic<u32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Compaction helper for layer 0: scans sh_l0_h_spike, writes
// sh_active_l0 + sh_n_active_l0.
fn build_active_l0(tid: u32, n: u32) {
    if (tid == 0u) { atomicStore(&sh_n_active_l0, 0u); }
    workgroupBarrier();
    var c: u32 = tid;
    loop {
        if (c >= n) { break; }
        if (sh_l0_h_spike[c] != 0.0) {
            let idx = atomicAdd(&sh_n_active_l0, 1u);
            sh_active_l0[idx] = c;
        }
        c = c + 256u;
    }
    workgroupBarrier();
}

fn build_active_l1(tid: u32, n: u32) {
    if (tid == 0u) { atomicStore(&sh_n_active_l1, 0u); }
    workgroupBarrier();
    var c: u32 = tid;
    loop {
        if (c >= n) { break; }
        if (sh_l1_h_spike[c] != 0.0) {
            let idx = atomicAdd(&sh_n_active_l1, 1u);
            sh_active_l1[idx] = c;
        }
        c = c + 256u;
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
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

    // Per-sample buffer offsets. Layout:
    //   initial_h_*[b * L * H + layer * H + h]    where L = 2
    //   h_*_packed[(b * L + layer) * (T or T+1) * H + slot * H + h]
    //   logits_packed[b * T * O + t * O + h]
    let l_count: u32 = 2u;
    let init_off_l0 = b * l_count * n + 0u * n;
    let init_off_l1 = b * l_count * n + 1u * n;
    let cache_init_off_l0 = (b * l_count + 0u) * (t_steps + 1u) * n;
    let cache_init_off_l1 = (b * l_count + 1u) * (t_steps + 1u) * n;
    let cache_no_init_off_l0 = (b * l_count + 0u) * t_steps * n;
    let cache_no_init_off_l1 = (b * l_count + 1u) * t_steps * n;
    let logits_off = b * t_steps * o_dim;
    let input_off = b * t_steps;

    // Initialize both layers' shared state from initial buffers.
    var hi: u32 = tid;
    loop {
        if (hi >= n) { break; }
        let init_s_l0 = initial_h_spike[init_off_l0 + hi];
        let init_v_l0 = initial_h_v_after[init_off_l0 + hi];
        let init_s_l1 = initial_h_spike[init_off_l1 + hi];
        let init_v_l1 = initial_h_v_after[init_off_l1 + hi];
        h_spike_packed[cache_init_off_l0 + hi] = init_s_l0;
        h_v_after_packed[cache_init_off_l0 + hi] = init_v_l0;
        h_spike_packed[cache_init_off_l1 + hi] = init_s_l1;
        h_v_after_packed[cache_init_off_l1 + hi] = init_v_l1;
        sh_l0_h_spike[hi] = init_s_l0;
        sh_l0_h_v_after[hi] = init_v_l0;
        sh_l1_h_spike[hi] = init_s_l1;
        sh_l1_h_v_after[hi] = init_v_l1;
        hi = hi + 256u;
    }
    workgroupBarrier();

    build_active_l0(tid, n);
    build_active_l1(tid, n);

    for (var t: u32 = 0u; t < t_steps; t = t + 1u) {
        let byte = input_bytes[input_off + t];
        let next_slot = (t + 1u) * n;

        // ===== Layer 0: byte input, sparse via active_l0 =====
        let n_act_0 = atomicLoad(&sh_n_active_l0);
        var h2: u32 = tid;
        loop {
            if (h2 >= n) { break; }
            var h_rec: f32 = 0.0;
            var gate_rec: f32 = 0.0;
            for (var k: u32 = 0u; k < n_act_0; k = k + 1u) {
                let c = sh_active_l0[k];
                h_rec = h_rec + w_rec_0[h2 * n + c];
                gate_rec = gate_rec + w_gate_rec_0[h2 * n + c];
            }
            let h_in_val = w_in_0[h2 * i_dim + byte];
            let gate_in = w_gate_in_0[h2 * i_dim + byte];
            let gate_pre = b_gate_0[h2] + gate_in + gate_rec;
            let g = sigmoid(gate_pre);
            let h_v_pre = clamp(sh_l0_h_v_after[h2] * g + h_in_val + h_rec + b_h_0[h2], -10.0, 10.0);
            let s = select(0.0, 1.0, h_v_pre >= theta);
            let h_v_after_t = h_v_pre - theta * s;

            h_v_pre_packed[cache_no_init_off_l0 + t * n + h2] = h_v_pre;
            gate_packed[cache_no_init_off_l0 + t * n + h2] = g;
            h_spike_packed[cache_init_off_l0 + next_slot + h2] = s;
            h_v_after_packed[cache_init_off_l0 + next_slot + h2] = h_v_after_t;

            h2 = h2 + 256u;
        }
        workgroupBarrier();

        // Update sh_l0 from packed (becomes input for layer 1 + previous-state for next iter's layer 0).
        h2 = tid;
        loop {
            if (h2 >= n) { break; }
            sh_l0_h_spike[h2] = h_spike_packed[cache_init_off_l0 + next_slot + h2];
            sh_l0_h_v_after[h2] = h_v_after_packed[cache_init_off_l0 + next_slot + h2];
            h2 = h2 + 256u;
        }
        workgroupBarrier();

        build_active_l0(tid, n);

        // ===== Layer 1: input is layer 0's just-computed h_spike via sh_active_l0 =====
        let n_act_l0_input = atomicLoad(&sh_n_active_l0);
        let n_act_l1 = atomicLoad(&sh_n_active_l1);
        h2 = tid;
        loop {
            if (h2 >= n) { break; }
            var h_in_val: f32 = 0.0;
            var gate_in: f32 = 0.0;
            for (var k: u32 = 0u; k < n_act_l0_input; k = k + 1u) {
                let c = sh_active_l0[k];
                h_in_val = h_in_val + w_in_1[h2 * n + c];
                gate_in = gate_in + w_gate_in_1[h2 * n + c];
            }
            var h_rec: f32 = 0.0;
            var gate_rec: f32 = 0.0;
            for (var k: u32 = 0u; k < n_act_l1; k = k + 1u) {
                let c = sh_active_l1[k];
                h_rec = h_rec + w_rec_1[h2 * n + c];
                gate_rec = gate_rec + w_gate_rec_1[h2 * n + c];
            }
            let gate_pre = b_gate_1[h2] + gate_in + gate_rec;
            let g = sigmoid(gate_pre);
            let h_v_pre = clamp(sh_l1_h_v_after[h2] * g + h_in_val + h_rec + b_h_1[h2], -10.0, 10.0);
            let s = select(0.0, 1.0, h_v_pre >= theta);
            let h_v_after_t = h_v_pre - theta * s;

            h_v_pre_packed[cache_no_init_off_l1 + t * n + h2] = h_v_pre;
            gate_packed[cache_no_init_off_l1 + t * n + h2] = g;
            h_spike_packed[cache_init_off_l1 + next_slot + h2] = s;
            h_v_after_packed[cache_init_off_l1 + next_slot + h2] = h_v_after_t;

            h2 = h2 + 256u;
        }
        workgroupBarrier();

        h2 = tid;
        loop {
            if (h2 >= n) { break; }
            sh_l1_h_spike[h2] = h_spike_packed[cache_init_off_l1 + next_slot + h2];
            sh_l1_h_v_after[h2] = h_v_after_packed[cache_init_off_l1 + next_slot + h2];
            h2 = h2 + 256u;
        }
        workgroupBarrier();

        build_active_l1(tid, n);

        // ===== Output: W_out @ layer 1 spikes =====
        let n_act_out = atomicLoad(&sh_n_active_l1);
        var oi: u32 = tid;
        loop {
            if (oi >= o_dim) { break; }
            var logit: f32 = b_o[oi];
            for (var k: u32 = 0u; k < n_act_out; k = k + 1u) {
                let c = sh_active_l1[k];
                logit = logit + w_out[oi * n + c];
            }
            logits_packed[logits_off + t * o_dim + oi] = logit;
            oi = oi + 256u;
        }
    }
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForwardMega2LayerParams {
    pub n_input: u32,
    pub n_hidden: u32,
    pub n_output: u32,
    pub n_timesteps: u32,
    pub threshold: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub struct ForwardMega2LayerKernel {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub bgl_l0: Arc<wgpu::BindGroupLayout>,
    pub bgl_l1: Arc<wgpu::BindGroupLayout>,
    pub bgl_io: Arc<wgpu::BindGroupLayout>,
}

impl ForwardMega2LayerKernel {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora forward_mega_2layer wgsl"),
                source: wgpu::ShaderSource::Wgsl(FORWARD_MEGA_2LAYER_WGSL.into()),
            });
        let bgl_l0 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora forward_mega_2layer bgl l0 weights"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                ],
            });
        let bgl_l1 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora forward_mega_2layer bgl l1 weights + top"),
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
                label: Some("aurora forward_mega_2layer bgl io"),
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
                label: Some("aurora forward_mega_2layer pl"),
                bind_group_layouts: &[&bgl_l0, &bgl_l1, &bgl_io],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("aurora forward_mega_2layer pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline: Arc::new(pipeline),
            bgl_l0: Arc::new(bgl_l0),
            bgl_l1: Arc::new(bgl_l1),
            bgl_io: Arc::new(bgl_io),
        }
    }

    /// Records a single dispatch into `encoder`. Each of the
    /// `batch_size` workgroups (64 threads) handles one full
    /// 2-layer sample chunk in parallel.
    pub fn dispatch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_l0: &wgpu::BindGroup,
        bind_l1: &wgpu::BindGroup,
        bind_io: &wgpu::BindGroup,
        params: &ForwardMega2LayerParams,
        batch_size: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aurora forward_mega_2layer pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_l0, &[]);
        pass.set_bind_group(1, bind_l1, &[]);
        pass.set_bind_group(2, bind_io, &[]);
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
