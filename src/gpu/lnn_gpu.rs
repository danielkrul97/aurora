//! GPU implementation of `MultiLayerLnnSnn`. Mirrors the CPU API:
//! `forward_lm_tokens_stateful_batch`, `forward_backward_batch_tokens_stateful`,
//! `zero_grad`, `clip_grads`, `step_adam`, plus checkpoint round-trip via
//! `from_cpu` / `to_cpu`.
//!
//! Buffer ownership:
//! - Weights + Adam state live in VRAM for the lifetime of the struct.
//! - Activation cache (f, g, h, alpha, x packed `[B, T, H]` per layer)
//!   is allocated lazily on the first forward call and reused whenever
//!   `(B, T)` match. Reallocates if the shape changes.
//! - Per-batch gradient buffers (`B`-leading slices for w_f/g/h, b_f/g/h,
//!   log_tau, w_out, b_o) are allocated lazily and reduced into a
//!   single total-grad buffer before each Adam step.

use std::sync::Arc;

use rand::SeedableRng;

use super::buffers::GpuBuffer;
use super::kernels_lnn::*;
use super::GpuContext;
use crate::lnn::{LnnState, MultiLayerLnnSnn};
use crate::optimizer::{bias_correction, AdamConfig};

const LOG_TAU_CLAMP: f32 = 3.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ForwardParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ProjParams {
    n_hidden: u32,
    n_output: u32,
    n_timesteps: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CeParams {
    n_output: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TopProjBackParams {
    n_hidden: u32,
    n_output: u32,
    n_timesteps: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BackParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    batch_size: u32,
    elem_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ZeroParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    n: u32,
    factor: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamParams {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ClampParams {
    n: u32,
    bound: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct L2Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn make_module(ctx: &GpuContext, label: &str, src: &str) -> Arc<wgpu::ShaderModule> {
    Arc::new(
        ctx.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            }),
    )
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

fn make_bgl(
    ctx: &GpuContext,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Arc<wgpu::BindGroupLayout> {
    Arc::new(
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            }),
    )
}

fn make_pipeline(
    ctx: &GpuContext,
    label: &str,
    module: &wgpu::ShaderModule,
    bgls: &[&wgpu::BindGroupLayout],
    push_size: u32,
) -> Arc<wgpu::ComputePipeline> {
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: bgls,
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..push_size,
            }],
        });
    Arc::new(
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
    )
}

fn buf_binding(buf: &GpuBuffer) -> wgpu::BindingResource<'_> {
    buf.inner.as_entire_binding()
}

/// Per-layer GPU resources.
pub struct LnnLayerGpu {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_z: usize,

    pub w_f: GpuBuffer,
    pub w_g: GpuBuffer,
    pub w_h: GpuBuffer,
    pub b_f: GpuBuffer,
    pub b_g: GpuBuffer,
    pub b_h: GpuBuffer,
    pub log_tau: GpuBuffer,

    pub m1_w_f: GpuBuffer,
    pub m2_w_f: GpuBuffer,
    pub m1_w_g: GpuBuffer,
    pub m2_w_g: GpuBuffer,
    pub m1_w_h: GpuBuffer,
    pub m2_w_h: GpuBuffer,
    pub m1_b_f: GpuBuffer,
    pub m2_b_f: GpuBuffer,
    pub m1_b_g: GpuBuffer,
    pub m2_b_g: GpuBuffer,
    pub m1_b_h: GpuBuffer,
    pub m2_b_h: GpuBuffer,
    pub m1_log_tau: GpuBuffer,
    pub m2_log_tau: GpuBuffer,

    pub w_f_grad: GpuBuffer,
    pub w_g_grad: GpuBuffer,
    pub w_h_grad: GpuBuffer,
    pub b_f_grad: GpuBuffer,
    pub b_g_grad: GpuBuffer,
    pub b_h_grad: GpuBuffer,
    pub log_tau_grad: GpuBuffer,
}

impl LnnLayerGpu {
    fn from_cpu(layer: &crate::lnn::LnnLayer, ctx: &GpuContext, idx: usize) -> Self {
        let n_in = layer.n_in;
        let n_hidden = layer.n_hidden;
        let n_z = layer.n_z;
        let weight_len = n_hidden * n_z;
        let bias_len = n_hidden;
        let lbl = |s: &str| format!("lnn_l{idx}_{s}");
        Self {
            n_in,
            n_hidden,
            n_z,
            w_f: GpuBuffer::from_slice(ctx, &lbl("w_f"), &layer.w_f),
            w_g: GpuBuffer::from_slice(ctx, &lbl("w_g"), &layer.w_g),
            w_h: GpuBuffer::from_slice(ctx, &lbl("w_h"), &layer.w_h),
            b_f: GpuBuffer::from_slice(ctx, &lbl("b_f"), &layer.b_f),
            b_g: GpuBuffer::from_slice(ctx, &lbl("b_g"), &layer.b_g),
            b_h: GpuBuffer::from_slice(ctx, &lbl("b_h"), &layer.b_h),
            log_tau: GpuBuffer::from_slice(ctx, &lbl("log_tau"), &layer.log_tau),

            m1_w_f: GpuBuffer::zeros(ctx, &lbl("m1_w_f"), weight_len),
            m2_w_f: GpuBuffer::zeros(ctx, &lbl("m2_w_f"), weight_len),
            m1_w_g: GpuBuffer::zeros(ctx, &lbl("m1_w_g"), weight_len),
            m2_w_g: GpuBuffer::zeros(ctx, &lbl("m2_w_g"), weight_len),
            m1_w_h: GpuBuffer::zeros(ctx, &lbl("m1_w_h"), weight_len),
            m2_w_h: GpuBuffer::zeros(ctx, &lbl("m2_w_h"), weight_len),
            m1_b_f: GpuBuffer::zeros(ctx, &lbl("m1_b_f"), bias_len),
            m2_b_f: GpuBuffer::zeros(ctx, &lbl("m2_b_f"), bias_len),
            m1_b_g: GpuBuffer::zeros(ctx, &lbl("m1_b_g"), bias_len),
            m2_b_g: GpuBuffer::zeros(ctx, &lbl("m2_b_g"), bias_len),
            m1_b_h: GpuBuffer::zeros(ctx, &lbl("m1_b_h"), bias_len),
            m2_b_h: GpuBuffer::zeros(ctx, &lbl("m2_b_h"), bias_len),
            m1_log_tau: GpuBuffer::zeros(ctx, &lbl("m1_log_tau"), bias_len),
            m2_log_tau: GpuBuffer::zeros(ctx, &lbl("m2_log_tau"), bias_len),

            w_f_grad: GpuBuffer::zeros(ctx, &lbl("w_f_grad"), weight_len),
            w_g_grad: GpuBuffer::zeros(ctx, &lbl("w_g_grad"), weight_len),
            w_h_grad: GpuBuffer::zeros(ctx, &lbl("w_h_grad"), weight_len),
            b_f_grad: GpuBuffer::zeros(ctx, &lbl("b_f_grad"), bias_len),
            b_g_grad: GpuBuffer::zeros(ctx, &lbl("b_g_grad"), bias_len),
            b_h_grad: GpuBuffer::zeros(ctx, &lbl("b_h_grad"), bias_len),
            log_tau_grad: GpuBuffer::zeros(ctx, &lbl("log_tau_grad"), bias_len),
        }
    }
}

/// Per-(B, T) chunk caches: activation packs, per-batch grads, tops &
/// inter-layer signals. Allocated once per `(B, T)` shape.
struct ChunkCache {
    batch_size: usize,
    n_timesteps: usize,
    n_output: usize,
    layer_dims: Vec<usize>,

    input_tokens: GpuBuffer,
    targets: GpuBuffer,

    f_packed: Vec<GpuBuffer>,
    g_packed: Vec<GpuBuffer>,
    h_packed: Vec<GpuBuffer>,
    alpha_packed: Vec<GpuBuffer>,
    x_packed: Vec<GpuBuffer>,
    initial_x: Vec<GpuBuffer>,
    final_x: Vec<GpuBuffer>,

    logits: GpuBuffer,
    d_logits: GpuBuffer,
    losses: GpuBuffer,
    d_x_top: GpuBuffer,

    d_input_per_t: Vec<GpuBuffer>,

    w_f_grad_pb: Vec<GpuBuffer>,
    w_g_grad_pb: Vec<GpuBuffer>,
    w_h_grad_pb: Vec<GpuBuffer>,
    b_f_grad_pb: Vec<GpuBuffer>,
    b_g_grad_pb: Vec<GpuBuffer>,
    b_h_grad_pb: Vec<GpuBuffer>,
    log_tau_grad_pb: Vec<GpuBuffer>,
    w_out_grad_pb: GpuBuffer,
    b_o_grad_pb: GpuBuffer,
}

impl ChunkCache {
    fn new(
        ctx: &GpuContext,
        batch_size: usize,
        n_timesteps: usize,
        n_output: usize,
        layer_dims: &[usize],
    ) -> Self {
        let n_layers = layer_dims.len() - 1;
        let mut f_packed = Vec::with_capacity(n_layers);
        let mut g_packed = Vec::with_capacity(n_layers);
        let mut h_packed = Vec::with_capacity(n_layers);
        let mut alpha_packed = Vec::with_capacity(n_layers);
        let mut x_packed = Vec::with_capacity(n_layers);
        let mut initial_x = Vec::with_capacity(n_layers);
        let mut final_x = Vec::with_capacity(n_layers);

        let mut w_f_pb = Vec::with_capacity(n_layers);
        let mut w_g_pb = Vec::with_capacity(n_layers);
        let mut w_h_pb = Vec::with_capacity(n_layers);
        let mut b_f_pb = Vec::with_capacity(n_layers);
        let mut b_g_pb = Vec::with_capacity(n_layers);
        let mut b_h_pb = Vec::with_capacity(n_layers);
        let mut lt_pb = Vec::with_capacity(n_layers);
        let mut d_input_per_t = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let n_in_l = layer_dims[l];
            let n_h = layer_dims[l + 1];
            let weight_len = n_h * (n_in_l + n_h);
            let bias_len = n_h;
            f_packed.push(GpuBuffer::zeros(ctx, "cache_f", batch_size * n_timesteps * n_h));
            g_packed.push(GpuBuffer::zeros(ctx, "cache_g", batch_size * n_timesteps * n_h));
            h_packed.push(GpuBuffer::zeros(ctx, "cache_h", batch_size * n_timesteps * n_h));
            alpha_packed.push(GpuBuffer::zeros(ctx, "cache_alpha", batch_size * n_timesteps * n_h));
            x_packed.push(GpuBuffer::zeros(ctx, "cache_x", batch_size * n_timesteps * n_h));
            initial_x.push(GpuBuffer::zeros(ctx, "cache_init_x", batch_size * n_h));
            final_x.push(GpuBuffer::zeros(ctx, "cache_final_x", batch_size * n_h));

            w_f_pb.push(GpuBuffer::zeros(ctx, "wf_pb", batch_size * weight_len));
            w_g_pb.push(GpuBuffer::zeros(ctx, "wg_pb", batch_size * weight_len));
            w_h_pb.push(GpuBuffer::zeros(ctx, "wh_pb", batch_size * weight_len));
            b_f_pb.push(GpuBuffer::zeros(ctx, "bf_pb", batch_size * bias_len));
            b_g_pb.push(GpuBuffer::zeros(ctx, "bg_pb", batch_size * bias_len));
            b_h_pb.push(GpuBuffer::zeros(ctx, "bh_pb", batch_size * bias_len));
            lt_pb.push(GpuBuffer::zeros(ctx, "lt_pb", batch_size * bias_len));

            let din_len = if l == 0 {
                1
            } else {
                batch_size * n_timesteps * n_in_l
            };
            d_input_per_t.push(GpuBuffer::zeros(ctx, "d_input_pt", din_len));
        }

        let h_top = *layer_dims.last().unwrap();

        Self {
            batch_size,
            n_timesteps,
            n_output,
            layer_dims: layer_dims.to_vec(),
            input_tokens: GpuBuffer::zeros(ctx, "input_tokens", batch_size * n_timesteps),
            targets: GpuBuffer::zeros(ctx, "targets", batch_size * n_timesteps),
            f_packed,
            g_packed,
            h_packed,
            alpha_packed,
            x_packed,
            initial_x,
            final_x,
            logits: GpuBuffer::zeros(ctx, "logits", batch_size * n_timesteps * n_output),
            d_logits: GpuBuffer::zeros(ctx, "d_logits", batch_size * n_timesteps * n_output),
            losses: GpuBuffer::zeros(ctx, "losses", batch_size * n_timesteps),
            d_x_top: GpuBuffer::zeros(ctx, "d_x_top", batch_size * n_timesteps * h_top),
            d_input_per_t,
            w_f_grad_pb: w_f_pb,
            w_g_grad_pb: w_g_pb,
            w_h_grad_pb: w_h_pb,
            b_f_grad_pb: b_f_pb,
            b_g_grad_pb: b_g_pb,
            b_h_grad_pb: b_h_pb,
            log_tau_grad_pb: lt_pb,
            w_out_grad_pb: GpuBuffer::zeros(ctx, "w_out_grad_pb", batch_size * n_output * h_top),
            b_o_grad_pb: GpuBuffer::zeros(ctx, "b_o_grad_pb", batch_size * n_output),
        }
    }

    fn matches(
        &self,
        batch_size: usize,
        n_timesteps: usize,
        n_output: usize,
        layer_dims: &[usize],
    ) -> bool {
        self.batch_size == batch_size
            && self.n_timesteps == n_timesteps
            && self.n_output == n_output
            && self.layer_dims == layer_dims
    }
}

struct Pipeline {
    pipeline: Arc<wgpu::ComputePipeline>,
    bgls: Vec<Arc<wgpu::BindGroupLayout>>,
}

pub struct MultiLayerLnnSnnGpu {
    ctx: GpuContext,
    pub n_input: usize,
    pub n_output: usize,
    pub layers: Vec<LnnLayerGpu>,
    pub w_out: GpuBuffer,
    pub b_o: GpuBuffer,

    pub m1_w_out: GpuBuffer,
    pub m2_w_out: GpuBuffer,
    pub m1_b_o: GpuBuffer,
    pub m2_b_o: GpuBuffer,
    pub w_out_grad: GpuBuffer,
    pub b_o_grad: GpuBuffer,

    cache: Option<ChunkCache>,

    p_forward_zero: Pipeline,
    p_forward_deep: Pipeline,
    p_output_proj: Pipeline,
    p_softmax_ce: Pipeline,
    p_topproj_back: Pipeline,
    p_back_zero: Pipeline,
    p_back_deep: Pipeline,
    p_reduce: Pipeline,
    p_zero: Pipeline,
    p_scale: Pipeline,
    p_adam: Pipeline,
    p_clamp: Pipeline,
    p_l2: Pipeline,

    adam_t: usize,
}

impl MultiLayerLnnSnnGpu {
    pub fn from_cpu(net: &MultiLayerLnnSnn, ctx: GpuContext) -> Self {
        let layers: Vec<LnnLayerGpu> = net
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LnnLayerGpu::from_cpu(l, &ctx, i))
            .collect();
        let h_top = net.layers.last().unwrap().n_hidden;
        let w_out = GpuBuffer::from_slice(&ctx, "w_out", &net.w_out);
        let b_o = GpuBuffer::from_slice(&ctx, "b_o", &net.b_o);
        let w_out_len = net.n_output * h_top;
        let b_o_len = net.n_output;

        let p_forward_zero = build_forward_zero(&ctx);
        let p_forward_deep = build_forward_deep(&ctx);
        let p_output_proj = build_output_proj(&ctx);
        let p_softmax_ce = build_softmax_ce(&ctx);
        let p_topproj_back = build_topproj_back(&ctx);
        let p_back_zero = build_back_zero(&ctx);
        let p_back_deep = build_back_deep(&ctx);
        let p_reduce = build_reduce(&ctx);
        let p_zero = build_zero(&ctx);
        let p_scale = build_scale(&ctx);
        let p_adam = build_adam(&ctx);
        let p_clamp = build_clamp(&ctx);
        let p_l2 = build_l2(&ctx);

        Self {
            n_input: net.n_input,
            n_output: net.n_output,
            layers,
            w_out,
            b_o,
            m1_w_out: GpuBuffer::zeros(&ctx, "m1_w_out", w_out_len),
            m2_w_out: GpuBuffer::zeros(&ctx, "m2_w_out", w_out_len),
            m1_b_o: GpuBuffer::zeros(&ctx, "m1_b_o", b_o_len),
            m2_b_o: GpuBuffer::zeros(&ctx, "m2_b_o", b_o_len),
            w_out_grad: GpuBuffer::zeros(&ctx, "w_out_grad", w_out_len),
            b_o_grad: GpuBuffer::zeros(&ctx, "b_o_grad", b_o_len),
            cache: None,
            p_forward_zero,
            p_forward_deep,
            p_output_proj,
            p_softmax_ce,
            p_topproj_back,
            p_back_zero,
            p_back_deep,
            p_reduce,
            p_zero,
            p_scale,
            p_adam,
            p_clamp,
            p_l2,
            adam_t: 0,
            ctx,
        }
    }

    pub fn to_cpu(&self) -> MultiLayerLnnSnn {
        let hidden_dims: Vec<usize> = self.layers.iter().map(|l| l.n_hidden).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut net = MultiLayerLnnSnn::new(self.n_input, &hidden_dims, self.n_output, &mut rng);
        for (i, layer) in self.layers.iter().enumerate() {
            net.layers[i].w_f = layer.w_f.download(&self.ctx);
            net.layers[i].w_g = layer.w_g.download(&self.ctx);
            net.layers[i].w_h = layer.w_h.download(&self.ctx);
            net.layers[i].b_f = layer.b_f.download(&self.ctx);
            net.layers[i].b_g = layer.b_g.download(&self.ctx);
            net.layers[i].b_h = layer.b_h.download(&self.ctx);
            net.layers[i].log_tau = layer.log_tau.download(&self.ctx);
        }
        net.w_out = self.w_out.download(&self.ctx);
        net.b_o = self.b_o.download(&self.ctx);
        net
    }

    pub fn fresh_states(&self) -> Vec<LnnState> {
        self.layers
            .iter()
            .map(|l| LnnState {
                x: vec![0.0; l.n_hidden],
            })
            .collect()
    }

    fn ensure_cache(&mut self, batch_size: usize, n_timesteps: usize) {
        let layer_dims: Vec<usize> = std::iter::once(self.n_input)
            .chain(self.layers.iter().map(|l| l.n_hidden))
            .collect();
        let needs_new = match &self.cache {
            Some(c) => !c.matches(batch_size, n_timesteps, self.n_output, &layer_dims),
            None => true,
        };
        if needs_new {
            self.cache = Some(ChunkCache::new(
                &self.ctx,
                batch_size,
                n_timesteps,
                self.n_output,
                &layer_dims,
            ));
        }
    }

    fn cache(&self) -> &ChunkCache {
        self.cache.as_ref().expect("cache not initialized")
    }

    /// Stateful forward over a single chunk for `batch_size` streams.
    /// Each row of `batch_inputs` is a slice of token IDs of length `T`.
    /// State is updated in place from the per-layer `final_x` GPU
    /// buffers.
    pub fn forward_lm_tokens_stateful_batch(
        &mut self,
        batch_inputs: &[&[u32]],
        states: &mut [Vec<LnnState>],
    ) {
        let batch_size = batch_inputs.len();
        assert_eq!(states.len(), batch_size);
        let n_timesteps = batch_inputs[0].len();
        for inp in batch_inputs {
            assert_eq!(inp.len(), n_timesteps);
        }
        self.ensure_cache(batch_size, n_timesteps);

        let mut tokens_flat: Vec<u32> = Vec::with_capacity(batch_size * n_timesteps);
        for inp in batch_inputs {
            tokens_flat.extend_from_slice(inp);
        }
        upload_u32(&self.ctx, &self.cache().input_tokens, &tokens_flat);

        for l in 0..self.layers.len() {
            let n_h = self.layers[l].n_hidden;
            let mut flat: Vec<f32> = Vec::with_capacity(batch_size * n_h);
            for st in states.iter() {
                assert_eq!(st[l].x.len(), n_h);
                flat.extend_from_slice(&st[l].x);
            }
            self.cache().initial_x[l].upload(&self.ctx, &flat);
        }

        self.run_forward();
        self.run_output_projection();

        for l in 0..self.layers.len() {
            let n_h = self.layers[l].n_hidden;
            let final_flat = self.cache().final_x[l].download(&self.ctx);
            for (b, st) in states.iter_mut().enumerate() {
                st[l].x.copy_from_slice(&final_flat[b * n_h..(b + 1) * n_h]);
            }
        }
    }

    fn run_forward(&self) {
        let cache = self.cache();
        let batch_size = cache.batch_size;
        let n_timesteps = cache.n_timesteps;

        for l in 0..self.layers.len() {
            let layer = &self.layers[l];
            let params = ForwardParams {
                n_in: layer.n_in as u32,
                n_hidden: layer.n_hidden as u32,
                n_timesteps: n_timesteps as u32,
                _pad: 0,
            };
            let bgl0 = if l == 0 {
                &self.p_forward_zero.bgls[0]
            } else {
                &self.p_forward_deep.bgls[0]
            };
            let bg0 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwd_bg0"),
                layout: bgl0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&layer.w_f) },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&layer.w_g) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&layer.w_h) },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&layer.b_f) },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&layer.b_g) },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&layer.b_h) },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&layer.log_tau) },
                ],
            });

            let bg1 = if l == 0 {
                self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fwd_bg1_zero"),
                    layout: &self.p_forward_zero.bgls[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.input_tokens) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.initial_x[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.f_packed[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.g_packed[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.h_packed[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.alpha_packed[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.x_packed[l]) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf_binding(&cache.final_x[l]) },
                    ],
                })
            } else {
                self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fwd_bg1_deep"),
                    layout: &self.p_forward_deep.bgls[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.x_packed[l - 1]) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.initial_x[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.f_packed[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.g_packed[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.h_packed[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.alpha_packed[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.x_packed[l]) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf_binding(&cache.final_x[l]) },
                    ],
                })
            };

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fwd_enc") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fwd_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if l == 0 {
                    &self.p_forward_zero.pipeline
                } else {
                    &self.p_forward_deep.pipeline
                });
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&params));
                pass.dispatch_workgroups(batch_size as u32, 1, 1);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
            // No explicit barrier between layers — wgpu inserts implicit
            // barriers when we re-bind storage buffers on a fresh pass.
        }
    }

    fn run_output_projection(&self) {
        let cache = self.cache();
        let top = self.layers.len() - 1;
        let h_top = self.layers[top].n_hidden;
        let params = ProjParams {
            n_hidden: h_top as u32,
            n_output: self.n_output as u32,
            n_timesteps: cache.n_timesteps as u32,
            _pad: 0,
        };
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proj_bg"),
            layout: &self.p_output_proj.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&self.w_out) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&self.b_o) },
                wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.x_packed[top]) },
                wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.logits) },
            ],
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("proj_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("proj_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_output_proj.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            let groups = (cache.batch_size * cache.n_timesteps) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Download logits per (b, t) for inspection / sample generation.
    /// Layout: `[B][T][V]` flattened.
    pub fn download_logits(&self) -> Vec<f32> {
        self.cache().logits.download(&self.ctx)
    }

    pub fn download_x_packed(&self, layer: usize) -> Vec<f32> {
        self.cache().x_packed[layer].download(&self.ctx)
    }
}

// -------------------- pipeline builders ----------------------

fn build_forward_zero(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_forward_zero", FORWARD_LAYER_ZERO_WGSL);
    let bgl0 = make_bgl(
        ctx,
        "fwd0_bgl0",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
        ],
    );
    let bgl1 = make_bgl(
        ctx,
        "fwd0_bgl1",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_forward_zero_pipeline",
        &module,
        &[&bgl0, &bgl1],
        std::mem::size_of::<ForwardParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl0, bgl1],
    }
}

fn build_forward_deep(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_forward_deep", FORWARD_LAYER_DEEP_WGSL);
    let bgl0 = make_bgl(
        ctx,
        "fwdd_bgl0",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
        ],
    );
    let bgl1 = make_bgl(
        ctx,
        "fwdd_bgl1",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_forward_deep_pipeline",
        &module,
        &[&bgl0, &bgl1],
        std::mem::size_of::<ForwardParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl0, bgl1],
    }
}

fn build_output_proj(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_output_proj", OUTPUT_PROJ_WGSL);
    let bgl = make_bgl(
        ctx,
        "proj_bgl",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_output_proj_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<ProjParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_softmax_ce(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_softmax_ce", SOFTMAX_CE_WGSL);
    let bgl = make_bgl(
        ctx,
        "ce_bgl",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_softmax_ce_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<CeParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_topproj_back(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_topproj_back", TOPPROJ_BACKWARD_WGSL);
    let bgl = make_bgl(
        ctx,
        "tpb_bgl",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_topproj_back_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<TopProjBackParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_back_zero(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_back_zero", BACKWARD_LAYER_ZERO_WGSL);
    let bgl0 = make_bgl(
        ctx,
        "bz_bgl0",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
        ],
    );
    let bgl1 = make_bgl(
        ctx,
        "bz_bgl1",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, true),
        ],
    );
    let bgl2 = make_bgl(
        ctx,
        "bz_bgl2",
        &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_back_zero_pipeline",
        &module,
        &[&bgl0, &bgl1, &bgl2],
        std::mem::size_of::<BackParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl0, bgl1, bgl2],
    }
}

fn build_back_deep(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_back_deep", BACKWARD_LAYER_DEEP_WGSL);
    let bgl0 = make_bgl(
        ctx,
        "bd_bgl0",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
        ],
    );
    let bgl1 = make_bgl(
        ctx,
        "bd_bgl1",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, true),
        ],
    );
    let bgl2 = make_bgl(
        ctx,
        "bd_bgl2",
        &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_back_deep_pipeline",
        &module,
        &[&bgl0, &bgl1, &bgl2],
        std::mem::size_of::<BackParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl0, bgl1, bgl2],
    }
}

fn build_reduce(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_reduce_pb", REDUCE_PER_BATCH_WGSL);
    let bgl = make_bgl(
        ctx,
        "reduce_bgl",
        &[storage_entry(0, true), storage_entry(1, false)],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_reduce_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<ReduceParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_zero(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_zero", ZERO_BUFFER_WGSL);
    let bgl = make_bgl(ctx, "zero_bgl", &[storage_entry(0, false)]);
    let pipeline = make_pipeline(
        ctx,
        "lnn_zero_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<ZeroParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_scale(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_scale", SCALE_BUFFER_WGSL);
    let bgl = make_bgl(ctx, "scale_bgl", &[storage_entry(0, false)]);
    let pipeline = make_pipeline(
        ctx,
        "lnn_scale_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<ScaleParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_adam(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_adam", ADAM_STEP_WGSL);
    let bgl = make_bgl(
        ctx,
        "adam_bgl",
        &[
            storage_entry(0, false),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_adam_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<AdamParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_clamp(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_clamp", CLAMP_LOG_TAU_WGSL);
    let bgl = make_bgl(ctx, "clamp_bgl", &[storage_entry(0, false)]);
    let pipeline = make_pipeline(
        ctx,
        "lnn_clamp_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<ClampParams>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

fn build_l2(ctx: &GpuContext) -> Pipeline {
    let module = make_module(ctx, "lnn_l2", L2_NORM_SQUARED_WGSL);
    let bgl = make_bgl(
        ctx,
        "l2_bgl",
        &[storage_entry(0, true), storage_entry(1, false)],
    );
    let pipeline = make_pipeline(
        ctx,
        "lnn_l2_pipeline",
        &module,
        &[&bgl],
        std::mem::size_of::<L2Params>() as u32,
    );
    Pipeline {
        pipeline,
        bgls: vec![bgl],
    }
}

// -------------------- helpers --------------------

fn upload_u32(ctx: &GpuContext, buf: &GpuBuffer, data: &[u32]) {
    debug_assert_eq!(data.len(), buf.len);
    ctx.queue
        .write_buffer(&buf.inner, 0, bytemuck::cast_slice(data));
}

fn dispatch_count(items: u32, workgroup: u32) -> u32 {
    items.div_ceil(workgroup)
}

// -------------------- backward + train + adam --------------------

impl MultiLayerLnnSnnGpu {
    /// Per-batch grad buffers + reduction kernel are bound to the
    /// per-layer total grads. Resets per-batch buffers to zero before
    /// the backward sweep so each backward call accumulates from a
    /// clean slate.
    fn zero_per_batch_grads(&self) {
        let cache = self.cache();
        for l in 0..self.layers.len() {
            self.dispatch_zero(&cache.w_f_grad_pb[l]);
            self.dispatch_zero(&cache.w_g_grad_pb[l]);
            self.dispatch_zero(&cache.w_h_grad_pb[l]);
            self.dispatch_zero(&cache.b_f_grad_pb[l]);
            self.dispatch_zero(&cache.b_g_grad_pb[l]);
            self.dispatch_zero(&cache.b_h_grad_pb[l]);
            self.dispatch_zero(&cache.log_tau_grad_pb[l]);
        }
        self.dispatch_zero(&cache.w_out_grad_pb);
        self.dispatch_zero(&cache.b_o_grad_pb);
    }

    /// Zero the aggregated per-parameter grad buffers (the targets of
    /// the reduction step). Called by `zero_grad` and at the start of
    /// each batch.
    pub fn zero_grad(&mut self) {
        for layer in &self.layers {
            self.dispatch_zero(&layer.w_f_grad);
            self.dispatch_zero(&layer.w_g_grad);
            self.dispatch_zero(&layer.w_h_grad);
            self.dispatch_zero(&layer.b_f_grad);
            self.dispatch_zero(&layer.b_g_grad);
            self.dispatch_zero(&layer.b_h_grad);
            self.dispatch_zero(&layer.log_tau_grad);
        }
        self.dispatch_zero(&self.w_out_grad);
        self.dispatch_zero(&self.b_o_grad);
    }

    fn dispatch_zero(&self, buf: &GpuBuffer) {
        let n = buf.len as u32;
        if n == 0 {
            return;
        }
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zero_bg"),
            layout: &self.p_zero.bgls[0],
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_binding(buf) }],
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero_enc") });
        let params = ZeroParams { n, _pad0: 0, _pad1: 0, _pad2: 0 };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zero_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_zero.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(dispatch_count(n, 256), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_reduce(&self, src: &GpuBuffer, dst: &GpuBuffer, batch_size: usize) {
        debug_assert_eq!(src.len, dst.len * batch_size);
        let elem_count = dst.len as u32;
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_bg"),
            layout: &self.p_reduce.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(src) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(dst) },
            ],
        });
        let params = ReduceParams {
            batch_size: batch_size as u32,
            elem_count,
            _pad0: 0,
            _pad1: 0,
        };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("reduce_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_reduce.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(dispatch_count(elem_count, 256), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_scale(&self, buf: &GpuBuffer, factor: f32) {
        let n = buf.len as u32;
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bg"),
            layout: &self.p_scale.bgls[0],
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_binding(buf) }],
        });
        let params = ScaleParams { n, factor, _pad0: 0, _pad1: 0 };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scale_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_scale.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(dispatch_count(n, 256), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_adam(
        &self,
        params_buf: &GpuBuffer,
        grad: &GpuBuffer,
        m1: &GpuBuffer,
        m2: &GpuBuffer,
        config: &AdamConfig,
        bc1: f32,
        bc2: f32,
    ) {
        let n = params_buf.len as u32;
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adam_bg"),
            layout: &self.p_adam.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(params_buf) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(grad) },
                wgpu::BindGroupEntry { binding: 2, resource: buf_binding(m1) },
                wgpu::BindGroupEntry { binding: 3, resource: buf_binding(m2) },
            ],
        });
        let params = AdamParams {
            n,
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            bc1,
            bc2,
            _pad: 0,
        };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("adam_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adam_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_adam.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(dispatch_count(n, 256), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_clamp(&self, buf: &GpuBuffer, bound: f32) {
        let n = buf.len as u32;
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clamp_bg"),
            layout: &self.p_clamp.bgls[0],
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf_binding(buf) }],
        });
        let params = ClampParams { n, bound, _pad0: 0, _pad1: 0 };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clamp_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clamp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_clamp.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(dispatch_count(n, 256), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Forward + backward over a chunked batch. Updates the activation
    /// cache, runs softmax+CE on the GPU producing `d_logits`, runs
    /// per-layer reverse-time backward, then reduces per-batch grads
    /// into the aggregated `*_grad` buffers.
    /// Returns the summed loss in nats over the chunk.
    pub fn forward_backward_batch_tokens_stateful(
        &mut self,
        batch_inputs: &[&[u32]],
        batch_targets: &[&[u32]],
        states: &mut [Vec<LnnState>],
    ) -> f32 {
        assert_eq!(batch_inputs.len(), batch_targets.len());
        let batch_size = batch_inputs.len();
        let n_timesteps = batch_inputs[0].len();
        for (i, t) in batch_targets.iter().enumerate() {
            assert_eq!(t.len(), n_timesteps, "target length mismatch at batch {i}");
        }

        self.forward_lm_tokens_stateful_batch(batch_inputs, states);

        let mut targets_flat: Vec<u32> = Vec::with_capacity(batch_size * n_timesteps);
        for tg in batch_targets {
            targets_flat.extend_from_slice(tg);
        }
        upload_u32(&self.ctx, &self.cache().targets, &targets_flat);

        self.run_softmax_ce();
        self.zero_per_batch_grads();
        self.run_topproj_backward();
        self.run_layers_backward();
        self.run_reductions();

        let losses = self.cache().losses.download(&self.ctx);
        losses.iter().sum()
    }

    fn run_softmax_ce(&self) {
        let cache = self.cache();
        let params = CeParams {
            n_output: self.n_output as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ce_bg"),
            layout: &self.p_softmax_ce.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.logits) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.targets) },
                wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.d_logits) },
                wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.losses) },
            ],
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ce_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_softmax_ce.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            let groups = (cache.batch_size * cache.n_timesteps) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn run_topproj_backward(&self) {
        let cache = self.cache();
        let top = self.layers.len() - 1;
        let h_top = self.layers[top].n_hidden;
        let params = TopProjBackParams {
            n_hidden: h_top as u32,
            n_output: self.n_output as u32,
            n_timesteps: cache.n_timesteps as u32,
            _pad: 0,
        };
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tpb_bg"),
            layout: &self.p_topproj_back.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&self.w_out) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.d_logits) },
                wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.x_packed[top]) },
                wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.d_x_top) },
                wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.w_out_grad_pb) },
                wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.b_o_grad_pb) },
            ],
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tpb_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tpb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_topproj_back.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(cache.batch_size as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn run_layers_backward(&self) {
        let cache = self.cache();
        let n_layers = self.layers.len();
        // d_x_above for each layer: top layer reads from d_x_top;
        // deeper layers read from the d_input_per_t that was written
        // by the LAYER ABOVE (l+1).
        for l in (0..n_layers).rev() {
            let layer = &self.layers[l];
            let params = BackParams {
                n_in: layer.n_in as u32,
                n_hidden: layer.n_hidden as u32,
                n_timesteps: cache.n_timesteps as u32,
                _pad: 0,
            };
            let d_x_above = if l == n_layers - 1 {
                &cache.d_x_top
            } else {
                &cache.d_input_per_t[l + 1]
            };
            if l == 0 {
                let bg0 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bz_bg0"),
                    layout: &self.p_back_zero.bgls[0],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&layer.w_f) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&layer.w_g) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&layer.w_h) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&layer.log_tau) },
                    ],
                });
                let bg1 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bz_bg1"),
                    layout: &self.p_back_zero.bgls[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.input_tokens) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.initial_x[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.f_packed[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.g_packed[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.h_packed[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.alpha_packed[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.x_packed[l]) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf_binding(d_x_above) },
                    ],
                });
                let bg2 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bz_bg2"),
                    layout: &self.p_back_zero.bgls[2],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.w_f_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.w_g_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.w_h_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.b_f_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.b_g_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.b_h_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.log_tau_grad_pb[l]) },
                    ],
                });
                let mut encoder = self
                    .ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bz_enc") });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bz_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.p_back_zero.pipeline);
                    pass.set_bind_group(0, &bg0, &[]);
                    pass.set_bind_group(1, &bg1, &[]);
                    pass.set_bind_group(2, &bg2, &[]);
                    pass.set_push_constants(0, bytemuck::bytes_of(&params));
                    pass.dispatch_workgroups(cache.batch_size as u32, 1, 1);
                }
                self.ctx.queue.submit(Some(encoder.finish()));
            } else {
                let bg0 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bd_bg0"),
                    layout: &self.p_back_deep.bgls[0],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&layer.w_f) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&layer.w_g) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&layer.w_h) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&layer.log_tau) },
                    ],
                });
                let bg1 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bd_bg1"),
                    layout: &self.p_back_deep.bgls[1],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.x_packed[l - 1]) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.initial_x[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.f_packed[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.g_packed[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.h_packed[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.alpha_packed[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.x_packed[l]) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf_binding(d_x_above) },
                    ],
                });
                let bg2 = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bd_bg2"),
                    layout: &self.p_back_deep.bgls[2],
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf_binding(&cache.w_f_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&cache.w_g_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 2, resource: buf_binding(&cache.w_h_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 3, resource: buf_binding(&cache.b_f_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 4, resource: buf_binding(&cache.b_g_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 5, resource: buf_binding(&cache.b_h_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 6, resource: buf_binding(&cache.log_tau_grad_pb[l]) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf_binding(&cache.d_input_per_t[l]) },
                    ],
                });
                let mut encoder = self
                    .ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bd_enc") });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bd_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.p_back_deep.pipeline);
                    pass.set_bind_group(0, &bg0, &[]);
                    pass.set_bind_group(1, &bg1, &[]);
                    pass.set_bind_group(2, &bg2, &[]);
                    pass.set_push_constants(0, bytemuck::bytes_of(&params));
                    pass.dispatch_workgroups(cache.batch_size as u32, 1, 1);
                }
                self.ctx.queue.submit(Some(encoder.finish()));
            }
        }
    }

    fn run_reductions(&self) {
        let cache = self.cache();
        let bs = cache.batch_size;
        for l in 0..self.layers.len() {
            let layer = &self.layers[l];
            self.dispatch_reduce(&cache.w_f_grad_pb[l], &layer.w_f_grad, bs);
            self.dispatch_reduce(&cache.w_g_grad_pb[l], &layer.w_g_grad, bs);
            self.dispatch_reduce(&cache.w_h_grad_pb[l], &layer.w_h_grad, bs);
            self.dispatch_reduce(&cache.b_f_grad_pb[l], &layer.b_f_grad, bs);
            self.dispatch_reduce(&cache.b_g_grad_pb[l], &layer.b_g_grad, bs);
            self.dispatch_reduce(&cache.b_h_grad_pb[l], &layer.b_h_grad, bs);
            self.dispatch_reduce(&cache.log_tau_grad_pb[l], &layer.log_tau_grad, bs);
        }
        self.dispatch_reduce(&cache.w_out_grad_pb, &self.w_out_grad, bs);
        self.dispatch_reduce(&cache.b_o_grad_pb, &self.b_o_grad, bs);
    }

    /// Compute the L2 norm of all aggregated grads and rescale them
    /// in-place to `max_norm` if larger.
    pub fn clip_grads(&self, max_norm: f32) {
        let mut sq = 0.0_f32;
        for layer in &self.layers {
            sq += self.l2_sum_sq(&layer.w_f_grad);
            sq += self.l2_sum_sq(&layer.w_g_grad);
            sq += self.l2_sum_sq(&layer.w_h_grad);
            sq += self.l2_sum_sq(&layer.b_f_grad);
            sq += self.l2_sum_sq(&layer.b_g_grad);
            sq += self.l2_sum_sq(&layer.b_h_grad);
            sq += self.l2_sum_sq(&layer.log_tau_grad);
        }
        sq += self.l2_sum_sq(&self.w_out_grad);
        sq += self.l2_sum_sq(&self.b_o_grad);
        let norm = sq.sqrt();
        if norm <= max_norm || norm == 0.0 {
            return;
        }
        let factor = max_norm / norm;
        for layer in &self.layers {
            self.dispatch_scale(&layer.w_f_grad, factor);
            self.dispatch_scale(&layer.w_g_grad, factor);
            self.dispatch_scale(&layer.w_h_grad, factor);
            self.dispatch_scale(&layer.b_f_grad, factor);
            self.dispatch_scale(&layer.b_g_grad, factor);
            self.dispatch_scale(&layer.b_h_grad, factor);
            self.dispatch_scale(&layer.log_tau_grad, factor);
        }
        self.dispatch_scale(&self.w_out_grad, factor);
        self.dispatch_scale(&self.b_o_grad, factor);
    }

    fn l2_sum_sq(&self, buf: &GpuBuffer) -> f32 {
        let n = buf.len as u32;
        if n == 0 {
            return 0.0;
        }
        let n_groups = dispatch_count(n, 256);
        let partial = GpuBuffer::zeros(&self.ctx, "l2_partial", n_groups as usize);
        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_bg"),
            layout: &self.p_l2.bgls[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_binding(buf) },
                wgpu::BindGroupEntry { binding: 1, resource: buf_binding(&partial) },
            ],
        });
        let params = L2Params { n, _pad0: 0, _pad1: 0, _pad2: 0 };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("l2_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_l2.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let cpu = partial.download(&self.ctx);
        cpu.iter().sum()
    }

    pub fn step_adam(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        for layer in &self.layers {
            self.dispatch_adam(&layer.w_f, &layer.w_f_grad, &layer.m1_w_f, &layer.m2_w_f, config, bc1, bc2);
            self.dispatch_adam(&layer.w_g, &layer.w_g_grad, &layer.m1_w_g, &layer.m2_w_g, config, bc1, bc2);
            self.dispatch_adam(&layer.w_h, &layer.w_h_grad, &layer.m1_w_h, &layer.m2_w_h, config, bc1, bc2);
            self.dispatch_adam(&layer.b_f, &layer.b_f_grad, &layer.m1_b_f, &layer.m2_b_f, config, bc1, bc2);
            self.dispatch_adam(&layer.b_g, &layer.b_g_grad, &layer.m1_b_g, &layer.m2_b_g, config, bc1, bc2);
            self.dispatch_adam(&layer.b_h, &layer.b_h_grad, &layer.m1_b_h, &layer.m2_b_h, config, bc1, bc2);
            self.dispatch_adam(
                &layer.log_tau,
                &layer.log_tau_grad,
                &layer.m1_log_tau,
                &layer.m2_log_tau,
                config,
                bc1,
                bc2,
            );
            self.dispatch_clamp(&layer.log_tau, LOG_TAU_CLAMP);
        }
        self.dispatch_adam(
            &self.w_out,
            &self.w_out_grad,
            &self.m1_w_out,
            &self.m2_w_out,
            config,
            bc1,
            bc2,
        );
        self.dispatch_adam(
            &self.b_o,
            &self.b_o_grad,
            &self.m1_b_o,
            &self.m2_b_o,
            config,
            bc1,
            bc2,
        );
    }
}
