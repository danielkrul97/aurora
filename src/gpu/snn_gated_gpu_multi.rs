#![allow(
    clippy::type_complexity,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

//! Sprint 25: GPU mirror of `MultiLayerGatedSnn`, hardcoded 2-layer.
//!
//! Forward runs as one megakernel dispatch per chunk (chains 2 gated
//! cells in-shader). Backward runs sequentially per layer per
//! timestep, calling existing single-layer backward kernels with
//! per-layer offsets into the packed cache buffers.
//!
//! Constraints (Sprint 25 v1):
//! - Hardcoded 2 layers.
//! - Uniform `n_hidden` across layers (constructor enforces).
//! - `n_hidden <= 512` (workgroup memory budget in `forward_mega_2layer`).
//! - Layer 0 reads byte input; layer 1 reads layer 0's spike output
//!   via `outer_product_accum` (hidden→hidden W_in_1).
//!
//! Sprint 26+ followups: generic `n_layers`, heterogeneous
//! `hidden_dims`, larger `n_hidden`.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::forward_mega_2layer::{ForwardMega2LayerKernel, ForwardMega2LayerParams};
use super::kernels::{
    AdamStepKernel, ColumnAccumKernel, DenseMatvecTransposeKernel, GatedCellBackwardKernel,
    OuterProductAccumKernel, ScaleBufferKernel, SoftmaxCrossEntropyKernel, VectorAddKernel,
    ZeroBufferKernel,
};
use super::GpuContext;
use crate::optimizer::{bias_correction, AdamConfig};
use crate::snn_gated_multi::{LayerGrads, MultiLayerGatedGrads, MultiLayerGatedSnn};

/// Per-layer + top-level GPU gradient buffers.
pub struct MultiLayerGpuGrads {
    pub w_in_0: GpuBuffer,
    pub w_rec_0: GpuBuffer,
    pub w_gate_in_0: GpuBuffer,
    pub w_gate_rec_0: GpuBuffer,
    pub b_h_0: GpuBuffer,
    pub b_gate_0: GpuBuffer,
    pub w_in_1: GpuBuffer,
    pub w_rec_1: GpuBuffer,
    pub w_gate_in_1: GpuBuffer,
    pub w_gate_rec_1: GpuBuffer,
    pub b_h_1: GpuBuffer,
    pub b_gate_1: GpuBuffer,
    pub w_out: GpuBuffer,
    pub b_o: GpuBuffer,
}

impl MultiLayerGpuGrads {
    fn zeros(ctx: &GpuContext, n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        Self {
            w_in_0: GpuBuffer::zeros(ctx, "g_w_in_0", n_hidden * n_input),
            w_rec_0: GpuBuffer::zeros(ctx, "g_w_rec_0", n_hidden * n_hidden),
            w_gate_in_0: GpuBuffer::zeros(ctx, "g_w_gate_in_0", n_hidden * n_input),
            w_gate_rec_0: GpuBuffer::zeros(ctx, "g_w_gate_rec_0", n_hidden * n_hidden),
            b_h_0: GpuBuffer::zeros(ctx, "g_b_h_0", n_hidden),
            b_gate_0: GpuBuffer::zeros(ctx, "g_b_gate_0", n_hidden),
            w_in_1: GpuBuffer::zeros(ctx, "g_w_in_1", n_hidden * n_hidden),
            w_rec_1: GpuBuffer::zeros(ctx, "g_w_rec_1", n_hidden * n_hidden),
            w_gate_in_1: GpuBuffer::zeros(ctx, "g_w_gate_in_1", n_hidden * n_hidden),
            w_gate_rec_1: GpuBuffer::zeros(ctx, "g_w_gate_rec_1", n_hidden * n_hidden),
            b_h_1: GpuBuffer::zeros(ctx, "g_b_h_1", n_hidden),
            b_gate_1: GpuBuffer::zeros(ctx, "g_b_gate_1", n_hidden),
            w_out: GpuBuffer::zeros(ctx, "g_w_out", n_output * n_hidden),
            b_o: GpuBuffer::zeros(ctx, "g_b_o", n_output),
        }
    }

    fn iter_buffers(&self) -> [&GpuBuffer; 14] {
        [
            &self.w_in_0,
            &self.w_rec_0,
            &self.w_gate_in_0,
            &self.w_gate_rec_0,
            &self.b_h_0,
            &self.b_gate_0,
            &self.w_in_1,
            &self.w_rec_1,
            &self.w_gate_in_1,
            &self.w_gate_rec_1,
            &self.b_h_1,
            &self.b_gate_1,
            &self.w_out,
            &self.b_o,
        ]
    }
}

/// Adam first/second moment buffers — one pair per parameter group
/// (14 groups for 2-layer + top).
pub struct MultiLayerGpuAdam {
    m1: [GpuBuffer; 14],
    m2: [GpuBuffer; 14],
}

impl MultiLayerGpuAdam {
    fn zeros(ctx: &GpuContext, n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        let sizes: [usize; 14] = [
            n_hidden * n_input,
            n_hidden * n_hidden,
            n_hidden * n_input,
            n_hidden * n_hidden,
            n_hidden,
            n_hidden,
            n_hidden * n_hidden,
            n_hidden * n_hidden,
            n_hidden * n_hidden,
            n_hidden * n_hidden,
            n_hidden,
            n_hidden,
            n_output * n_hidden,
            n_output,
        ];
        let labels: [&str; 14] = [
            "m_w_in_0", "m_w_rec_0", "m_w_gate_in_0", "m_w_gate_rec_0", "m_b_h_0", "m_b_gate_0",
            "m_w_in_1", "m_w_rec_1", "m_w_gate_in_1", "m_w_gate_rec_1", "m_b_h_1", "m_b_gate_1",
            "m_w_out", "m_b_o",
        ];
        let mut m1: Vec<GpuBuffer> = Vec::with_capacity(14);
        let mut m2: Vec<GpuBuffer> = Vec::with_capacity(14);
        for (i, &sz) in sizes.iter().enumerate() {
            m1.push(GpuBuffer::zeros(ctx, labels[i], sz));
            m2.push(GpuBuffer::zeros(ctx, labels[i], sz));
        }
        Self {
            m1: m1.try_into().ok().expect("14 buffers"),
            m2: m2.try_into().ok().expect("14 buffers"),
        }
    }
}

/// Per-chunk forward+backward scratch. Cache buffers carry layer dim
/// (`B × L × T × N` or `B × L × (T+1) × N`).
struct FbScratch {
    capacity_t: usize,
    capacity_b: usize,
    n_hidden: usize,
    n_output: usize,

    h_v_pre_packed: GpuBuffer,
    h_v_after_packed: GpuBuffer,
    h_spike_packed: GpuBuffer,
    gate_packed: GpuBuffer,
    logits_packed: GpuBuffer,
    d_logits_packed: GpuBuffer,
    loss_seq: GpuBuffer,
    input_bytes_buf: GpuBuffer,
    target_bytes_buf: GpuBuffer,

    /// Backward scratch per layer (slot 0 = layer 0, slot 1 = layer 1).
    d_h_spike_from_above: [GpuBuffer; 2],     // [B*N] each
    d_h_spike_recurrent: [GpuBuffer; 2],
    d_h_v_after_carry_a: [GpuBuffer; 2],
    d_h_v_after_carry_b: [GpuBuffer; 2],
    d_cell_packed: [GpuBuffer; 2],            // [3 * B * N] each, slot-major
    d_h_v_pre: [GpuBuffer; 2],
    d_gate_pre: [GpuBuffer; 2],

    initial_h_v_after: GpuBuffer,
    initial_h_spike: GpuBuffer,
}

impl FbScratch {
    fn new(
        ctx: &GpuContext,
        n_hidden: usize,
        n_output: usize,
        initial_capacity_t: usize,
        initial_capacity_b: usize,
    ) -> Self {
        let t = initial_capacity_t.max(1);
        let b = initial_capacity_b.max(1);
        let l = 2_usize;
        Self {
            capacity_t: t,
            capacity_b: b,
            n_hidden,
            n_output,
            h_v_pre_packed: GpuBuffer::zeros(ctx, "h_v_pre_packed", b * l * t * n_hidden),
            h_v_after_packed: GpuBuffer::zeros(ctx, "h_v_after_packed", b * l * (t + 1) * n_hidden),
            h_spike_packed: GpuBuffer::zeros(ctx, "h_spike_packed", b * l * (t + 1) * n_hidden),
            gate_packed: GpuBuffer::zeros(ctx, "gate_packed", b * l * t * n_hidden),
            logits_packed: GpuBuffer::zeros(ctx, "logits_packed", b * t * n_output),
            d_logits_packed: GpuBuffer::zeros(ctx, "d_logits_packed", b * t * n_output),
            loss_seq: GpuBuffer::zeros(ctx, "loss_seq", b * t),
            input_bytes_buf: GpuBuffer::zeros(ctx, "input_bytes", b * t),
            target_bytes_buf: GpuBuffer::zeros(ctx, "target_bytes", b * t),
            d_h_spike_from_above: [
                GpuBuffer::zeros(ctx, "d_h_spike_from_above_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_h_spike_from_above_l1", b * n_hidden),
            ],
            d_h_spike_recurrent: [
                GpuBuffer::zeros(ctx, "d_h_spike_rec_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_h_spike_rec_l1", b * n_hidden),
            ],
            d_h_v_after_carry_a: [
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_a_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_a_l1", b * n_hidden),
            ],
            d_h_v_after_carry_b: [
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_b_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_b_l1", b * n_hidden),
            ],
            d_cell_packed: [
                GpuBuffer::zeros(ctx, "d_cell_packed_l0", 3 * b * n_hidden),
                GpuBuffer::zeros(ctx, "d_cell_packed_l1", 3 * b * n_hidden),
            ],
            d_h_v_pre: [
                GpuBuffer::zeros(ctx, "d_h_v_pre_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_h_v_pre_l1", b * n_hidden),
            ],
            d_gate_pre: [
                GpuBuffer::zeros(ctx, "d_gate_pre_l0", b * n_hidden),
                GpuBuffer::zeros(ctx, "d_gate_pre_l1", b * n_hidden),
            ],
            initial_h_v_after: GpuBuffer::zeros(ctx, "initial_h_v_after", b * l * n_hidden),
            initial_h_spike: GpuBuffer::zeros(ctx, "initial_h_spike", b * l * n_hidden),
        }
    }

    fn ensure_capacity(&mut self, ctx: &GpuContext, t: usize, b: usize) -> bool {
        if t == self.capacity_t && b == self.capacity_b {
            return false;
        }
        *self = Self::new(ctx, self.n_hidden, self.n_output, t, b);
        true
    }
}

pub struct MultiLayerGatedSnnGpu {
    pub ctx: GpuContext,
    pub n_input: u32,
    pub n_hidden: u32,
    pub n_output: u32,
    pub threshold: f32,
    pub alpha: f32,

    // Layer 0 weights
    pub w_in_0: GpuBuffer,
    pub w_rec_0: GpuBuffer,
    pub w_gate_in_0: GpuBuffer,
    pub w_gate_rec_0: GpuBuffer,
    pub b_h_0: GpuBuffer,
    pub b_gate_0: GpuBuffer,
    // Layer 1 weights
    pub w_in_1: GpuBuffer,
    pub w_rec_1: GpuBuffer,
    pub w_gate_in_1: GpuBuffer,
    pub w_gate_rec_1: GpuBuffer,
    pub b_h_1: GpuBuffer,
    pub b_gate_1: GpuBuffer,
    // Top
    pub w_out: GpuBuffer,
    pub b_o: GpuBuffer,

    pub grads: MultiLayerGpuGrads,
    adam: MultiLayerGpuAdam,
    pub adam_t: usize,

    forward_mega: Arc<ForwardMega2LayerKernel>,
    transpose: Arc<DenseMatvecTransposeKernel>,
    cell_back: Arc<GatedCellBackwardKernel>,
    outer: Arc<OuterProductAccumKernel>,
    column: Arc<ColumnAccumKernel>,
    vector_add: Arc<VectorAddKernel>,
    softmax_ce: Arc<SoftmaxCrossEntropyKernel>,
    adam_kernel: Arc<AdamStepKernel>,
    scale: Arc<ScaleBufferKernel>,
    zero: Arc<ZeroBufferKernel>,

    bind_l0: wgpu::BindGroup,
    bind_l1: wgpu::BindGroup,
    bind_io: Option<wgpu::BindGroup>,

    fb: FbScratch,
}

impl MultiLayerGatedSnnGpu {
    /// Upload CPU multi-layer network (must be 2 layers, uniform
    /// `n_hidden`) to GPU.
    ///
    /// # Panics
    /// Panics if `net.layers.len() != 2` or layer hidden dims differ.
    #[must_use]
    pub fn from_cpu(net: &MultiLayerGatedSnn, ctx: GpuContext) -> Self {
        assert_eq!(net.layers.len(), 2, "Sprint 25 supports exactly 2 layers");
        let n_hidden = net.layers[0].n_hidden;
        assert_eq!(
            net.layers[1].n_hidden, n_hidden,
            "Sprint 25 requires uniform hidden across layers"
        );
        assert_eq!(
            net.layers[1].n_in, n_hidden,
            "layer 1 input dim must equal layer 0 hidden"
        );
        let n_input = net.n_input;
        let n_output = net.n_output;

        let l0 = &net.layers[0];
        let l1 = &net.layers[1];
        let w_in_0 = GpuBuffer::from_slice(&ctx, "w_in_0", &l0.w_in);
        let w_rec_0 = GpuBuffer::from_slice(&ctx, "w_rec_0", &l0.w_rec);
        let w_gate_in_0 = GpuBuffer::from_slice(&ctx, "w_gate_in_0", &l0.w_gate_in);
        let w_gate_rec_0 = GpuBuffer::from_slice(&ctx, "w_gate_rec_0", &l0.w_gate_rec);
        let b_h_0 = GpuBuffer::from_slice(&ctx, "b_h_0", &l0.b_h);
        let b_gate_0 = GpuBuffer::from_slice(&ctx, "b_gate_0", &l0.b_gate);
        let w_in_1 = GpuBuffer::from_slice(&ctx, "w_in_1", &l1.w_in);
        let w_rec_1 = GpuBuffer::from_slice(&ctx, "w_rec_1", &l1.w_rec);
        let w_gate_in_1 = GpuBuffer::from_slice(&ctx, "w_gate_in_1", &l1.w_gate_in);
        let w_gate_rec_1 = GpuBuffer::from_slice(&ctx, "w_gate_rec_1", &l1.w_gate_rec);
        let b_h_1 = GpuBuffer::from_slice(&ctx, "b_h_1", &l1.b_h);
        let b_gate_1 = GpuBuffer::from_slice(&ctx, "b_gate_1", &l1.b_gate);
        let w_out = GpuBuffer::from_slice(&ctx, "w_out", &net.w_out);
        let b_o = GpuBuffer::from_slice(&ctx, "b_o", &net.b_o);

        let forward_mega = Arc::new(ForwardMega2LayerKernel::new(&ctx));
        let transpose = Arc::new(DenseMatvecTransposeKernel::new(&ctx));
        let cell_back = Arc::new(GatedCellBackwardKernel::new(&ctx));
        let outer = Arc::new(OuterProductAccumKernel::new(&ctx));
        let column = Arc::new(ColumnAccumKernel::new(&ctx));
        let vector_add = Arc::new(VectorAddKernel::new(&ctx));
        let softmax_ce = Arc::new(SoftmaxCrossEntropyKernel::new(&ctx));
        let adam_kernel = Arc::new(AdamStepKernel::new(&ctx));
        let scale = Arc::new(ScaleBufferKernel::new(&ctx));
        let zero = Arc::new(ZeroBufferKernel::new(&ctx));

        let grads = MultiLayerGpuGrads::zeros(&ctx, n_input, n_hidden, n_output);
        let adam = MultiLayerGpuAdam::zeros(&ctx, n_input, n_hidden, n_output);
        let fb = FbScratch::new(&ctx, n_hidden, n_output, 64, 4);

        let bind_l0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora multi 2L bind l0"),
            layout: &forward_mega.bgl_l0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_in_0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_rec_0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_gate_in_0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: w_gate_rec_0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_h_0.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_gate_0.inner.as_entire_binding() },
            ],
        });
        let bind_l1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora multi 2L bind l1+top"),
            layout: &forward_mega.bgl_l1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_in_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_rec_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w_gate_in_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: w_gate_rec_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_h_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_gate_1.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: w_out.inner.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b_o.inner.as_entire_binding() },
            ],
        });

        Self {
            n_input: n_input as u32,
            n_hidden: n_hidden as u32,
            n_output: n_output as u32,
            threshold: net.spike_fn.threshold,
            alpha: net.spike_fn.alpha,
            w_in_0, w_rec_0, w_gate_in_0, w_gate_rec_0, b_h_0, b_gate_0,
            w_in_1, w_rec_1, w_gate_in_1, w_gate_rec_1, b_h_1, b_gate_1,
            w_out, b_o,
            grads,
            adam,
            adam_t: 0,
            forward_mega,
            transpose,
            cell_back,
            outer,
            column,
            vector_add,
            softmax_ce,
            adam_kernel,
            scale,
            zero,
            bind_l0,
            bind_l1,
            bind_io: None,
            fb,
            ctx,
        }
    }

    fn ensure_fb_capacity(&mut self, t: usize, b: usize) {
        if self.fb.ensure_capacity(&self.ctx, t, b) {
            self.bind_io = None;
        }
    }

    fn ensure_bind_io(&mut self) {
        if self.bind_io.is_none() {
            let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora multi 2L bind io"),
                layout: &self.forward_mega.bgl_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.fb.input_bytes_buf.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.fb.initial_h_v_after.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.fb.initial_h_spike.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.fb.h_v_pre_packed.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.fb.h_v_after_packed.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.fb.h_spike_packed.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.fb.gate_packed.inner.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.fb.logits_packed.inner.as_entire_binding() },
                ],
            });
            self.bind_io = Some(bind);
        }
    }

    pub fn zero_grad_gpu(&mut self) {
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi zero_grad"),
        });
        for buf in self.grads.iter_buffers() {
            self.zero.dispatch_into(&self.ctx, &mut encoder, buf);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn clip_grads_gpu(&self, max_norm: f32) {
        let g = self.download_grads();
        let norm = g.l2_norm();
        if norm <= max_norm || norm == 0.0 {
            return;
        }
        let factor = max_norm / norm;
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi clip_grads"),
        });
        for buf in self.grads.iter_buffers() {
            self.scale.dispatch_into(&self.ctx, &mut encoder, buf, factor);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn step_adam_gpu(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi step_adam"),
        });
        let params: [(&GpuBuffer, &GpuBuffer); 14] = [
            (&self.grads.w_in_0, &self.w_in_0),
            (&self.grads.w_rec_0, &self.w_rec_0),
            (&self.grads.w_gate_in_0, &self.w_gate_in_0),
            (&self.grads.w_gate_rec_0, &self.w_gate_rec_0),
            (&self.grads.b_h_0, &self.b_h_0),
            (&self.grads.b_gate_0, &self.b_gate_0),
            (&self.grads.w_in_1, &self.w_in_1),
            (&self.grads.w_rec_1, &self.w_rec_1),
            (&self.grads.w_gate_in_1, &self.w_gate_in_1),
            (&self.grads.w_gate_rec_1, &self.w_gate_rec_1),
            (&self.grads.b_h_1, &self.b_h_1),
            (&self.grads.b_gate_1, &self.b_gate_1),
            (&self.grads.w_out, &self.w_out),
            (&self.grads.b_o, &self.b_o),
        ];
        for (i, (g, p)) in params.iter().enumerate() {
            self.adam_kernel.dispatch_into(
                &self.ctx,
                &mut encoder,
                g,
                p,
                &self.adam.m1[i],
                &self.adam.m2[i],
                p.len as u32,
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                bc1,
                bc2,
            );
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Download all 14 grad buffers back to a CPU
    /// `MultiLayerGatedGrads` (used for parity tests + clip_grads).
    #[must_use]
    pub fn download_grads(&self) -> MultiLayerGatedGrads {
        let n_input = self.n_input as usize;
        let n_hidden = self.n_hidden as usize;
        let n_output = self.n_output as usize;
        let layer_dims = [n_input, n_hidden, n_hidden];
        let mut g = MultiLayerGatedGrads::zeros(&layer_dims, n_output);
        // Layer 0
        g.layers[0] = LayerGrads {
            w_in: self.grads.w_in_0.download(&self.ctx),
            w_rec: self.grads.w_rec_0.download(&self.ctx),
            w_gate_in: self.grads.w_gate_in_0.download(&self.ctx),
            w_gate_rec: self.grads.w_gate_rec_0.download(&self.ctx),
            b_h: self.grads.b_h_0.download(&self.ctx),
            b_gate: self.grads.b_gate_0.download(&self.ctx),
        };
        g.layers[1] = LayerGrads {
            w_in: self.grads.w_in_1.download(&self.ctx),
            w_rec: self.grads.w_rec_1.download(&self.ctx),
            w_gate_in: self.grads.w_gate_in_1.download(&self.ctx),
            w_gate_rec: self.grads.w_gate_rec_1.download(&self.ctx),
            b_h: self.grads.b_h_1.download(&self.ctx),
            b_gate: self.grads.b_gate_1.download(&self.ctx),
        };
        g.w_out = self.grads.w_out.download(&self.ctx);
        g.b_o = self.grads.b_o.download(&self.ctx);
        g
    }

    /// Download all 14 weight buffers into a fresh CPU `MultiLayerGatedSnn`.
    /// Used by the example for per-epoch validation + sample generation
    /// (avoids needing GPU autoregressive `step_lm_byte`). Adam state is
    /// discarded.
    #[must_use]
    pub fn to_cpu(&self) -> MultiLayerGatedSnn {
        use crate::surrogate::SpikeFn;
        use rand::SeedableRng;
        let n_input = self.n_input as usize;
        let n_hidden = self.n_hidden as usize;
        let n_output = self.n_output as usize;
        let mut net = MultiLayerGatedSnn::new(
            n_input,
            &[n_hidden, n_hidden],
            n_output,
            SpikeFn {
                threshold: self.threshold,
                alpha: self.alpha,
            },
            &mut rand::rngs::StdRng::seed_from_u64(0),
        );
        net.layers[0].w_in = self.w_in_0.download(&self.ctx);
        net.layers[0].w_rec = self.w_rec_0.download(&self.ctx);
        net.layers[0].w_gate_in = self.w_gate_in_0.download(&self.ctx);
        net.layers[0].w_gate_rec = self.w_gate_rec_0.download(&self.ctx);
        net.layers[0].b_h = self.b_h_0.download(&self.ctx);
        net.layers[0].b_gate = self.b_gate_0.download(&self.ctx);
        net.layers[1].w_in = self.w_in_1.download(&self.ctx);
        net.layers[1].w_rec = self.w_rec_1.download(&self.ctx);
        net.layers[1].w_gate_in = self.w_gate_in_1.download(&self.ctx);
        net.layers[1].w_gate_rec = self.w_gate_rec_1.download(&self.ctx);
        net.layers[1].b_h = self.b_h_1.download(&self.ctx);
        net.layers[1].b_gate = self.b_gate_1.download(&self.ctx);
        net.w_out = self.w_out.download(&self.ctx);
        net.b_o = self.b_o.download(&self.ctx);
        net
    }

    fn upload_batch_io_tokens(&self, inputs: &[&[u32]], targets: &[&[u32]]) {
        let stride = self.fb.capacity_t;
        let total = self.fb.capacity_b * stride;
        let mut input_buf = vec![0_u32; total];
        let mut target_buf = vec![0_u32; total];
        for (b, ins) in inputs.iter().enumerate() {
            let off = b * stride;
            input_buf[off..off + ins.len()].copy_from_slice(ins);
        }
        for (b, ts) in targets.iter().enumerate() {
            let off = b * stride;
            target_buf[off..off + ts.len()].copy_from_slice(ts);
        }
        self.ctx.queue.write_buffer(&self.fb.input_bytes_buf.inner, 0, bytemuck::cast_slice(&input_buf));
        self.ctx.queue.write_buffer(&self.fb.target_bytes_buf.inner, 0, bytemuck::cast_slice(&target_buf));
    }

    fn upload_batch_initial_state(&self, states: &[(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)]) {
        // states[b] = (l0_h_v_after, l0_h_spike, l1_h_v_after, l1_h_spike)
        let n = self.n_hidden as usize;
        let total = self.fb.capacity_b * 2 * n;
        let mut h_v_after_buf = vec![0.0_f32; total];
        let mut h_spike_buf = vec![0.0_f32; total];
        for (b, (v0, s0, v1, s1)) in states.iter().enumerate() {
            let off_l0 = b * 2 * n;
            let off_l1 = (b * 2 + 1) * n;
            h_v_after_buf[off_l0..off_l0 + n].copy_from_slice(v0);
            h_spike_buf[off_l0..off_l0 + n].copy_from_slice(s0);
            h_v_after_buf[off_l1..off_l1 + n].copy_from_slice(v1);
            h_spike_buf[off_l1..off_l1 + n].copy_from_slice(s1);
        }
        self.fb.initial_h_v_after.upload(&self.ctx, &h_v_after_buf);
        self.fb.initial_h_spike.upload(&self.ctx, &h_spike_buf);
    }

    /// Forward + backward + grad accum over a batch of `B` sequences,
    /// 2-layer network. `states[b]` is `(l0_h_v_after, l0_h_spike,
    /// l1_h_v_after, l1_h_spike)`.
    pub fn forward_backward_batch_gpu_multi_fused(
        &mut self,
        inputs: &[&[u8]],
        targets: &[&[u8]],
        states: &mut [(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> f32 {
        let in_u32: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| s.iter().map(|&b| u32::from(b)).collect())
            .collect();
        let tg_u32: Vec<Vec<u32>> = targets
            .iter()
            .map(|s| s.iter().map(|&b| u32::from(b)).collect())
            .collect();
        let in_refs: Vec<&[u32]> = in_u32.iter().map(Vec::as_slice).collect();
        let tg_refs: Vec<&[u32]> = tg_u32.iter().map(Vec::as_slice).collect();
        self.forward_backward_batch_gpu_multi_fused_tokens(&in_refs, &tg_refs, states)
    }

    /// Sprint 28: token-level variant. Same as the byte version but
    /// inputs/targets are arbitrary `u32` token IDs (`< n_input`).
    /// Required for BPE training on the GPU stack.
    pub fn forward_backward_batch_gpu_multi_fused_tokens(
        &mut self,
        inputs: &[&[u32]],
        targets: &[&[u32]],
        states: &mut [(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len(), states.len());
        let batch_size = inputs.len();
        if batch_size == 0 {
            return 0.0;
        }
        let t_steps = inputs[0].len();
        for b in 0..batch_size {
            assert_eq!(inputs[b].len(), t_steps);
            assert_eq!(targets[b].len(), t_steps);
        }
        if t_steps == 0 {
            return 0.0;
        }
        self.ensure_fb_capacity(t_steps, batch_size);

        let owned_states: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = states
            .iter()
            .map(|(a, b, c, d)| (a.clone(), b.clone(), c.clone(), d.clone()))
            .collect();
        self.upload_batch_initial_state(&owned_states);
        self.upload_batch_io_tokens(inputs, targets);

        self.gpu_forward(t_steps, batch_size);
        let total_loss = self.compute_d_logits_gpu(batch_size, t_steps);
        self.gpu_backward(t_steps, batch_size);

        let n = self.n_hidden as usize;
        let h_v_after_full = self.fb.h_v_after_packed.download(&self.ctx);
        let h_spike_full = self.fb.h_spike_packed.download(&self.ctx);
        let stride = (t_steps + 1) * n;
        for (b, (v0, s0, v1, s1)) in states.iter_mut().enumerate() {
            let l0_base = b * 2 * stride;
            let l1_base = (b * 2 + 1) * stride;
            let last = t_steps * n;
            v0.copy_from_slice(&h_v_after_full[l0_base + last..l0_base + last + n]);
            s0.copy_from_slice(&h_spike_full[l0_base + last..l0_base + last + n]);
            v1.copy_from_slice(&h_v_after_full[l1_base + last..l1_base + last + n]);
            s1.copy_from_slice(&h_spike_full[l1_base + last..l1_base + last + n]);
        }

        total_loss
    }

    fn gpu_forward(&mut self, t_steps: usize, batch_size: usize) {
        let params = ForwardMega2LayerParams {
            n_input: self.n_input,
            n_hidden: self.n_hidden,
            n_output: self.n_output,
            n_timesteps: t_steps as u32,
            threshold: self.threshold,
            pad0: 0, pad1: 0, pad2: 0,
        };
        self.ensure_bind_io();
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi gpu_forward"),
        });
        let bind_io = self.bind_io.as_ref().unwrap();
        self.forward_mega.dispatch_into(
            &mut encoder,
            &self.bind_l0,
            &self.bind_l1,
            bind_io,
            &params,
            batch_size as u32,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn compute_d_logits_gpu(&mut self, batch_size: usize, n_timesteps: usize) -> f32 {
        let o = self.n_output;
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi compute_d_logits"),
        });
        self.softmax_ce.dispatch_into(
            &self.ctx,
            &mut encoder,
            &self.fb.logits_packed,
            &self.fb.d_logits_packed,
            &self.fb.loss_seq,
            &self.fb.target_bytes_buf,
            o,
            n_timesteps as u32,
            batch_size as u32,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
        let losses = self.fb.loss_seq.download(&self.ctx);
        let used = batch_size * n_timesteps;
        losses.iter().take(used).sum()
    }

    fn gpu_backward(&mut self, t_steps: usize, batch_size: usize) {
        let h = self.n_hidden;
        let i_dim = self.n_input;
        let o = self.n_output;
        let b32 = batch_size as u32;
        let t_steps_u = t_steps as u32;
        let t_capacity = self.fb.capacity_t as u32;
        let cache_no_init_per_layer = t_steps_u * h;
        let cache_init_per_layer = (t_steps_u + 1) * h;
        let cache_out_total = t_steps_u * o;
        // Buffer strides per sample (multi-layer flat layout):
        let cache_no_init_per_sample = 2u32 * cache_no_init_per_layer;
        let cache_init_per_sample = 2u32 * cache_init_per_layer;

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aurora multi gpu_backward"),
        });

        // Init carry buffers per layer to zero.
        for layer in 0..2 {
            self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_v_after_carry_a[layer]);
            self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_v_after_carry_b[layer]);
            self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_spike_recurrent[layer]);
            self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_spike_from_above[layer]);
        }

        // Per-layer weight references (for cleaner indexing).
        let w_rec = [&self.w_rec_0, &self.w_rec_1];
        let w_gate_rec = [&self.w_gate_rec_0, &self.w_gate_rec_1];
        let w_in_l = [&self.w_in_0, &self.w_in_1];
        let w_gate_in_l = [&self.w_gate_in_0, &self.w_gate_in_1];
        let g_w_rec = [&self.grads.w_rec_0, &self.grads.w_rec_1];
        let g_w_gate_rec = [&self.grads.w_gate_rec_0, &self.grads.w_gate_rec_1];
        let g_w_in = [&self.grads.w_in_0, &self.grads.w_in_1];
        let g_w_gate_in = [&self.grads.w_gate_in_0, &self.grads.w_gate_in_1];
        let g_b_h = [&self.grads.b_h_0, &self.grads.b_h_1];
        let g_b_gate = [&self.grads.b_gate_0, &self.grads.b_gate_1];

        for t in (0..t_steps).rev() {
            let t32 = t as u32;
            let off_t_n = t32 * h; // within per-layer per-sample slice
            let off_t1_n = (t32 + 1) * h;
            let off_t_o = t32 * o;

            // Layer 1 first (top-down), then layer 0.
            // d_h_spike_from_above for top layer = W_out^T @ d_logits[t].
            self.transpose.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_out,
                &self.fb.d_logits_packed,
                &self.fb.d_h_spike_from_above[1],
                o,
                h,
                false,
                off_t_o,
                cache_out_total,
                h,
                b32,
            );

            for layer in (0..2).rev() {
                // Layer-specific cache offset within per-sample-per-layer.
                let layer_no_init_off = (layer as u32) * cache_no_init_per_layer;
                let layer_init_off = (layer as u32) * cache_init_per_layer;
                let h_v_pre_off = layer_no_init_off + off_t_n;
                let gate_off = h_v_pre_off;
                let h_v_after_prev_off = layer_init_off + off_t_n;

                let (carry_in, carry_out): (&GpuBuffer, &GpuBuffer) = if t.is_multiple_of(2) {
                    (&self.fb.d_h_v_after_carry_a[layer], &self.fb.d_h_v_after_carry_b[layer])
                } else {
                    (&self.fb.d_h_v_after_carry_b[layer], &self.fb.d_h_v_after_carry_a[layer])
                };

                self.cell_back.dispatch_into(
                    &self.ctx,
                    &mut encoder,
                    &self.fb.d_h_spike_from_above[layer],
                    &self.fb.d_h_spike_recurrent[layer],
                    carry_in,
                    &self.fb.h_v_pre_packed,
                    &self.fb.h_v_after_packed,
                    &self.fb.gate_packed,
                    &self.fb.d_cell_packed[layer],
                    h,
                    self.threshold,
                    self.alpha,
                    h_v_pre_off,
                    h_v_after_prev_off,
                    gate_off,
                    cache_no_init_per_sample,
                    cache_init_per_sample,
                    b32,
                );
                let slot_size = b32 * h;
                Self::copy_packed_slot_into(&mut encoder, &self.fb.d_cell_packed[layer], &self.fb.d_h_v_pre[layer], 0, slot_size);
                Self::copy_packed_slot_into(&mut encoder, &self.fb.d_cell_packed[layer], &self.fb.d_gate_pre[layer], 1, slot_size);
                Self::copy_packed_slot_into(&mut encoder, &self.fb.d_cell_packed[layer], carry_out, 2, slot_size);

                // Bias gradients.
                self.vector_add.dispatch_into(
                    &self.ctx, &mut encoder,
                    &self.fb.d_h_v_pre[layer], g_b_h[layer], h, 0, h, b32,
                );
                self.vector_add.dispatch_into(
                    &self.ctx, &mut encoder,
                    &self.fb.d_gate_pre[layer], g_b_gate[layer], h, 0, h, b32,
                );

                // Output bias only updated for top layer's d_logits.
                if layer == 1 {
                    self.vector_add.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_logits_packed, &self.grads.b_o,
                        o, off_t_o, cache_out_total, b32,
                    );
                    // W_out += d_logits[b, t] ⊗ h_spike_l1[b, t]
                    // h_spike[t] = slot t+1 in (T+1)*N per-layer per-sample.
                    let h_spike_t_off = cache_init_per_layer + off_t1_n;
                    self.outer.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_logits_packed, &self.fb.h_spike_packed,
                        &self.grads.w_out,
                        o, h,
                        off_t_o, h_spike_t_off,
                        cache_out_total, cache_init_per_sample, b32,
                    );
                }

                // W_rec / W_gate_rec for this layer.
                let h_spike_prev_off = (layer as u32) * cache_init_per_layer + off_t_n;
                self.outer.dispatch_into(
                    &self.ctx, &mut encoder,
                    &self.fb.d_h_v_pre[layer], &self.fb.h_spike_packed,
                    g_w_rec[layer],
                    h, h,
                    0, h_spike_prev_off,
                    h, cache_init_per_sample, b32,
                );
                self.outer.dispatch_into(
                    &self.ctx, &mut encoder,
                    &self.fb.d_gate_pre[layer], &self.fb.h_spike_packed,
                    g_w_gate_rec[layer],
                    h, h,
                    0, h_spike_prev_off,
                    h, cache_init_per_sample, b32,
                );

                // W_in / W_gate_in.
                if layer == 0 {
                    // Layer 0 input is byte: column_accum at byte_idx[b, t].
                    self.column.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_h_v_pre[layer], g_w_in[layer], &self.fb.input_bytes_buf,
                        h, i_dim, t_capacity, t32, h, b32,
                    );
                    self.column.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_gate_pre[layer], g_w_gate_in[layer], &self.fb.input_bytes_buf,
                        h, i_dim, t_capacity, t32, h, b32,
                    );
                } else {
                    // Layer 1 input is layer 0's spike output at t (slot t+1
                    // in layer 0's init-prepended slice; layer 0 base = 0).
                    let l0_h_spike_t_off = off_t1_n;
                    self.outer.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_h_v_pre[layer], &self.fb.h_spike_packed,
                        g_w_in[layer],
                        h, h,
                        0, l0_h_spike_t_off,
                        h, cache_init_per_sample, b32,
                    );
                    self.outer.dispatch_into(
                        &self.ctx, &mut encoder,
                        &self.fb.d_gate_pre[layer], &self.fb.h_spike_packed,
                        g_w_gate_in[layer],
                        h, h,
                        0, l0_h_spike_t_off,
                        h, cache_init_per_sample, b32,
                    );
                }

                // d_h_spike_recurrent[layer] = W_rec^T @ d_h_v_pre + W_gate_rec^T @ d_gate_pre
                // (consumed by next iteration's cell_back of this layer).
                self.transpose.dispatch_into(
                    &self.ctx, &mut encoder,
                    w_rec[layer], &self.fb.d_h_v_pre[layer], &self.fb.d_h_spike_recurrent[layer],
                    h, h, false, 0, h, h, b32,
                );
                self.transpose.dispatch_into(
                    &self.ctx, &mut encoder,
                    w_gate_rec[layer], &self.fb.d_gate_pre[layer], &self.fb.d_h_spike_recurrent[layer],
                    h, h, true, 0, h, h, b32,
                );

                // Cross-layer credit: layer 1 → layer 0's d_h_spike_from_above.
                // For layer 0, no further down — skip.
                if layer == 1 {
                    let target = &self.fb.d_h_spike_from_above[0];
                    self.transpose.dispatch_into(
                        &self.ctx, &mut encoder,
                        w_in_l[layer], &self.fb.d_h_v_pre[layer], target,
                        h, h, false, 0, h, h, b32,
                    );
                    self.transpose.dispatch_into(
                        &self.ctx, &mut encoder,
                        w_gate_in_l[layer], &self.fb.d_gate_pre[layer], target,
                        h, h, true, 0, h, h, b32,
                    );
                }
            }
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn copy_packed_slot_into(
        encoder: &mut wgpu::CommandEncoder,
        packed: &GpuBuffer,
        dst: &GpuBuffer,
        slot: u32,
        slot_size: u32,
    ) {
        let bytes = u64::from(slot_size) * 4;
        let src_offset = u64::from(slot) * bytes;
        encoder.copy_buffer_to_buffer(&packed.inner, src_offset, &dst.inner, 0, bytes);
    }
}
