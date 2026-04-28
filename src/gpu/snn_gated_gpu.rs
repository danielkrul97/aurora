//! GPU mirror of `GatedRecurrentSnn` — full forward + backward + Adam.
//!
//! Sprint 14 delivered the forward path (`forward_lm_bytes_gpu`).
//! Sprint 15 adds:
//! - Per-timestep forward cache resident on GPU.
//! - GPU backward (`backward_lm_bytes_gpu_internal`) over the full BPTT
//!   chain, accumulating into resident GPU gradient buffers.
//! - GPU Adam state + `step_adam_gpu`, `clip_grads_gpu`, `zero_grad_gpu`.
//! - End-to-end training entry points: `forward_backward_lm_bytes_gpu`
//!   (single sample), `forward_backward_batch_gpu` (sequential per
//!   sample, sharing GPU grad buffers).
//! - Weight readback via `to_cpu`.
//!
//! The CPU↔GPU forward parity test (`tests/gpu_parity.rs`) still passes
//! through the simpler stateless `forward_lm_bytes_gpu`. Sprint 15 adds
//! parity tests for backward gradients and end-to-end training direction.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::forward_mega::{ForwardMegaKernel, ForwardMegaParams};
use super::kernels::{
    AdamStepKernel, ColumnAccumKernel, DenseMatvecKernel, DenseMatvecTransposeKernel,
    FusedGatePreKernel, GatedCellBackwardKernel, GatedCellStepKernel, OuterProductAccumKernel,
    ScaleBufferKernel, SoftmaxCrossEntropyKernel, SparseMatvecOneHotKernel, VectorAddKernel,
    ZeroBufferKernel,
};
use super::GpuContext;
use crate::optimizer::{bias_correction, AdamConfig};
use crate::snn_gated::{GatedGrads, GatedRecurrentSnn};

/// Eight gradient buffers — GPU mirror of `GatedGrads`.
pub struct GatedGpuGrads {
    pub w_in: GpuBuffer,
    pub w_rec: GpuBuffer,
    pub w_out: GpuBuffer,
    pub w_gate_in: GpuBuffer,
    pub w_gate_rec: GpuBuffer,
    pub b_h: GpuBuffer,
    pub b_o: GpuBuffer,
    pub b_gate: GpuBuffer,
}

impl GatedGpuGrads {
    fn zeros(ctx: &GpuContext, n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        Self {
            w_in: GpuBuffer::zeros(ctx, "grad_w_in", n_hidden * n_input),
            w_rec: GpuBuffer::zeros(ctx, "grad_w_rec", n_hidden * n_hidden),
            w_out: GpuBuffer::zeros(ctx, "grad_w_out", n_output * n_hidden),
            w_gate_in: GpuBuffer::zeros(ctx, "grad_w_gate_in", n_hidden * n_input),
            w_gate_rec: GpuBuffer::zeros(ctx, "grad_w_gate_rec", n_hidden * n_hidden),
            b_h: GpuBuffer::zeros(ctx, "grad_b_h", n_hidden),
            b_o: GpuBuffer::zeros(ctx, "grad_b_o", n_output),
            b_gate: GpuBuffer::zeros(ctx, "grad_b_gate", n_hidden),
        }
    }
}

/// Adam first/second moment buffers — one pair per parameter group.
pub struct GatedGpuAdam {
    pub m1_w_in: GpuBuffer,
    pub m1_w_rec: GpuBuffer,
    pub m1_w_out: GpuBuffer,
    pub m1_w_gate_in: GpuBuffer,
    pub m1_w_gate_rec: GpuBuffer,
    pub m1_b_h: GpuBuffer,
    pub m1_b_o: GpuBuffer,
    pub m1_b_gate: GpuBuffer,
    pub m2_w_in: GpuBuffer,
    pub m2_w_rec: GpuBuffer,
    pub m2_w_out: GpuBuffer,
    pub m2_w_gate_in: GpuBuffer,
    pub m2_w_gate_rec: GpuBuffer,
    pub m2_b_h: GpuBuffer,
    pub m2_b_o: GpuBuffer,
    pub m2_b_gate: GpuBuffer,
}

impl GatedGpuAdam {
    fn zeros(ctx: &GpuContext, n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        Self {
            m1_w_in: GpuBuffer::zeros(ctx, "m1_w_in", n_hidden * n_input),
            m1_w_rec: GpuBuffer::zeros(ctx, "m1_w_rec", n_hidden * n_hidden),
            m1_w_out: GpuBuffer::zeros(ctx, "m1_w_out", n_output * n_hidden),
            m1_w_gate_in: GpuBuffer::zeros(ctx, "m1_w_gate_in", n_hidden * n_input),
            m1_w_gate_rec: GpuBuffer::zeros(ctx, "m1_w_gate_rec", n_hidden * n_hidden),
            m1_b_h: GpuBuffer::zeros(ctx, "m1_b_h", n_hidden),
            m1_b_o: GpuBuffer::zeros(ctx, "m1_b_o", n_output),
            m1_b_gate: GpuBuffer::zeros(ctx, "m1_b_gate", n_hidden),
            m2_w_in: GpuBuffer::zeros(ctx, "m2_w_in", n_hidden * n_input),
            m2_w_rec: GpuBuffer::zeros(ctx, "m2_w_rec", n_hidden * n_hidden),
            m2_w_out: GpuBuffer::zeros(ctx, "m2_w_out", n_output * n_hidden),
            m2_w_gate_in: GpuBuffer::zeros(ctx, "m2_w_gate_in", n_hidden * n_input),
            m2_w_gate_rec: GpuBuffer::zeros(ctx, "m2_w_gate_rec", n_hidden * n_hidden),
            m2_b_h: GpuBuffer::zeros(ctx, "m2_b_h", n_hidden),
            m2_b_o: GpuBuffer::zeros(ctx, "m2_b_o", n_output),
            m2_b_gate: GpuBuffer::zeros(ctx, "m2_b_gate", n_hidden),
        }
    }
}

/// Forward cache + per-step scratch buffers. Sprint 18 packed the
/// per-timestep dimension into single buffers; Sprint 19 adds the
/// **batch** dimension on top, so all packed cache buffers are sized
/// `[B * T * N]` (or `B * (T+1) * N` for `h_v_after_packed` /
/// `h_spike_packed` which prepend the per-sample initial state).
///
/// Layout convention (`b` = sample, `t` = timestep, `h` = neuron):
/// - `h_v_pre_packed[b * T * N + t * N + h]`
/// - `gate_packed[b * T * N + t * N + h]`
/// - `h_v_after_packed[b * (T+1) * N + slot * N + h]` where slot 0 =
///   initial state for sample b, slot t+1 = post-step state at t
/// - `h_spike_packed`: same as `h_v_after_packed`
/// - `logits_packed[b * T * O + t * O + h]`
/// - `d_logits_packed[b * T * O + t * O + h]`
/// - `loss_seq[b * T + t]`
/// - `input_bytes_buf[b * T + t]`
/// - `target_bytes_buf[b * T + t]`
/// - `initial_h_v_after[b * N + h]`, `initial_h_spike[b * N + h]`
///
/// Backward per-step scratch buffers (`d_h_v_pre`, `d_gate_pre`, etc.)
/// keep a single-sample shape (`[N]`); backward is sequential per
/// sample (Sprint 19 only batches forward + softmax_ce).
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

    d_h_spike_from_out: GpuBuffer,
    d_h_spike_recurrent: GpuBuffer,
    d_h_v_after_carry_a: GpuBuffer,
    d_h_v_after_carry_b: GpuBuffer,
    d_cell_packed: GpuBuffer,
    d_h_v_pre: GpuBuffer,
    d_gate_pre: GpuBuffer,

    initial_h_v_after: GpuBuffer,
    initial_h_spike: GpuBuffer,

    /// Sprint 34: dummy d_h_spike_packed (single-layer doesn't compute
    /// α gradient, but the cell_back kernel signature now requires this
    /// binding for parity with the multi-layer code path).
    d_h_spike_packed: GpuBuffer,
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
        Self {
            capacity_t: t,
            capacity_b: b,
            n_hidden,
            n_output,
            h_v_pre_packed: GpuBuffer::zeros(ctx, "h_v_pre_packed", b * t * n_hidden),
            h_v_after_packed: GpuBuffer::zeros(ctx, "h_v_after_packed", b * (t + 1) * n_hidden),
            h_spike_packed: GpuBuffer::zeros(ctx, "h_spike_packed", b * (t + 1) * n_hidden),
            gate_packed: GpuBuffer::zeros(ctx, "gate_packed", b * t * n_hidden),
            logits_packed: GpuBuffer::zeros(ctx, "logits_packed", b * t * n_output),
            d_logits_packed: GpuBuffer::zeros(ctx, "d_logits_packed", b * t * n_output),
            loss_seq: GpuBuffer::zeros(ctx, "loss_seq", b * t),
            input_bytes_buf: GpuBuffer::zeros(ctx, "input_bytes", b * t),
            target_bytes_buf: GpuBuffer::zeros(ctx, "target_bytes", b * t),
            d_h_spike_from_out: GpuBuffer::zeros(ctx, "d_h_spike_from_out", b * n_hidden),
            d_h_spike_recurrent: GpuBuffer::zeros(ctx, "d_h_spike_recurrent", b * n_hidden),
            d_h_v_after_carry_a: GpuBuffer::zeros(ctx, "d_h_v_after_carry_a", b * n_hidden),
            d_h_v_after_carry_b: GpuBuffer::zeros(ctx, "d_h_v_after_carry_b", b * n_hidden),
            d_cell_packed: GpuBuffer::zeros(ctx, "d_cell_packed", 3 * b * n_hidden),
            d_h_v_pre: GpuBuffer::zeros(ctx, "d_h_v_pre", b * n_hidden),
            d_gate_pre: GpuBuffer::zeros(ctx, "d_gate_pre", b * n_hidden),
            initial_h_v_after: GpuBuffer::zeros(ctx, "initial_h_v_after", b * n_hidden),
            initial_h_spike: GpuBuffer::zeros(ctx, "initial_h_spike", b * n_hidden),
            d_h_spike_packed: GpuBuffer::zeros(ctx, "d_h_spike_packed", b * t * n_hidden),
        }
    }

    /// Reallocate packed buffers to exactly `(t, b)` if they don't
    /// already match. Returns `true` if any buffer was reallocated;
    /// the caller MUST then invalidate cached bind groups.
    ///
    /// Sprint 19: buffer stride must match kernel-side stride exactly,
    /// otherwise sample b's data overlaps sample b-1's trailing slots.
    /// Training loops with constant `(t, b)` reallocate once on the
    /// first call and reuse thereafter.
    fn ensure_capacity(&mut self, ctx: &GpuContext, t: usize, b: usize) -> bool {
        if t == self.capacity_t && b == self.capacity_b {
            return false;
        }
        let new_t = t.max(1);
        let new_b = b.max(1);
        let n = self.n_hidden;
        let o = self.n_output;
        self.h_v_pre_packed = GpuBuffer::zeros(ctx, "h_v_pre_packed", new_b * new_t * n);
        self.d_h_spike_packed =
            GpuBuffer::zeros(ctx, "d_h_spike_packed", new_b * new_t * n);
        self.h_v_after_packed =
            GpuBuffer::zeros(ctx, "h_v_after_packed", new_b * (new_t + 1) * n);
        self.h_spike_packed = GpuBuffer::zeros(ctx, "h_spike_packed", new_b * (new_t + 1) * n);
        self.gate_packed = GpuBuffer::zeros(ctx, "gate_packed", new_b * new_t * n);
        self.logits_packed = GpuBuffer::zeros(ctx, "logits_packed", new_b * new_t * o);
        self.d_logits_packed = GpuBuffer::zeros(ctx, "d_logits_packed", new_b * new_t * o);
        self.loss_seq = GpuBuffer::zeros(ctx, "loss_seq", new_b * new_t);
        self.input_bytes_buf = GpuBuffer::zeros(ctx, "input_bytes", new_b * new_t);
        self.target_bytes_buf = GpuBuffer::zeros(ctx, "target_bytes", new_b * new_t);
        if new_b != self.capacity_b {
            self.initial_h_v_after = GpuBuffer::zeros(ctx, "initial_h_v_after", new_b * n);
            self.initial_h_spike = GpuBuffer::zeros(ctx, "initial_h_spike", new_b * n);
            self.d_h_spike_from_out =
                GpuBuffer::zeros(ctx, "d_h_spike_from_out", new_b * n);
            self.d_h_spike_recurrent =
                GpuBuffer::zeros(ctx, "d_h_spike_recurrent", new_b * n);
            self.d_h_v_after_carry_a =
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_a", new_b * n);
            self.d_h_v_after_carry_b =
                GpuBuffer::zeros(ctx, "d_h_v_after_carry_b", new_b * n);
            self.d_cell_packed = GpuBuffer::zeros(ctx, "d_cell_packed", 3 * new_b * n);
            self.d_h_v_pre = GpuBuffer::zeros(ctx, "d_h_v_pre", new_b * n);
            self.d_gate_pre = GpuBuffer::zeros(ctx, "d_gate_pre", new_b * n);
        }
        self.capacity_t = new_t;
        self.capacity_b = new_b;
        true
    }
}

pub struct GatedRecurrentSnnGpu {
    pub ctx: GpuContext,
    pub n_input: u32,
    pub n_hidden: u32,
    pub n_output: u32,
    pub threshold: f32,
    pub alpha: f32,

    pub w_in: GpuBuffer,
    pub w_rec: GpuBuffer,
    pub w_out: GpuBuffer,
    pub w_gate_in: GpuBuffer,
    pub w_gate_rec: GpuBuffer,
    pub b_h: GpuBuffer,
    pub b_o: GpuBuffer,
    pub b_gate: GpuBuffer,

    pub grads: GatedGpuGrads,
    pub adam: GatedGpuAdam,
    pub adam_t: usize,

    onehot: Arc<SparseMatvecOneHotKernel>,
    dense: Arc<DenseMatvecKernel>,
    cell: Arc<GatedCellStepKernel>,
    transpose: Arc<DenseMatvecTransposeKernel>,
    cell_back: Arc<GatedCellBackwardKernel>,
    outer: Arc<OuterProductAccumKernel>,
    column: Arc<ColumnAccumKernel>,
    vector_add: Arc<VectorAddKernel>,
    adam_kernel: Arc<AdamStepKernel>,
    scale: Arc<ScaleBufferKernel>,
    zero: Arc<ZeroBufferKernel>,
    fused_gate_pre: Arc<FusedGatePreKernel>,
    softmax_ce: Arc<SoftmaxCrossEntropyKernel>,
    forward_mega: Arc<ForwardMegaKernel>,

    /// Cached bind group for the forward megakernel's weight bindings
    /// (group 0). Stable across timesteps and chunks; weights are
    /// uploaded once at `from_cpu`.
    fwd_mega_bind_weights: wgpu::BindGroup,
    /// Cached bind group for the forward megakernel's per-chunk i/o
    /// bindings (group 1). Invalidated and recreated when `fb`
    /// reallocates its packed buffers in `ensure_fb_capacity`.
    fwd_mega_bind_io: Option<wgpu::BindGroup>,

    no_bias_placeholder: GpuBuffer,
    fb: FbScratch,
}

impl GatedRecurrentSnnGpu {
    /// Upload CPU network weights to GPU. Compiles all kernels (forward,
    /// backward, Adam) and allocates resident gradient + Adam state
    /// buffers. The forward+backward cache is sized lazily for the chunk
    /// length actually used.
    #[must_use]
    pub fn from_cpu(net: &GatedRecurrentSnn, ctx: GpuContext) -> Self {
        let no_bias_placeholder = GpuBuffer::zeros(&ctx, "no-bias-placeholder", 1);
        let w_in = GpuBuffer::from_slice(&ctx, "w_in", &net.w_in);
        let w_rec = GpuBuffer::from_slice(&ctx, "w_rec", &net.w_rec);
        let w_out = GpuBuffer::from_slice(&ctx, "w_out", &net.w_out);
        let w_gate_in = GpuBuffer::from_slice(&ctx, "w_gate_in", &net.w_gate_in);
        let w_gate_rec = GpuBuffer::from_slice(&ctx, "w_gate_rec", &net.w_gate_rec);
        let b_h = GpuBuffer::from_slice(&ctx, "b_h", &net.b_h);
        let b_o = GpuBuffer::from_slice(&ctx, "b_o", &net.b_o);
        let b_gate = GpuBuffer::from_slice(&ctx, "b_gate", &net.b_gate);

        let onehot = Arc::new(SparseMatvecOneHotKernel::new(&ctx));
        let dense = Arc::new(DenseMatvecKernel::new(&ctx));
        let cell = Arc::new(GatedCellStepKernel::new(&ctx));
        let transpose = Arc::new(DenseMatvecTransposeKernel::new(&ctx));
        let cell_back = Arc::new(GatedCellBackwardKernel::new(&ctx));
        let outer = Arc::new(OuterProductAccumKernel::new(&ctx));
        let column = Arc::new(ColumnAccumKernel::new(&ctx));
        let vector_add = Arc::new(VectorAddKernel::new(&ctx));
        let adam_kernel = Arc::new(AdamStepKernel::new(&ctx));
        let scale = Arc::new(ScaleBufferKernel::new(&ctx));
        let zero = Arc::new(ZeroBufferKernel::new(&ctx));
        let fused_gate_pre = Arc::new(FusedGatePreKernel::new(&ctx));
        let softmax_ce = Arc::new(SoftmaxCrossEntropyKernel::new(&ctx));
        let forward_mega = Arc::new(ForwardMegaKernel::new(&ctx));

        let grads = GatedGpuGrads::zeros(&ctx, net.n_input, net.n_hidden, net.n_output);
        let adam = GatedGpuAdam::zeros(&ctx, net.n_input, net.n_hidden, net.n_output);

        // Default chunk capacity 64, batch capacity 4; lazy-grown by
        // `ensure_fb_capacity`.
        let fb = FbScratch::new(&ctx, net.n_hidden, net.n_output, 64, 4);

        let fwd_mega_bind_weights = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora forward_mega weights bind"),
            layout: &forward_mega.bgl_weights,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_in.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_rec.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_out.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: w_gate_in.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: w_gate_rec.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_h.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_o.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_gate.inner.as_entire_binding(),
                },
            ],
        });

        Self {
            n_input: net.n_input as u32,
            n_hidden: net.n_hidden as u32,
            n_output: net.n_output as u32,
            threshold: net.spike_fn.threshold,
            alpha: net.spike_fn.alpha,
            w_in,
            w_rec,
            w_out,
            w_gate_in,
            w_gate_rec,
            b_h,
            b_o,
            b_gate,
            grads,
            adam,
            adam_t: 0,
            onehot,
            dense,
            cell,
            transpose,
            cell_back,
            outer,
            column,
            vector_add,
            adam_kernel,
            scale,
            zero,
            fused_gate_pre,
            softmax_ce,
            forward_mega,
            fwd_mega_bind_weights,
            fwd_mega_bind_io: None,
            no_bias_placeholder,
            fb,
            ctx,
        }
    }

    /// Grow `self.fb` to fit chunk length `t` and batch size `b` if
    /// needed. If buffers were reallocated, invalidate cached bind
    /// groups that reference them.
    fn ensure_fb_capacity(&mut self, t: usize, b: usize) {
        if self.fb.ensure_capacity(&self.ctx, t, b) {
            self.fwd_mega_bind_io = None;
        }
    }

    /// Lazily build (or rebuild after fb resize) the I/O bind group
    /// for the forward megakernel. Idempotent.
    fn ensure_fwd_mega_bind_io(&mut self) {
        if self.fwd_mega_bind_io.is_none() {
            let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora forward_mega io bind"),
                layout: &self.forward_mega.bgl_io,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.fb.input_bytes_buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.fb.initial_h_v_after.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.fb.initial_h_spike.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.fb.h_v_pre_packed.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.fb.h_v_after_packed.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.fb.h_spike_packed.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.fb.gate_packed.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.fb.logits_packed.inner.as_entire_binding(),
                    },
                ],
            });
            self.fwd_mega_bind_io = Some(bind);
        }
    }

    /// Stateless forward: hidden state starts at zero and is discarded
    /// at the end. Returns `[T][n_output]` logits — same shape as
    /// `LmCache::logits` from the CPU path.
    ///
    /// # Panics
    /// Panics if any byte index is `>= n_input`.
    pub fn forward_lm_bytes_gpu(&self, input_bytes: &[u8]) -> Vec<Vec<f32>> {
        let h = self.n_hidden;
        let i_dim = self.n_input;
        let o = self.n_output;
        let t_steps = input_bytes.len();

        let h_in = GpuBuffer::zeros(&self.ctx, "h_in", h as usize);
        let h_rec = GpuBuffer::zeros(&self.ctx, "h_rec", h as usize);
        let gate_pre = GpuBuffer::zeros(&self.ctx, "gate_pre", h as usize);
        let gate_in_contrib = GpuBuffer::zeros(&self.ctx, "gate_in_contrib", h as usize);
        let gate_rec_contrib = GpuBuffer::zeros(&self.ctx, "gate_rec_contrib", h as usize);
        let h_v_after_a = GpuBuffer::zeros(&self.ctx, "h_v_after_a", h as usize);
        let h_v_after_b = GpuBuffer::zeros(&self.ctx, "h_v_after_b", h as usize);
        let h_spike_a = GpuBuffer::zeros(&self.ctx, "h_spike_a", h as usize);
        let h_spike_b = GpuBuffer::zeros(&self.ctx, "h_spike_b", h as usize);
        let packed_cell = GpuBuffer::zeros(&self.ctx, "packed_cell", 4 * h as usize);
        let logits_buf = GpuBuffer::zeros(&self.ctx, "logits", o as usize);

        let mut logits_seq: Vec<Vec<f32>> = Vec::with_capacity(t_steps);

        for (t, &byte) in input_bytes.iter().enumerate() {
            assert!(u32::from(byte) < i_dim);

            let (h_v_after_prev, h_spike_prev, h_v_after_next, h_spike_next) =
                if t.is_multiple_of(2) {
                    (&h_v_after_a, &h_spike_a, &h_v_after_b, &h_spike_b)
                } else {
                    (&h_v_after_b, &h_spike_b, &h_v_after_a, &h_spike_a)
                };

            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("aurora forward_lm_bytes_gpu step"),
                    });

            self.onehot
                .dispatch_into(&self.ctx, &mut encoder, &self.w_in, &h_in, h, i_dim, u32::from(byte));
            self.onehot.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_gate_in,
                &gate_in_contrib,
                h,
                i_dim,
                u32::from(byte),
            );
            self.dense.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_rec,
                h_spike_prev,
                &self.no_bias_placeholder,
                &h_rec,
                h,
                h,
                false,
            );
            self.dense.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_gate_rec,
                h_spike_prev,
                &self.no_bias_placeholder,
                &gate_rec_contrib,
                h,
                h,
                false,
            );

            self.fused_gate_pre.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.b_gate,
                &gate_in_contrib,
                &gate_rec_contrib,
                &gate_pre,
                h,
            );

            self.cell.dispatch_into(
                &self.ctx,
                &mut encoder,
                &gate_pre,
                &h_in,
                &h_rec,
                h_v_after_prev,
                &self.b_h,
                &packed_cell,
                h,
                self.threshold,
            );

            Self::copy_packed_slot_into(&mut encoder, &packed_cell, h_spike_next, 2, h);
            Self::copy_packed_slot_into(&mut encoder, &packed_cell, h_v_after_next, 3, h);

            self.dense.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_out,
                h_spike_next,
                &self.b_o,
                &logits_buf,
                o,
                h,
                true,
            );
            self.ctx.queue.submit(Some(encoder.finish()));
            logits_seq.push(logits_buf.download(&self.ctx));
        }

        logits_seq
    }

    /// Stateful forward+backward over a single byte chunk. Loss is
    /// accumulated into resident GPU gradient buffers (`self.grads`).
    /// Hidden state at t = -1 is taken from `initial_h_v_after` /
    /// `initial_h_spike` slices (zeros for stateless calls); the chunk's
    /// final hidden state is written back into them so the caller can
    /// carry it across chunks for TBPTT-with-detach.
    ///
    /// # Panics
    /// Panics if `input_bytes.len() != target_bytes.len()`, if any byte
    /// index is `>= n_input`, or if `initial_*` slice lengths don't
    /// match `n_hidden`.
    pub fn forward_backward_lm_bytes_gpu(
        &mut self,
        input_bytes: &[u8],
        target_bytes: &[u8],
        initial_h_v_after: &mut [f32],
        initial_h_spike: &mut [f32],
    ) -> f32 {
        assert_eq!(input_bytes.len(), target_bytes.len());
        assert_eq!(initial_h_v_after.len(), self.n_hidden as usize);
        assert_eq!(initial_h_spike.len(), self.n_hidden as usize);
        let t_steps = input_bytes.len();
        if t_steps == 0 {
            return 0.0;
        }
        self.ensure_fb_capacity(t_steps, 1);

        // Upload initial state to b=0 slice (first N elements of the
        // [B*N] buffer).
        let n = self.n_hidden as usize;
        let total_init = self.fb.capacity_b * n;
        let mut h_v_after_buf = vec![0.0_f32; total_init];
        let mut h_spike_buf = vec![0.0_f32; total_init];
        h_v_after_buf[..n].copy_from_slice(initial_h_v_after);
        h_spike_buf[..n].copy_from_slice(initial_h_spike);
        self.fb
            .initial_h_v_after
            .upload(&self.ctx, &h_v_after_buf);
        self.fb.initial_h_spike.upload(&self.ctx, &h_spike_buf);

        // Upload single-sample inputs / targets into b=0 slice.
        let inputs_owned: Vec<&[u8]> = vec![input_bytes];
        let targets_owned: Vec<&[u8]> = vec![target_bytes];
        self.upload_batch_io(&inputs_owned, &targets_owned);

        // Forward (B=1), then softmax_ce (B=1, T), then batched
        // backward (Sprint 20: B=1 still goes through the fused path).
        self.gpu_forward_batched(t_steps, 1);
        let total_loss = self.compute_d_logits_gpu_batched(1, t_steps);
        self.gpu_backward_batched(t_steps, 1);

        // Read final hidden state back to caller-owned buffers from
        // b=0 slice (slot `t_steps` of the per-sample (T+1)*N region).
        let h_v_after_full = self.fb.h_v_after_packed.download(&self.ctx);
        let h_spike_full = self.fb.h_spike_packed.download(&self.ctx);
        let start = t_steps * n;
        initial_h_v_after.copy_from_slice(&h_v_after_full[start..start + n]);
        initial_h_spike.copy_from_slice(&h_spike_full[start..start + n]);

        total_loss
    }

    /// Stateless wrapper: zero initial state, discards final state.
    pub fn forward_backward_lm_bytes_gpu_stateless(
        &mut self,
        input_bytes: &[u8],
        target_bytes: &[u8],
    ) -> f32 {
        let mut h0 = vec![0.0_f32; self.n_hidden as usize];
        let mut s0 = vec![0.0_f32; self.n_hidden as usize];
        self.forward_backward_lm_bytes_gpu(input_bytes, target_bytes, &mut h0, &mut s0)
    }

    /// Sequential per-sample forward+backward. Each sample's gradients
    /// are accumulated into `self.grads` (no zero in between). Caller
    /// is responsible for `zero_grad_gpu` before the batch and
    /// `clip_grads_gpu` + `step_adam_gpu` after.
    ///
    /// # Panics
    /// Panics if `inputs.len() != targets.len()` or `inputs.len() !=
    /// states.len()`. Each `states[i]` is a `(h_v_after, h_spike)` pair
    /// of length `n_hidden`.
    pub fn forward_backward_batch_gpu(
        &mut self,
        inputs: &[&[u8]],
        targets: &[&[u8]],
        states: &mut [(Vec<f32>, Vec<f32>)],
    ) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len(), states.len());
        let mut total_loss = 0.0_f32;
        for (((inp, tgt), state), _) in inputs
            .iter()
            .zip(targets.iter())
            .zip(states.iter_mut())
            .zip(0..)
        {
            let (h_v_after, h_spike) = state;
            total_loss += self.forward_backward_lm_bytes_gpu(inp, tgt, h_v_after, h_spike);
        }
        total_loss
    }

    /// Sprint 19: forward + softmax_ce **batch-fused** on GPU.
    /// Forward megakernel runs all `B` samples in parallel via
    /// `dispatch_workgroups(B, 1, 1)`; softmax_ce dispatches a 2D grid
    /// `(B, T)` of workgroups in one launch. Backward stays
    /// per-sample sequential (Sprint 20+ work) but reads from the
    /// shared packed cache via per-sample base offsets.
    ///
    /// Caller manages `states` (one pair per sample). Each
    /// `inputs[b]` and `targets[b]` must have the same length T;
    /// gradients accumulate across all B samples into `self.grads`
    /// (caller `zero_grad_gpu` before, `clip_grads_gpu` + `step_adam_gpu`
    /// after). Returns total loss summed over all (sample, timestep).
    ///
    /// # Panics
    /// Panics on length mismatch between inputs/targets/states or if
    /// any byte is `>= n_input`.
    pub fn forward_backward_batch_gpu_fused(
        &mut self,
        inputs: &[&[u8]],
        targets: &[&[u8]],
        states: &mut [(Vec<f32>, Vec<f32>)],
    ) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs.len(), states.len());
        let batch_size = inputs.len();
        if batch_size == 0 {
            return 0.0;
        }
        let t_steps = inputs[0].len();
        for b in 0..batch_size {
            assert_eq!(inputs[b].len(), t_steps, "all batch streams must have same T");
            assert_eq!(targets[b].len(), t_steps, "target length must equal input length");
            for &byte in inputs[b] {
                assert!(u32::from(byte) < self.n_input);
            }
        }
        if t_steps == 0 {
            return 0.0;
        }
        let n = self.n_hidden as usize;
        for (b, (h_v_after, h_spike)) in states.iter().enumerate() {
            assert_eq!(h_v_after.len(), n, "state {b} h_v_after wrong len");
            assert_eq!(h_spike.len(), n, "state {b} h_spike wrong len");
        }

        self.ensure_fb_capacity(t_steps, batch_size);

        // Convert &mut states into a slice for upload helper.
        let states_owned: Vec<(Vec<f32>, Vec<f32>)> = states
            .iter()
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        self.upload_batch_initial_state(&states_owned);
        self.upload_batch_io(inputs, targets);

        // Forward megakernel: 1 dispatch covers all B samples.
        self.gpu_forward_batched(t_steps, batch_size);
        // Softmax + CE + d_logits: 1 dispatch covers all (B, T).
        let total_loss = self.compute_d_logits_gpu_batched(batch_size, t_steps);
        // Sprint 20: batch-fused backward — one dispatch per kernel
        // covers all B samples in parallel (per timestep).
        self.gpu_backward_batched(t_steps, batch_size);

        // Read final hidden state of each sample at slot t_steps in
        // its own (T+1)*N slice, write back to caller-owned states.
        let h_v_after_full = self.fb.h_v_after_packed.download(&self.ctx);
        let h_spike_full = self.fb.h_spike_packed.download(&self.ctx);
        let stride = (t_steps + 1) * n;
        for (b, (out_v, out_s)) in states.iter_mut().enumerate() {
            let start = b * stride + t_steps * n;
            out_v.copy_from_slice(&h_v_after_full[start..start + n]);
            out_s.copy_from_slice(&h_spike_full[start..start + n]);
        }

        total_loss
    }

    pub fn zero_grad_gpu(&mut self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora zero_grad_gpu"),
            });
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_in);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_rec);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_out);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_gate_in);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_gate_rec);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_h);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_o);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_gate);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// L2-norm clip across all 8 grad buffers. Downloads each grad to
    /// CPU, computes norm, dispatches scale kernel if above threshold.
    pub fn clip_grads_gpu(&self, max_norm: f32) {
        let g = self.download_grads();
        let norm = g.l2_norm();
        if norm <= max_norm || norm == 0.0 {
            return;
        }
        let factor = max_norm / norm;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora clip_grads_gpu"),
            });
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_in, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_rec, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_out, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_gate_in, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.w_gate_rec, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_h, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_o, factor);
        self.scale.dispatch_into(&self.ctx, &mut encoder, &self.grads.b_gate, factor);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn step_adam_gpu(&mut self, config: &AdamConfig) {
        self.adam_t += 1;
        let (bc1, bc2) = bias_correction(self.adam_t, config);
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora step_adam_gpu"),
            });
        let groups: [(&GpuBuffer, &GpuBuffer, &GpuBuffer, &GpuBuffer); 8] = [
            (&self.grads.w_in, &self.w_in, &self.adam.m1_w_in, &self.adam.m2_w_in),
            (&self.grads.w_rec, &self.w_rec, &self.adam.m1_w_rec, &self.adam.m2_w_rec),
            (&self.grads.w_out, &self.w_out, &self.adam.m1_w_out, &self.adam.m2_w_out),
            (
                &self.grads.w_gate_in,
                &self.w_gate_in,
                &self.adam.m1_w_gate_in,
                &self.adam.m2_w_gate_in,
            ),
            (
                &self.grads.w_gate_rec,
                &self.w_gate_rec,
                &self.adam.m1_w_gate_rec,
                &self.adam.m2_w_gate_rec,
            ),
            (&self.grads.b_h, &self.b_h, &self.adam.m1_b_h, &self.adam.m2_b_h),
            (&self.grads.b_o, &self.b_o, &self.adam.m1_b_o, &self.adam.m2_b_o),
            (&self.grads.b_gate, &self.b_gate, &self.adam.m1_b_gate, &self.adam.m2_b_gate),
        ];
        for (g, p, m1, m2) in groups {
            self.adam_kernel.dispatch_into(
                &self.ctx,
                &mut encoder,
                g,
                p,
                m1,
                m2,
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

    /// Download current weights into a fresh CPU `GatedRecurrentSnn`.
    /// Adam state is discarded — restart Adam from t=0 if you continue
    /// training on CPU.
    #[must_use]
    pub fn to_cpu(&self) -> GatedRecurrentSnn {
        use crate::surrogate::SpikeFn;
        use rand::SeedableRng;
        let n_input = self.n_input as usize;
        let n_hidden = self.n_hidden as usize;
        let n_output = self.n_output as usize;
        let mut net = GatedRecurrentSnn::new(
            n_input,
            n_hidden,
            n_output,
            SpikeFn {
                threshold: self.threshold,
                alpha: self.alpha,
            },
            &mut rand::rngs::StdRng::seed_from_u64(0),
        );
        net.w_in = self.w_in.download(&self.ctx);
        net.w_rec = self.w_rec.download(&self.ctx);
        net.w_out = self.w_out.download(&self.ctx);
        net.w_gate_in = self.w_gate_in.download(&self.ctx);
        net.w_gate_rec = self.w_gate_rec.download(&self.ctx);
        net.b_h = self.b_h.download(&self.ctx);
        net.b_o = self.b_o.download(&self.ctx);
        net.b_gate = self.b_gate.download(&self.ctx);
        net
    }

    /// Read all 8 grad buffers back to CPU. Useful for tests and
    /// `clip_grads_gpu` (which needs the global L2 norm).
    #[must_use]
    pub fn download_grads(&self) -> GatedGrads {
        GatedGrads {
            w_in: self.grads.w_in.download(&self.ctx),
            w_rec: self.grads.w_rec.download(&self.ctx),
            w_out: self.grads.w_out.download(&self.ctx),
            w_gate_in: self.grads.w_gate_in.download(&self.ctx),
            w_gate_rec: self.grads.w_gate_rec.download(&self.ctx),
            b_h: self.grads.b_h.download(&self.ctx),
            b_o: self.grads.b_o.download(&self.ctx),
            b_gate: self.grads.b_gate.download(&self.ctx),
        }
    }

    fn copy_packed_slot_into(
        encoder: &mut wgpu::CommandEncoder,
        packed: &GpuBuffer,
        dst: &GpuBuffer,
        slot: u32,
        n_hidden: u32,
    ) {
        let bytes = u64::from(n_hidden) * 4;
        let src_offset = u64::from(slot) * bytes;
        encoder.copy_buffer_to_buffer(&packed.inner, src_offset, &dst.inner, 0, bytes);
    }

    /// Sprint 19 batched megakernel forward: a single dispatch runs the
    /// full multi-timestep forward pass for **all `batch_size` samples
    /// in parallel** via `(batch_size, 1, 1)` workgroups. Each
    /// workgroup handles one sample independently with its own
    /// workgroup-shared hidden state. Cache buffers are sized
    /// `[batch * (T or T+1) * N]` and the kernel uses
    /// `workgroup_id.x` as the sample index.
    ///
    /// Caller is responsible for having uploaded `input_bytes_buf` and
    /// initial state buffers in the `[batch][T or N]` layout for the
    /// first `batch_size` samples, plus enough capacity for `t_steps`.
    fn gpu_forward_batched(&mut self, t_steps: usize, batch_size: usize) {
        let params = ForwardMegaParams {
            n_input: self.n_input,
            n_hidden: self.n_hidden,
            n_output: self.n_output,
            n_timesteps: t_steps as u32,
            threshold: self.threshold,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        self.ensure_fwd_mega_bind_io();
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora gpu_forward_mega"),
            });
        let bind_io = self
            .fwd_mega_bind_io
            .as_ref()
            .expect("ensure_fwd_mega_bind_io populated");
        self.forward_mega.dispatch_into(
            &mut encoder,
            &self.fwd_mega_bind_weights,
            bind_io,
            &params,
            batch_size as u32,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Upload concatenated batch input bytes to `input_bytes_buf`.
    /// `batch_inputs[b]` becomes the `[b * capacity_t .. (b+1) *
    /// capacity_t]` slice. `targets` populates the analogous
    /// `target_bytes_buf`. After `ensure_fb_capacity` matched
    /// `(t, b)` exactly to the chunk dims, every slot is filled.
    fn upload_batch_io(&self, batch_inputs: &[&[u8]], batch_targets: &[&[u8]]) {
        let stride = self.fb.capacity_t;
        let total = self.fb.capacity_b * stride;
        let mut input_buf = vec![0_u32; total];
        let mut target_buf = vec![0_u32; total];
        for (b, inputs) in batch_inputs.iter().enumerate() {
            let off = b * stride;
            debug_assert!(inputs.len() <= stride);
            for (t, &byte) in inputs.iter().enumerate() {
                input_buf[off + t] = u32::from(byte);
            }
        }
        for (b, targets) in batch_targets.iter().enumerate() {
            let off = b * stride;
            debug_assert!(targets.len() <= stride);
            for (t, &byte) in targets.iter().enumerate() {
                target_buf[off + t] = u32::from(byte);
            }
        }
        self.ctx.queue.write_buffer(
            &self.fb.input_bytes_buf.inner,
            0,
            bytemuck::cast_slice(&input_buf),
        );
        self.ctx.queue.write_buffer(
            &self.fb.target_bytes_buf.inner,
            0,
            bytemuck::cast_slice(&target_buf),
        );
    }

    /// Upload per-sample initial state into the `[B*N]`-sized
    /// `initial_h_v_after` and `initial_h_spike` buffers.
    fn upload_batch_initial_state(&self, states: &[(Vec<f32>, Vec<f32>)]) {
        let n = self.n_hidden as usize;
        let total = self.fb.capacity_b * n;
        let mut h_v_after_buf = vec![0.0_f32; total];
        let mut h_spike_buf = vec![0.0_f32; total];
        for (b, (hv, hs)) in states.iter().enumerate() {
            let off = b * n;
            h_v_after_buf[off..off + n].copy_from_slice(hv);
            h_spike_buf[off..off + n].copy_from_slice(hs);
        }
        self.fb
            .initial_h_v_after
            .upload(&self.ctx, &h_v_after_buf);
        self.fb.initial_h_spike.upload(&self.ctx, &h_spike_buf);
    }

    /// GPU-side softmax + cross-entropy: per-timestep `softmax_ce`
    /// kernel writes `d_logits_seq[t]` and `loss_seq[t]`. Single
    /// `loss_seq` download at the end gives total chunk loss with no
    /// per-timestep CPU sync (Sprint 16).
    /// Dispatch the batched softmax+CE+grad kernel over `(B, T, 1)`
    /// workgroups in one call. Targets are read from
    /// `self.fb.target_bytes_buf` (caller uploads them before this).
    /// Returns the sum of per-(sample, timestep) loss values from the
    /// first `batch_size * n_timesteps` entries of `loss_seq`.
    fn compute_d_logits_gpu_batched(&mut self, batch_size: usize, n_timesteps: usize) -> f32 {
        let o = self.n_output;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora compute_d_logits_gpu_batched"),
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
        // loss_seq is laid out as `[B][T]`; we use the first
        // `batch_size * n_timesteps` entries (the rest is leftover from
        // a previously larger chunk).
        let used = batch_size * n_timesteps;
        losses.iter().take(used).sum()
    }

    /// Sprint 20: fully batch-fused backward. Per timestep, ONE
    /// dispatch of each backward kernel covers all `B` samples in
    /// parallel. Per-element kernels (transpose, cell_back) use 2D
    /// dispatch `(h_chunks, B, 1)`; accumulator kernels (outer,
    /// column, vector_add) reduce across the batch dim in-thread
    /// when writing into shared weight grads.
    ///
    /// `t_capacity` (= `self.fb.capacity_t`) is the per-sample stride
    /// in the packed cache buffers. `batch_size = self.fb.capacity_b`.
    /// First `inputs[b]` of each sample is read for the timestep's
    /// byte index; targets are NOT used here (already consumed by
    /// `compute_d_logits_gpu_batched`).
    fn gpu_backward_batched(&mut self, t_steps: usize, batch_size: usize) {
        let h = self.n_hidden;
        let i_dim = self.n_input;
        let o = self.n_output;
        let b32 = batch_size as u32;
        let t_steps_u = t_steps as u32;
        let t_capacity = self.fb.capacity_t as u32;
        let cache_stride_no_init = t_steps_u * h;
        let cache_stride_init = (t_steps_u + 1) * h;
        let cache_stride_out = t_steps_u * o;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora gpu_backward_batched"),
            });

        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_v_after_carry_a);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_v_after_carry_b);
        self.zero.dispatch_into(&self.ctx, &mut encoder, &self.fb.d_h_spike_recurrent);

        for t in (0..t_steps).rev() {
            let t32 = t as u32;
            let off_t_n = t32 * h;
            let off_t1_n = (t32 + 1) * h;
            let off_t_o = t32 * o;

            // d_h_spike_from_out[b, h] = (W_out^T @ d_logits[b, t])[h]
            self.transpose.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_out,
                &self.fb.d_logits_packed,
                &self.fb.d_h_spike_from_out,
                o,
                h,
                false,
                off_t_o,
                cache_stride_out,
                h,
                b32,
            );

            let (carry_in, carry_out): (&GpuBuffer, &GpuBuffer) = if t.is_multiple_of(2) {
                (&self.fb.d_h_v_after_carry_a, &self.fb.d_h_v_after_carry_b)
            } else {
                (&self.fb.d_h_v_after_carry_b, &self.fb.d_h_v_after_carry_a)
            };

            // Per-element fused backward across all (h, b).
            self.cell_back.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_h_spike_from_out,
                &self.fb.d_h_spike_recurrent,
                carry_in,
                &self.fb.h_v_pre_packed,
                &self.fb.h_v_after_packed,
                &self.fb.gate_packed,
                &self.fb.d_cell_packed,
                &self.fb.d_h_spike_packed,
                h,
                self.threshold,
                self.alpha,
                off_t_n,
                off_t_n,
                off_t_n,
                cache_stride_no_init,
                cache_stride_init,
                b32,
            );
            // Slot layout in d_cell_packed: [slot][b][h] where slot
            // stride = B * N. Single copies extract each slot's full
            // batch worth of data.
            let slot_size = b32 * h;
            Self::copy_packed_slot_into(
                &mut encoder,
                &self.fb.d_cell_packed,
                &self.fb.d_h_v_pre,
                0,
                slot_size,
            );
            Self::copy_packed_slot_into(
                &mut encoder,
                &self.fb.d_cell_packed,
                &self.fb.d_gate_pre,
                1,
                slot_size,
            );
            Self::copy_packed_slot_into(
                &mut encoder,
                &self.fb.d_cell_packed,
                carry_out,
                2,
                slot_size,
            );

            // Bias gradients accumulate across batch.
            // b_o += sum_b d_logits[b, t]
            self.vector_add.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_logits_packed,
                &self.grads.b_o,
                o,
                off_t_o,
                cache_stride_out,
                b32,
            );
            // b_h += sum_b d_h_v_pre[b]
            self.vector_add.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_h_v_pre,
                &self.grads.b_h,
                h,
                0,
                h,
                b32,
            );
            self.vector_add.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_gate_pre,
                &self.grads.b_gate,
                h,
                0,
                h,
                b32,
            );

            // W_out grad: M += sum_b d_logits[b, t] ⊗ h_spike[b, t]
            // h_spike[t] is at slot t+1 of (T+1)*N per-sample slice.
            self.outer.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_logits_packed,
                &self.fb.h_spike_packed,
                &self.grads.w_out,
                o,
                h,
                off_t_o,
                off_t1_n,
                cache_stride_out,
                cache_stride_init,
                b32,
            );

            // W_rec / W_gate_rec: sum_b d_h_v_pre ⊗ h_spike_prev (slot t).
            self.outer.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_h_v_pre,
                &self.fb.h_spike_packed,
                &self.grads.w_rec,
                h,
                h,
                0,
                off_t_n,
                h,
                cache_stride_init,
                b32,
            );
            self.outer.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_gate_pre,
                &self.fb.h_spike_packed,
                &self.grads.w_gate_rec,
                h,
                h,
                0,
                off_t_n,
                h,
                cache_stride_init,
                b32,
            );

            // W_in / W_gate_in: column-add at this timestep's byte
            // for each sample. Kernel reads byte_idx[b * t_capacity +
            // t] from input_bytes_buf.
            self.column.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_h_v_pre,
                &self.grads.w_in,
                &self.fb.input_bytes_buf,
                h,
                i_dim,
                t_capacity,
                t32,
                h,
                b32,
            );
            self.column.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.fb.d_gate_pre,
                &self.grads.w_gate_in,
                &self.fb.input_bytes_buf,
                h,
                i_dim,
                t_capacity,
                t32,
                h,
                b32,
            );

            // d_h_spike_recurrent[b] = W_rec^T @ d_h_v_pre[b]
            //                        + W_gate_rec^T @ d_gate_pre[b]
            self.transpose.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_rec,
                &self.fb.d_h_v_pre,
                &self.fb.d_h_spike_recurrent,
                h,
                h,
                false,
                0,
                h,
                h,
                b32,
            );
            self.transpose.dispatch_into(
                &self.ctx,
                &mut encoder,
                &self.w_gate_rec,
                &self.fb.d_gate_pre,
                &self.fb.d_h_spike_recurrent,
                h,
                h,
                true,
                0,
                h,
                h,
                b32,
            );
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}
