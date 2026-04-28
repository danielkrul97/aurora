//! GPU mirror of `GatedRecurrentSnn` forward path (Sprint 14).
//!
//! Holds GPU-resident weight buffers and a per-instance set of compute
//! pipelines. `forward_lm_bytes_gpu` runs the full multi-timestep
//! forward over a byte sequence and returns logits per timestep,
//! matching `GatedRecurrentSnn::forward_lm_bytes` to f32 numerics.
//!
//! Backward and Adam port are Sprint 15 work; the CPU path remains the
//! authoritative training loop until those land.

use std::sync::Arc;

use super::buffers::GpuBuffer;
use super::kernels::{DenseMatvecKernel, GatedCellStepKernel, SparseMatvecOneHotKernel};
use super::GpuContext;
use crate::snn_gated::GatedRecurrentSnn;

/// GPU-resident weights + cached compute pipelines for one network.
pub struct GatedRecurrentSnnGpu {
    pub ctx: GpuContext,
    pub n_input: u32,
    pub n_hidden: u32,
    pub n_output: u32,
    pub threshold: f32,

    // Weights (uploaded once at construction).
    pub w_in: GpuBuffer,
    pub w_rec: GpuBuffer,
    pub w_out: GpuBuffer,
    pub w_gate_in: GpuBuffer,
    pub w_gate_rec: GpuBuffer,
    pub b_h: GpuBuffer,
    pub b_o: GpuBuffer,
    pub b_gate: GpuBuffer,

    // Compute pipelines.
    onehot: Arc<SparseMatvecOneHotKernel>,
    dense: Arc<DenseMatvecKernel>,
    cell: Arc<GatedCellStepKernel>,

    // Persistent placeholder buffer for matvecs that don't add a bias
    // (W_rec, W_gate_rec). Length 0 would fail bind validation, so we
    // keep a length-1 buffer with bias_mode=0.
    no_bias_placeholder: GpuBuffer,
}

impl GatedRecurrentSnnGpu {
    /// Upload CPU network weights to GPU. Compiles all forward kernels.
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

        Self {
            n_input: net.n_input as u32,
            n_hidden: net.n_hidden as u32,
            n_output: net.n_output as u32,
            threshold: net.spike_fn.threshold,
            w_in,
            w_rec,
            w_out,
            w_gate_in,
            w_gate_rec,
            b_h,
            b_o,
            b_gate,
            onehot,
            dense,
            cell,
            no_bias_placeholder,
            ctx,
        }
    }

    /// GPU multi-timestep forward over a byte sequence. Returns
    /// `[T][n_output]` logits — same shape as `LmCache::logits` from
    /// the CPU path. Hidden state starts at zero and is discarded at the
    /// end (stateless). Streaming/stateful GPU forward is Sprint 15.
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
        let bg_cpu = self.b_gate.download(&self.ctx);

        for (t, &byte) in input_bytes.iter().enumerate() {
            assert!(u32::from(byte) < i_dim);

            let (h_v_after_prev, h_spike_prev, h_v_after_next, h_spike_next) =
                if t.is_multiple_of(2) {
                    (&h_v_after_a, &h_spike_a, &h_v_after_b, &h_spike_b)
                } else {
                    (&h_v_after_b, &h_spike_b, &h_v_after_a, &h_spike_a)
                };

            self.onehot
                .dispatch(&self.ctx, &self.w_in, &h_in, h, i_dim, u32::from(byte));
            self.onehot.dispatch(
                &self.ctx,
                &self.w_gate_in,
                &gate_in_contrib,
                h,
                i_dim,
                u32::from(byte),
            );
            self.dense.dispatch(
                &self.ctx,
                &self.w_rec,
                h_spike_prev,
                &self.no_bias_placeholder,
                &h_rec,
                h,
                h,
                false,
            );
            self.dense.dispatch(
                &self.ctx,
                &self.w_gate_rec,
                h_spike_prev,
                &self.no_bias_placeholder,
                &gate_rec_contrib,
                h,
                h,
                false,
            );

            // gate_pre = b_gate + W_gate_in[:, byte] + W_gate_rec @ h_spike_prev.
            // CPU sum is acceptable here (3 buffers × n_hidden × 4 bytes
            // ≈ 3 KB at hidden=256). A fused-add WGSL kernel is a Sprint
            // 14.1 follow-up.
            let gin = gate_in_contrib.download(&self.ctx);
            let grec = gate_rec_contrib.download(&self.ctx);
            let mut gp = vec![0.0_f32; h as usize];
            for k in 0..h as usize {
                gp[k] = bg_cpu[k] + gin[k] + grec[k];
            }
            gate_pre.upload(&self.ctx, &gp);

            self.cell.dispatch(
                &self.ctx,
                &gate_pre,
                &h_in,
                &h_rec,
                h_v_after_prev,
                &self.b_h,
                &packed_cell,
                h,
                self.threshold,
            );

            // Extract h_spike (slot 2) and h_v_after (slot 3) for the
            // subsequent dispatches: w_out reads h_spike, next iter's
            // membrane integration reads h_v_after.
            self.copy_packed_slot(&packed_cell, h_spike_next, 2, h);
            self.copy_packed_slot(&packed_cell, h_v_after_next, 3, h);

            self.dense.dispatch(
                &self.ctx,
                &self.w_out,
                h_spike_next,
                &self.b_o,
                &logits_buf,
                o,
                h,
                true,
            );
            logits_seq.push(logits_buf.download(&self.ctx));
        }

        logits_seq
    }

    fn copy_packed_slot(&self, packed: &GpuBuffer, dst: &GpuBuffer, slot: u32, n_hidden: u32) {
        let bytes = u64::from(n_hidden) * 4;
        let src_offset = u64::from(slot) * bytes;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora packed slot copy"),
            });
        enc.copy_buffer_to_buffer(&packed.inner, src_offset, &dst.inner, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
    }
}
