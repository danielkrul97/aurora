#![cfg(feature = "gpu")]

//! Sprint 13 GPU parity foundation: confirm that the WGSL kernels we
//! ship produce the same numbers the CPU implementation does.
//!
//! Currently covers:
//! - `sparse_matvec_onehot` — one-hot input matvec, the hot path of
//!   `forward_lm_bytes` for the LM input layer.
//!
//! Backward kernels and full forward/backward GPU integration land in
//! Sprint 13 followup turns; their parity assertions go here.

use aurora::gpu::buffers::GpuBuffer;
use aurora::gpu::kernels::SparseMatvecOneHotKernel;
use aurora::gpu::{GatedRecurrentSnnGpu, GpuContext};
use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::VOCAB;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn sparse_matvec_onehot_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    let info = ctx.adapter_info();
    eprintln!("GPU adapter: {} ({:?})", info.name, info.backend);

    let n_hidden = 256_usize;
    let n_input = 256_usize;
    let w: Vec<f32> = (0..n_hidden * n_input)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();

    let kernel = SparseMatvecOneHotKernel::new(&ctx);
    let w_buf = GpuBuffer::from_slice(&ctx, "w", &w);
    let out_buf = GpuBuffer::zeros(&ctx, "out", n_hidden);

    for byte_idx in [0u32, 13, 99, 200, 255] {
        kernel.dispatch(
            &ctx,
            &w_buf,
            &out_buf,
            n_hidden as u32,
            n_input as u32,
            byte_idx,
        );
        let gpu = out_buf.download(&ctx);
        let cpu: Vec<f32> = (0..n_hidden)
            .map(|h| w[h * n_input + byte_idx as usize])
            .collect();
        for (h, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(
                g, c,
                "mismatch at byte_idx={byte_idx}, h={h}: gpu={g} cpu={c}"
            );
        }
    }
}

#[test]
fn gated_forward_lm_matches_cpu_within_tolerance() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };

    // Tiny network so the test runs in a few seconds even with the
    // current CPU-roundtrip-per-step gate_pre fusion.
    let n_hidden = 32_usize;
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let net_cpu = GatedRecurrentSnn::new(VOCAB, n_hidden, VOCAB, SpikeFn::default(), &mut rng);

    let net_gpu = GatedRecurrentSnnGpu::from_cpu(&net_cpu, ctx);

    let mut byte_rng = StdRng::seed_from_u64(0xBEEF);
    let bytes: Vec<u8> = (0..16).map(|_| byte_rng.random::<u8>()).collect();

    let cache_cpu = net_cpu.forward_lm_bytes(&bytes);
    let logits_gpu = net_gpu.forward_lm_bytes_gpu(&bytes);

    assert_eq!(logits_gpu.len(), cache_cpu.logits.len());
    let mut max_diff = 0.0_f32;
    for (t, (gpu_step, cpu_step)) in logits_gpu.iter().zip(cache_cpu.logits.iter()).enumerate() {
        assert_eq!(
            gpu_step.len(),
            cpu_step.len(),
            "logits dim mismatch at t={t}"
        );
        for (i, (g, c)) in gpu_step.iter().zip(cpu_step.iter()).enumerate() {
            let diff = (g - c).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-3,
                "logits diff at t={t} i={i}: gpu={g} cpu={c} diff={diff}"
            );
        }
    }
    eprintln!("GPU forward parity OK, max diff = {max_diff:.2e}");
}
