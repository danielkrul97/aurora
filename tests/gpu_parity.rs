#![cfg(feature = "gpu")]

//! GPU parity tests:
//! - Sprint 13: `sparse_matvec_onehot` bit-equality with CPU.
//! - Sprint 14: full `forward_lm_bytes` GPU vs CPU within FP tolerance.
//! - Sprint 15: backward gradient parity (8 buffers within tolerance)
//!   and end-to-end training direction (loss strictly decreases over
//!   N Adam steps on a fixed input).

use aurora::gpu::buffers::GpuBuffer;
use aurora::gpu::kernels::SparseMatvecOneHotKernel;
use aurora::gpu::{GatedRecurrentSnnGpu, GpuContext};
use aurora::optimizer::AdamConfig;
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
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora gpu_parity test enc"),
            });
        kernel.dispatch_into(
            &ctx,
            &mut enc,
            &w_buf,
            &out_buf,
            n_hidden as u32,
            n_input as u32,
            byte_idx,
        );
        ctx.queue.submit(Some(enc.finish()));
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

#[test]
fn gated_backward_lm_grads_match_cpu_within_tolerance() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };

    let n_hidden = 32_usize;
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let mut net_cpu = GatedRecurrentSnn::new(VOCAB, n_hidden, VOCAB, SpikeFn::default(), &mut rng);
    let mut net_gpu = GatedRecurrentSnnGpu::from_cpu(&net_cpu, ctx);

    let mut byte_rng = StdRng::seed_from_u64(0xBEEF);
    let bytes: Vec<u8> = (0..16).map(|_| byte_rng.random::<u8>()).collect();
    // Targets = inputs shifted by one (next-byte prediction).
    let targets: Vec<u8> = bytes.iter().skip(1).copied().chain([0]).collect();

    // CPU forward + backward into a fresh grad buffer.
    let cache_cpu = net_cpu.forward_lm_bytes(&bytes);
    let mut grads_cpu = net_cpu.alloc_grads();
    let loss_cpu = net_cpu.backward_lm_bytes_into(&cache_cpu, &bytes, &targets, &mut grads_cpu);

    // GPU forward + backward into resident grad buffers (start zeroed).
    net_gpu.zero_grad_gpu();
    let loss_gpu = net_gpu.forward_backward_lm_bytes_gpu_stateless(&bytes, &targets);
    let grads_gpu = net_gpu.download_grads();

    let loss_diff = (loss_gpu - loss_cpu).abs();
    assert!(
        loss_diff < 1e-2,
        "loss mismatch: gpu={loss_gpu} cpu={loss_cpu} diff={loss_diff}"
    );

    let groups: [(&str, &[f32], &[f32], f32); 8] = [
        ("w_in", &grads_gpu.w_in, &grads_cpu.w_in, 5e-3),
        ("w_rec", &grads_gpu.w_rec, &grads_cpu.w_rec, 5e-3),
        ("w_out", &grads_gpu.w_out, &grads_cpu.w_out, 5e-3),
        ("w_gate_in", &grads_gpu.w_gate_in, &grads_cpu.w_gate_in, 5e-3),
        ("w_gate_rec", &grads_gpu.w_gate_rec, &grads_cpu.w_gate_rec, 5e-3),
        ("b_h", &grads_gpu.b_h, &grads_cpu.b_h, 5e-3),
        ("b_o", &grads_gpu.b_o, &grads_cpu.b_o, 5e-3),
        ("b_gate", &grads_gpu.b_gate, &grads_cpu.b_gate, 5e-3),
    ];
    for (name, gpu, cpu, tol) in groups {
        assert_eq!(gpu.len(), cpu.len(), "len mismatch in {name}");
        let mut max_diff = 0.0_f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < tol,
                "{name} grad diff at idx={i}: gpu={g} cpu={c} diff={diff}"
            );
        }
        eprintln!("{name}: max diff {max_diff:.2e}");
    }
    let _ = &mut net_cpu;
}

#[test]
fn gated_end_to_end_gpu_training_reduces_loss() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };

    let n_hidden = 32_usize;
    let mut rng = StdRng::seed_from_u64(0xFACE);
    let net_cpu = GatedRecurrentSnn::new(VOCAB, n_hidden, VOCAB, SpikeFn::default(), &mut rng);
    let mut net_gpu = GatedRecurrentSnnGpu::from_cpu(&net_cpu, ctx);

    // Predictable repeating cycle: easy to overfit, so loss should
    // strictly drop within a handful of Adam steps.
    let cycle: Vec<u8> = (0..32).map(|i| (i * 7 + 13) as u8).collect();
    let inputs: Vec<u8> = cycle.iter().cycle().take(64).copied().collect();
    let targets: Vec<u8> = inputs.iter().skip(1).copied().chain([inputs[0]]).collect();

    let cfg = AdamConfig {
        lr: 0.02,
        ..AdamConfig::default()
    };
    let mut first_loss = None;
    let mut last_loss = 0.0_f32;
    for step in 0..40 {
        net_gpu.zero_grad_gpu();
        let loss = net_gpu.forward_backward_lm_bytes_gpu_stateless(&inputs, &targets);
        net_gpu.clip_grads_gpu(1.5);
        net_gpu.step_adam_gpu(&cfg);
        if first_loss.is_none() {
            first_loss = Some(loss);
        }
        last_loss = loss;
        if step % 5 == 0 {
            eprintln!("step {step}: loss = {loss:.4}");
        }
    }
    let first = first_loss.expect("at least one step");
    assert!(
        last_loss < first * 0.85,
        "expected GPU training to reduce loss: first={first} last={last_loss}"
    );
}

#[test]
fn batch_fused_matches_sequential_per_sample() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };

    let n_hidden = 32_usize;
    let batch_size = 4_usize;
    let t_steps = 12_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let net_cpu = GatedRecurrentSnn::new(VOCAB, n_hidden, VOCAB, SpikeFn::default(), &mut rng);

    let mut byte_rng = StdRng::seed_from_u64(0xBADD_F00D);
    let inputs: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| {
            (0..t_steps)
                .map(|_| byte_rng.random::<u8>())
                .collect::<Vec<_>>()
        })
        .collect();
    let targets: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| {
            (0..t_steps)
                .map(|_| byte_rng.random::<u8>())
                .collect::<Vec<_>>()
        })
        .collect();
    let inputs_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let targets_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();

    // Path A: 4× single-sample sequential.
    let mut net_a = GatedRecurrentSnnGpu::from_cpu(&net_cpu, ctx.clone());
    net_a.zero_grad_gpu();
    let mut states_a: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .map(|_| (vec![0.0_f32; n_hidden], vec![0.0_f32; n_hidden]))
        .collect();
    let loss_a = net_a.forward_backward_batch_gpu(&inputs_refs, &targets_refs, &mut states_a);
    let grads_a = net_a.download_grads();

    // Path B: batch-fused (1 forward megakernel + 1 softmax_ce dispatch
    // covers all B samples; backward sequential per sample).
    let mut net_b = GatedRecurrentSnnGpu::from_cpu(&net_cpu, ctx);
    net_b.zero_grad_gpu();
    let mut states_b: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .map(|_| (vec![0.0_f32; n_hidden], vec![0.0_f32; n_hidden]))
        .collect();
    let loss_b =
        net_b.forward_backward_batch_gpu_fused(&inputs_refs, &targets_refs, &mut states_b);
    let grads_b = net_b.download_grads();

    let loss_diff = (loss_a - loss_b).abs();
    assert!(
        loss_diff < 1e-2,
        "batch-fused loss diverged from sequential: a={loss_a} b={loss_b} diff={loss_diff}"
    );

    let groups: [(&str, &[f32], &[f32]); 8] = [
        ("w_in", &grads_a.w_in, &grads_b.w_in),
        ("w_rec", &grads_a.w_rec, &grads_b.w_rec),
        ("w_out", &grads_a.w_out, &grads_b.w_out),
        ("w_gate_in", &grads_a.w_gate_in, &grads_b.w_gate_in),
        ("w_gate_rec", &grads_a.w_gate_rec, &grads_b.w_gate_rec),
        ("b_h", &grads_a.b_h, &grads_b.b_h),
        ("b_o", &grads_a.b_o, &grads_b.b_o),
        ("b_gate", &grads_a.b_gate, &grads_b.b_gate),
    ];
    for (name, a, b) in groups {
        let mut max_diff = 0.0_f32;
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let d = (av - bv).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 5e-3,
                "{name} grad diff at {i}: seq={av} fused={bv} diff={d}"
            );
        }
        eprintln!("{name}: batch-fused max diff {max_diff:.2e}");
    }

    // Final state should match too.
    for b in 0..batch_size {
        for i in 0..n_hidden {
            assert!((states_a[b].0[i] - states_b[b].0[i]).abs() < 1e-3);
            assert!((states_a[b].1[i] - states_b[b].1[i]).abs() < 1e-3);
        }
    }
}
