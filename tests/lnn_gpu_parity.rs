//! CPU↔GPU parity for the LNN-CfC stack.
//!
//! 1. Forward parity: identical weights, identical inputs, identical
//!    initial state → `x_packed` and logits agree to ~1e-3.
//! 2. Backward parity: same forward setup, then one Adam step on both
//!    sides → updated weights agree to ~1e-3.
//! 3. Multi-step parity: 10 Adam steps on identical inputs → final
//!    weights still agree (drift bounded by FP32 noise).
//!
//! CfC has only smooth activations (sigmoid, tanh, exp), so parity
//! tolerance is much tighter than the SNN heaviside surrogate path
//! (which used ~5e-2). We assert max-abs drift < 1e-3 on every weight
//! tensor.

#![cfg(feature = "gpu")]

use aurora::gpu::{GpuContext, MultiLayerLnnSnnGpu};
use aurora::lnn::{LnnState, MultiLayerLnnSnn};
use aurora::optimizer::AdamConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL_FORWARD: f32 = 1.0e-3;
const TOL_BACKWARD: f32 = 1.0e-3;
const TOL_MULTI: f32 = 5.0e-3;

fn build_net(seed: u64, n_in: usize, hidden: &[usize], n_out: usize) -> MultiLayerLnnSnn {
    let mut rng = StdRng::seed_from_u64(seed);
    MultiLayerLnnSnn::new(n_in, hidden, n_out, &mut rng)
}

fn random_tokens(seed: u64, batch: usize, t: usize, vocab: usize) -> Vec<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch)
        .map(|_| (0..t).map(|_| rng.random_range(0..vocab as u32)).collect())
        .collect()
}

fn init_states(net: &MultiLayerLnnSnn, batch: usize, seed: u64) -> Vec<Vec<LnnState>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch)
        .map(|_| {
            net.layers
                .iter()
                .map(|l| {
                    let v: Vec<f32> = (0..l.n_hidden)
                        .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
                        .collect();
                    LnnState { x: v }
                })
                .collect()
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn forward_parity_single_layer() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let net_cpu = build_net(1, 16, &[8], 16);
    let mut net_gpu = MultiLayerLnnSnnGpu::from_cpu(&net_cpu, ctx);

    let batch = 4;
    let t = 12;
    let vocab = 16;
    let inputs = random_tokens(7, batch, t, vocab);
    let mut states_cpu = init_states(&net_cpu, batch, 11);
    let mut states_gpu = states_cpu.clone();

    // CPU forward (collect logits + final x)
    let mut cpu_logits: Vec<Vec<f32>> = Vec::new();
    for (b, inp) in inputs.iter().enumerate() {
        let cache = net_cpu.forward_lm_tokens_stateful(inp, &mut states_cpu[b]);
        cpu_logits.extend(cache.logits.into_iter().flatten().map(|v| vec![v]));
        let _ = cache;
    }
    // Re-do CPU collect into flat logits per batch.
    let mut cpu_flat: Vec<f32> = Vec::with_capacity(batch * t * vocab);
    for (b, inp) in inputs.iter().enumerate() {
        let mut local_state = init_states(&net_cpu, batch, 11)[b].clone();
        let cache = net_cpu.forward_lm_tokens_stateful(inp, &mut local_state);
        for logits in &cache.logits {
            cpu_flat.extend_from_slice(logits);
        }
    }

    let inp_refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    net_gpu.forward_lm_tokens_stateful_batch(&inp_refs, &mut states_gpu);
    let gpu_flat = net_gpu.download_logits();

    let drift = max_abs(&cpu_flat, &gpu_flat);
    assert!(
        drift < TOL_FORWARD,
        "single-layer forward drift {drift} exceeds {TOL_FORWARD}"
    );
}

#[test]
fn forward_parity_two_layer() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let net_cpu = build_net(2, 32, &[16, 16], 32);
    let mut net_gpu = MultiLayerLnnSnnGpu::from_cpu(&net_cpu, ctx);

    let batch = 3;
    let t = 8;
    let vocab = 32;
    let inputs = random_tokens(13, batch, t, vocab);

    let mut cpu_flat: Vec<f32> = Vec::with_capacity(batch * t * vocab);
    let states_seed = init_states(&net_cpu, batch, 21);
    for (b, inp) in inputs.iter().enumerate() {
        let mut s = states_seed[b].clone();
        let cache = net_cpu.forward_lm_tokens_stateful(inp, &mut s);
        for logits in &cache.logits {
            cpu_flat.extend_from_slice(logits);
        }
    }

    let mut states_gpu = states_seed.clone();
    let inp_refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    net_gpu.forward_lm_tokens_stateful_batch(&inp_refs, &mut states_gpu);
    let gpu_flat = net_gpu.download_logits();

    let drift = max_abs(&cpu_flat, &gpu_flat);
    assert!(
        drift < TOL_FORWARD,
        "two-layer forward drift {drift} exceeds {TOL_FORWARD}"
    );

    // Final-state parity: GPU writes back into states_gpu.
    for b in 0..batch {
        for l in 0..net_cpu.layers.len() {
            let mut s = states_seed[b].clone();
            net_cpu.forward_lm_tokens_stateful(&inputs[b], &mut s);
            let drift_state = max_abs(&s[l].x, &states_gpu[b][l].x);
            assert!(
                drift_state < TOL_FORWARD,
                "state drift batch={b} layer={l}: {drift_state}"
            );
        }
    }
}

#[test]
fn backward_parity_single_step() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut net_cpu = build_net(3, 16, &[8, 8], 16);
    let mut net_gpu = MultiLayerLnnSnnGpu::from_cpu(&net_cpu, ctx);

    let batch = 4;
    let t = 8;
    let vocab = 16;
    let inputs = random_tokens(31, batch, t, vocab);
    let targets = random_tokens(32, batch, t, vocab);
    let states_seed = init_states(&net_cpu, batch, 41);

    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    // CPU: forward + backward + Adam.
    let inp_refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let tgt_refs: Vec<&[u32]> = targets.iter().map(|v| v.as_slice()).collect();
    let mut cpu_states = states_seed.clone();
    net_cpu.zero_grad();
    let _ = net_cpu.forward_backward_batch_tokens_stateful(&inp_refs, &tgt_refs, &mut cpu_states);
    net_cpu.clip_grads(1.0);
    net_cpu.step_adam(&adam);

    // GPU: same.
    let mut gpu_states = states_seed.clone();
    net_gpu.zero_grad();
    let _ = net_gpu.forward_backward_batch_tokens_stateful(&inp_refs, &tgt_refs, &mut gpu_states);
    net_gpu.clip_grads(1.0);
    net_gpu.step_adam(&adam);

    let net_gpu_back = net_gpu.to_cpu();
    for (li, (cpu_l, gpu_l)) in net_cpu.layers.iter().zip(&net_gpu_back.layers).enumerate() {
        let drift_w_f = max_abs(&cpu_l.w_f, &gpu_l.w_f);
        let drift_w_g = max_abs(&cpu_l.w_g, &gpu_l.w_g);
        let drift_w_h = max_abs(&cpu_l.w_h, &gpu_l.w_h);
        let drift_b_f = max_abs(&cpu_l.b_f, &gpu_l.b_f);
        let drift_b_g = max_abs(&cpu_l.b_g, &gpu_l.b_g);
        let drift_b_h = max_abs(&cpu_l.b_h, &gpu_l.b_h);
        let drift_lt = max_abs(&cpu_l.log_tau, &gpu_l.log_tau);
        assert!(
            drift_w_f < TOL_BACKWARD,
            "L{li} w_f drift {drift_w_f}"
        );
        assert!(
            drift_w_g < TOL_BACKWARD,
            "L{li} w_g drift {drift_w_g}"
        );
        assert!(
            drift_w_h < TOL_BACKWARD,
            "L{li} w_h drift {drift_w_h}"
        );
        assert!(
            drift_b_f < TOL_BACKWARD,
            "L{li} b_f drift {drift_b_f}"
        );
        assert!(
            drift_b_g < TOL_BACKWARD,
            "L{li} b_g drift {drift_b_g}"
        );
        assert!(
            drift_b_h < TOL_BACKWARD,
            "L{li} b_h drift {drift_b_h}"
        );
        assert!(drift_lt < TOL_BACKWARD, "L{li} log_tau drift {drift_lt}");
    }
    let drift_w_out = max_abs(&net_cpu.w_out, &net_gpu_back.w_out);
    let drift_b_o = max_abs(&net_cpu.b_o, &net_gpu_back.b_o);
    assert!(drift_w_out < TOL_BACKWARD, "w_out drift {drift_w_out}");
    assert!(drift_b_o < TOL_BACKWARD, "b_o drift {drift_b_o}");
}

#[test]
fn multi_step_drift_bounded() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut net_cpu = build_net(4, 16, &[8], 16);
    let mut net_gpu = MultiLayerLnnSnnGpu::from_cpu(&net_cpu, ctx);

    let batch = 4;
    let t = 8;
    let vocab = 16;
    let adam = AdamConfig {
        lr: 0.01,
        ..AdamConfig::default()
    };

    let mut cpu_states = init_states(&net_cpu, batch, 51);
    let mut gpu_states = cpu_states.clone();

    for step in 0..10 {
        let inputs = random_tokens(100 + step, batch, t, vocab);
        let targets = random_tokens(200 + step, batch, t, vocab);
        let inp_refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let tgt_refs: Vec<&[u32]> = targets.iter().map(|v| v.as_slice()).collect();

        net_cpu.zero_grad();
        let _ = net_cpu.forward_backward_batch_tokens_stateful(&inp_refs, &tgt_refs, &mut cpu_states);
        net_cpu.clip_grads(1.0);
        net_cpu.step_adam(&adam);

        net_gpu.zero_grad();
        let _ = net_gpu.forward_backward_batch_tokens_stateful(&inp_refs, &tgt_refs, &mut gpu_states);
        net_gpu.clip_grads(1.0);
        net_gpu.step_adam(&adam);
    }

    let net_gpu_back = net_gpu.to_cpu();
    for (li, (cpu_l, gpu_l)) in net_cpu.layers.iter().zip(&net_gpu_back.layers).enumerate() {
        let drift = max_abs(&cpu_l.w_f, &gpu_l.w_f).max(max_abs(&cpu_l.w_g, &gpu_l.w_g)).max(
            max_abs(&cpu_l.w_h, &gpu_l.w_h)
                .max(max_abs(&cpu_l.b_f, &gpu_l.b_f))
                .max(max_abs(&cpu_l.b_g, &gpu_l.b_g))
                .max(max_abs(&cpu_l.b_h, &gpu_l.b_h))
                .max(max_abs(&cpu_l.log_tau, &gpu_l.log_tau)),
        );
        assert!(
            drift < TOL_MULTI,
            "L{li} 10-step drift {drift} exceeds {TOL_MULTI}"
        );
    }
    let drift_w_out = max_abs(&net_cpu.w_out, &net_gpu_back.w_out);
    assert!(
        drift_w_out < TOL_MULTI,
        "w_out 10-step drift {drift_w_out}"
    );
}
