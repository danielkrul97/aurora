//! Sprint 50 smoke: closed-form continuous-time (CfC) LNN cell.
//!
//! 1. Forward shape sanity (logits + per-layer caches).
//! 2. Bounded state — `x ∈ [-1, 1]` by construction (alpha·tanh + (1-alpha)·tanh).
//! 3. End-to-end Adam training reduces loss on a tiny LM problem.
//! 4. log_tau stays inside the clamp envelope after Adam steps.
//! 5. Five central-difference numerical gradient checks covering every
//!    backward path: `b_f`, `b_g`, `b_h`, `log_tau`, and one weight in
//!    `W_f` (input-projection column).
//!
//! CfC has smooth activations (sigmoid, tanh, exp) — no surrogate
//! gradient — so finite-difference equality should hold tightly. We
//! demand abs error < 1e-2 (or rel < 1 %) on each check.

use aurora::lnn::{LnnState, MultiLayerLnnSnn};
use aurora::optimizer::AdamConfig;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn build_net(seed: u64, n_in: usize, n_h: usize, n_out: usize) -> MultiLayerLnnSnn {
    let mut rng = StdRng::seed_from_u64(seed);
    MultiLayerLnnSnn::new(n_in, &[n_h, n_h], n_out, &mut rng)
}

fn forward_loss_nats(net: &MultiLayerLnnSnn, inputs: &[u32], targets: &[u32]) -> f32 {
    let cache = net.forward_lm_tokens(inputs);
    let mut loss = 0.0_f32;
    for (logits, &target) in cache.logits.iter().zip(targets.iter()) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
        loss += lse - logits[target as usize];
    }
    loss
}

#[test]
fn forward_produces_expected_shapes() {
    let net = build_net(0, 16, 8, 16);
    let tokens: Vec<u32> = (0..32).map(|t| (t % 16) as u32).collect();
    let cache = net.forward_lm_tokens(&tokens);
    assert_eq!(cache.logits.len(), tokens.len());
    for logits in &cache.logits {
        assert_eq!(logits.len(), 16);
    }
    assert_eq!(cache.layers.len(), 2);
    for layer in &cache.layers {
        assert_eq!(layer.f.len(), tokens.len());
        assert_eq!(layer.g.len(), tokens.len());
        assert_eq!(layer.h_act.len(), tokens.len());
        assert_eq!(layer.alpha.len(), tokens.len());
        assert_eq!(layer.x.len(), tokens.len());
        assert_eq!(layer.x[0].len(), 8);
    }
}

#[test]
fn state_is_bounded_by_construction() {
    // x = α·tanh(g_pre) + (1-α)·tanh(h_pre); both tanh outputs ∈ [-1, 1]
    // and α ∈ (0, 1), so x stays in [-1, 1] regardless of weights.
    let net = build_net(99, 32, 16, 32);
    let tokens: Vec<u32> = (0..64).map(|t| (t % 32) as u32).collect();
    let cache = net.forward_lm_tokens(&tokens);
    for (l, layer) in cache.layers.iter().enumerate() {
        for (t, row) in layer.x.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                assert!(
                    v.abs() <= 1.0 + 1e-5,
                    "layer {l} t {t} neuron {i}: x = {v} outside [-1, 1]"
                );
            }
        }
    }
}

#[test]
fn training_reduces_loss() {
    let mut net = build_net(0xBEEF, 16, 32, 16);
    let inputs_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..32).map(|t| ((t + b) % 16) as u32).collect())
        .collect();
    let targets_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..32).map(|t| ((t + b + 1) % 16) as u32).collect())
        .collect();
    let inputs: Vec<&[u32]> = inputs_owned.iter().map(Vec::as_slice).collect();
    let targets: Vec<&[u32]> = targets_owned.iter().map(Vec::as_slice).collect();
    let mut states: Vec<Vec<LnnState>> = (0..4).map(|_| net.fresh_states()).collect();
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };
    let mut losses = Vec::new();
    for _ in 0..50 {
        net.zero_grad();
        let l = net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
        assert!(l.is_finite(), "loss became non-finite");
        losses.push(l);
    }
    let initial = losses[0..3].iter().sum::<f32>() / 3.0;
    let final_ = losses[47..50].iter().sum::<f32>() / 3.0;
    assert!(
        final_ < initial,
        "loss did not decrease: initial={initial:.3}, final={final_:.3}"
    );
}

#[test]
fn log_tau_stays_in_clamp_envelope() {
    let mut net = build_net(7, 16, 16, 16);
    let inputs_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b) % 16) as u32).collect())
        .collect();
    let targets_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b + 1) % 16) as u32).collect())
        .collect();
    let inputs: Vec<&[u32]> = inputs_owned.iter().map(Vec::as_slice).collect();
    let targets: Vec<&[u32]> = targets_owned.iter().map(Vec::as_slice).collect();
    let mut states: Vec<Vec<LnnState>> = (0..4).map(|_| net.fresh_states()).collect();
    let adam = AdamConfig {
        lr: 0.05,
        ..AdamConfig::default()
    };
    for _ in 0..40 {
        net.zero_grad();
        let _ = net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
    }
    for (l, layer) in net.layers.iter().enumerate() {
        for (i, &v) in layer.log_tau.iter().enumerate() {
            assert!(
                v.abs() <= 3.0 + 1e-5,
                "layer {l} neuron {i}: log_tau = {v} outside clamp envelope"
            );
        }
    }
}

/// Generic finite-difference grad check: perturb `params[idx]` by ±eps,
/// compare to analytical grad. Smooth activations (sigmoid + tanh +
/// exp + linear) → tight tolerance.
fn check_grad(
    label: &str,
    net: &mut MultiLayerLnnSnn,
    inputs: &[u32],
    targets: &[u32],
    extract: impl Fn(&MultiLayerLnnSnn) -> f32,
    extract_grad: impl Fn(&aurora::lnn::MultiLayerLnnGrads) -> f32,
    set: impl Fn(&mut MultiLayerLnnSnn, f32),
) {
    let original = extract(net);
    net.zero_grad();
    let cache = net.forward_lm_tokens(inputs);
    let _ = net.backward_lm_tokens(&cache, inputs, targets);
    let g_analytic = extract_grad(net.grads());

    let eps = 1e-3_f32;
    set(net, original + eps);
    let loss_p = forward_loss_nats(net, inputs, targets);
    set(net, original - eps);
    let loss_m = forward_loss_nats(net, inputs, targets);
    set(net, original);

    let g_numeric = (loss_p - loss_m) / (2.0 * eps);
    let abs_err = (g_analytic - g_numeric).abs();
    let rel_err = abs_err / (g_numeric.abs() + 1e-6);
    assert!(
        abs_err < 1e-2 || rel_err < 0.01,
        "{label}: analytic={g_analytic:.6}, numeric={g_numeric:.6}, abs_err={abs_err:.6}, rel_err={rel_err:.6}"
    );
}

#[test]
fn b_f_gradient_matches_finite_difference() {
    let mut net = build_net(11, 8, 4, 8);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_f[0][2]",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[0].b_f[2],
        |g| g.layers[0].b_f[2],
        |n, v| n.layers[0].b_f[2] = v,
    );
}

#[test]
fn b_g_gradient_matches_finite_difference() {
    let mut net = build_net(13, 8, 4, 8);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_g[0][2]",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[0].b_g[2],
        |g| g.layers[0].b_g[2],
        |n, v| n.layers[0].b_g[2] = v,
    );
}

#[test]
fn b_h_gradient_matches_finite_difference() {
    let mut net = build_net(17, 8, 4, 8);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_h[0][2]",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[0].b_h[2],
        |g| g.layers[0].b_h[2],
        |n, v| n.layers[0].b_h[2] = v,
    );
}

#[test]
fn log_tau_gradient_matches_finite_difference() {
    let mut net = build_net(23, 8, 4, 8);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "log_tau[1][1]",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[1].log_tau[1],
        |g| g.layers[1].log_tau[1],
        |n, v| n.layers[1].log_tau[1] = v,
    );
}

#[test]
fn w_f_recurrent_gradient_matches_finite_difference() {
    // Pick a recurrent column (j >= n_in) so the test exercises the
    // x_prev path through the cell — the trickiest backward route.
    let mut net = build_net(29, 8, 4, 8);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    let n_in = net.layers[0].n_in;
    let n_z = net.layers[0].n_z;
    // Layer 0, neuron i=1, recurrent col j=2 (so absolute col = n_in + 2).
    let idx = 1 * n_z + n_in + 2;
    check_grad(
        "W_f[0][i=1, rec j=2]",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[0].w_f[idx],
        |g| g.layers[0].w_f[idx],
        |n, v| n.layers[0].w_f[idx] = v,
    );
}

fn build_mltc_net(seed: u64) -> MultiLayerLnnSnn {
    // 8-vocab, two h=4 layers, the second has 2 memory neurons (M-LTC).
    let mut rng = StdRng::seed_from_u64(seed);
    MultiLayerLnnSnn::new_with_memory(8, &[4, 4], &[0, 2], 8, 0.0, 1.5, &mut rng)
}

#[test]
fn mltc_b_g_memory_neuron_grad_matches_finite_difference() {
    // Memory neuron (layer 1, i=0): activation is sin(g_pre); deriv is
    // cos(g_pre). This test would fail if backward used 1-g² for it.
    let mut net = build_mltc_net(101);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_g[1][0] (memory neuron)",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[1].b_g[0],
        |g| g.layers[1].b_g[0],
        |n, v| n.layers[1].b_g[0] = v,
    );
}

#[test]
fn mltc_b_h_memory_neuron_grad_matches_finite_difference() {
    let mut net = build_mltc_net(103);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_h[1][1] (memory neuron)",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[1].b_h[1],
        |g| g.layers[1].b_h[1],
        |n, v| n.layers[1].b_h[1] = v,
    );
}

#[test]
fn mltc_b_g_learning_neuron_grad_still_matches() {
    // Sanity that adding memory neurons didn't break the tanh path.
    let mut net = build_mltc_net(107);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "b_g[1][3] (learning neuron in M-LTC layer)",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[1].b_g[3],
        |g| g.layers[1].b_g[3],
        |n, v| n.layers[1].b_g[3] = v,
    );
}

#[test]
fn mltc_log_tau_memory_neuron_grad_matches() {
    let mut net = build_mltc_net(109);
    let inputs: Vec<u32> = (0..12).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();
    check_grad(
        "log_tau[1][0] (memory neuron)",
        &mut net,
        &inputs,
        &targets,
        |n| n.layers[1].log_tau[0],
        |g| g.layers[1].log_tau[0],
        |n, v| n.layers[1].log_tau[0] = v,
    );
}

#[test]
fn mltc_state_remains_bounded() {
    let net = build_mltc_net(113);
    let tokens: Vec<u32> = (0..32).map(|t| (t % 8) as u32).collect();
    let cache = net.forward_lm_tokens(&tokens);
    for (l, layer) in cache.layers.iter().enumerate() {
        for (t, row) in layer.x.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                assert!(
                    v.abs() <= 1.0 + 1e-5,
                    "layer {l} t {t} neuron {i}: x = {v} outside [-1, 1]"
                );
            }
        }
    }
}

#[test]
fn mltc_log_tau_memory_floor_holds_after_adam() {
    // Even with high lr pushing log_tau down, memory neurons must stay
    // above LOG_TAU_MEM_MIN = 1.0.
    let mut net = build_mltc_net(127);
    let inputs_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b) % 8) as u32).collect())
        .collect();
    let targets_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b + 1) % 8) as u32).collect())
        .collect();
    let inputs: Vec<&[u32]> = inputs_owned.iter().map(Vec::as_slice).collect();
    let targets: Vec<&[u32]> = targets_owned.iter().map(Vec::as_slice).collect();
    let mut states: Vec<Vec<LnnState>> = (0..4).map(|_| net.fresh_states()).collect();
    let adam = AdamConfig {
        lr: 0.1,
        ..AdamConfig::default()
    };
    for _ in 0..40 {
        net.zero_grad();
        let _ = net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
    }
    let mem_layer = &net.layers[1];
    for i in 0..mem_layer.n_memory {
        assert!(
            mem_layer.log_tau[i] >= 1.0 - 1e-5,
            "memory neuron {i}: log_tau = {} dropped below floor",
            mem_layer.log_tau[i]
        );
    }
}
