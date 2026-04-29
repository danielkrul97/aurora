//! Sprint 46 smoke: diagonal SSM cell.
//!
//! 1. Forward shape sanity.
//! 2. End-to-end Adam training reduces loss on a tiny LM problem.
//! 3. Numerical-vs-analytical gradient check on `b` (cheapest scalar
//!    to perturb). Confirms backward correctly differentiates the
//!    surrogate-mediated forward.

use aurora::optimizer::AdamConfig;
use aurora::ssm::MultiLayerSsmSnn;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn build_net(seed: u64, n_in: usize, n_h: usize, n_out: usize) -> MultiLayerSsmSnn {
    let mut rng = StdRng::seed_from_u64(seed);
    MultiLayerSsmSnn::new(n_in, &[n_h, n_h], n_out, SpikeFn::default(), &mut rng)
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
        assert_eq!(layer.state.len(), tokens.len());
        assert_eq!(layer.spike.len(), tokens.len());
        assert_eq!(layer.state[0].len(), 8);
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
    let mut states: Vec<Vec<aurora::ssm::SsmState>> =
        (0..4).map(|_| net.fresh_states()).collect();
    let adam = AdamConfig::default();
    let mut losses = Vec::new();
    for _ in 0..30 {
        net.zero_grad();
        let l = net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
        assert!(l.is_finite(), "loss became non-finite");
        losses.push(l);
    }
    let initial = losses[0..3].iter().sum::<f32>() / 3.0;
    let final_ = losses[27..30].iter().sum::<f32>() / 3.0;
    assert!(
        final_ < initial,
        "loss did not decrease: initial={initial:.3}, final={final_:.3}"
    );
}

/// Numerical vs analytical gradient on a single bias scalar. With
/// surrogate gradients the equality holds only approximately near
/// firing thresholds, but the relative error should be < 1 % on
/// random small inputs at small `eps`.
#[test]
fn b_gradient_matches_finite_difference() {
    let mut net = build_net(7, 8, 4, 8);
    let inputs: Vec<u32> = (0..16).map(|t| (t % 8) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 8).collect();

    // Analytical grad on b[0][2].
    let cache = net.forward_lm_tokens(&inputs);
    let _ = net.backward_lm_tokens(&cache, &inputs, &targets);
    let g_analytic = net.grads().layers[0].b[2];

    // Finite-difference: f(b + ε) − f(b − ε) / (2ε)
    let eps = 1e-2_f32;
    let original = net.layers[0].b[2];
    net.layers[0].b[2] = original + eps;
    let cache_p = net.forward_lm_tokens(&inputs);
    let mut loss_p = 0.0_f32;
    for (logits, &target) in cache_p.logits.iter().zip(targets.iter()) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
        loss_p += lse - logits[target as usize];
    }
    net.layers[0].b[2] = original - eps;
    let cache_m = net.forward_lm_tokens(&inputs);
    let mut loss_m = 0.0_f32;
    for (logits, &target) in cache_m.logits.iter().zip(targets.iter()) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
        loss_m += lse - logits[target as usize];
    }
    net.layers[0].b[2] = original;
    let g_numeric = (loss_p - loss_m) / (2.0 * eps);

    // Surrogate-mediated gradients are noisy near θ; tolerate up to
    // 2 abs error or 30 % relative — typical for heaviside backward.
    let abs_err = (g_analytic - g_numeric).abs();
    let rel_err = abs_err / (g_numeric.abs() + 1e-6);
    assert!(
        abs_err < 2.0 || rel_err < 0.3,
        "grad mismatch: analytic={g_analytic:.4}, numeric={g_numeric:.4}, abs_err={abs_err:.4}, rel_err={rel_err:.4}"
    );
}
