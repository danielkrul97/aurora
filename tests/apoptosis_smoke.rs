//! Sprint 42 smoke: activity-dependent neuron pruning.
//!
//! Verifies the masking semantics required by the experiment:
//! 1. dead neurons emit zero spike in forward;
//! 2. dead neurons receive zero gradient in backward (no update to
//!    `b_h`, weight rows, or alpha);
//! 3. forward+backward+adam runs end-to-end with apoptosis on, no
//!    NaN, no panics, loss stays finite.

use aurora::apoptosis::{ApoptosisConfig, ApoptosisMode, ApoptosisState};
use aurora::optimizer::AdamConfig;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn build_net(seed: u64) -> MultiLayerGatedSnn {
    let mut rng = StdRng::seed_from_u64(seed);
    MultiLayerGatedSnn::new(
        16,
        &[8, 8],
        16,
        SpikeFn::default(),
        &mut rng,
    )
}

fn manually_kill(net: &mut MultiLayerGatedSnn, layer: usize, neuron: usize) {
    let cfg = ApoptosisConfig {
        mode: ApoptosisMode::Off,
        ..Default::default()
    };
    if net.apoptosis.is_none() {
        let layer_sizes: Vec<usize> = net.layers.iter().map(|l| l.n_hidden).collect();
        net.apoptosis = Some(ApoptosisState::new(cfg, &layer_sizes));
    }
    let apop = net.apoptosis.as_mut().unwrap();
    apop.dead_mask[layer][neuron] = true;
}

#[test]
fn dead_neuron_produces_zero_spike() {
    let mut net = build_net(42);
    manually_kill(&mut net, 0, 3);
    manually_kill(&mut net, 0, 5);
    manually_kill(&mut net, 1, 1);

    let tokens: Vec<u32> = (0..32).map(|t| (t % 16) as u32).collect();
    let cache = net.forward_lm_tokens(&tokens);

    for row in &cache.layers[0].h_spike {
        assert_eq!(row[3], 0.0, "dead neuron 0/3 spiked");
        assert_eq!(row[5], 0.0, "dead neuron 0/5 spiked");
    }
    for row in &cache.layers[1].h_spike {
        assert_eq!(row[1], 0.0, "dead neuron 1/1 spiked");
    }
}

#[test]
fn dead_neuron_receives_zero_gradient() {
    let mut net = build_net(7);
    manually_kill(&mut net, 0, 2);
    manually_kill(&mut net, 1, 4);

    let inputs: Vec<u32> = (0..32).map(|t| (t % 16) as u32).collect();
    let targets: Vec<u32> = inputs.iter().map(|&t| (t + 1) % 16).collect();

    let cache = net.forward_lm_tokens(&inputs);
    let _loss = net.backward_lm_tokens(&cache, &inputs, &targets);

    let g0 = &net.grads().layers[0];
    let n_in_0 = net.layers[0].n_in;
    let n_h_0 = net.layers[0].n_hidden;
    assert_eq!(g0.b_h[2], 0.0, "b_h grad for dead 0/2 nonzero");
    for c in 0..n_in_0 {
        assert_eq!(g0.w_in[2 * n_in_0 + c], 0.0, "w_in row of dead 0/2 nonzero");
    }
    for j in 0..n_h_0 {
        assert_eq!(g0.w_rec[2 * n_h_0 + j], 0.0, "w_rec row of dead 0/2 nonzero");
    }

    let g1 = &net.grads().layers[1];
    let n_in_1 = net.layers[1].n_in;
    let n_h_1 = net.layers[1].n_hidden;
    assert_eq!(g1.b_h[4], 0.0, "b_h grad for dead 1/4 nonzero");
    for c in 0..n_in_1 {
        assert_eq!(g1.w_in[4 * n_in_1 + c], 0.0, "w_in row of dead 1/4 nonzero");
    }
    for j in 0..n_h_1 {
        assert_eq!(g1.w_rec[4 * n_h_1 + j], 0.0, "w_rec row of dead 1/4 nonzero");
    }

    // w_out column for top-layer dead neuron 1/4 must also stay 0
    // (top-layer dead → never in active_top_t → no update).
    for o in 0..net.n_output {
        assert_eq!(net.grads().w_out[o * n_h_1 + 4], 0.0, "w_out col of dead 1/4 nonzero");
    }
}

#[test]
fn end_to_end_train_stays_finite_with_kills() {
    let mut net = build_net(0xABCDEF);
    let cfg = ApoptosisConfig {
        mode: ApoptosisMode::Activity,
        warmup_chunks: 2,
        ema_alpha: 0.5,
        kill_threshold: 0.05, // aggressive — likely kills several
        check_interval: 1,
    };
    net.enable_apoptosis(cfg);

    let inputs_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b) % 16) as u32).collect())
        .collect();
    let targets_owned: Vec<Vec<u32>> = (0..4)
        .map(|b| (0..16).map(|t| ((t + b + 1) % 16) as u32).collect())
        .collect();
    let inputs: Vec<&[u32]> = inputs_owned.iter().map(Vec::as_slice).collect();
    let targets: Vec<&[u32]> = targets_owned.iter().map(Vec::as_slice).collect();
    let mut states: Vec<Vec<aurora::snn_gated::GatedState>> =
        (0..4).map(|_| net.fresh_states()).collect();

    let adam = AdamConfig::default();
    let mut rng = StdRng::seed_from_u64(99);

    for _ in 0..10 {
        net.zero_grad();
        let (loss, rates) = net.forward_backward_batch_tokens_stateful_with_rates(
            &inputs, &targets, &mut states,
        );
        assert!(loss.is_finite(), "loss not finite");
        net.clip_grads(1.0);
        net.step_adam(&adam);
        let apop = net.apoptosis.as_mut().unwrap();
        apop.observe_rates(&rates);
        apop.tick();
        let _ = apop.maybe_kill(&mut rng);
    }

    let total_dead: usize = net
        .apoptosis
        .as_ref()
        .unwrap()
        .dead_mask
        .iter()
        .map(|m| m.iter().filter(|d| **d).count())
        .sum();
    // With aggressive threshold + tiny network, we should see at least
    // one kill across 10 steps × 16 neurons × 2 layers.
    assert!(total_dead > 0, "expected at least one apoptosis kill, got 0");
    // But not everyone should die or the network couldn't have learned.
    assert!(total_dead < 16, "too many killed ({total_dead}) — bad config?");
}
