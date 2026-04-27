//! Eligibility trace mechanics: it must (1) decay exponentially with `τ_e`
//! and (2) survive long enough that a reward delivered ~τ_e/2 later still
//! drives a measurable weight change.

use aurora::eligibility::EligibilityParams;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::neuromodulator::{Modulator, ModulatorParams};
use aurora::types::NeuronId;

const DT: f64 = 0.0001;
const TAU_E: f64 = 0.500;

fn build_two_neuron_net() -> Network {
    let mut net = Network::new();
    net.eligibility_params = Some(EligibilityParams {
        tau: TAU_E,
        learning_rate: 1.0,
    });
    net.modulator = Some(Modulator::new(ModulatorParams {
        decay_tau: 0.200,
        baseline: 0.0,
    }));
    let n0 = net.add_neuron(LifParams::default());
    let n1 = net.add_neuron(LifParams::default());
    net.add_synapse(n0, n1, None, 0.001, 0.001);
    net
}

fn force_pre_post_pair(net: &mut Network, pre_t: f64, post_t: f64) {
    let amp = 0.030;
    net.schedule_input(pre_t, NeuronId(0), None, amp);
    net.schedule_input(post_t, NeuronId(1), None, amp);
}

fn run_for(net: &mut Network, duration: f64) {
    let n_steps = (duration / DT).ceil() as usize;
    for _ in 0..n_steps {
        net.step(DT);
    }
}

#[test]
fn eligibility_decays_exponentially() {
    let mut net = build_two_neuron_net();
    force_pre_post_pair(&mut net, 0.001, 0.005);
    run_for(&mut net, 0.010);

    net.flush_eligibility_to_now();
    let e_initial = net.synapses[0].eligibility;
    assert!(
        e_initial > 0.0,
        "eligibility should be positive after pre→post pair, got {e_initial}"
    );

    let t_initial = net.now;
    run_for(&mut net, TAU_E);
    net.flush_eligibility_to_now();
    let e_after = net.synapses[0].eligibility;

    let elapsed = net.now - t_initial;
    let expected_ratio = (-elapsed / TAU_E).exp();
    let actual_ratio = e_after / e_initial;
    let err = (actual_ratio - expected_ratio).abs();
    assert!(
        err < 0.02,
        "eligibility ratio after {elapsed}s should be ≈ {expected_ratio} (1/e), got {actual_ratio} (err={err})"
    );
}

#[test]
fn late_reward_still_drives_weight_change() {
    let mut net_with = build_two_neuron_net();
    let mut net_without = build_two_neuron_net();

    force_pre_post_pair(&mut net_with, 0.001, 0.005);
    force_pre_post_pair(&mut net_without, 0.001, 0.005);

    let initial_w = net_with.synapses[0].weight;

    run_for(&mut net_with, 0.260);
    run_for(&mut net_without, 0.260);

    net_with.reward(1.0);

    run_for(&mut net_with, 0.300);
    run_for(&mut net_without, 0.300);

    let with_w = net_with.synapses[0].weight;
    let without_w = net_without.synapses[0].weight;

    assert!(
        with_w > initial_w + 1e-5,
        "late reward (~τ_e/2 after pair) should grow weight: {initial_w} → {with_w}"
    );
    assert!(
        (without_w - initial_w).abs() < 1e-9,
        "without reward weight must not drift: {initial_w} → {without_w}"
    );
    assert!(
        with_w > without_w + 1e-4,
        "rewarded weight should clearly exceed unrewarded: with={with_w}, without={without_w}"
    );
}
