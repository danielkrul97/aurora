//! Pre→post pairing must monotonically grow synaptic weight (potentiation).
//! Post→pre pairing must monotonically shrink it (depression).

use aurora::stdp::StdpParams;
use aurora::synapse::Synapse;
use aurora::types::NeuronId;

#[test]
fn pre_then_post_pairing_potentiates_monotonically() {
    let params = StdpParams::default();
    let mut syn = Synapse::new(NeuronId(0), NeuronId(1), None, 0.5, 0.0);
    let initial = syn.weight;

    let pair_dt = 0.005;
    let pair_period = 0.100;
    let n_pairs = 50;

    let mut last = syn.weight;
    let mut t = 0.0;
    for _ in 0..n_pairs {
        syn.weight += syn.on_pre_spike(t, &params);
        syn.weight += syn.on_post_spike(t + pair_dt, &params);
        assert!(
            syn.weight >= last,
            "weight must grow monotonically: {last} → {}",
            syn.weight
        );
        last = syn.weight;
        t += pair_period;
    }

    assert!(
        syn.weight > initial,
        "weight should grow after {n_pairs} pre→post pairs: {initial} → {}",
        syn.weight
    );
}

#[test]
fn post_then_pre_pairing_depresses_monotonically() {
    let params = StdpParams::default();
    let mut syn = Synapse::new(NeuronId(0), NeuronId(1), None, 0.5, 0.0);
    let initial = syn.weight;

    let pair_dt = 0.005;
    let pair_period = 0.100;
    let n_pairs = 50;

    let mut last = syn.weight;
    let mut t = 0.0;
    for _ in 0..n_pairs {
        syn.weight += syn.on_post_spike(t, &params);
        syn.weight += syn.on_pre_spike(t + pair_dt, &params);
        assert!(
            syn.weight <= last,
            "weight must shrink monotonically: {last} → {}",
            syn.weight
        );
        last = syn.weight;
        t += pair_period;
    }

    assert!(
        syn.weight < initial,
        "weight should shrink after {n_pairs} post→pre pairs: {initial} → {}",
        syn.weight
    );
}
