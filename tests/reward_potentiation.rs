//! Three-factor STDP: only the modulator gates persistent weight change.
//! Repeated pre→post pairs with reward must grow weight; the same pairs
//! without reward must leave weight untouched.

use aurora::eligibility::EligibilityParams;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::neuromodulator::{Modulator, ModulatorParams};
use aurora::types::NeuronId;

const DT: f64 = 0.0001;
const PAIR_PERIOD: f64 = 0.100;
const N_PAIRS: usize = 30;
const PAIR_DT: f64 = 0.004;

fn build() -> Network {
    let mut net = Network::new();
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.500,
        learning_rate: 1.0,
    });
    net.modulator = Some(Modulator::new(ModulatorParams {
        decay_tau: 0.150,
        baseline: 0.0,
    }));
    let n0 = net.add_neuron(LifParams::default());
    let n1 = net.add_neuron(LifParams::default());
    net.add_synapse(n0, n1, None, 0.001, 0.001);
    net
}

fn run_until(net: &mut Network, t_target: f64) {
    while net.now < t_target {
        net.step(DT);
    }
}

fn run_paired(reward_amount: f64) -> f64 {
    let mut net = build();
    let amp = 0.030;
    for i in 0..N_PAIRS {
        let t_pre = (i as f64) * PAIR_PERIOD + 0.001;
        let t_post = t_pre + PAIR_DT;
        net.schedule_input(t_pre, NeuronId(0), None, amp);
        net.schedule_input(t_post, NeuronId(1), None, amp);
    }
    for i in 0..N_PAIRS {
        let reward_t = (i as f64) * PAIR_PERIOD + 0.020;
        run_until(&mut net, reward_t);
        if reward_amount != 0.0 {
            net.reward(reward_amount);
        }
    }
    run_until(&mut net, (N_PAIRS as f64) * PAIR_PERIOD + 0.500);
    net.synapses[0].weight
}

#[test]
fn reward_drives_potentiation() {
    let with_reward = run_paired(1.0);
    let without_reward = run_paired(0.0);
    let initial = 0.001;

    assert!(
        (without_reward - initial).abs() < 1e-9,
        "without reward, weight must not drift: {initial} → {without_reward}"
    );
    assert!(
        with_reward > initial + 0.001,
        "with reward, weight should grow significantly: {initial} → {with_reward}"
    );
    assert!(
        with_reward > without_reward + 0.001,
        "rewarded > unrewarded by clear margin: with={with_reward}, without={without_reward}"
    );
}

#[test]
fn negative_reward_drives_depression() {
    let depressed = run_paired(-1.0);
    let initial = 0.001;
    assert!(
        depressed < initial - 0.001,
        "negative reward should shrink weight: {initial} → {depressed}"
    );
}
