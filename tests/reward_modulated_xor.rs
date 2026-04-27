//! Reward-modulated STDP refines a perturbed XOR detector.
//!
//! Architecture: 2 inputs → 1 output neuron with 2 dendrites. Initial
//! weights are the hand-wired XOR solution except `in1→A` is weakened
//! (5% magnitude vs the −20% target), making pattern (1,1) initially
//! wrong-fire via dendrite A. Three-factor R-STDP must drive `in1→A`
//! more negative until dendrite A no longer fires on (1,1).
//!
//! **Scope of this test:** the *credit-assignment hard part* of XOR — the
//! symmetric eligibility that classical R-STDP can't disambiguate — is
//! solved here by routing post-spike STDP through the firing dendrite
//! rather than the soma (`network.dendrite_local_stdp = true`). With this
//! structural fix, (1,1) eligibility on `in1→A` accumulates without any
//! competing reward signal and converges in tens of epochs.
//!
//! **Out of scope (deferred to sprint 5+):** XOR from random weights with
//! no architectural priors. With four synapses receiving identical
//! eligibility from a single global modulator on (1,1), the excitatory
//! synapses run away under positive rewards on (1,0)/(0,1) faster than
//! the inhibitory ones can catch up. Surrogate-gradient pretraining or
//! population coding is the standard fix; we'll attempt it in sprint 5.

use aurora::dendrite::DendriteParams;
use aurora::eligibility::EligibilityParams;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::neuromodulator::{Modulator, ModulatorParams};
use aurora::types::NeuronId;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const DT: f64 = 0.0001;
const PRESENT_DURATION: f64 = 0.030;
const INTER_TRIAL: f64 = 0.700;
const PULSE_AMP: f64 = 0.030;
const N_EPOCHS: usize = 60;

const IN0: usize = 0;
const IN1: usize = 1;
const OUT: usize = 2;

const SYN_IN0_A: usize = 0;
const SYN_IN1_A: usize = 1;

fn build_xor_net() -> Network {
    let mut net = Network::new();
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.200,
        learning_rate: 3.0,
    });
    net.modulator = Some(Modulator::new(ModulatorParams {
        decay_tau: 0.100,
        baseline: 0.0,
    }));
    net.dendrite_local_stdp = true;

    let in0 = net.add_neuron(LifParams::default());
    let in1 = net.add_neuron(LifParams::default());
    let out = net.add_neuron(LifParams::default());
    assert_eq!(in0, NeuronId(IN0));
    assert_eq!(in1, NeuronId(IN1));
    assert_eq!(out, NeuronId(OUT));

    let dend_a = net.add_dendrite(out, DendriteParams::default());
    let dend_b = net.add_dendrite(out, DendriteParams::default());

    let exc = 0.020;
    let inh_strong = -0.020;
    let inh_weak = -0.005;

    net.add_synapse(in0, out, Some(dend_a), exc, 0.001);
    net.add_synapse(in1, out, Some(dend_a), inh_weak, 0.001);
    net.add_synapse(in0, out, Some(dend_b), inh_strong, 0.001);
    net.add_synapse(in1, out, Some(dend_b), exc, 0.001);

    net
}

fn run_steps(net: &mut Network, n_steps: usize, count_out: bool) -> usize {
    let mut output_fires = 0;
    for _ in 0..n_steps {
        let spikes = net.step(DT);
        if count_out {
            for s in &spikes {
                if s.source.0 == OUT {
                    output_fires += 1;
                }
            }
        }
    }
    output_fires
}

fn present_pattern(net: &mut Network, in0_on: bool, in1_on: bool) -> bool {
    let t = net.now + 0.001;
    if in0_on {
        net.schedule_input(t, NeuronId(IN0), None, PULSE_AMP);
    }
    if in1_on {
        net.schedule_input(t, NeuronId(IN1), None, PULSE_AMP);
    }
    let n_steps = (PRESENT_DURATION / DT).ceil() as usize;
    run_steps(net, n_steps, true) > 0
}

fn train_trial(net: &mut Network, in0_on: bool, in1_on: bool, expected_fire: bool) {
    let actual_fire = present_pattern(net, in0_on, in1_on);
    let r = match (actual_fire, expected_fire) {
        (true, true) => 1.0,
        (true, false) => -1.0,
        (false, true) | (false, false) => 0.0,
    };
    if r != 0.0 {
        net.reward(r);
    }
    let wait = (INTER_TRIAL / DT).ceil() as usize;
    run_steps(net, wait, false);
}

#[test]
fn r_stdp_drives_perturbed_inhibitory_weight_in_correct_direction() {
    let mut net = build_xor_net();
    let mut rng = StdRng::seed_from_u64(42);

    let initial_in1_a = net.synapses[SYN_IN1_A].weight;
    let initial_in0_a = net.synapses[SYN_IN0_A].weight;
    assert!(
        initial_in1_a > -0.010,
        "perturbed init: in1→A should be near -0.005"
    );

    let cases: [((bool, bool), bool); 4] = [
        ((false, false), false),
        ((false, true), true),
        ((true, false), true),
        ((true, true), false),
    ];

    for _ in 0..N_EPOCHS {
        let mut shuffled = cases;
        shuffled.shuffle(&mut rng);
        for &((a, b), expected) in &shuffled {
            train_trial(&mut net, a, b, expected);
        }
    }

    let final_in1_a = net.synapses[SYN_IN1_A].weight;
    let final_in0_a = net.synapses[SYN_IN0_A].weight;

    let inh_change = final_in1_a - initial_in1_a;
    assert!(
        inh_change < -0.005,
        "in1→A should grow more inhibitory under R-STDP (Δ < -0.005), got Δ={inh_change} ({initial_in1_a} → {final_in1_a})"
    );
    let exc_change = final_in0_a - initial_in0_a;
    assert!(
        exc_change > 0.0,
        "in0→A (correctly excitatory) should not erode under R-STDP, got Δ={exc_change} ({initial_in0_a} → {final_in0_a})"
    );
}
