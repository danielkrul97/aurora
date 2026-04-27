//! Hand-wired XOR using two dendritic compartments with signed synapses.
//!
//! Architecture:
//!   in0, in1 → hidden (2 dendrites: A = +in0 −in1, B = −in0 +in1) → out
//!
//! Truth table:
//!   (0,0): both dendrites silent           → out silent ✓
//!   (1,0): dendrite A's net charge > θ_d   → out fires  ✓
//!   (0,1): dendrite B's net charge > θ_d   → out fires  ✓
//!   (1,1): both dendrites cancel to ~0     → out silent ✓

use aurora::dendrite::DendriteParams;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::stdp::StdpParams;
use aurora::types::{NeuronId, Spike};

fn build_xor_network() -> (Network, NeuronId, NeuronId, NeuronId) {
    let lif = LifParams::default();
    let dend = DendriteParams {
        tau: 0.010,
        v_rest: 0.0,
        v_reset: 0.0,
        threshold: 0.015,
        boost: 0.025,
    };

    let mut net = Network::new();
    net.stdp_params = StdpParams {
        a_plus: 0.0,
        a_minus: 0.0,
        tau_plus: 0.020,
        tau_minus: 0.020,
    };

    let in0 = net.add_neuron(lif);
    let in1 = net.add_neuron(lif);

    let hidden = net.add_neuron(lif);
    let dend_a = net.add_dendrite(hidden, dend);
    let dend_b = net.add_dendrite(hidden, dend);
    debug_assert_eq!((dend_a, dend_b), (0, 1));

    let out = net.add_neuron(lif);

    let exc = 0.020;
    let inh = -0.020;
    let delay = 0.001;

    net.add_synapse(in0, hidden, Some(dend_a), exc, delay);
    net.add_synapse(in1, hidden, Some(dend_a), inh, delay);
    net.add_synapse(in0, hidden, Some(dend_b), inh, delay);
    net.add_synapse(in1, hidden, Some(dend_b), exc, delay);

    net.add_synapse(hidden, out, None, 0.025, delay);

    (net, in0, in1, out)
}

fn run_pattern(
    net: &mut Network,
    in0: NeuronId,
    in1: NeuronId,
    out: NeuronId,
    pattern: (bool, bool),
    duration_s: f64,
    dt: f64,
) -> usize {
    let drive_on = 0.030;
    net.set_external_input(in0, if pattern.0 { drive_on } else { 0.0 });
    net.set_external_input(in1, if pattern.1 { drive_on } else { 0.0 });

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_steps = (duration_s / dt) as usize;
    let mut out_spikes = 0_usize;
    for _ in 0..n_steps {
        let spikes: Vec<Spike> = net.step(dt);
        out_spikes += spikes.iter().filter(|s| s.source == out).count();
    }
    out_spikes
}

fn count_output_spikes(pattern: (bool, bool)) -> usize {
    let (mut net, in0, in1, out) = build_xor_network();
    let dt = 0.0001;
    let _ = run_pattern(&mut net, in0, in1, out, (false, false), 0.050, dt);
    run_pattern(&mut net, in0, in1, out, pattern, 0.200, dt)
}

#[test]
fn handwired_dendritic_xor() {
    let n00 = count_output_spikes((false, false));
    let n10 = count_output_spikes((true, false));
    let n01 = count_output_spikes((false, true));
    let n11 = count_output_spikes((true, true));

    assert_eq!(n00, 0, "(0,0) must produce no output spikes, got {n00}");
    assert_eq!(n11, 0, "(1,1) must produce no output spikes, got {n11}");
    assert!(n10 > 0, "(1,0) must produce output spikes, got {n10}");
    assert!(n01 > 0, "(0,1) must produce output spikes, got {n01}");
}
