//! Discrete and event-driven engines must agree on spike trains for the
//! same deterministic stimulus. Discrete rounds events to its `dt` grid,
//! so per-fire-time diffs ≤ a few dt — counts must match exactly.

use aurora::engine::EventNetwork;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::stdp::StdpParams;
use aurora::types::{NeuronId, Spike};

const N_NEURONS: usize = 5;
const PULSE_AMP: f64 = 0.030;
const SYN_WEIGHT: f64 = 0.030;
const SYN_DELAY: f64 = 0.005;
const PULSE_TIMES: &[f64] = &[
    0.000, 0.020, 0.040, 0.060, 0.080, 0.100, 0.120, 0.140, 0.160, 0.180,
];
const T_END: f64 = 0.250;
const DT: f64 = 0.0001;

fn build_discrete() -> (Network, Vec<NeuronId>) {
    let mut net = Network::new();
    net.stdp_params = StdpParams {
        a_plus: 0.0,
        a_minus: 0.0,
        ..Default::default()
    };
    let neurons: Vec<NeuronId> = (0..N_NEURONS)
        .map(|_| net.add_neuron(LifParams::default()))
        .collect();
    for w in neurons.windows(2) {
        net.add_synapse(w[0], w[1], None, SYN_WEIGHT, SYN_DELAY);
    }
    for &t in PULSE_TIMES {
        net.schedule_input(t, neurons[0], None, PULSE_AMP);
    }
    (net, neurons)
}

fn build_event() -> (EventNetwork, Vec<NeuronId>) {
    let mut net = EventNetwork::new();
    net.stdp_params = StdpParams {
        a_plus: 0.0,
        a_minus: 0.0,
        ..Default::default()
    };
    let neurons: Vec<NeuronId> = (0..N_NEURONS)
        .map(|_| net.add_neuron(LifParams::default()))
        .collect();
    for w in neurons.windows(2) {
        net.add_synapse(w[0], w[1], None, SYN_WEIGHT, SYN_DELAY);
    }
    for &t in PULSE_TIMES {
        net.schedule_input(t, neurons[0], None, PULSE_AMP);
    }
    (net, neurons)
}

fn run_discrete(net: &mut Network) -> Vec<Spike> {
    let mut all = Vec::new();
    let n_steps = (T_END / DT).ceil() as usize;
    for _ in 0..n_steps {
        all.extend(net.step(DT));
    }
    all
}

fn bin_spikes(spikes: &[Spike]) -> Vec<Vec<f64>> {
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); N_NEURONS];
    for s in spikes {
        bins[s.source.0].push(s.t);
    }
    for b in &mut bins {
        b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    bins
}

#[test]
fn chain_parity_spike_counts() {
    let (mut d, _) = build_discrete();
    let (mut e, _) = build_event();

    let sd = run_discrete(&mut d);
    let se = e.run_until(T_END);

    let bd = bin_spikes(&sd);
    let be = bin_spikes(&se);

    let counts_d: Vec<usize> = bd.iter().map(Vec::len).collect();
    let counts_e: Vec<usize> = be.iter().map(Vec::len).collect();

    assert_eq!(
        counts_d, counts_e,
        "per-neuron spike counts must match (discrete={counts_d:?}, event={counts_e:?})"
    );
    assert_eq!(
        counts_d,
        vec![PULSE_TIMES.len(); N_NEURONS],
        "each chain neuron should fire once per pulse"
    );
}

#[test]
fn chain_parity_fire_times_close() {
    let (mut d, _) = build_discrete();
    let (mut e, _) = build_event();

    let sd = run_discrete(&mut d);
    let se = e.run_until(T_END);

    let bd = bin_spikes(&sd);
    let be = bin_spikes(&se);

    let tol = 5.0 * DT;
    for (i, (d_times, e_times)) in bd.iter().zip(be.iter()).enumerate() {
        assert_eq!(
            d_times.len(),
            e_times.len(),
            "neuron {i}: spike count mismatch"
        );
        for (k, (d_t, e_t)) in d_times.iter().zip(e_times.iter()).enumerate() {
            let diff = (d_t - e_t).abs();
            assert!(
                diff < tol,
                "neuron {i} spike {k}: discrete t={d_t}, event t={e_t}, diff={diff}, tol={tol}"
            );
        }
    }
}
