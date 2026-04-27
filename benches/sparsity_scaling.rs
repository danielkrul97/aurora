//! Throughput at varying input rates. Discrete cost is `≈ N × T/dt`
//! independent of activity; event cost scales with delivered events,
//! so the gap widens as activity drops.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

use aurora::engine::EventNetwork;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::stdp::StdpParams;
use aurora::types::NeuronId;

const N_NEURONS: usize = 10_000;
const FAN_OUT: usize = 20;
const SYN_WEIGHT: f64 = 0.0006;
const SYN_DELAY: f64 = 0.002;
const T_END: f64 = 0.05;
const DT: f64 = 0.0001;
const PULSE_AMP: f64 = 0.025;
const SEED: u64 = 0xA1FB_2024;

const NO_STDP: StdpParams = StdpParams {
    a_plus: 0.0,
    a_minus: 0.0,
    tau_plus: 0.020,
    tau_minus: 0.020,
};

fn make_topology(rng: &mut StdRng) -> Vec<(usize, usize)> {
    let mut conns = Vec::with_capacity(N_NEURONS * FAN_OUT);
    for src in 0..N_NEURONS {
        for _ in 0..FAN_OUT {
            let dst = rng.random_range(0..N_NEURONS);
            conns.push((src, dst));
        }
    }
    conns
}

fn make_pulses(rng: &mut StdRng, target_rate_hz: f64) -> Vec<(f64, usize)> {
    let mean_isi = 1.0 / target_rate_hz;
    let mut out = Vec::new();
    for i in 0..N_NEURONS {
        let mut t = -mean_isi * rng.random::<f64>().ln();
        while t < T_END {
            out.push((t, i));
            t += -mean_isi * rng.random::<f64>().ln();
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

fn build_discrete(conns: &[(usize, usize)], pulses: &[(f64, usize)]) -> Network {
    let mut net = Network::new();
    net.stdp_params = NO_STDP;
    let neurons: Vec<NeuronId> = (0..N_NEURONS)
        .map(|_| net.add_neuron(LifParams::default()))
        .collect();
    for &(s, d) in conns {
        net.add_synapse(neurons[s], neurons[d], None, SYN_WEIGHT, SYN_DELAY);
    }
    for &(t, i) in pulses {
        net.schedule_input(t, neurons[i], None, PULSE_AMP);
    }
    net
}

fn build_event(conns: &[(usize, usize)], pulses: &[(f64, usize)]) -> EventNetwork {
    let mut net = EventNetwork::new();
    net.stdp_params = NO_STDP;
    let neurons: Vec<NeuronId> = (0..N_NEURONS)
        .map(|_| net.add_neuron(LifParams::default()))
        .collect();
    for &(s, d) in conns {
        net.add_synapse(neurons[s], neurons[d], None, SYN_WEIGHT, SYN_DELAY);
    }
    for &(t, i) in pulses {
        net.schedule_input(t, neurons[i], None, PULSE_AMP);
    }
    net
}

fn run_discrete(net: &mut Network) -> usize {
    let n_steps = (T_END / DT).ceil() as usize;
    let mut total = 0usize;
    for _ in 0..n_steps {
        total += net.step(DT).len();
    }
    total
}

fn run_event(net: &mut EventNetwork) -> usize {
    net.run_until(T_END).len()
}

fn bench_engines(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));

    let mut rng = StdRng::seed_from_u64(SEED);
    let conns = make_topology(&mut rng);

    for &rate in &[1.0_f64, 5.0, 10.0, 50.0] {
        let mut prng = StdRng::seed_from_u64(SEED.wrapping_add(rate.to_bits()));
        let pulses = make_pulses(&mut prng, rate);

        group.bench_with_input(BenchmarkId::new("discrete", rate as u64), &rate, |b, _| {
            b.iter_batched(
                || build_discrete(&conns, &pulses),
                |mut net| run_discrete(&mut net),
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("event", rate as u64), &rate, |b, _| {
            b.iter_batched(
                || build_event(&conns, &pulses),
                |mut net| run_event(&mut net),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_engines);
criterion_main!(benches);
