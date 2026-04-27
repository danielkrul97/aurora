//! Constant-current LIF neuron must fire at the analytical rate within ±5%.

use aurora::lif::{LifParams, Neuron};
use aurora::types::NeuronId;

#[test]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn lif_firing_rate_matches_analytical() {
    let params = LifParams::default();
    let mut neuron = Neuron::new(NeuronId(0), params);

    let input = 0.030;
    let dt = 0.0001;
    let t_total = 2.0;
    let n_steps = (t_total / dt) as usize;

    let mut spike_count = 0_usize;
    for _ in 0..n_steps {
        if neuron.step(dt, input).is_some() {
            spike_count += 1;
        }
    }

    let v_thr_minus_rest = params.v_threshold - params.v_rest;
    let analytical_isi = params.tau_ref - params.tau_m * (1.0 - v_thr_minus_rest / input).ln();
    let analytical_rate = 1.0 / analytical_isi;
    let measured_rate = spike_count as f64 / t_total;

    let rel_error = (measured_rate - analytical_rate).abs() / analytical_rate;
    assert!(
        rel_error < 0.05,
        "firing rate {measured_rate:.3} Hz vs analytical {analytical_rate:.3} Hz \
         (relative error {:.2}%)",
        rel_error * 100.0
    );
}

#[test]
fn lif_below_threshold_never_fires() {
    let params = LifParams::default();
    let mut neuron = Neuron::new(NeuronId(0), params);

    let sub_threshold_input = (params.v_threshold - params.v_rest) * 0.5;
    let dt = 0.0001;
    let n_steps = 10_000;

    for _ in 0..n_steps {
        assert!(neuron.step(dt, sub_threshold_input).is_none());
    }
}
