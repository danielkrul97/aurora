use aurora::homeostasis::{HomeoParams, Homeostasis};
use aurora::lif::LifParams;
use aurora::neuron::Neuron;
use aurora::types::NeuronId;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn population_converges_to_target_rate() {
    let n_neurons = 100;
    let target_rate = 3.0;
    let dt = 0.001;
    let warmup = 5.0;
    let measure = 5.0;
    let warmup_steps = (warmup / dt) as usize;
    let measure_steps = (measure / dt) as usize;

    let lif = LifParams::default();
    let homeo_params = HomeoParams {
        tau: 1.0,
        target_rate,
        eta: 0.05,
    };

    let mut neurons: Vec<Neuron> = (0..n_neurons)
        .map(|i| {
            let mut n = Neuron::new(NeuronId(i), lif);
            n.enable_homeostasis(Homeostasis::new(homeo_params));
            n
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(42);
    let drives: Vec<f64> = (0..n_neurons)
        .map(|_| 0.020 + rng.random::<f64>() * 0.030)
        .collect();

    let mut now = 0.0;
    for _ in 0..warmup_steps {
        now += dt;
        for (n, &drive) in neurons.iter_mut().zip(drives.iter()) {
            let _ = n.step(now, dt, drive);
        }
    }

    let mut spike_counts = vec![0_usize; n_neurons];
    for _ in 0..measure_steps {
        now += dt;
        for (i, (n, &drive)) in neurons.iter_mut().zip(drives.iter()).enumerate() {
            if n.step(now, dt, drive).is_some() {
                spike_counts[i] += 1;
            }
        }
    }

    let mean_rate: f64 =
        spike_counts.iter().map(|&c| c as f64).sum::<f64>() / f64::from(n_neurons as u32) / measure;

    assert!(
        (1.5..=5.5).contains(&mean_rate),
        "mean firing rate {mean_rate:.2} Hz outside [1.5, 5.5] Hz target band \
         (target {target_rate} Hz)"
    );
}
