//! End-to-end pipeline smoke test using sprint 6 WTA primitives.
//!
//! Exercises encoding → topology → lateral inhibition → weight clamp →
//! three-factor R-STDP → readout. We do NOT assert above-chance
//! accuracy: the WTA hyperparameter regime that produces class-selective
//! features (Diehl & Cook 2015 style) requires careful tuning of
//! inhibition strength, threshold adaptation, and weight init that is
//! beyond the scope of this smoke test. The bar here is mechanism
//! correctness — primitives wire up, network spikes, predictions are
//! returned, plastic weights actually move.

use aurora::eligibility::EligibilityParams;
use aurora::encoding::encode_image_rate;
use aurora::homeostasis::{HomeoParams, Homeostasis};
use aurora::network::Network;
use aurora::neuromodulator::ModulatorParams;
use aurora::readout::predict_class;
use aurora::stdp::StdpParams;
use aurora::topology::{add_lateral_inhibition, build_feedforward};
use aurora::types::{NeuronId, Spike};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SIDE: usize = 4;
const N_INPUTS: usize = SIDE * SIDE;
const N_HIDDEN: usize = 8;
const N_CLASSES: usize = 2;
const DT: f64 = 0.001;
const PRESENT: f64 = 0.030;
const WAIT: f64 = 0.300;
const MAX_RATE_HZ: f64 = 80.0;
const PULSE_AMP: f64 = 0.025;
const TEACHER_PULSE: f64 = 0.040;

fn make_horizontal(rng: &mut impl Rng, noise: f64) -> Vec<f64> {
    let mut img = vec![0.0; N_INPUTS];
    for c in 0..SIDE {
        img[SIDE + c] = 1.0;
        img[2 * SIDE + c] = 1.0;
    }
    add_noise(&mut img, rng, noise);
    img
}

fn make_vertical(rng: &mut impl Rng, noise: f64) -> Vec<f64> {
    let mut img = vec![0.0; N_INPUTS];
    for r in 0..SIDE {
        img[r * SIDE + 1] = 1.0;
        img[r * SIDE + 2] = 1.0;
    }
    add_noise(&mut img, rng, noise);
    img
}

fn add_noise(img: &mut [f64], rng: &mut impl Rng, noise: f64) {
    for v in img.iter_mut() {
        *v = (*v + rng.random::<f64>() * noise).clamp(0.0, 1.0);
    }
}

fn make_sample(class: usize, rng: &mut impl Rng) -> Vec<f64> {
    if class == 0 {
        make_horizontal(rng, 0.15)
    } else {
        make_vertical(rng, 0.15)
    }
}

fn build_wta_net(rng: &mut impl Rng) -> (Network, Vec<NeuronId>, Vec<NeuronId>) {
    let (mut net, layers) =
        build_feedforward(&[N_INPUTS, N_HIDDEN, N_CLASSES], (0.003, 0.008), 0.001, rng);
    add_lateral_inhibition(&mut net, &layers[1], 0.050, 0.001);
    add_lateral_inhibition(&mut net, &layers[2], 0.150, 0.001);
    let homeo_params = HomeoParams {
        tau: 1.0,
        target_rate: 5.0,
        eta: 0.005,
    };
    for &h in &layers[1] {
        net.neurons[h.0].enable_homeostasis(Homeostasis::new(homeo_params));
    }
    let mod_params = ModulatorParams {
        decay_tau: 0.100,
        baseline: 0.0,
    };
    net.modulator = Some(aurora::neuromodulator::Modulator::new(mod_params));
    for &out in &layers[2] {
        net.enable_neuron_modulator(out, mod_params);
    }
    net.stdp_params = StdpParams {
        a_plus: 0.01,
        a_minus: 0.0,
        tau_plus: 0.020,
        tau_minus: 0.020,
    };
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.200,
        learning_rate: 1.0,
    });
    net.weight_clamp = Some((0.0, 0.025));
    let inputs = layers[0].clone();
    let outputs = layers[2].clone();
    (net, inputs, outputs)
}

fn present_sample(
    net: &mut Network,
    image: &[f64],
    inputs: &[NeuronId],
    teacher: Option<(&[NeuronId], usize)>,
    rng: &mut impl Rng,
) -> (Vec<Spike>, f64) {
    let t_start = net.now;
    let t0 = t_start + 0.001;
    let encoded = encode_image_rate(image, PRESENT, MAX_RATE_HZ, rng);
    for (i, times) in encoded.iter().enumerate() {
        for &t in times {
            net.schedule_input(t0 + t, inputs[i], None, PULSE_AMP);
        }
    }
    if let Some((outputs, target_class)) = teacher {
        let n_inhib = 10;
        for k in 0..n_inhib {
            #[allow(clippy::cast_precision_loss)]
            let t_inh = t0 + (k as f64 + 0.5) * (PRESENT / n_inhib as f64);
            for (j, &out) in outputs.iter().enumerate() {
                if j != target_class {
                    net.schedule_input(t_inh, out, None, -0.500);
                }
            }
        }
        net.schedule_input(t0 + 0.005, outputs[target_class], None, TEACHER_PULSE);
    }
    let n_steps = (PRESENT / DT).ceil() as usize;
    let mut spikes = Vec::new();
    for _ in 0..n_steps {
        spikes.extend(net.step(DT));
    }
    (spikes, t_start)
}

fn settle(net: &mut Network) {
    let n = (WAIT / DT).ceil() as usize;
    for _ in 0..n {
        net.step(DT);
    }
}

#[test]
fn wta_pipeline_smoke() {
    let mut rng = StdRng::seed_from_u64(42);
    let (mut net, inputs, outputs) = build_wta_net(&mut rng);

    let initial_in_hid: f64 = net
        .synapses
        .iter()
        .filter(|s| s.plastic && (16..24).contains(&s.post_neuron.0))
        .map(|s| s.weight)
        .sum::<f64>()
        / 128.0;

    let n_train_per_class = 30;
    for _ in 0..n_train_per_class {
        for class in 0..N_CLASSES {
            let img = make_sample(class, &mut rng);
            let _ = present_sample(&mut net, &img, &inputs, Some((&outputs, class)), &mut rng);
            net.reward(1.0);
            net.reward_neuron(outputs[class], 1.0);
            settle(&mut net);
        }
    }

    let final_in_hid: f64 = net
        .synapses
        .iter()
        .filter(|s| s.plastic && (16..24).contains(&s.post_neuron.0))
        .map(|s| s.weight)
        .sum::<f64>()
        / 128.0;
    assert!(
        (final_in_hid - initial_in_hid).abs() > 1e-4,
        "in→hid weights must move under three-factor R-STDP: \
         initial={initial_in_hid:.5}, final={final_in_hid:.5}"
    );

    let n_test = 20;
    let mut predictions = 0;
    for _ in 0..n_test {
        let class = rng.random_range(0..N_CLASSES);
        let img = make_sample(class, &mut rng);
        let (spikes, t_start) = present_sample(&mut net, &img, &inputs, None, &mut rng);
        if predict_class(&spikes, &outputs, t_start, t_start + PRESENT).is_some() {
            predictions += 1;
        }
        settle(&mut net);
    }
    assert!(
        predictions >= n_test / 2,
        "readout must produce predictions on at least half the test set, got {predictions}/{n_test}"
    );

    let weight_clamp_max = 0.025;
    for syn in &net.synapses {
        if syn.plastic {
            assert!(
                syn.weight <= weight_clamp_max + 1e-9,
                "weight clamp violated: {} > {}",
                syn.weight,
                weight_clamp_max
            );
        }
    }
}
