//! Train an SNN on (a subset of) MNIST with three-factor R-STDP.
//!
//! Usage:
//!   cargo run --release --example mnist_train -- [n_train] [n_epochs]
//! Expects MNIST IDX files in `data/mnist/` (run `scripts/get_mnist.sh`).
//!
//! Saves trained weights to `data/aurora_mnist_weights.bin` for the eval
//! example to load.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use aurora::eligibility::EligibilityParams;
use aurora::encoding::encode_image_rate;
use aurora::mnist;
use aurora::network::Network;
use aurora::neuromodulator::{Modulator, ModulatorParams};
use aurora::stdp::StdpParams;
use aurora::topology::{add_lateral_inhibition, build_feedforward};
use aurora::types::NeuronId;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const N_INPUTS: usize = mnist::IMAGE_PIXELS;
const N_HIDDEN: usize = 100;
const N_CLASSES: usize = 10;
const DT: f64 = 0.001;
const PRESENT: f64 = 0.030;
const WAIT: f64 = 0.300;
const MAX_RATE_HZ: f64 = 80.0;
const PULSE_AMP: f64 = 0.025;
const TEACHER_PULSE: f64 = 0.040;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let n_train: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let n_epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let train_images = match mnist::load_images("data/mnist/train-images-idx3-ubyte") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read training images: {e}");
            eprintln!("run `bash scripts/get_mnist.sh` to fetch MNIST");
            return ExitCode::FAILURE;
        }
    };
    let train_labels = match mnist::load_labels("data/mnist/train-labels-idx1-ubyte") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read training labels: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_train = n_train.min(train_images.len());
    println!(
        "loaded {} train images, using {n_train}",
        train_images.len()
    );

    let mut rng = StdRng::seed_from_u64(0xA1FB);
    let (mut net, layers) = build_feedforward(
        &[N_INPUTS, N_HIDDEN, N_CLASSES],
        (0.001, 0.004),
        0.001,
        &mut rng,
    );
    add_lateral_inhibition(&mut net, &layers[1], 0.040, 0.001);
    add_lateral_inhibition(&mut net, &layers[2], 0.150, 0.001);
    net.stdp_params = StdpParams {
        a_plus: 0.01,
        a_minus: 0.0,
        tau_plus: 0.020,
        tau_minus: 0.020,
    };
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.200,
        learning_rate: 0.3,
    });
    net.modulator = Some(Modulator::new(ModulatorParams {
        decay_tau: 0.100,
        baseline: 0.0,
    }));
    net.weight_clamp = Some((0.0, 0.020));

    let inputs = layers[0].clone();
    let outputs = layers[2].clone();

    let mut indices: Vec<usize> = (0..n_train).collect();
    let start = Instant::now();
    for epoch in 0..n_epochs {
        indices.shuffle(&mut rng);
        for (k, &i) in indices.iter().enumerate() {
            let img = &train_images[i];
            let label = train_labels[i] as usize;
            present_sample(&mut net, img, &inputs, Some((&outputs, label)), &mut rng);
            net.reward(1.0);
            settle(&mut net);
            if k % 100 == 0 {
                println!(
                    "epoch {epoch} sample {k}/{n_train} (elapsed {:.1}s)",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
    println!("training done in {:.1}s", start.elapsed().as_secs_f64());

    if let Err(e) = save_weights(&net, "data/aurora_mnist_weights.bin") {
        eprintln!("failed to save weights: {e}");
        return ExitCode::FAILURE;
    }
    println!("weights saved to data/aurora_mnist_weights.bin");
    ExitCode::SUCCESS
}

fn present_sample(
    net: &mut Network,
    image: &[f64],
    inputs: &[NeuronId],
    teacher: Option<(&[NeuronId], usize)>,
    rng: &mut impl rand::Rng,
) {
    let t0 = net.now + 0.001;
    let encoded = encode_image_rate(image, PRESENT, MAX_RATE_HZ, rng);
    for (i, times) in encoded.iter().enumerate() {
        for &t in times {
            net.schedule_input(t0 + t, inputs[i], None, PULSE_AMP);
        }
    }
    if let Some((outputs, target)) = teacher {
        let n_inhib = 6;
        for k in 0..n_inhib {
            #[allow(clippy::cast_precision_loss)]
            let t_inh = t0 + (k as f64 + 0.5) * (PRESENT / n_inhib as f64);
            for (j, &out) in outputs.iter().enumerate() {
                if j != target {
                    net.schedule_input(t_inh, out, None, -0.030);
                }
            }
        }
        net.schedule_input(t0 + PRESENT - 0.003, outputs[target], None, TEACHER_PULSE);
    }
    let n_steps = (PRESENT / DT).ceil() as usize;
    for _ in 0..n_steps {
        net.step(DT);
    }
}

fn settle(net: &mut Network) {
    let n = (WAIT / DT).ceil() as usize;
    for _ in 0..n {
        net.step(DT);
    }
}

fn save_weights<P: AsRef<Path>>(net: &Network, path: P) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let count = net.synapses.len() as u32;
    w.write_all(&count.to_le_bytes())?;
    for s in &net.synapses {
        w.write_all(&s.weight.to_le_bytes())?;
    }
    Ok(())
}
