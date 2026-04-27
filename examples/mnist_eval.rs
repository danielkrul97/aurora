//! Evaluate the SNN trained by `mnist_train` on the MNIST test set.
//!
//! Usage:
//!   cargo run --release --example mnist_eval -- [n_test]
//! Expects `data/aurora_mnist_weights.bin` from a prior `mnist_train` run.

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use aurora::encoding::encode_image_rate;
use aurora::mnist;
use aurora::network::Network;
use aurora::readout::predict_class;
use aurora::topology::build_feedforward;
use aurora::types::{NeuronId, Spike};
use rand::rngs::StdRng;
use rand::SeedableRng;

const N_INPUTS: usize = mnist::IMAGE_PIXELS;
const N_HIDDEN: usize = 100;
const N_CLASSES: usize = 10;
const DT: f64 = 0.001;
const PRESENT: f64 = 0.030;
const WAIT: f64 = 0.100;
const MAX_RATE_HZ: f64 = 80.0;
const PULSE_AMP: f64 = 0.025;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let n_test: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);

    let test_images = match mnist::load_images("data/mnist/t10k-images-idx3-ubyte") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read test images: {e}");
            return ExitCode::FAILURE;
        }
    };
    let test_labels = match mnist::load_labels("data/mnist/t10k-labels-idx1-ubyte") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read test labels: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut rng = StdRng::seed_from_u64(0xA1FB);
    let (mut net, layers) = build_feedforward(
        &[N_INPUTS, N_HIDDEN, N_CLASSES],
        (0.001, 0.004),
        0.001,
        &mut rng,
    );
    if let Err(e) = load_weights(&mut net, "data/aurora_mnist_weights.bin") {
        eprintln!("failed to load weights: {e}");
        eprintln!("run `cargo run --release --example mnist_train` first");
        return ExitCode::FAILURE;
    }

    let inputs = layers[0].clone();
    let outputs = layers[2].clone();

    let n_test = n_test.min(test_images.len());

    let avg_in_hid = mean_weight(&net, &layers[1]);
    let avg_hid_out = mean_weight(&net, &layers[2]);
    println!("loaded weights: avg in→hid={avg_in_hid:.5}, avg hid→out={avg_hid_out:.5}");

    let mut correct = 0_usize;
    let mut predicted = 0_usize;
    let mut total_hid_spikes = 0_usize;
    let mut total_out_spikes = 0_usize;
    let start = Instant::now();
    let mut rng2 = StdRng::seed_from_u64(0xBEEF);
    for i in 0..n_test {
        let img = &test_images[i];
        let label = test_labels[i] as usize;
        let (spikes, t_start) = present_sample(&mut net, img, &inputs, &mut rng2);
        let hid = spikes
            .iter()
            .filter(|s| layers[1].contains(&s.source))
            .count();
        let outs = spikes
            .iter()
            .filter(|s| layers[2].contains(&s.source))
            .count();
        total_hid_spikes += hid;
        total_out_spikes += outs;
        let pred = predict_class(&spikes, &outputs, t_start, t_start + PRESENT);
        if let Some(c) = pred {
            predicted += 1;
            if c == label {
                correct += 1;
            }
        }
        settle(&mut net);
        if i % 100 == 0 {
            println!(
                "{i}/{n_test} correct={correct} avg_hid_fires/sample={:.1} avg_out_fires/sample={:.1} (elapsed {:.1}s)",
                total_hid_spikes as f64 / (i + 1) as f64,
                total_out_spikes as f64 / (i + 1) as f64,
                start.elapsed().as_secs_f64()
            );
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let acc = correct as f64 / n_test as f64;
    #[allow(clippy::cast_precision_loss)]
    let coverage = predicted as f64 / n_test as f64;
    println!("\n=== MNIST eval ===");
    println!("test set:      {n_test} images");
    println!("predicted:     {predicted} ({:.1}%)", coverage * 100.0);
    println!("correct:       {correct}");
    println!("accuracy:      {:.2}% (random baseline: 10%)", acc * 100.0);
    ExitCode::SUCCESS
}

fn present_sample(
    net: &mut Network,
    image: &[f64],
    inputs: &[NeuronId],
    rng: &mut impl rand::Rng,
) -> (Vec<Spike>, f64) {
    let t_start = net.now;
    let t0 = t_start + 0.001;
    let encoded = encode_image_rate(image, PRESENT, MAX_RATE_HZ, rng);
    for (i, times) in encoded.iter().enumerate() {
        for &t in times {
            net.schedule_input(t0 + t, inputs[i], None, PULSE_AMP);
        }
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

fn mean_weight(net: &Network, target_layer: &[NeuronId]) -> f64 {
    let (sum, n) = net
        .synapses
        .iter()
        .filter(|s| target_layer.contains(&s.post_neuron))
        .fold((0.0, 0_usize), |(s, n), syn| (s + syn.weight, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn load_weights<P: AsRef<Path>>(net: &mut Network, path: P) -> std::io::Result<()> {
    let mut r = BufReader::new(File::open(path)?);
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;
    if count != net.synapses.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "weight file has {count} synapses, network expects {}",
                net.synapses.len()
            ),
        ));
    }
    let mut buf8 = [0u8; 8];
    for s in &mut net.synapses {
        r.read_exact(&mut buf8)?;
        s.weight = f64::from_le_bytes(buf8);
    }
    Ok(())
}
