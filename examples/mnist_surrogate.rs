#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! Train + evaluate a surrogate-gradient SNN on MNIST.
//! Usage: cargo run --release --example mnist_surrogate -- [n_train] [n_test] [n_epochs]

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use aurora::encoding::rate_encode_bernoulli;
use aurora::mnist;
use aurora::snn_dense::SurrogateSnn;
use aurora::snn_loss::argmax;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const N_HIDDEN: usize = 200;
const N_CLASSES: usize = 10;
const T_STEPS: usize = 25;
const MAX_RATE: f32 = 0.4;
const LR: f32 = 0.001;
const TAU: f32 = 5.0;
const DT: f32 = 1.0;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let n_train: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let n_test: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000);
    let n_epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

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

    let n_train = n_train.min(train_images.len());
    let n_test = n_test.min(test_images.len());

    let train_f32: Vec<Vec<f32>> = train_images[..n_train]
        .iter()
        .map(|img| img.iter().map(|&p| p as f32).collect())
        .collect();
    let test_f32: Vec<Vec<f32>> = test_images[..n_test]
        .iter()
        .map(|img| img.iter().map(|&p| p as f32).collect())
        .collect();

    let mut rng = StdRng::seed_from_u64(0xA1FB);
    let mut net = SurrogateSnn::new(
        mnist::IMAGE_PIXELS,
        N_HIDDEN,
        N_CLASSES,
        SpikeFn::default(),
        TAU,
        DT,
        &mut rng,
    );

    print_header(n_train, n_test, n_epochs);

    let mut indices: Vec<usize> = (0..n_train).collect();
    let start = Instant::now();
    let mut best_test = 0.0_f64;
    let mut last_test = 0.0_f64;
    let mut last_train = 0.0_f64;

    for epoch in 0..n_epochs {
        indices.shuffle(&mut rng);
        let mut correct = 0_usize;
        let mut loss_sum = 0.0_f32;
        let report_every = (n_train / 10).max(1);
        let epoch_start = Instant::now();

        epoch_open(epoch);

        for (k, &i) in indices.iter().enumerate() {
            let label = train_labels[i] as usize;
            let encoded = rate_encode_bernoulli(&train_f32[i], T_STEPS, MAX_RATE, &mut rng);
            let cache = net.forward(&encoded);
            net.zero_grad();
            let (loss, pred) = net.backward(&cache, &encoded, label);
            loss_sum += loss;
            if pred == label {
                correct += 1;
            }
            net.step_sgd(LR);

            if (k + 1) % report_every == 0 {
                let frac = (k + 1) as f64 / n_train as f64;
                let acc = f64::from(correct as u32) / f64::from((k + 1) as u32);
                let avg_loss = loss_sum / (k + 1) as f32;
                epoch_progress(frac, acc, avg_loss, epoch_start.elapsed().as_secs_f64());
            }
        }

        let train_acc = f64::from(correct as u32) / f64::from(n_train as u32);
        let test_acc = evaluate(&net, &test_f32, &test_labels, &mut rng);
        last_train = train_acc;
        last_test = test_acc;
        if test_acc > best_test {
            best_test = test_acc;
        }
        epoch_done(train_acc, test_acc, epoch_start.elapsed().as_secs_f64());
        epoch_close();
    }

    print_footer(
        last_train,
        last_test,
        best_test,
        start.elapsed().as_secs_f64(),
    );
    ExitCode::SUCCESS
}

const BOX_WIDTH: usize = 60;

fn print_header(n_train: usize, n_test: usize, n_epochs: usize) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    println!("{}", center("Aurora · surrogate-gradient SNN · MNIST"));
    println!("╠{bar_h}╣");
    println!(
        "{}",
        kv(
            "architecture",
            &format!("{} → {} → {}", mnist::IMAGE_PIXELS, N_HIDDEN, N_CLASSES)
        )
    );
    println!(
        "{}",
        kv(
            "timesteps",
            &format!("T = {T_STEPS}, dt = {DT}, tau = {TAU}")
        )
    );
    println!(
        "{}",
        kv(
            "encoding",
            &format!("rate (Bernoulli, max {MAX_RATE} per step)")
        )
    );
    println!(
        "{}",
        kv(
            "spike fn",
            &format!(
                "θ = {}, surrogate α = {}",
                SpikeFn::default().threshold,
                SpikeFn::default().alpha
            )
        )
    );
    println!(
        "{}",
        kv(
            "optimizer",
            &format!("SGD, lr = {LR}, BPTT, CE on spike counts")
        )
    );
    println!(
        "{}",
        kv(
            "dataset",
            &format!("MNIST {n_train} train / {n_test} test, {n_epochs} epochs")
        )
    );
    println!("╚{bar_h}╝");
}

fn print_footer(last_train: f64, last_test: f64, best_test: f64, elapsed_s: f64) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("\n╔{bar_h}╗");
    println!("{}", center("summary"));
    println!("╠{bar_h}╣");
    println!(
        "{}",
        kv(
            "final train accuracy",
            &format!("{:>6.2}%", last_train * 100.0)
        )
    );
    println!(
        "{}",
        kv(
            "final test  accuracy",
            &format!("{:>6.2}%", last_test * 100.0)
        )
    );
    println!(
        "{}",
        kv(
            "best  test  accuracy",
            &format!("{:>6.2}%", best_test * 100.0)
        )
    );
    println!(
        "{}",
        kv(
            "random baseline",
            &format!("{:>6.2}%", 100.0 / N_CLASSES as f64)
        )
    );
    println!(
        "{}",
        kv(
            "total elapsed",
            &format!("{:.1} s ({:.1} min)", elapsed_s, elapsed_s / 60.0)
        )
    );
    println!("╚{bar_h}╝");
}

fn epoch_open(epoch: usize) {
    let label = format!(" epoch {epoch} ");
    let dashes = BOX_WIDTH.saturating_sub(label.chars().count());
    println!("\n┌{label}{}┐", "─".repeat(dashes));
}

fn epoch_close() {
    println!("└{}┘", "─".repeat(BOX_WIDTH));
}

fn epoch_progress(frac: f64, acc: f64, loss: f32, elapsed: f64) {
    let bar = render_bar(frac, 16);
    let body = format!(
        " {bar} {:>5.1}%  acc {:>5.1}%  loss {:>6.3}  ({:>5.1}s)",
        frac * 100.0,
        acc * 100.0,
        loss,
        elapsed
    );
    println!("{}", line(&body));
}

fn epoch_done(train: f64, test: f64, elapsed: f64) {
    let body = format!(
        " epoch done — train {:>5.2}%  test {:>5.2}%  ({:>5.1}s)",
        train * 100.0,
        test * 100.0,
        elapsed
    );
    println!("{}", line(&body));
}

fn line(body: &str) -> String {
    let pad = BOX_WIDTH.saturating_sub(body.chars().count());
    format!("│{body}{}│", " ".repeat(pad))
}

fn kv(key: &str, value: &str) -> String {
    let body = format!("  {key:<14} : {value}");
    let pad = BOX_WIDTH.saturating_sub(body.chars().count());
    format!("║{body}{}║", " ".repeat(pad))
}

fn center(text: &str) -> String {
    let width = BOX_WIDTH;
    let len = text.chars().count();
    let left = width.saturating_sub(len) / 2;
    let right = width.saturating_sub(len + left);
    format!("║{}{text}{}║", " ".repeat(left), " ".repeat(right))
}

fn render_bar(frac: f64, width: usize) -> String {
    let filled = ((frac * width as f64).round() as usize).min(width);
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for i in 0..width {
        s.push(if i < filled { '█' } else { '·' });
    }
    s.push(']');
    s
}

fn evaluate(net: &SurrogateSnn, images: &[Vec<f32>], labels: &[u8], rng: &mut impl Rng) -> f64 {
    let mut correct = 0_usize;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let encoded = rate_encode_bernoulli(img, T_STEPS, MAX_RATE, rng);
        let cache = net.forward(&encoded);
        if argmax(&cache.spike_counts) == label as usize {
            correct += 1;
        }
    }
    f64::from(correct as u32) / f64::from(images.len() as u32)
}
