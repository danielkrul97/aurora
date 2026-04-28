#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! Char-level LM: byte-level recurrent SNN trained on a text corpus to
//! predict the next byte. Reports bits-per-byte and generates a sample
//! continuation after each epoch.
//!
//! Usage:
//!   cargo run --release --example char_lm -- <corpus_path> [n_epochs] [chunk] [lr]
//!   cargo run --release --example char_lm -- data/tiny.txt
//!
//! For tiny-shakespeare, fetch with:
//!   curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tiny.txt

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use aurora::optimizer::AdamConfig;
use aurora::snn_recurrent::RecurrentSurrogateSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::one_hot;
use aurora::text_io::{bits_per_byte, load_corpus, sample_byte, VOCAB};
use rand::rngs::StdRng;
use rand::SeedableRng;

const N_HIDDEN: usize = 256;
const TAU: f32 = 5.0;
const DT: f32 = 1.0;
const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data/tiny.txt".into());
    let n_epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let chunk: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.002);

    let corpus = match load_corpus(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read corpus from {path}: {e}");
            eprintln!("hint: download tiny-shakespeare into data/tiny.txt");
            return ExitCode::FAILURE;
        }
    };
    let n_train = (corpus.len() * 9) / 10;
    let train = &corpus[..n_train];
    let val = &corpus[n_train..];

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut net = RecurrentSurrogateSnn::new(
        VOCAB,
        N_HIDDEN,
        VOCAB,
        SpikeFn::default(),
        TAU,
        DT,
        &mut rng,
    );
    let adam = AdamConfig {
        lr,
        ..AdamConfig::default()
    };

    print_header(&path, corpus.len(), n_train, val.len(), n_epochs, chunk, lr);

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();
        epoch_open(epoch);

        let mut total_nats = 0.0_f32;
        let mut total_bytes = 0_usize;
        let mut pos = 0;
        let n_chunks = (train.len().saturating_sub(1)) / chunk;
        let report_every = (n_chunks / 10).max(1);
        let mut chunk_idx = 0_usize;

        while pos + chunk < train.len() {
            let inputs: &[u8] = &train[pos..pos + chunk];
            let targets: &[u8] = &train[pos + 1..pos + chunk + 1];
            let cache = net.forward_lm_bytes(inputs);
            net.zero_grad();
            let loss = net.backward_lm_bytes(&cache, inputs, targets);
            net.clip_grads(GRAD_CLIP);
            net.step_adam(&adam);
            total_nats += loss;
            total_bytes += chunk;
            pos += chunk;
            chunk_idx += 1;

            if chunk_idx.is_multiple_of(report_every) {
                let frac = chunk_idx as f64 / n_chunks as f64;
                let bpb = bits_per_byte(total_nats, total_bytes);
                epoch_progress(frac, bpb, epoch_start.elapsed().as_secs_f64());
            }
        }

        let train_bpb = bits_per_byte(total_nats, total_bytes);
        let val_bpb = evaluate(&net, val, chunk);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        epoch_done(train_bpb, val_bpb, epoch_start.elapsed().as_secs_f64());
        epoch_close();

        let seed = if val.len() >= 32 { &val[..32] } else { val };
        let sample = generate(&net, seed, SAMPLE_LEN, SAMPLE_TEMP, &mut rng);
        sample_block(seed, &sample);
    }

    print_footer(best_val, total_start.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

fn evaluate(net: &RecurrentSurrogateSnn, val: &[u8], chunk: usize) -> f32 {
    if val.len() < chunk + 1 {
        return f32::INFINITY;
    }
    let mut total_nats = 0.0_f32;
    let mut total_bytes = 0_usize;
    let mut pos = 0;
    while pos + chunk < val.len() {
        let inputs: &[u8] = &val[pos..pos + chunk];
        let targets: &[u8] = &val[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_bytes(inputs);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            let max = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max + logits_t.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            total_nats += log_sum_exp - logits_t[target as usize];
        }
        total_bytes += chunk;
        pos += chunk;
    }
    bits_per_byte(total_nats, total_bytes)
}

fn generate(
    net: &RecurrentSurrogateSnn,
    seed: &[u8],
    n: usize,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> Vec<u8> {
    let mut state = net.fresh_state();
    for &byte in seed {
        let _ = net.step_lm(&one_hot(byte), &mut state);
    }
    let mut last = *seed.last().unwrap_or(&b'\n');
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = net.step_lm(&one_hot(last), &mut state);
        last = sample_byte(&logits, temperature, rng);
        out.push(last);
    }
    out
}

const BOX_WIDTH: usize = 60;

fn print_header(
    path: &str,
    total: usize,
    n_train: usize,
    n_val: usize,
    n_epochs: usize,
    chunk: usize,
    lr: f32,
) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    println!("{}", center("Aurora · char-level LM (recurrent SNN)"));
    println!("╠{bar_h}╣");
    println!("{}", kv("corpus", &format!("{path} ({total} bytes)")));
    println!("{}", kv("split", &format!("{n_train} train / {n_val} val")));
    println!(
        "{}",
        kv(
            "architecture",
            &format!("{VOCAB} → {N_HIDDEN} (rec) → {VOCAB}")
        )
    );
    println!("{}", kv("chunk (BPTT len)", &format!("{chunk}")));
    println!(
        "{}",
        kv("optimizer", &format!("Adam lr={lr}, grad clip {GRAD_CLIP}"))
    );
    println!("{}", kv("epochs", &format!("{n_epochs}")));
    println!("╚{bar_h}╝");
}

fn print_footer(best_val: f32, elapsed_s: f64) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("\n╔{bar_h}╗");
    println!("{}", center("summary"));
    println!("╠{bar_h}╣");
    println!("{}", kv("best val bpb", &format!("{best_val:.3}")));
    println!("{}", kv("uniform baseline", "8.000 bpb (256-class random)"));
    println!(
        "{}",
        kv(
            "elapsed",
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

fn epoch_progress(frac: f64, bpb: f32, elapsed: f64) {
    let bar = render_bar(frac, 16);
    let body = format!(
        " {bar} {:>5.1}%  bpb {:>5.3}  ({:>5.1}s)",
        frac * 100.0,
        bpb,
        elapsed
    );
    println!("{}", line(&body));
}

fn epoch_done(train: f32, val: f32, elapsed: f64) {
    let body = format!(
        " epoch done — train bpb {:>5.3}  val bpb {:>5.3}  ({:>5.1}s)",
        train, val, elapsed
    );
    println!("{}", line(&body));
}

fn sample_block(seed: &[u8], generated: &[u8]) {
    let bar_h = "─".repeat(BOX_WIDTH);
    println!("┌{bar_h}┐");
    println!("{}", line(" sample (seed in [], generated continues):"));
    println!("{}", line(""));
    let seed_str = String::from_utf8_lossy(seed).replace('\n', "↵");
    let gen_str = String::from_utf8_lossy(generated).replace('\n', "↵");
    let combined = format!("[{seed_str}]{gen_str}");
    for chunk in combined.as_bytes().chunks(BOX_WIDTH - 2) {
        let s = String::from_utf8_lossy(chunk);
        println!("{}", line(&format!(" {s}")));
    }
    println!("└{bar_h}┘");
}

fn line(body: &str) -> String {
    let pad = BOX_WIDTH.saturating_sub(body.chars().count());
    format!("│{body}{}│", " ".repeat(pad))
}

fn kv(key: &str, value: &str) -> String {
    let body = format!("  {key:<22} : {value}");
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
