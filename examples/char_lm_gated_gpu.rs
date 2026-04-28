#![cfg(feature = "gpu")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! GPU-trained char-level LM (Sprint 15). Drop-in replacement for
//! `char_lm_gated` using `GatedRecurrentSnnGpu` for forward+backward.
//! Sampling and validation still run on CPU after downloading weights
//! at end of each epoch — this avoids implementing GPU `step_lm_byte`
//! while keeping training entirely on the device.
//!
//! Usage:
//!   cargo run --release --features gpu --example char_lm_gated_gpu \
//!     -- <corpus_path> [n_epochs] [chunk] [lr]

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use aurora::gpu::{GatedRecurrentSnnGpu, GpuContext};
use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, load_corpus, sample_byte, VOCAB};
use rand::rngs::StdRng;
use rand::SeedableRng;

const N_HIDDEN: usize = 256;
const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;
const BATCH_SIZE: usize = 4;

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
            return ExitCode::FAILURE;
        }
    };
    let n_train = (corpus.len() * 9) / 10;
    let train = &corpus[..n_train];
    let val = &corpus[n_train..];

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU init failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = ctx.adapter_info();

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let net_init = GatedRecurrentSnn::new(VOCAB, N_HIDDEN, VOCAB, SpikeFn::default(), &mut rng);
    let mut net_gpu = GatedRecurrentSnnGpu::from_cpu(&net_init, ctx);
    drop(net_init);
    let adam = AdamConfig {
        lr,
        ..AdamConfig::default()
    };

    print_header(
        &path,
        corpus.len(),
        n_train,
        val.len(),
        n_epochs,
        chunk,
        lr,
        &info,
    );

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    let h = N_HIDDEN;
    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();
        epoch_open(epoch);

        let stream_len = train.len() / BATCH_SIZE;
        let n_chunks_per_stream = stream_len.saturating_sub(1) / chunk;
        let report_every = (n_chunks_per_stream / 50).max(1);
        let mut total_nats = 0.0_f32;
        let mut total_bytes = 0_usize;
        let mut states: Vec<(Vec<f32>, Vec<f32>)> = (0..BATCH_SIZE)
            .map(|_| (vec![0.0_f32; h], vec![0.0_f32; h]))
            .collect();
        let mut inputs_owned: Vec<&[u8]> = Vec::with_capacity(BATCH_SIZE);
        let mut targets_owned: Vec<&[u8]> = Vec::with_capacity(BATCH_SIZE);

        for chunk_idx in 0..n_chunks_per_stream {
            inputs_owned.clear();
            targets_owned.clear();
            for b in 0..BATCH_SIZE {
                let stream_start = b * stream_len;
                let p = stream_start + chunk_idx * chunk;
                inputs_owned.push(&train[p..p + chunk]);
                targets_owned.push(&train[p + 1..p + chunk + 1]);
            }
            net_gpu.zero_grad_gpu();
            let loss = net_gpu.forward_backward_batch_gpu_fused(
                &inputs_owned,
                &targets_owned,
                &mut states,
            );
            net_gpu.clip_grads_gpu(GRAD_CLIP);
            net_gpu.step_adam_gpu(&adam);
            total_nats += loss;
            total_bytes += chunk * BATCH_SIZE;

            if (chunk_idx + 1).is_multiple_of(report_every) {
                let frac = (chunk_idx + 1) as f64 / n_chunks_per_stream as f64;
                let bpb = bits_per_byte(total_nats, total_bytes);
                epoch_progress(frac, bpb, epoch_start.elapsed().as_secs_f64());
            }
        }

        let train_bpb = bits_per_byte(total_nats, total_bytes);
        let net_snapshot = net_gpu.to_cpu();
        let val_bpb = evaluate(&net_snapshot, val, chunk);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        epoch_done(train_bpb, val_bpb, epoch_start.elapsed().as_secs_f64());
        epoch_close();

        let seed = if val.len() >= 32 { &val[..32] } else { val };
        let sample = generate(&net_snapshot, seed, SAMPLE_LEN, SAMPLE_TEMP, &mut rng);
        sample_block(seed, &sample);
    }

    print_footer(best_val, total_start.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

fn evaluate(net: &GatedRecurrentSnn, val: &[u8], chunk: usize) -> f32 {
    if val.len() < chunk + 1 {
        return f32::INFINITY;
    }
    let mut total_nats = 0.0_f32;
    let mut total_bytes = 0_usize;
    let mut pos = 0;
    while pos + chunk < val.len() {
        let inputs = &val[pos..pos + chunk];
        let targets = &val[pos + 1..pos + chunk + 1];
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
    net: &GatedRecurrentSnn,
    seed: &[u8],
    n: usize,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> Vec<u8> {
    let mut state = net.fresh_state();
    for &byte in seed {
        let _ = net.step_lm_byte(byte, &mut state);
    }
    let mut last = *seed.last().unwrap_or(&b'\n');
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = net.step_lm_byte(last, &mut state);
        last = sample_byte(&logits, temperature, rng);
        out.push(last);
    }
    out
}

const BOX_WIDTH: usize = 60;

#[allow(clippy::too_many_arguments)]
fn print_header(
    path: &str,
    total: usize,
    n_train: usize,
    n_val: usize,
    n_epochs: usize,
    chunk: usize,
    lr: f32,
    adapter: &wgpu::AdapterInfo,
) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    println!("{}", center("Aurora · gated recurrent SNN · char-LM (GPU)"));
    println!("╠{bar_h}╣");
    println!(
        "{}",
        kv("gpu", &format!("{} ({:?})", adapter.name, adapter.backend))
    );
    println!("{}", kv("corpus", &format!("{path} ({total} bytes)")));
    println!("{}", kv("split", &format!("{n_train} train / {n_val} val")));
    println!(
        "{}",
        kv(
            "architecture",
            &format!("{VOCAB} → {N_HIDDEN} (gated rec) → {VOCAB}")
        )
    );
    println!("{}", kv("chunk (BPTT len)", &format!("{chunk}")));
    println!(
        "{}",
        kv(
            "optimizer",
            &format!("Adam lr={lr}, grad clip {GRAD_CLIP}, batch {BATCH_SIZE}")
        )
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
