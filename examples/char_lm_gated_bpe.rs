#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! Sprint 26: BPE-tokenized multi-layer gated SNN (CPU).
//!
//! Trains (or loads) a BPE tokenizer first, then trains a multi-layer
//! gated SNN on the token stream rather than raw bytes. With vocab >
//! 256, each token covers multiple bytes — effective context per
//! chunk grows ~2-4× and the model learns at sub-word level.
//!
//! Usage:
//!   cargo run --release --example char_lm_gated_bpe -- \
//!     <corpus_path> [n_epochs] [chunk_tokens] [lr] [n_layers] \
//!     [hidden] [vocab_size]
//!
//! `chunk_tokens` is the BPTT length **in tokens** (not bytes). Since
//! tokens compress ~2-4× in English, chunk=64 tokens ≈ 128-256 bytes
//! of context — comparable to byte-level chunk=128-256.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedState;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use rand::rngs::StdRng;
use rand::SeedableRng;

const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;
const BATCH_SIZE: usize = 16;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data/tiny.txt".into());
    let n_epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let chunk: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    let n_layers: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2);
    let hidden_per_layer: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(256);
    let vocab_size: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1024);

    if n_layers == 0 {
        eprintln!("n_layers must be ≥ 1");
        return ExitCode::FAILURE;
    }
    let hidden_dims: Vec<usize> = vec![hidden_per_layer; n_layers];

    let corpus = match load_corpus(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read corpus from {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Try to load a cached tokenizer for this (corpus, vocab) pair.
    let stem = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("corpus");
    let bpe_path: PathBuf = PathBuf::from(format!("data/{stem}_{vocab_size}.bpe"));
    let bpe = if bpe_path.exists() {
        println!("loading BPE from {}", bpe_path.display());
        match BpeTokenizer::load(&bpe_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("BPE load failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!(
            "no cached BPE at {}; training fresh on corpus…",
            bpe_path.display()
        );
        let t0 = Instant::now();
        let tok = BpeTokenizer::train(&corpus, vocab_size);
        println!("  trained {} merges in {:.1}s", tok.merges().len(), t0.elapsed().as_secs_f64());
        if let Err(e) = tok.save(&bpe_path) {
            eprintln!("BPE save failed (continuing): {e}");
        } else {
            println!("  saved to {}", bpe_path.display());
        }
        tok
    };
    let actual_vocab = bpe.vocab_size();

    // Encode the full corpus once.
    let t0 = Instant::now();
    let tokens = bpe.encode(&corpus);
    let encode_s = t0.elapsed().as_secs_f64();
    let comp = (corpus.len() as f32) / (tokens.len() as f32);
    println!(
        "encoded {} bytes → {} tokens ({comp:.2}× compression) in {encode_s:.1}s",
        corpus.len(),
        tokens.len()
    );

    let n_train = (tokens.len() * 9) / 10;
    let train_toks = &tokens[..n_train];
    let val_toks = &tokens[n_train..];
    let val_bytes_est = corpus.len() - (n_train * corpus.len() / tokens.len());

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut net = MultiLayerGatedSnn::new(
        actual_vocab,
        &hidden_dims,
        actual_vocab,
        SpikeFn::default(),
        &mut rng,
    );
    let adam = AdamConfig {
        lr,
        ..AdamConfig::default()
    };

    print_header(
        &path,
        corpus.len(),
        tokens.len(),
        actual_vocab,
        n_train,
        val_toks.len(),
        n_epochs,
        chunk,
        lr,
        n_layers,
        hidden_per_layer,
    );

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();
        epoch_open(epoch);

        let stream_len = train_toks.len() / BATCH_SIZE;
        let n_chunks_per_stream = stream_len.saturating_sub(1) / chunk;
        let report_every = (n_chunks_per_stream / 10).max(1);
        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;
        let mut states: Vec<Vec<GatedState>> =
            (0..BATCH_SIZE).map(|_| net.fresh_states()).collect();
        let mut inputs_owned: Vec<&[u32]> = Vec::with_capacity(BATCH_SIZE);
        let mut targets_owned: Vec<&[u32]> = Vec::with_capacity(BATCH_SIZE);

        for chunk_idx in 0..n_chunks_per_stream {
            inputs_owned.clear();
            targets_owned.clear();
            for b in 0..BATCH_SIZE {
                let stream_start = b * stream_len;
                let p = stream_start + chunk_idx * chunk;
                inputs_owned.push(&train_toks[p..p + chunk]);
                targets_owned.push(&train_toks[p + 1..p + chunk + 1]);
            }
            let loss = net.forward_backward_batch_tokens_stateful(
                &inputs_owned,
                &targets_owned,
                &mut states,
            );
            net.clip_grads(GRAD_CLIP);
            net.step_adam(&adam);
            total_nats += loss;
            total_tokens += chunk * BATCH_SIZE;

            if (chunk_idx + 1).is_multiple_of(report_every) {
                let frac = (chunk_idx + 1) as f64 / n_chunks_per_stream as f64;
                // BPB normalizes to the original byte count (so it
                // compares to byte-level baselines).
                let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
                epoch_progress(frac, bpb, epoch_start.elapsed().as_secs_f64());
            }
        }

        let train_bpb =
            bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
        let val_bpb = evaluate(&net, val_toks, chunk, comp);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        epoch_done(train_bpb, val_bpb, epoch_start.elapsed().as_secs_f64());
        epoch_close();

        let seed_tokens: Vec<u32> = val_toks.iter().take(16).copied().collect();
        let sample_tokens = generate(
            &net,
            &seed_tokens,
            SAMPLE_LEN,
            SAMPLE_TEMP,
            actual_vocab,
            &mut rng,
        );
        let seed_bytes = bpe.decode(&seed_tokens);
        let sample_bytes = bpe.decode(&sample_tokens);
        sample_block(&seed_bytes, &sample_bytes);
    }

    print_footer(best_val, total_start.elapsed().as_secs_f64(), val_bytes_est);
    ExitCode::SUCCESS
}

fn evaluate(net: &MultiLayerGatedSnn, val_toks: &[u32], chunk: usize, comp: f32) -> f32 {
    if val_toks.len() < chunk + 1 {
        return f32::INFINITY;
    }
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut pos = 0;
    while pos + chunk < val_toks.len() {
        let inputs = &val_toks[pos..pos + chunk];
        let targets = &val_toks[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_tokens(inputs);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            let max = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max + logits_t.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            total_nats += log_sum_exp - logits_t[target as usize];
        }
        total_tokens += chunk;
        pos += chunk;
    }
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}

fn generate(
    net: &MultiLayerGatedSnn,
    seed: &[u32],
    n: usize,
    temperature: f32,
    vocab: usize,
    rng: &mut impl rand::Rng,
) -> Vec<u32> {
    let mut states = net.fresh_states();
    for &t in seed {
        let _ = net.step_lm_token(t, &mut states);
    }
    let mut last = *seed.last().unwrap_or(&0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = net.step_lm_token(last, &mut states);
        last = sample_token(&logits, temperature, rng);
        if (last as usize) >= vocab {
            last = (vocab - 1) as u32;
        }
        out.push(last);
    }
    out
}

const BOX_WIDTH: usize = 60;

#[allow(clippy::too_many_arguments)]
fn print_header(
    path: &str,
    corpus_bytes: usize,
    n_tokens: usize,
    vocab: usize,
    n_train_tokens: usize,
    n_val_tokens: usize,
    n_epochs: usize,
    chunk: usize,
    lr: f32,
    n_layers: usize,
    hidden_per_layer: usize,
) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    println!("{}", center("Aurora · BPE multi-layer gated SNN · char-LM"));
    println!("╠{bar_h}╣");
    println!(
        "{}",
        kv("corpus", &format!("{path} ({corpus_bytes} bytes)"))
    );
    println!(
        "{}",
        kv(
            "tokens",
            &format!(
                "{n_tokens} (vocab {vocab}, {:.2}× comp)",
                (corpus_bytes as f32) / (n_tokens as f32)
            )
        )
    );
    println!(
        "{}",
        kv(
            "split",
            &format!("{n_train_tokens} train / {n_val_tokens} val toks")
        )
    );
    let arch = format!("{vocab} → ({hidden_per_layer} × {n_layers} layers) → {vocab}");
    println!("{}", kv("architecture", &arch));
    println!("{}", kv("chunk (BPTT toks)", &format!("{chunk}")));
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

fn print_footer(best_val: f32, elapsed_s: f64, val_bytes_est: usize) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("\n╔{bar_h}╗");
    println!("{}", center("summary"));
    println!("╠{bar_h}╣");
    println!("{}", kv("best val bpb", &format!("{best_val:.3}")));
    println!("{}", kv("val bytes (approx)", &format!("{val_bytes_est}")));
    println!("{}", kv("uniform baseline", "8.000 bpb"));
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
