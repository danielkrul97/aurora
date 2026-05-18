#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

//! Aurora training entry point — decoder-only transformer (Sprint 54+).
//!
//! Usage:
//!   cargo run --release --example aurora_train -- \
//!     --corpus data/enwik8_10m.txt \
//!     --vocab 4096 \
//!     --d-model 384 \
//!     --n-layer 6 \
//!     --n-head 6 \
//!     --d-ff 1536 \
//!     --seq-len 256 \
//!     --batch-size 16 \
//!     --epochs 8 \
//!     --lr 6e-4 \
//!     --warmup-steps 2000 \
//!     --weight-decay 0.1 \
//!     --grad-clip 1.0
//!
//! Sprint 54 ships the CPU path. Sprint 58 will add a `--gpu` backend.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
#[cfg(feature = "gpu")]
use aurora::gpu::{DecoderTransformerGpu, GpuContext};
use aurora::optimizer::AdamWConfig;
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use aurora::transformer::loss::softmax;
use aurora::transformer::{
    cosine_warmup, DecoderTransformer, ModelGrads, TransformerConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    d_model: usize,
    n_layer: usize,
    n_head: usize,
    d_ff: usize,
    seq_len: usize,
    batch_size: usize,
    epochs: usize,
    lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    weight_decay: f32,
    grad_clip: f32,
    checkpoint: PathBuf,
    bpe_path: Option<PathBuf>,
    metrics: Option<PathBuf>,
    seed: u64,
    sample_len: usize,
    sample_temp: f32,
    gpu: bool,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut corpus: PathBuf = "data/tiny_200k.txt".into();
        let mut vocab: usize = 1024;
        let mut d_model: usize = 128;
        let mut n_layer: usize = 4;
        let mut n_head: usize = 4;
        let mut d_ff: usize = 512;
        let mut seq_len: usize = 64;
        let mut batch_size: usize = 16;
        let mut epochs: usize = 2;
        let mut lr: f32 = 6e-4;
        let mut min_lr: f32 = 6e-5;
        let mut warmup_steps: usize = 200;
        let mut weight_decay: f32 = 0.1;
        let mut grad_clip: f32 = 1.0;
        let mut checkpoint: Option<PathBuf> = None;
        let mut bpe_path: Option<PathBuf> = None;
        let mut metrics: Option<PathBuf> = None;
        let mut seed: u64 = 0xC0FFEE;
        let mut sample_len: usize = 128;
        let mut sample_temp: f32 = 0.8;
        let mut gpu: bool = false;

        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            let val = |idx: usize, name: &str| -> Result<&String, String> {
                args.get(idx + 1)
                    .ok_or_else(|| format!("missing value for {name}"))
            };
            match arg.as_str() {
                "--corpus" => {
                    corpus = val(i, arg)?.into();
                    i += 2;
                }
                "--vocab" => {
                    vocab = val(i, arg)?.parse().map_err(|e| format!("--vocab: {e}"))?;
                    i += 2;
                }
                "--d-model" => {
                    d_model = val(i, arg)?.parse().map_err(|e| format!("--d-model: {e}"))?;
                    i += 2;
                }
                "--n-layer" => {
                    n_layer = val(i, arg)?.parse().map_err(|e| format!("--n-layer: {e}"))?;
                    i += 2;
                }
                "--n-head" => {
                    n_head = val(i, arg)?.parse().map_err(|e| format!("--n-head: {e}"))?;
                    i += 2;
                }
                "--d-ff" => {
                    d_ff = val(i, arg)?.parse().map_err(|e| format!("--d-ff: {e}"))?;
                    i += 2;
                }
                "--seq-len" => {
                    seq_len = val(i, arg)?.parse().map_err(|e| format!("--seq-len: {e}"))?;
                    i += 2;
                }
                "--batch-size" => {
                    batch_size = val(i, arg)?.parse().map_err(|e| format!("--batch-size: {e}"))?;
                    i += 2;
                }
                "--epochs" => {
                    epochs = val(i, arg)?.parse().map_err(|e| format!("--epochs: {e}"))?;
                    i += 2;
                }
                "--lr" => {
                    lr = val(i, arg)?.parse().map_err(|e| format!("--lr: {e}"))?;
                    i += 2;
                }
                "--min-lr" => {
                    min_lr = val(i, arg)?.parse().map_err(|e| format!("--min-lr: {e}"))?;
                    i += 2;
                }
                "--warmup-steps" => {
                    warmup_steps = val(i, arg)?.parse().map_err(|e| format!("--warmup-steps: {e}"))?;
                    i += 2;
                }
                "--weight-decay" => {
                    weight_decay = val(i, arg)?.parse().map_err(|e| format!("--weight-decay: {e}"))?;
                    i += 2;
                }
                "--grad-clip" => {
                    grad_clip = val(i, arg)?.parse().map_err(|e| format!("--grad-clip: {e}"))?;
                    i += 2;
                }
                "--checkpoint" => {
                    checkpoint = Some(val(i, arg)?.into());
                    i += 2;
                }
                "--bpe" => {
                    bpe_path = Some(val(i, arg)?.into());
                    i += 2;
                }
                "--metrics" => {
                    metrics = Some(val(i, arg)?.into());
                    i += 2;
                }
                "--seed" => {
                    let v = val(i, arg)?;
                    seed = if let Some(rest) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))
                    {
                        u64::from_str_radix(rest, 16).map_err(|e| format!("--seed (hex): {e}"))?
                    } else {
                        v.parse::<u64>().map_err(|e| format!("--seed: {e}"))?
                    };
                    i += 2;
                }
                "--sample-len" => {
                    sample_len = val(i, arg)?.parse().map_err(|e| format!("--sample-len: {e}"))?;
                    i += 2;
                }
                "--sample-temp" => {
                    sample_temp = val(i, arg)?.parse().map_err(|e| format!("--sample-temp: {e}"))?;
                    i += 2;
                }
                "--gpu" => {
                    gpu = true;
                    i += 1;
                }
                "-h" | "--help" => {
                    return Err("see file header for usage".into());
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }

        if d_model % n_head != 0 {
            return Err(format!("d_model ({d_model}) must be divisible by n_head ({n_head})"));
        }

        let checkpoint = checkpoint.unwrap_or_else(|| {
            let stem = corpus
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("corpus");
            PathBuf::from(format!("data/{stem}_d{d_model}_l{n_layer}_v{vocab}.ckpt"))
        });

        Ok(Self {
            corpus,
            vocab,
            d_model,
            n_layer,
            n_head,
            d_ff,
            seq_len,
            batch_size,
            epochs,
            lr,
            min_lr,
            warmup_steps,
            weight_decay,
            grad_clip,
            checkpoint,
            bpe_path,
            metrics,
            seed,
            sample_len,
            sample_temp,
            gpu,
        })
    }
}

fn main() -> ExitCode {
    let cli = match Cli::parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("usage error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let corpus = match load_corpus(&cli.corpus) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read {}: {e}", cli.corpus.display());
            return ExitCode::FAILURE;
        }
    };

    let bpe_path = cli.bpe_path.clone().unwrap_or_else(|| {
        let stem = cli
            .corpus
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("corpus");
        PathBuf::from(format!("data/{stem}_{}.bpe", cli.vocab))
    });
    let bpe = match load_or_train_bpe(&corpus, cli.vocab, &bpe_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("BPE error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let actual_vocab = bpe.vocab_size();

    let t_encode = Instant::now();
    let tokens = bpe.encode(&corpus);
    let encode_s = t_encode.elapsed().as_secs_f64();
    let comp = (corpus.len() as f32) / (tokens.len() as f32);

    let n_train = (tokens.len() * 9) / 10;
    let train_toks = &tokens[..n_train];
    let val_toks = &tokens[n_train..];

    let mut rng = StdRng::seed_from_u64(cli.seed);

    // Load checkpoint if compatible, else fresh init.
    let (mut model, start_epoch, mut step) = match checkpoint::load(&cli.checkpoint) {
        Ok(ckpt) => {
            if ckpt.model.config.vocab == actual_vocab
                && ckpt.model.config.d_model == cli.d_model
                && ckpt.model.config.n_layer == cli.n_layer
                && ckpt.model.config.n_head == cli.n_head
                && ckpt.model.config.d_ff == cli.d_ff
            {
                println!(
                    "resuming from {} (epoch {}, step {})",
                    cli.checkpoint.display(),
                    ckpt.epoch,
                    ckpt.step
                );
                (ckpt.model, ckpt.epoch, ckpt.step)
            } else {
                println!(
                    "checkpoint {} architecture mismatch — fresh init",
                    cli.checkpoint.display()
                );
                let cfg = TransformerConfig {
                    vocab: actual_vocab,
                    d_model: cli.d_model,
                    n_layer: cli.n_layer,
                    n_head: cli.n_head,
                    d_ff: cli.d_ff,
                    seq_len_max: cli.seq_len,
                };
                (DecoderTransformer::new(cfg, &mut rng), 0, 0u64)
            }
        }
        Err(_) => {
            println!("no checkpoint at {} — fresh init", cli.checkpoint.display());
            let cfg = TransformerConfig {
                vocab: actual_vocab,
                d_model: cli.d_model,
                n_layer: cli.n_layer,
                n_head: cli.n_head,
                d_ff: cli.d_ff,
                seq_len_max: cli.seq_len,
            };
            (DecoderTransformer::new(cfg, &mut rng), 0, 0u64)
        }
    };

    #[cfg(feature = "gpu")]
    let mut gpu_backend: Option<DecoderTransformerGpu> = None;
    #[cfg(feature = "gpu")]
    let backend_name = if cli.gpu {
        match GpuContext::new() {
            Ok(ctx) => {
                let name = ctx.adapter_info().name.clone();
                let gpu = DecoderTransformerGpu::from_cpu(&model, ctx);
                gpu_backend = Some(gpu);
                format!("GPU ({name})")
            }
            Err(e) => {
                eprintln!("⚠ GPU init failed: {e}; falling back to CPU");
                "CPU".to_string()
            }
        }
    } else {
        "CPU".to_string()
    };
    #[cfg(not(feature = "gpu"))]
    let backend_name = {
        if cli.gpu {
            eprintln!("⚠ --gpu given but binary was built without `--features gpu`; running on CPU");
        }
        "CPU".to_string()
    };
    let metrics_path = cli.metrics.clone().unwrap_or_else(|| {
        let stem = cli
            .checkpoint
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("aurora_train");
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from(format!("logs/{stem}_{ts}.jsonl"))
    });
    let mut metrics = match MetricsWriter::open(&metrics_path) {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("⚠ metrics log open failed ({e}); continuing without metrics");
            None
        }
    };

    print_header(
        &cli,
        corpus.len(),
        tokens.len(),
        actual_vocab,
        n_train,
        val_toks.len(),
        encode_s,
        start_epoch,
        &backend_name,
        model.param_count(),
    );

    let stream_len = train_toks.len() / cli.batch_size;
    let max_chunk_advance = cli.seq_len.max(1);
    let n_chunks_per_stream = stream_len.saturating_sub(1) / max_chunk_advance;
    let total_steps_planned = (cli.epochs - start_epoch as usize) * n_chunks_per_stream;
    let total_steps_for_lr = (cli.epochs * n_chunks_per_stream).max(1);

    if let Some(m) = metrics.as_mut() {
        m.run_meta(
            &cli,
            corpus.len(),
            tokens.len(),
            actual_vocab,
            start_epoch,
            &backend_name,
            model.param_count(),
        );
    }

    let mut grads = ModelGrads::zeros(&model.config);
    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;

    for epoch in start_epoch..cli.epochs as u32 {
        let epoch_start = Instant::now();
        println!("\n── epoch {epoch} ───────────────────────────────");

        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;
        let report_every = (n_chunks_per_stream / 50).max(1);

        for chunk_idx in 0..n_chunks_per_stream {
            // Compute LR.
            let lr_t = cosine_warmup(
                step as usize,
                cli.warmup_steps,
                total_steps_for_lr,
                cli.lr,
                cli.min_lr,
            );
            let adam = AdamWConfig {
                lr: lr_t,
                weight_decay: cli.weight_decay,
                ..AdamWConfig::default()
            };

            // Build the batch: one window per stream.
            let mut batch_loss = 0.0_f32;
            let grad_norm: f32;

            #[cfg(feature = "gpu")]
            if let Some(gpu) = gpu_backend.as_mut() {
                gpu.zero_grad();
                for b in 0..cli.batch_size {
                    let stream_start = b * stream_len;
                    let p = stream_start + chunk_idx * cli.seq_len;
                    let input = &train_toks[p..p + cli.seq_len];
                    let target = &train_toks[p + 1..p + cli.seq_len + 1];
                    let loss = gpu.train_step(input, target);
                    batch_loss += loss;
                }
                // train_step accumulates batch_size separate calls. Average grads.
                gpu.clip_grads(cli.grad_clip * cli.batch_size as f32);
                grad_norm = gpu.grad_norm();
                gpu.step_adamw(&adam);
                step += 1;
            } else {
                grads.zero();
                for b in 0..cli.batch_size {
                    let stream_start = b * stream_len;
                    let p = stream_start + chunk_idx * cli.seq_len;
                    let input = &train_toks[p..p + cli.seq_len];
                    let target = &train_toks[p + 1..p + cli.seq_len + 1];
                    let (_logits, cache) = model.forward(input);
                    let loss = model.backward(&cache, target, &mut grads);
                    batch_loss += loss;
                }
                grads.scale(1.0 / cli.batch_size as f32);
                grad_norm = DecoderTransformer::clip_grads(&mut grads, cli.grad_clip);
                model.step_adamw(&grads, &adam);
                step += 1;
            }
            #[cfg(not(feature = "gpu"))]
            {
                grads.zero();
                for b in 0..cli.batch_size {
                    let stream_start = b * stream_len;
                    let p = stream_start + chunk_idx * cli.seq_len;
                    let input = &train_toks[p..p + cli.seq_len];
                    let target = &train_toks[p + 1..p + cli.seq_len + 1];
                    let (_logits, cache) = model.forward(input);
                    let loss = model.backward(&cache, target, &mut grads);
                    batch_loss += loss;
                }
                grads.scale(1.0 / cli.batch_size as f32);
                grad_norm = DecoderTransformer::clip_grads(&mut grads, cli.grad_clip);
                model.step_adamw(&grads, &adam);
                step += 1;
            }

            let avg_loss = batch_loss / cli.batch_size as f32;
            total_nats += avg_loss * cli.seq_len as f32;
            total_tokens += cli.seq_len;

            if (chunk_idx + 1).is_multiple_of(report_every) {
                let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
                let elapsed = epoch_start.elapsed().as_secs_f64();
                println!(
                    "  step {step:>7}  chunk {:>5}/{:<5}  loss {:.3}  bpb {:.3}  lr {:.2e}  gn {:.2}  {:>6.1}s",
                    chunk_idx + 1,
                    n_chunks_per_stream,
                    avg_loss,
                    bpb,
                    lr_t,
                    grad_norm,
                    elapsed
                );
                if let Some(m) = metrics.as_mut() {
                    m.step(epoch, step, avg_loss, bpb, lr_t, grad_norm, elapsed);
                }
            }
        }

        // If GPU backend is active, snapshot weights back to CPU model
        // before validation + checkpoint.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu_backend.as_ref() {
            model = gpu.to_cpu();
        }

        let train_bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
        let val_bpb = evaluate(&model, val_toks, cli.seq_len, comp);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        println!(
            "  done   train bpb {train_bpb:.3}  val bpb {val_bpb:.3}  {epoch_elapsed:.1}s"
        );
        if let Some(m) = metrics.as_mut() {
            m.epoch_done(epoch, train_bpb, val_bpb, epoch_elapsed);
        }

        if let Err(e) = checkpoint::save(&cli.checkpoint, &model, epoch + 1, step) {
            eprintln!("⚠ checkpoint save failed: {e}");
        } else {
            println!("  checkpoint → {}", cli.checkpoint.display());
            if let Some(m) = metrics.as_mut() {
                m.checkpoint(epoch + 1, step, &cli.checkpoint);
            }
        }

        // Sample generation.
        let seed_tokens: Vec<u32> = val_toks.iter().take(8).copied().collect();
        let sample = generate(&model, &seed_tokens, cli.sample_len, cli.sample_temp, &mut rng);
        let seed_text = bpe.decode(&seed_tokens);
        let sample_text = bpe.decode(&sample);
        println!(
            "  sample:\n    seed: {:?}\n    cont: {:?}",
            String::from_utf8_lossy(&seed_text),
            String::from_utf8_lossy(&sample_text)
        );
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!(
        "\nTotal time {:.1}s   best val bpb {:.3}   steps planned {}",
        total_elapsed, best_val, total_steps_planned
    );
    ExitCode::SUCCESS
}

fn load_or_train_bpe(
    corpus: &[u8],
    vocab: usize,
    bpe_path: &Path,
) -> Result<BpeTokenizer, Box<dyn std::error::Error>> {
    if bpe_path.exists() {
        let tok = BpeTokenizer::load(bpe_path)?;
        Ok(tok)
    } else {
        if let Some(parent) = bpe_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let tok = BpeTokenizer::train(corpus, vocab);
        if let Err(e) = tok.save(bpe_path) {
            eprintln!("  ⚠ BPE save failed (continuing): {e}");
        }
        Ok(tok)
    }
}

/// Compute val BPB by sliding a window of length `seq_len` over `val_toks`.
fn evaluate(model: &DecoderTransformer, val_toks: &[u32], seq_len: usize, comp: f32) -> f32 {
    if val_toks.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let stride = seq_len;
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut p = 0;
    while p + seq_len + 1 < val_toks.len() {
        let input = &val_toks[p..p + seq_len];
        let target = &val_toks[p + 1..p + seq_len + 1];
        let (logits, _) = model.forward(input);
        let v = model.config.vocab;
        for ti in 0..seq_len {
            let row = &logits[ti * v..(ti + 1) * v];
            let probs = softmax(row);
            let p_t = probs[target[ti] as usize].max(1e-12);
            total_nats += -p_t.ln();
            total_tokens += 1;
        }
        p += stride;
    }
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}

fn generate(
    model: &DecoderTransformer,
    seed: &[u32],
    n: usize,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> Vec<u32> {
    let mut tokens: Vec<u32> = seed.to_vec();
    let v = model.config.vocab;
    let max_ctx = model.config.seq_len_max.min(model.embed.seq_len_max);
    for _ in 0..n {
        let start = tokens.len().saturating_sub(max_ctx);
        let ctx = &tokens[start..];
        let (logits, _) = model.forward(ctx);
        let t = ctx.len();
        let last = &logits[(t - 1) * v..t * v];
        let next = sample_token(last, temperature, rng);
        tokens.push(next);
    }
    tokens[seed.len()..].to_vec()
}

fn print_header(
    cli: &Cli,
    corpus_bytes: usize,
    n_tokens: usize,
    actual_vocab: usize,
    n_train: usize,
    n_val: usize,
    encode_s: f64,
    start_epoch: u32,
    backend: &str,
    params: usize,
) {
    let bar = "═".repeat(60);
    println!("{bar}");
    println!(
        " Aurora train  ·  decoder-only transformer  ·  {} params  ·  {backend}",
        format_n(params)
    );
    println!("{bar}");
    println!(" corpus       : {} ({} bytes)", cli.corpus.display(), corpus_bytes);
    println!(" tokens       : {n_tokens} (BPE vocab {actual_vocab}, encode {encode_s:.1}s)");
    println!(" train / val  : {n_train} / {n_val}");
    println!(
        " model        : d={} L={} h={} dff={} ctx={}",
        cli.d_model, cli.n_layer, cli.n_head, cli.d_ff, cli.seq_len
    );
    println!(
        " optim        : lr {:.2e} → {:.2e}, warmup {} step, wd {}, clip {}",
        cli.lr, cli.min_lr, cli.warmup_steps, cli.weight_decay, cli.grad_clip
    );
    println!(" epochs       : {} (from {start_epoch}) batch {}", cli.epochs, cli.batch_size);
    println!(" checkpoint   : {}", cli.checkpoint.display());
    println!("{bar}");
}

fn format_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

// --- JSONL metrics ------------------------------------------------------

struct MetricsWriter {
    inner: BufWriter<File>,
}

impl MetricsWriter {
    fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            inner: BufWriter::new(f),
        })
    }

    fn flush(&mut self) {
        let _ = self.inner.flush();
    }

    fn run_meta(
        &mut self,
        cli: &Cli,
        corpus_bytes: usize,
        n_tokens: usize,
        actual_vocab: usize,
        start_epoch: u32,
        backend: &str,
        params: usize,
    ) {
        let line = format!(
            "{{\"event\":\"run\",\"corpus\":\"{}\",\"corpus_bytes\":{corpus_bytes},\"tokens\":{n_tokens},\"vocab\":{actual_vocab},\"d_model\":{},\"n_layer\":{},\"n_head\":{},\"d_ff\":{},\"seq_len\":{},\"batch_size\":{},\"epochs\":{},\"start_epoch\":{start_epoch},\"lr\":{:.6e},\"min_lr\":{:.6e},\"warmup_steps\":{},\"weight_decay\":{},\"grad_clip\":{},\"seed\":{},\"backend\":\"{backend}\",\"params\":{params}}}\n",
            cli.corpus.display(),
            cli.d_model, cli.n_layer, cli.n_head, cli.d_ff, cli.seq_len,
            cli.batch_size, cli.epochs, cli.lr, cli.min_lr, cli.warmup_steps,
            cli.weight_decay, cli.grad_clip, cli.seed,
        );
        let _ = self.inner.write_all(line.as_bytes());
        self.flush();
    }

    fn step(&mut self, epoch: u32, step: u64, loss: f32, bpb: f32, lr: f32, grad_norm: f32, elapsed_s: f64) {
        let line = format!(
            "{{\"event\":\"step\",\"epoch\":{epoch},\"step\":{step},\"loss\":{},\"bpb\":{},\"lr\":{:.6e},\"grad_norm\":{},\"elapsed_s\":{:.3}}}\n",
            json_f32(loss), json_f32(bpb), lr, json_f32(grad_norm), elapsed_s
        );
        let _ = self.inner.write_all(line.as_bytes());
        self.flush();
    }

    fn epoch_done(&mut self, epoch: u32, train_bpb: f32, val_bpb: f32, elapsed_s: f64) {
        let line = format!(
            "{{\"event\":\"epoch_done\",\"epoch\":{epoch},\"train_bpb\":{},\"val_bpb\":{},\"elapsed_s\":{:.3}}}\n",
            json_f32(train_bpb), json_f32(val_bpb), elapsed_s
        );
        let _ = self.inner.write_all(line.as_bytes());
        self.flush();
    }

    fn checkpoint(&mut self, epoch: u32, step: u64, path: &Path) {
        let line = format!(
            "{{\"event\":\"checkpoint\",\"epoch\":{epoch},\"step\":{step},\"path\":\"{}\"}}\n",
            path.display()
        );
        let _ = self.inner.write_all(line.as_bytes());
        self.flush();
    }
}

fn json_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else if v.is_nan() {
        "\"NaN\"".into()
    } else if v.is_sign_positive() {
        "\"Inf\"".into()
    } else {
        "\"-Inf\"".into()
    }
}
