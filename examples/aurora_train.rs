#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

//! Aurora — bipolar HDC language-model trainer.
//!
//! Default backend: GPU (wgpu over Vulkan/Metal/DX12). Pass `--cpu` to
//! force the rayon CPU path. Both backends share the same parameter
//! layout (`crate::hdc::HdcLm`) and the same `AURORA-HDC01` checkpoint
//! format, so the GPU run is bit-equivalent to CPU within FP32 noise
//! (see `tests/hdc_gpu_parity.rs`).
//!
//! Usage:
//!   cargo run --release --features gpu --example aurora_train -- \
//!     --corpus data/tiny_200k.txt \
//!     --vocab 512 --dim 2048 \
//!     --epochs 3 --chunk 64 --lr 0.005 \
//!     --checkpoint data/tiny_200k_d2048.ckpt
//!
//! All flags optional; defaults are tuned for a quick smoke comparison
//! against the LNN-era baseline.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
use aurora::hdc::{HdcLm, HdcState};
use aurora::loss::cross_entropy_loss;
use aurora::optimizer::AdamConfig;
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use textplots::{Chart, Plot, Shape};

#[cfg(feature = "gpu")]
use aurora::gpu::{GpuContext, HdcGpu};

const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;
const DEFAULT_BATCH: usize = 16;
const BOX_WIDTH: usize = 60;

#[derive(Debug, Clone)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    dim: usize,
    epochs: usize,
    chunk: usize,
    lr: f32,
    batch: usize,
    /// If true, carry the bundle accumulator across chunks within one
    /// epoch (long-context HDC). Default false = stateless per chunk.
    stateful: bool,
    bpe_path: Option<PathBuf>,
    metrics: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    seed: u64,
    cpu: bool,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut corpus: PathBuf = "data/tiny_200k.txt".into();
        let mut vocab: usize = 512;
        let mut dim: usize = 2048;
        let mut epochs: usize = 3;
        let mut chunk: usize = 64;
        let mut lr: f32 = 0.005;
        let mut batch: usize = DEFAULT_BATCH;
        let mut stateful = false;
        let mut bpe_path: Option<PathBuf> = None;
        let mut metrics: Option<PathBuf> = None;
        let mut ckpt: Option<PathBuf> = None;
        let mut seed: u64 = 0xC0FFEE;
        let mut cpu = false;

        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            let want_value = |i: usize| -> Result<&String, String> {
                args.get(i + 1).ok_or_else(|| format!("{key} needs a value"))
            };
            match key.as_str() {
                "--corpus" => {
                    corpus = want_value(i)?.into();
                    i += 2;
                }
                "--vocab" => {
                    vocab = want_value(i)?.parse().map_err(|e| format!("--vocab: {e}"))?;
                    i += 2;
                }
                "--dim" => {
                    dim = want_value(i)?.parse().map_err(|e| format!("--dim: {e}"))?;
                    i += 2;
                }
                "--epochs" => {
                    epochs = want_value(i)?.parse().map_err(|e| format!("--epochs: {e}"))?;
                    i += 2;
                }
                "--chunk" => {
                    chunk = want_value(i)?.parse().map_err(|e| format!("--chunk: {e}"))?;
                    i += 2;
                }
                "--lr" => {
                    lr = want_value(i)?.parse().map_err(|e| format!("--lr: {e}"))?;
                    i += 2;
                }
                "--batch" => {
                    batch = want_value(i)?.parse().map_err(|e| format!("--batch: {e}"))?;
                    i += 2;
                }
                "--stateful" => {
                    stateful = true;
                    i += 1;
                }
                "--bpe-path" => {
                    bpe_path = Some(want_value(i)?.into());
                    i += 2;
                }
                "--metrics" => {
                    metrics = Some(want_value(i)?.into());
                    i += 2;
                }
                "--checkpoint" => {
                    ckpt = Some(want_value(i)?.into());
                    i += 2;
                }
                "--seed" => {
                    seed = want_value(i)?.parse().map_err(|e| format!("--seed: {e}"))?;
                    i += 2;
                }
                "--cpu" => {
                    cpu = true;
                    i += 1;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}")),
            }
        }

        if vocab < 16 {
            return Err("--vocab must be ≥ 16".into());
        }
        if dim < 64 {
            return Err("--dim must be ≥ 64".into());
        }
        if dim > 4096 {
            return Err("--dim must be ≤ 4096 (GPU workgroup-shared budget)".into());
        }
        if chunk < 1 {
            return Err("--chunk must be ≥ 1".into());
        }
        if epochs < 1 {
            return Err("--epochs must be ≥ 1".into());
        }
        if batch < 1 {
            return Err("--batch must be ≥ 1".into());
        }
        // wgpu binding-size cap (typically 128 MB on consumer GPUs).
        // Per-batch grad buffer = batch * vocab * dim * 4 bytes.
        let pb_bytes = batch as u64 * vocab as u64 * dim as u64 * 4;
        if pb_bytes > 128 * 1024 * 1024 && !cpu {
            let hint = (128 * 1024 * 1024) / (vocab as u64 * dim as u64 * 4);
            return Err(format!(
                "GPU per-batch grad buffer = {} MB exceeds 128 MB binding limit. \
                 Reduce --batch (try ≤ {hint}), --vocab, or --dim, or pass --cpu.",
                pb_bytes / (1024 * 1024)
            ));
        }

        Ok(Self {
            corpus,
            vocab,
            dim,
            epochs,
            chunk,
            lr,
            batch,
            stateful,
            bpe_path,
            metrics,
            checkpoint: ckpt,
            seed,
            cpu,
        })
    }
}

fn print_usage() {
    println!("aurora_train — bipolar HDC language-model trainer (GPU default)");
    println!();
    println!("flags (all optional):");
    println!("  --corpus PATH     UTF-8 byte corpus           [default: data/tiny_200k.txt]");
    println!("  --vocab N         BPE vocab size              [default: 512]");
    println!("  --dim N           hyperdim D                  [default: 2048, max 4096 on GPU]");
    println!("  --epochs N        training epochs             [default: 3]");
    println!("  --chunk N         chunk length (tokens)       [default: 64]");
    println!("  --lr F            Adam learning rate          [default: 0.005]");
    println!("  --batch N         batch size                  [default: 16; lower if GPU OOM]");
    println!("  --stateful        carry bundle across chunks  [default: stateless per chunk]");
    println!("  --bpe-path PATH   cache BPE merges            [default: <corpus>.<vocab>.bpe]");
    println!("  --metrics PATH    JSONL metrics log           [default: logs/<stem>_<ts>.jsonl]");
    println!("  --checkpoint PATH binary AURORA-HDC01 ckpt    [default: none]");
    println!("  --seed N          RNG seed                    [default: 0xC0FFEE]");
    println!("  --cpu             force CPU (rayon) backend   [default: GPU when feature on]");
}

enum Backend {
    Cpu(HdcLm),
    #[cfg(feature = "gpu")]
    Gpu(HdcGpu),
}

impl Backend {
    fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "CPU HDC (rayon)",
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => "GPU HDC (wgpu)",
        }
    }

    fn fresh_state(&self) -> HdcState {
        match self {
            Self::Cpu(net) => net.fresh_state(),
            #[cfg(feature = "gpu")]
            Self::Gpu(net) => net.fresh_state(),
        }
    }

    fn forward_backward_chunk(
        &mut self,
        inputs: &[&[u32]],
        targets: &[&[u32]],
        states: &mut [HdcState],
        adam: &AdamConfig,
    ) -> f32 {
        match self {
            Self::Cpu(net) => {
                let loss = net.forward_backward_batch_tokens_stateful(inputs, targets, states);
                net.clip_grads(GRAD_CLIP);
                net.step_adam(adam);
                loss
            }
            #[cfg(feature = "gpu")]
            Self::Gpu(net) => {
                net.zero_grad();
                let loss = net.forward_backward_batch_tokens_stateful(inputs, targets, states);
                net.clip_grads(GRAD_CLIP);
                net.step_adam(adam);
                loss
            }
        }
    }

    /// Snapshot CPU view of current weights — for evaluate, generate, save.
    fn cpu_snapshot(&self) -> HdcLm {
        match self {
            Self::Cpu(net) => net.clone(),
            #[cfg(feature = "gpu")]
            Self::Gpu(net) => net.to_cpu(),
        }
    }
}

fn main() -> ExitCode {
    let cli = match Cli::parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let corpus = match load_corpus(&cli.corpus) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to load {}: {e}", cli.corpus.display());
            return ExitCode::FAILURE;
        }
    };

    let bpe_path = cli.bpe_path.clone().unwrap_or_else(|| {
        let mut p = cli.corpus.clone();
        p.set_extension(format!("v{}.bpe", cli.vocab));
        p
    });
    let bpe = match load_or_train_bpe(&corpus, cli.vocab, &bpe_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("BPE failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let actual_vocab = bpe.vocab_size();

    let encode_t0 = Instant::now();
    let tokens: Vec<u32> = bpe.encode(&corpus);
    let encode_s = encode_t0.elapsed().as_secs_f64();
    let n_tokens = tokens.len();
    if n_tokens < (cli.chunk + 1) * cli.batch * 2 {
        eprintln!("corpus too short for chunk={} batch={}", cli.chunk, cli.batch);
        return ExitCode::FAILURE;
    }
    let split = (n_tokens as f32 * 0.95) as usize;
    let train_toks = &tokens[..split];
    let val_toks = &tokens[split..];
    let comp = corpus.len() as f32 / n_tokens as f32;

    let (mut backend, start_epoch) = build_backend(&cli, actual_vocab);

    let adam = AdamConfig {
        lr: cli.lr,
        ..AdamConfig::default()
    };

    let metrics_path = cli.metrics.clone().unwrap_or_else(|| {
        let stem = cli
            .corpus
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("aurora");
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
    let backend_name = backend.name();
    if let Some(m) = metrics.as_mut() {
        m.run_meta(&cli, corpus.len(), n_tokens, actual_vocab, backend_name);
    }

    print_header(
        &cli,
        corpus.len(),
        n_tokens,
        actual_vocab,
        train_toks.len(),
        val_toks.len(),
        encode_s,
        backend_name,
        start_epoch,
    );

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    let mut train_history: Vec<(f32, f32)> = Vec::new();
    let mut val_history: Vec<(f32, f32)> = Vec::new();
    let mut chunks_seen: u64 = 0;

    let bar_template = "  [{bar:32.cyan/blue}] {percent:>3}%  bpb {msg}  ({elapsed:>4} / ETA {eta:>4})";
    let bar_style = ProgressStyle::with_template(bar_template)
        .expect("static template parses")
        .progress_chars("█·");

    let mut rng = StdRng::seed_from_u64(cli.seed);

    for epoch in start_epoch..cli.epochs as u32 {
        let epoch_start = Instant::now();

        let stream_len = train_toks.len() / cli.batch;
        let n_chunks_per_stream = stream_len.saturating_sub(1) / cli.chunk;
        let report_every = (n_chunks_per_stream / 50).max(1);
        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;

        println!("\n  epoch {}/{}", epoch + 1, cli.epochs);
        let pb = ProgressBar::new(n_chunks_per_stream as u64).with_style(bar_style.clone());
        pb.set_message("·");

        // Stateful: keep states across the whole epoch's chunks (per-stream).
        // Stateless: reset before each chunk.
        let mut states: Vec<HdcState> =
            (0..cli.batch).map(|_| backend.fresh_state()).collect();
        let mut inputs_owned: Vec<&[u32]> = Vec::with_capacity(cli.batch);
        let mut targets_owned: Vec<&[u32]> = Vec::with_capacity(cli.batch);

        for chunk_idx in 0..n_chunks_per_stream {
            inputs_owned.clear();
            targets_owned.clear();
            for b in 0..cli.batch {
                let stream_start = b * stream_len;
                let p = stream_start + chunk_idx * cli.chunk;
                inputs_owned.push(&train_toks[p..p + cli.chunk]);
                targets_owned.push(&train_toks[p + 1..p + cli.chunk + 1]);
            }
            if !cli.stateful {
                for s in states.iter_mut() {
                    *s = backend.fresh_state();
                }
            }
            let loss = backend.forward_backward_chunk(
                &inputs_owned,
                &targets_owned,
                &mut states,
                &adam,
            );
            total_nats += loss;
            total_tokens += cli.chunk * cli.batch;
            chunks_seen += 1;
            pb.inc(1);

            if (chunk_idx + 1) % report_every == 0 {
                let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
                pb.set_message(format!("{bpb:.3}"));
                let elapsed = epoch_start.elapsed().as_secs_f64();
                train_history.push((chunks_seen as f32, bpb));
                if let Some(m) = metrics.as_mut() {
                    m.chunk(epoch, chunk_idx + 1, bpb, elapsed);
                }
                if pb.is_hidden() {
                    let pct = ((chunk_idx + 1) as f64 / n_chunks_per_stream as f64) * 100.0;
                    println!("  [{pct:>5.1}%]  bpb {bpb:.3}  ({elapsed:>5.1}s)");
                }
            }
        }

        let train_bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
        pb.finish_with_message(format!("{train_bpb:.3}"));

        let net_cpu = backend.cpu_snapshot();
        let (val_bpb, bundle_norm, cleanup_top1) =
            evaluate_hdc(&net_cpu, val_toks, cli.chunk, comp, cli.stateful);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        val_history.push((chunks_seen as f32, val_bpb));
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        println!(
            "  done — train {train_bpb:.3}  val {val_bpb:.3}  ({epoch_elapsed:.1}s)"
        );
        epoch_diagnostics(bundle_norm, cleanup_top1);
        render_bpb_chart(&train_history, &val_history, chunks_seen);

        if let Some(m) = metrics.as_mut() {
            m.epoch_done(
                epoch,
                train_bpb,
                val_bpb,
                bundle_norm,
                cleanup_top1,
                epoch_elapsed,
            );
        }

        if let Some(ckpt_path) = cli.checkpoint.as_ref() {
            if let Err(e) = checkpoint::save(ckpt_path, &net_cpu, epoch + 1) {
                eprintln!("⚠ checkpoint save failed ({e})");
            } else {
                println!("  ckpt → {} (epoch {})", ckpt_path.display(), epoch + 1);
                if let Some(m) = metrics.as_mut() {
                    m.checkpoint(epoch + 1, ckpt_path);
                }
            }
        }

        let seed_tokens: Vec<u32> = val_toks.iter().take(16).copied().collect();
        if seed_tokens.is_empty() {
            continue;
        }
        let sample_tokens =
            generate(&net_cpu, &seed_tokens, SAMPLE_LEN, SAMPLE_TEMP, actual_vocab, &mut rng);
        let seed_bytes = bpe.decode(&seed_tokens);
        let sample_bytes = bpe.decode(&sample_tokens);
        sample_block(&seed_bytes, &sample_bytes);
    }

    let val_bytes_est = corpus.len().saturating_sub(split * corpus.len() / n_tokens);
    print_footer(best_val, total_start.elapsed().as_secs_f64(), val_bytes_est);
    ExitCode::SUCCESS
}

fn build_backend(cli: &Cli, actual_vocab: usize) -> (Backend, u32) {
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut fresh_net = || HdcLm::new(cli.dim, actual_vocab, &mut rng);

    let (cpu_net, start_epoch) = match cli.checkpoint.as_ref().filter(|p| p.exists()) {
        Some(path) => match checkpoint::load(path) {
            Ok(c) => {
                let arch_ok = c.net.dim == cli.dim && c.net.vocab == actual_vocab;
                if arch_ok {
                    println!("loaded checkpoint {} (epoch {})", path.display(), c.epoch);
                    (c.net, c.epoch)
                } else {
                    eprintln!(
                        "⚠ checkpoint at {} has different architecture; starting fresh",
                        path.display()
                    );
                    (fresh_net(), 0)
                }
            }
            Err(e) => {
                eprintln!(
                    "⚠ checkpoint at {} failed to load ({e}); starting fresh",
                    path.display()
                );
                (fresh_net(), 0)
            }
        },
        None => (fresh_net(), 0),
    };

    if cli.cpu {
        return (Backend::Cpu(cpu_net), start_epoch);
    }

    #[cfg(feature = "gpu")]
    {
        match GpuContext::new() {
            Ok(ctx) => {
                let info = ctx.adapter_info();
                println!(
                    "GPU adapter: {} ({:?}, {:?})",
                    info.name, info.device_type, info.backend
                );
                return (Backend::Gpu(HdcGpu::from_cpu(&cpu_net, ctx)), start_epoch);
            }
            Err(e) => {
                eprintln!("⚠ GPU init failed ({e}); falling back to CPU");
                return (Backend::Cpu(cpu_net), start_epoch);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("note: built without `gpu` feature — using CPU backend");
        (Backend::Cpu(cpu_net), start_epoch)
    }
}

fn load_or_train_bpe(
    corpus: &[u8],
    vocab: usize,
    bpe_path: &Path,
) -> Result<BpeTokenizer, Box<dyn std::error::Error>> {
    if bpe_path.exists() {
        println!("loading BPE from {}", bpe_path.display());
        let tok = BpeTokenizer::load(bpe_path)?;
        if tok.vocab_size() != vocab {
            println!(
                "  cached BPE vocab is {} (requested {}) — using cached",
                tok.vocab_size(),
                vocab
            );
        }
        Ok(tok)
    } else {
        println!(
            "no cached BPE at {}; training fresh on corpus…",
            bpe_path.display()
        );
        let t0 = Instant::now();
        let tok = BpeTokenizer::train(corpus, vocab);
        println!(
            "  trained {} merges in {:.1}s",
            tok.merges().len(),
            t0.elapsed().as_secs_f64()
        );
        if let Err(e) = tok.save(bpe_path) {
            eprintln!("  ⚠ BPE save failed (continuing): {e}");
        } else {
            println!("  saved → {}", bpe_path.display());
        }
        Ok(tok)
    }
}

/// Evaluate val BPB and two HDC-specific diagnostics:
/// - mean L2 norm of context HV (`bundle_norm`),
/// - top-1 cleanup accuracy: how often `argmax_v cosine(ctx, codebook_v)`
///   matches the target. Pure-HDC signal independent of the readout.
fn evaluate_hdc(
    net: &HdcLm,
    val_toks: &[u32],
    chunk: usize,
    comp: f32,
    stateful: bool,
) -> (f32, f32, f32) {
    if val_toks.len() < chunk + 1 {
        return (f32::INFINITY, 0.0, 0.0);
    }
    let mut state = net.fresh_state();
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut bundle_norm_sum = 0.0_f32;
    let mut bundle_norm_chunks = 0_usize;
    let mut cleanup_hits = 0_u64;
    let mut pos = 0;
    while pos + chunk < val_toks.len() {
        if !stateful {
            state = net.fresh_state();
        }
        let inputs = &val_toks[pos..pos + chunk];
        let targets = &val_toks[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_tokens_stateful(inputs, &mut state);
        for (t, &target) in targets.iter().enumerate() {
            let logits_off = t * net.vocab;
            total_nats += cross_entropy_loss(
                &cache.logits[logits_off..logits_off + net.vocab],
                target as usize,
            );
            // Cleanup: cosine similarity of context_t against every
            // codebook row → argmax. Frozen codebook = pure HDC signal.
            let ctx_off = t * net.dim;
            let ctx = &cache.contexts[ctx_off..ctx_off + net.dim];
            let mut best = 0_u32;
            let mut best_score = f32::NEG_INFINITY;
            for v in 0..net.vocab {
                let row = &net.codebook.items[v * net.dim..(v + 1) * net.dim];
                let mut s = 0.0_f32;
                for i in 0..net.dim {
                    s += row[i] * ctx[i];
                }
                if s > best_score {
                    best_score = s;
                    best = v as u32;
                }
            }
            if best == target {
                cleanup_hits += 1;
            }
        }
        bundle_norm_sum += cache.bundle_norm(net.dim);
        bundle_norm_chunks += 1;
        total_tokens += chunk;
        pos += chunk;
    }
    let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
    let bundle_norm = if bundle_norm_chunks == 0 {
        0.0
    } else {
        bundle_norm_sum / bundle_norm_chunks as f32
    };
    let top1 = if total_tokens == 0 {
        0.0
    } else {
        cleanup_hits as f32 / total_tokens as f32
    };
    (bpb, bundle_norm, top1)
}

fn generate(
    net: &HdcLm,
    seed: &[u32],
    n: usize,
    temperature: f32,
    vocab: usize,
    rng: &mut impl rand::Rng,
) -> Vec<u32> {
    let mut state = net.fresh_state();
    for &t in seed {
        let _ = net.step_lm_token(t, &mut state);
    }
    let mut last = *seed.last().unwrap_or(&0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = net.step_lm_token(last, &mut state);
        last = sample_token(&logits, temperature, rng);
        if (last as usize) >= vocab {
            last = (vocab - 1) as u32;
        }
        out.push(last);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn print_header(
    cli: &Cli,
    corpus_bytes: usize,
    n_tokens: usize,
    vocab: usize,
    n_train: usize,
    n_val: usize,
    encode_s: f64,
    backend: &str,
    start_epoch: u32,
) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    let title = format!("Aurora · bipolar HDC · D = {}", cli.dim);
    println!("{}", center(&title));
    println!("╠{bar_h}╣");
    println!("{}", kv("backend", backend));
    println!(
        "{}",
        kv(
            "corpus",
            &format!("{} ({corpus_bytes} bytes)", cli.corpus.display())
        )
    );
    println!(
        "{}",
        kv(
            "tokens",
            &format!(
                "{n_tokens} (vocab {vocab}, {:.2}× comp, encode {encode_s:.1}s)",
                (corpus_bytes as f32) / (n_tokens as f32)
            )
        )
    );
    println!(
        "{}",
        kv("split", &format!("{n_train} train / {n_val} val toks"))
    );
    let arch = format!("{vocab} → HDC(D={dim}) → {vocab}", dim = cli.dim);
    println!("{}", kv("architecture", &arch));
    println!("{}", kv("chunk (tokens)", &format!("{}", cli.chunk)));
    println!(
        "{}",
        kv(
            "state mode",
            if cli.stateful { "stateful (carry across chunks)" } else { "stateless (reset per chunk)" }
        )
    );
    println!(
        "{}",
        kv(
            "optimizer",
            &format!("Adam lr={}, grad clip {GRAD_CLIP}, batch {}", cli.lr, cli.batch)
        )
    );
    println!(
        "{}",
        kv("epochs", &format!("{} → {}", start_epoch, cli.epochs))
    );
    println!("╚{bar_h}╝");
}

fn print_footer(best_val: f32, elapsed_s: f64, val_bytes_est: usize) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("\n╔{bar_h}╗");
    println!("{}", center("summary"));
    println!("╠{bar_h}╣");
    println!("{}", kv("best val bpb", &format!("{best_val:.3}")));
    println!("{}", kv("val bytes (approx)", &format!("{val_bytes_est}")));
    println!("{}", kv("uniform baseline", "8.000 bpb (256-class random)"));
    println!(
        "{}",
        kv(
            "elapsed",
            &format!("{elapsed_s:.1} s ({:.1} min)", elapsed_s / 60.0)
        )
    );
    println!("╚{bar_h}╝");
}

fn epoch_diagnostics(bundle_norm: f32, cleanup_top1: f32) {
    println!(
        "  bundle_norm {bundle_norm:.3}  cleanup_top1 {:.4}",
        cleanup_top1
    );
}

fn render_bpb_chart(
    train_history: &[(f32, f32)],
    val_history: &[(f32, f32)],
    chunks_seen: u64,
) {
    if train_history.len() < 2 {
        return;
    }
    let x_max = chunks_seen.max(1) as f32;
    println!("  bpb history (— train, • val per epoch)");
    Chart::new(72, 20, 0.0, x_max)
        .lineplot(&Shape::Lines(train_history))
        .lineplot(&Shape::Points(val_history))
        .display();
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

// =====================================================================
// Hand-rolled JSONL metrics writer (no serde_json dep).
// =====================================================================

struct MetricsWriter {
    out: BufWriter<File>,
}

impl MetricsWriter {
    fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = File::create(path)?;
        Ok(Self {
            out: BufWriter::new(f),
        })
    }

    fn run_meta(
        &mut self,
        cli: &Cli,
        corpus_bytes: usize,
        n_tokens: usize,
        actual_vocab: usize,
        backend: &str,
    ) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"run","cell":"hdc","corpus":{corpus:?},"corpus_bytes":{corpus_bytes},"n_tokens":{n_tokens},"vocab":{actual_vocab},"dim":{dim},"chunk":{chunk},"lr":{lr},"stateful":{stateful},"epochs":{epochs},"seed":{seed},"backend":{backend:?}}}"#,
            corpus = cli.corpus.display().to_string(),
            dim = cli.dim,
            chunk = cli.chunk,
            lr = format_finite(cli.lr),
            stateful = cli.stateful,
            epochs = cli.epochs,
            seed = cli.seed,
        );
        let _ = self.out.flush();
    }

    fn chunk(&mut self, epoch: u32, chunk_idx: usize, train_bpb: f32, elapsed_s: f64) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"chunk","epoch":{epoch},"chunk":{chunk_idx},"train_bpb":{bpb},"elapsed_s":{elapsed:.3}}}"#,
            bpb = format_finite(train_bpb),
            elapsed = elapsed_s,
        );
        let _ = self.out.flush();
    }

    fn epoch_done(
        &mut self,
        epoch: u32,
        train_bpb: f32,
        val_bpb: f32,
        bundle_norm: f32,
        cleanup_top1: f32,
        elapsed_s: f64,
    ) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"epoch_done","epoch":{epoch},"train_bpb":{train},"val_bpb":{val},"bundle_norm":{bn},"cleanup_top1":{c1},"elapsed_s":{elapsed:.3}}}"#,
            train = format_finite(train_bpb),
            val = format_finite(val_bpb),
            bn = format_finite(bundle_norm),
            c1 = format_finite(cleanup_top1),
            elapsed = elapsed_s,
        );
        let _ = self.out.flush();
    }

    fn checkpoint(&mut self, epoch: u32, path: &Path) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"checkpoint","epoch":{epoch},"path":{path:?}}}"#,
            path = path.display().to_string(),
        );
        let _ = self.out.flush();
    }
}

fn format_finite(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        format!("\"{v}\"")
    }
}
