#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

//! Aurora — closed-form continuous-time (CfC) Liquid Neural Network
//! language-model trainer.
//!
//! Default backend: GPU (wgpu over Vulkan/Metal/DX12). Pass `--cpu` to
//! force the rayon CPU path. Both backends share the same parameter
//! layout (`crate::lnn::MultiLayerLnnSnn`) and the same `AURORA-LNN01`
//! checkpoint format, so the GPU run is bit-equivalent to CPU within
//! FP32 noise (see `tests/lnn_gpu_parity.rs`).
//!
//! Usage:
//!   cargo run --release --features gpu --example aurora_train -- \
//!     --corpus data/tiny_600k.txt \
//!     --vocab 1024 --hidden 128 --layers 2 \
//!     --epochs 3 --chunk 64 --lr 0.005 \
//!     --checkpoint data/tiny_600k_h128.ckpt
//!
//! All flags optional; defaults are tuned for the tiny_600k smoke
//! comparison against the Sprint 48 baseline.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
use aurora::lnn::{LnnState, MultiLayerLnnSnn};
use aurora::loss::cross_entropy_loss;
use aurora::optimizer::AdamConfig;
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use textplots::{Chart, Plot, Shape};

#[cfg(feature = "gpu")]
use aurora::gpu::{GpuContext, MultiLayerLnnSnnGpu};

const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;
const BATCH_SIZE: usize = 16;
const BOX_WIDTH: usize = 60;

#[derive(Debug, Clone)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    hidden: usize,
    layers: usize,
    epochs: usize,
    chunk: usize,
    lr: f32,
    tau_init: f32,
    /// M-LTC: fraction of *top-layer* neurons that run the memory
    /// variant (sin/cos activation, log_tau floored at LOG_TAU_MEM_MIN).
    /// `0.0` (default) reproduces classic CfC.
    memory_frac: f32,
    /// Initial log_tau for memory neurons (clamped to ≥ LOG_TAU_MEM_MIN).
    tau_mem_init: f32,
    bpe_path: Option<PathBuf>,
    metrics: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    seed: u64,
    cpu: bool,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut corpus: PathBuf = "data/tiny_600k.txt".into();
        let mut vocab: usize = 1024;
        let mut hidden: usize = 128;
        let mut layers: usize = 2;
        let mut epochs: usize = 3;
        let mut chunk: usize = 64;
        let mut lr: f32 = 0.005;
        let mut tau_init: f32 = 0.0;
        let mut memory_frac: f32 = 0.0;
        let mut tau_mem_init: f32 = 1.5;
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
                "--hidden" => {
                    hidden = want_value(i)?.parse().map_err(|e| format!("--hidden: {e}"))?;
                    i += 2;
                }
                "--layers" => {
                    layers = want_value(i)?.parse().map_err(|e| format!("--layers: {e}"))?;
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
                "--tau-init" => {
                    tau_init = want_value(i)?
                        .parse()
                        .map_err(|e| format!("--tau-init: {e}"))?;
                    i += 2;
                }
                "--memory-frac" => {
                    memory_frac = want_value(i)?
                        .parse()
                        .map_err(|e| format!("--memory-frac: {e}"))?;
                    i += 2;
                }
                "--tau-mem-init" => {
                    tau_mem_init = want_value(i)?
                        .parse()
                        .map_err(|e| format!("--tau-mem-init: {e}"))?;
                    i += 2;
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

        if layers < 1 {
            return Err("--layers must be ≥ 1".into());
        }
        if vocab < 256 {
            return Err("--vocab must be ≥ 256 (byte-level minimum)".into());
        }
        if chunk < 1 {
            return Err("--chunk must be ≥ 1".into());
        }
        if epochs < 1 {
            return Err("--epochs must be ≥ 1".into());
        }
        if !(0.0..=1.0).contains(&memory_frac) {
            return Err("--memory-frac must be in [0, 1]".into());
        }
        // M-LTC currently has no GPU kernel; force CPU when enabled.
        if memory_frac > 0.0 && !cpu {
            eprintln!(
                "note: --memory-frac > 0 forces CPU backend (no GPU kernel for sin/cos cell yet)"
            );
            cpu = true;
        }

        Ok(Self {
            corpus,
            vocab,
            hidden,
            layers,
            epochs,
            chunk,
            lr,
            tau_init,
            memory_frac,
            tau_mem_init,
            bpe_path,
            metrics,
            checkpoint: ckpt,
            seed,
            cpu,
        })
    }
}

fn print_usage() {
    println!("aurora_train — CfC LNN language-model trainer (GPU default)");
    println!();
    println!("flags (all optional):");
    println!("  --corpus PATH     UTF-8 byte corpus           [default: data/tiny_600k.txt]");
    println!("  --vocab N         BPE vocab size              [default: 1024]");
    println!("  --hidden N        per-layer hidden dim        [default: 128, ≤ 1024 on GPU]");
    println!("  --layers N        number of LNN layers        [default: 2]");
    println!("  --epochs N        training epochs             [default: 3]");
    println!("  --chunk N         BPTT chunk length (toks)    [default: 64]");
    println!("  --lr F            Adam learning rate          [default: 0.005]");
    println!("  --tau-init F      log_tau init                [default: 0.0]");
    println!("  --memory-frac F   M-LTC memory fraction in    [default: 0.0; 0 = pure CfC]");
    println!("                    top layer (forces --cpu)");
    println!("  --tau-mem-init F  log_tau init for memory     [default: 1.5]");
    println!("  --bpe-path PATH   cache BPE merges            [default: <corpus>.bpe]");
    println!("  --metrics PATH    JSONL metrics log           [default: logs/<stem>_<ts>.jsonl]");
    println!("  --checkpoint PATH binary AURORA-LNN01 ckpt    [default: none]");
    println!("  --seed N          RNG seed                    [default: 0xC0FFEE]");
    println!("  --cpu             force CPU (rayon) backend   [default: GPU when feature on]");
}

/// Backend choice. CPU keeps the network struct directly; GPU keeps a
/// `MultiLayerLnnSnnGpu` plus a CPU-side mirror for evaluation /
/// sampling / checkpoint save (refreshed via `to_cpu` once per epoch).
enum Backend {
    Cpu(MultiLayerLnnSnn),
    #[cfg(feature = "gpu")]
    Gpu(MultiLayerLnnSnnGpu),
}

impl Backend {
    fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "CPU LNN-CfC (rayon)",
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => "GPU LNN-CfC (wgpu)",
        }
    }

    fn fresh_states(&self) -> Vec<LnnState> {
        match self {
            Self::Cpu(net) => net.fresh_states(),
            #[cfg(feature = "gpu")]
            Self::Gpu(net) => net.fresh_states(),
        }
    }

    fn forward_backward_chunk(
        &mut self,
        inputs: &[&[u32]],
        targets: &[&[u32]],
        states: &mut [Vec<LnnState>],
        adam: &AdamConfig,
    ) -> f32 {
        match self {
            Self::Cpu(net) => {
                net.zero_grad();
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
    fn cpu_snapshot(&self) -> MultiLayerLnnSnn {
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
    if n_tokens < (cli.chunk + 1) * BATCH_SIZE * 2 {
        eprintln!("corpus too short for chunk={} batch={BATCH_SIZE}", cli.chunk);
        return ExitCode::FAILURE;
    }
    let split = (n_tokens as f32 * 0.95) as usize;
    let train_toks = &tokens[..split];
    let val_toks = &tokens[split..];
    let comp = corpus.len() as f32 / n_tokens as f32;

    // Net + start_epoch (0 unless we resumed from a checkpoint).
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

        let stream_len = train_toks.len() / BATCH_SIZE;
        let n_chunks_per_stream = stream_len.saturating_sub(1) / cli.chunk;
        let report_every = (n_chunks_per_stream / 50).max(1);
        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;

        println!("\n  epoch {}/{}", epoch + 1, cli.epochs);
        let pb = ProgressBar::new(n_chunks_per_stream as u64).with_style(bar_style.clone());
        pb.set_message("·");

        let mut states: Vec<Vec<LnnState>> =
            (0..BATCH_SIZE).map(|_| backend.fresh_states()).collect();
        let mut inputs_owned: Vec<&[u32]> = Vec::with_capacity(BATCH_SIZE);
        let mut targets_owned: Vec<&[u32]> = Vec::with_capacity(BATCH_SIZE);

        for chunk_idx in 0..n_chunks_per_stream {
            inputs_owned.clear();
            targets_owned.clear();
            for b in 0..BATCH_SIZE {
                let stream_start = b * stream_len;
                let p = stream_start + chunk_idx * cli.chunk;
                inputs_owned.push(&train_toks[p..p + cli.chunk]);
                targets_owned.push(&train_toks[p + 1..p + cli.chunk + 1]);
            }
            let loss = backend.forward_backward_chunk(
                &inputs_owned,
                &targets_owned,
                &mut states,
                &adam,
            );
            total_nats += loss;
            total_tokens += cli.chunk * BATCH_SIZE;
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

        // Snapshot CPU view for evaluation / sampling / checkpointing.
        let net_cpu = backend.cpu_snapshot();
        let (val_bpb, activity, alpha_stats) = evaluate_lnn(&net_cpu, val_toks, cli.chunk, comp);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        val_history.push((chunks_seen as f32, val_bpb));
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        let tau_means: Vec<f32> = net_cpu
            .layers
            .iter()
            .map(|l| {
                let n = l.log_tau.len() as f32;
                if n == 0.0 {
                    0.0
                } else {
                    l.log_tau.iter().map(|v| v.exp()).sum::<f32>() / n
                }
            })
            .collect();
        println!(
            "  done — train {train_bpb:.3}  val {val_bpb:.3}  ({epoch_elapsed:.1}s)"
        );
        epoch_diagnostics(&activity, &alpha_stats, &tau_means);
        render_bpb_chart(&train_history, &val_history, chunks_seen);

        if let Some(m) = metrics.as_mut() {
            let alpha_means: Vec<f32> = alpha_stats.iter().map(|(m, _)| *m).collect();
            let alpha_devs: Vec<f32> = alpha_stats.iter().map(|(_, d)| *d).collect();
            m.epoch_done(
                epoch,
                train_bpb,
                val_bpb,
                &activity,
                &alpha_means,
                &alpha_devs,
                &tau_means,
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
    let hidden_dims = vec![cli.hidden; cli.layers];
    // M-LTC: only the *top* layer gets memory neurons. Lower layers stay
    // as fast/sensory CfC.
    let n_mem_top = ((cli.hidden as f32) * cli.memory_frac).round() as usize;
    let mut n_memory_per_layer = vec![0_usize; cli.layers];
    if cli.memory_frac > 0.0 {
        if let Some(last) = n_memory_per_layer.last_mut() {
            *last = n_mem_top.min(cli.hidden);
        }
    }
    let n_memory_per_layer_for_closure = n_memory_per_layer.clone();
    let mut fresh_net = || {
        MultiLayerLnnSnn::new_with_memory(
            actual_vocab,
            &hidden_dims,
            &n_memory_per_layer_for_closure,
            actual_vocab,
            cli.tau_init,
            cli.tau_mem_init,
            &mut rng,
        )
    };

    let (cpu_net, start_epoch) = match cli.checkpoint.as_ref().filter(|p| p.exists()) {
        Some(path) => match checkpoint::load(path) {
            Ok(c) => {
                let arch_ok = c.net.n_input == actual_vocab
                    && c.net.n_output == actual_vocab
                    && c.net.layers.len() == cli.layers
                    && c.net.layers.iter().all(|l| l.n_hidden == cli.hidden);
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
                return (
                    Backend::Gpu(MultiLayerLnnSnnGpu::from_cpu(&cpu_net, ctx)),
                    start_epoch,
                );
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

fn evaluate_lnn(
    net: &MultiLayerLnnSnn,
    val_toks: &[u32],
    chunk: usize,
    comp: f32,
) -> (f32, Vec<f32>, Vec<(f32, f32)>) {
    if val_toks.len() < chunk + 1 {
        return (
            f32::INFINITY,
            vec![0.0; net.layers.len()],
            vec![(0.0, 0.0); net.layers.len()],
        );
    }
    let mut states = net.fresh_states();
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut activity_acc = vec![0.0_f32; net.layers.len()];
    let mut alpha_acc: Vec<(f32, f32)> = vec![(0.0, 0.0); net.layers.len()];
    let mut chunks_seen = 0_usize;
    let mut pos = 0;
    while pos + chunk < val_toks.len() {
        let inputs = &val_toks[pos..pos + chunk];
        let targets = &val_toks[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_tokens_stateful(inputs, &mut states);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            total_nats += cross_entropy_loss(logits_t, target as usize);
        }
        for (acc, a) in activity_acc.iter_mut().zip(cache.activity()) {
            *acc += a;
        }
        for (acc, (m, d)) in alpha_acc.iter_mut().zip(cache.alpha_stats()) {
            acc.0 += m;
            if d > acc.1 {
                acc.1 = d;
            }
        }
        chunks_seen += 1;
        total_tokens += chunk;
        pos += chunk;
    }
    let mean_activity: Vec<f32> = activity_acc
        .into_iter()
        .map(|s| if chunks_seen == 0 { 0.0 } else { s / chunks_seen as f32 })
        .collect();
    let mean_alpha: Vec<(f32, f32)> = alpha_acc
        .into_iter()
        .map(|(m, d)| {
            (
                if chunks_seen == 0 { 0.0 } else { m / chunks_seen as f32 },
                d,
            )
        })
        .collect();
    let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
    (bpb, mean_activity, mean_alpha)
}

fn generate(
    net: &MultiLayerLnnSnn,
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
    let title = format!("Aurora · {}-layer LNN-CfC · BPE", cli.layers);
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
    let arch = format!(
        "{vocab} → ({hidden} × {layers} layers) → {vocab}",
        hidden = cli.hidden,
        layers = cli.layers,
    );
    println!("{}", kv("architecture", &arch));
    println!("{}", kv("chunk (BPTT toks)", &format!("{}", cli.chunk)));
    println!(
        "{}",
        kv(
            "optimizer",
            &format!("Adam lr={}, grad clip {GRAD_CLIP}, batch {BATCH_SIZE}", cli.lr)
        )
    );
    println!(
        "{}",
        kv("tau init", &format!("log_tau = {:.2}", cli.tau_init))
    );
    if cli.memory_frac > 0.0 {
        let n_mem = ((cli.hidden as f32) * cli.memory_frac).round() as usize;
        println!(
            "{}",
            kv(
                "M-LTC top layer",
                &format!(
                    "{n_mem}/{} memory neurons (sin), log_tau_mem = {:.2}",
                    cli.hidden, cli.tau_mem_init
                )
            )
        );
    }
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

fn epoch_diagnostics(activity: &[f32], alpha_stats: &[(f32, f32)], tau_means: &[f32]) {
    let act_s = activity
        .iter()
        .map(|r| format!("{r:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    let alpha_s = alpha_stats
        .iter()
        .map(|(m, d)| format!("{m:.3}±{d:.2}"))
        .collect::<Vec<_>>()
        .join(", ");
    let tau_s = tau_means
        .iter()
        .map(|t| format!("{t:.2}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("  |x| [{act_s}]  α [{alpha_s}]  τ [{tau_s}]");
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
            r#"{{"event":"run","cell":"lnn-cfc","corpus":{corpus:?},"corpus_bytes":{corpus_bytes},"n_tokens":{n_tokens},"vocab":{actual_vocab},"hidden":{hidden},"layers":{layers},"chunk":{chunk},"lr":{lr},"tau_init":{tau_init},"memory_frac":{memory_frac},"tau_mem_init":{tau_mem_init},"epochs":{epochs},"seed":{seed},"backend":{backend:?}}}"#,
            corpus = cli.corpus.display().to_string(),
            hidden = cli.hidden,
            layers = cli.layers,
            chunk = cli.chunk,
            lr = format_finite(cli.lr),
            tau_init = format_finite(cli.tau_init),
            memory_frac = format_finite(cli.memory_frac),
            tau_mem_init = format_finite(cli.tau_mem_init),
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

    #[allow(clippy::too_many_arguments)]
    fn epoch_done(
        &mut self,
        epoch: u32,
        train_bpb: f32,
        val_bpb: f32,
        activity: &[f32],
        alpha_means: &[f32],
        alpha_devs: &[f32],
        tau_means: &[f32],
        elapsed_s: f64,
    ) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"epoch_done","epoch":{epoch},"train_bpb":{train},"val_bpb":{val},"activity":{activity},"alpha_means":{alpha_means},"alpha_devs":{alpha_devs},"tau_means":{tau_means},"elapsed_s":{elapsed:.3}}}"#,
            train = format_finite(train_bpb),
            val = format_finite(val_bpb),
            activity = format_array(activity),
            alpha_means = format_array(alpha_means),
            alpha_devs = format_array(alpha_devs),
            tau_means = format_array(tau_means),
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

fn format_array(xs: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format_finite(*x));
    }
    s.push(']');
    s
}
