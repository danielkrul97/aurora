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
use aurora::optimizer::{AdamWConfig, MuonConfig};
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
#[cfg(feature = "viz")]
use aurora::transformer::ForwardCache;
use aurora::transformer::{
    cosine_warmup, DecoderTransformer, ModelGrads, TransformerConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[cfg(feature = "viz")]
use std::sync::Arc;

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
    viz: bool,
    muon_lr: f32,
    muon_momentum: f32,
    muon_wd: f32,
    rope: bool,
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
        let mut viz: bool = false;
        let mut muon_lr: f32 = 0.02;
        let mut muon_momentum: f32 = 0.95;
        let mut muon_wd: f32 = 0.0;
        let mut rope: bool = false;

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
                "--viz" => {
                    viz = true;
                    i += 1;
                }
                "--muon-lr" => {
                    muon_lr = val(i, arg)?.parse().map_err(|e| format!("--muon-lr: {e}"))?;
                    i += 2;
                }
                "--muon-momentum" => {
                    muon_momentum = val(i, arg)?.parse().map_err(|e| format!("--muon-momentum: {e}"))?;
                    i += 2;
                }
                "--muon-wd" => {
                    muon_wd = val(i, arg)?.parse().map_err(|e| format!("--muon-wd: {e}"))?;
                    i += 2;
                }
                "--rope" => {
                    rope = true;
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
            viz,
            muon_lr,
            muon_momentum,
            muon_wd,
            rope,
        })
    }
}

struct TrainCtx {
    cli: Cli,
    model: DecoderTransformer,
    #[cfg(feature = "gpu")]
    gpu_backend: Option<DecoderTransformerGpu>,
    backend_name: String,
    tokens: Vec<u32>,
    n_train: usize,
    actual_vocab: usize,
    corpus_bytes: usize,
    bpe: BpeTokenizer,
    comp: f32,
    metrics_path: PathBuf,
    start_epoch: u32,
    step: u64,
    #[cfg(feature = "viz")]
    snapshot: Option<aurora::viz::SnapshotArc>,
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
    let corpus_bytes = corpus.len();

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
    drop(corpus);

    let n_train = (tokens.len() * 9) / 10;

    let mut rng = StdRng::seed_from_u64(cli.seed);

    // Load checkpoint if compatible, else fresh init.
    let (model, start_epoch, step) = match checkpoint::load(&cli.checkpoint) {
        Ok(ckpt) => {
            if ckpt.model.config.vocab == actual_vocab
                && ckpt.model.config.d_model == cli.d_model
                && ckpt.model.config.n_layer == cli.n_layer
                && ckpt.model.config.n_head == cli.n_head
                && ckpt.model.config.d_ff == cli.d_ff
                && ckpt.model.config.rope == cli.rope
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
                    rope: cli.rope,
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
                rope: cli.rope,
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
                let gpu = DecoderTransformerGpu::from_cpu(&model, ctx, cli.batch_size);
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

    let val_len = tokens.len() - n_train;
    print_header(
        &cli,
        corpus_bytes,
        tokens.len(),
        actual_vocab,
        n_train,
        val_len,
        encode_s,
        start_epoch,
        &backend_name,
        model.param_count(),
    );

    let viz_requested = cli.viz;

    #[cfg(not(feature = "viz"))]
    if viz_requested {
        eprintln!(
            "⚠ --viz given but binary was built without `--features viz`; running headless"
        );
    }

    #[cfg(feature = "viz")]
    let snapshot: Option<aurora::viz::SnapshotArc> = if viz_requested {
        Some(aurora::viz::new_snapshot(
            model.config.n_layer,
            cli.epochs as u32,
            backend_name.clone(),
            model.param_count(),
        ))
    } else {
        None
    };

    let ctx = TrainCtx {
        cli,
        model,
        #[cfg(feature = "gpu")]
        gpu_backend,
        backend_name,
        tokens,
        n_train,
        actual_vocab,
        corpus_bytes,
        bpe,
        comp,
        metrics_path,
        start_epoch,
        step,
        #[cfg(feature = "viz")]
        snapshot: snapshot.as_ref().map(Arc::clone),
    };

    #[cfg(feature = "viz")]
    if let Some(snap) = snapshot {
        let n_layer = ctx.model.config.n_layer;
        let train_handle = std::thread::spawn(move || {
            run_training_loop(ctx);
        });
        aurora::viz::run_window(snap, n_layer);
        // Window closed — training continues; this main thread blocks
        // until the worker finishes (so the user can interrupt with C-c
        // if they want to abort).
        if !train_handle.is_finished() {
            println!(
                "\n(viz window closed — training continues headless; press Ctrl-C to abort)"
            );
        }
        let _ = train_handle.join();
        return ExitCode::SUCCESS;
    }

    run_training_loop(ctx);
    ExitCode::SUCCESS
}

/// One CPU optimizer step over a full batch, parallelized across batch
/// elements with rayon. Each element runs forward+backward into its own
/// `batch_grads[b]` (no shared mutable state); the per-element grads are
/// then summed into `grads`, scaled by 1/B, clipped over the AdamW subset,
/// and applied with the Muon-hybrid stepper. Returns `(batch_loss,
/// grad_norm)`; `batch_loss` is the sum of per-element mean cross-entropies
/// — identical to the previous serial accumulation.
#[cfg(not(feature = "viz"))]
#[allow(clippy::too_many_arguments)]
fn cpu_optimizer_step(
    model: &mut DecoderTransformer,
    train_toks: &[u32],
    stream_len: usize,
    chunk_idx: usize,
    seq_len: usize,
    grad_clip: f32,
    grads: &mut ModelGrads,
    batch_grads: &mut [ModelGrads],
    muon_cfg: &MuonConfig,
    adam: &AdamWConfig,
) -> (f32, f32) {
    use rayon::prelude::*;

    let batch_size = batch_grads.len();
    let model_ref: &DecoderTransformer = model;
    let batch_loss: f32 = batch_grads
        .par_iter_mut()
        .enumerate()
        .map(|(b, g)| {
            g.zero();
            let p = b * stream_len + chunk_idx * seq_len;
            let input = &train_toks[p..p + seq_len];
            let target = &train_toks[p + 1..p + seq_len + 1];
            let (logits, cache) = model_ref.forward(input);
            model_ref.backward(&cache, &logits, target, g)
        })
        .sum();

    grads.zero();
    for g in batch_grads.iter() {
        grads.add_assign(g);
    }
    grads.scale(1.0 / batch_size as f32);
    let grad_norm = DecoderTransformer::clip_grads_adamw_subset(grads, grad_clip);
    model.step_muon_hybrid(grads, muon_cfg, adam);
    (batch_loss, grad_norm)
}

fn run_training_loop(mut ctx: TrainCtx) {
    let train_toks = &ctx.tokens[..ctx.n_train];
    let val_toks = &ctx.tokens[ctx.n_train..];

    let mut metrics = match MetricsWriter::open(&ctx.metrics_path) {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("⚠ metrics log open failed ({e}); continuing without metrics");
            None
        }
    };

    let stream_len = train_toks.len() / ctx.cli.batch_size;
    let max_chunk_advance = ctx.cli.seq_len.max(1);
    let n_chunks_per_stream = stream_len.saturating_sub(1) / max_chunk_advance;
    let total_steps_planned =
        (ctx.cli.epochs - ctx.start_epoch as usize) * n_chunks_per_stream;
    let total_steps_for_lr = (ctx.cli.epochs * n_chunks_per_stream).max(1);

    if let Some(m) = metrics.as_mut() {
        m.run_meta(
            &ctx.cli,
            ctx.corpus_bytes,
            ctx.tokens.len(),
            ctx.actual_vocab,
            ctx.start_epoch,
            &ctx.backend_name,
            ctx.model.param_count(),
        );
    }

    let mut grads = ModelGrads::zeros(&ctx.model.config);

    // Per-batch-element gradient scratch for the CPU stepper: rayon runs
    // forward+backward for each element into its own buffer (no shared
    // mutable state), then we reduce into `grads`. Allocated once, only
    // for the CPU backend (the GPU path accumulates on-device). The viz
    // build keeps the serial path and doesn't use these.
    #[cfg(all(feature = "gpu", not(feature = "viz")))]
    let backend_is_cpu = ctx.gpu_backend.is_none();
    #[cfg(all(not(feature = "gpu"), not(feature = "viz")))]
    let backend_is_cpu = true;
    #[cfg(not(feature = "viz"))]
    let mut batch_grads: Vec<ModelGrads> = if backend_is_cpu {
        (0..ctx.cli.batch_size)
            .map(|_| ModelGrads::zeros(&ctx.model.config))
            .collect()
    } else {
        Vec::new()
    };

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    let mut rng = StdRng::seed_from_u64(ctx.cli.seed.wrapping_add(0xBADC0FFEE0DDF00D));

    // Muon LR cosine targets the same min/max ratio as AdamW (default
    // 0.1×): muon_min_lr = muon_lr * (min_lr / lr).
    let muon_min_lr = if ctx.cli.lr > 0.0 {
        ctx.cli.muon_lr * (ctx.cli.min_lr / ctx.cli.lr)
    } else {
        ctx.cli.muon_lr * 0.1
    };

    for epoch in ctx.start_epoch..ctx.cli.epochs as u32 {
        let epoch_start = Instant::now();
        println!("\n── epoch {epoch} ───────────────────────────────");

        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;
        let report_every = (n_chunks_per_stream / 50).max(1);

        for chunk_idx in 0..n_chunks_per_stream {
            // Compute LR (cosine warmup) for both possible steppers.
            let lr_t = cosine_warmup(
                ctx.step as usize,
                ctx.cli.warmup_steps,
                total_steps_for_lr,
                ctx.cli.lr,
                ctx.cli.min_lr,
            );
            let muon_lr_t = cosine_warmup(
                ctx.step as usize,
                ctx.cli.warmup_steps,
                total_steps_for_lr,
                ctx.cli.muon_lr,
                muon_min_lr,
            );
            let adam = AdamWConfig {
                lr: lr_t,
                weight_decay: ctx.cli.weight_decay,
                ..AdamWConfig::default()
            };
            let muon_cfg = MuonConfig {
                lr: muon_lr_t,
                momentum: ctx.cli.muon_momentum,
                weight_decay: ctx.cli.muon_wd,
                ..MuonConfig::default()
            };

            // The pre-clip grad norm is only displayed on report steps
            // (and every step under viz). On other steps we skip the
            // GPU→host readback that fetching it would force.
            #[cfg(feature = "gpu")]
            let want_grad_norm = (chunk_idx + 1).is_multiple_of(report_every) || ctx.cli.viz;

            // Build the batch: one window per stream.
            let mut batch_loss = 0.0_f32;
            let grad_norm: f32;
            #[cfg(feature = "viz")]
            let mut last_cache: Option<ForwardCache> = None;
            #[cfg(feature = "viz")]
            let mut last_input: Vec<u32> = Vec::new();
            #[cfg(feature = "viz")]
            let mut last_target: Vec<u32> = Vec::new();
            #[cfg(feature = "viz")]
            let mut last_logits: Vec<f32> = Vec::new();

            #[cfg(feature = "gpu")]
            if let Some(gpu) = ctx.gpu_backend.as_mut() {
                gpu.zero_grad();
                let mut inputs: Vec<&[u32]> = Vec::with_capacity(ctx.cli.batch_size);
                let mut tgts: Vec<&[u32]> = Vec::with_capacity(ctx.cli.batch_size);
                for b in 0..ctx.cli.batch_size {
                    let p = b * stream_len + chunk_idx * ctx.cli.seq_len;
                    inputs.push(&train_toks[p..p + ctx.cli.seq_len]);
                    tgts.push(&train_toks[p + 1..p + ctx.cli.seq_len + 1]);
                }
                // Whole batch in one encoder / one submit / one loss readback.
                batch_loss += gpu.train_step_batch(&inputs, &tgts);
                #[cfg(feature = "viz")]
                {
                    last_input = inputs[ctx.cli.batch_size - 1].to_vec();
                    last_target = tgts[ctx.cli.batch_size - 1].to_vec();
                }
                // Grads are summed (not averaged) across the batch, so the
                // clip threshold scales with batch size. Clip touches only
                // the AdamW slice (embed + LN); matrix-grad magnitudes are
                // bounded by Newton–Schulz inside the Muon step.
                let clip_max = ctx.cli.grad_clip * ctx.cli.batch_size as f32;
                grad_norm = gpu.clip_grads_adamw_subset(clip_max, want_grad_norm);
                gpu.step_muon_hybrid(&muon_cfg, &adam);
                ctx.step += 1;
            } else {
                #[cfg(not(feature = "viz"))]
                {
                    let (bl, gn) = cpu_optimizer_step(
                        &mut ctx.model,
                        train_toks,
                        stream_len,
                        chunk_idx,
                        ctx.cli.seq_len,
                        ctx.cli.grad_clip,
                        &mut grads,
                        &mut batch_grads,
                        &muon_cfg,
                        &adam,
                    );
                    batch_loss += bl;
                    grad_norm = gn;
                }
                #[cfg(feature = "viz")]
                {
                    grads.zero();
                    for b in 0..ctx.cli.batch_size {
                        let stream_start = b * stream_len;
                        let p = stream_start + chunk_idx * ctx.cli.seq_len;
                        let input = &train_toks[p..p + ctx.cli.seq_len];
                        let target = &train_toks[p + 1..p + ctx.cli.seq_len + 1];
                        let (logits, cache) = ctx.model.forward(input);
                        let loss = ctx.model.backward(&cache, &logits, target, &mut grads);
                        batch_loss += loss;
                        if b + 1 == ctx.cli.batch_size {
                            last_cache = Some(cache);
                            last_input = input.to_vec();
                            last_target = target.to_vec();
                            last_logits = logits;
                        }
                    }
                    grads.scale(1.0 / ctx.cli.batch_size as f32);
                    grad_norm = DecoderTransformer::clip_grads_adamw_subset(
                        &mut grads,
                        ctx.cli.grad_clip,
                    );
                    ctx.model.step_muon_hybrid(&grads, &muon_cfg, &adam);
                }
                ctx.step += 1;
            }
            #[cfg(not(feature = "gpu"))]
            {
                #[cfg(not(feature = "viz"))]
                {
                    let (bl, gn) = cpu_optimizer_step(
                        &mut ctx.model,
                        train_toks,
                        stream_len,
                        chunk_idx,
                        ctx.cli.seq_len,
                        ctx.cli.grad_clip,
                        &mut grads,
                        &mut batch_grads,
                        &muon_cfg,
                        &adam,
                    );
                    batch_loss += bl;
                    grad_norm = gn;
                }
                #[cfg(feature = "viz")]
                {
                    grads.zero();
                    for b in 0..ctx.cli.batch_size {
                        let stream_start = b * stream_len;
                        let p = stream_start + chunk_idx * ctx.cli.seq_len;
                        let input = &train_toks[p..p + ctx.cli.seq_len];
                        let target = &train_toks[p + 1..p + ctx.cli.seq_len + 1];
                        let (logits, cache) = ctx.model.forward(input);
                        let loss = ctx.model.backward(&cache, &logits, target, &mut grads);
                        batch_loss += loss;
                        if b + 1 == ctx.cli.batch_size {
                            last_cache = Some(cache);
                            last_input = input.to_vec();
                            last_target = target.to_vec();
                            last_logits = logits;
                        }
                    }
                    grads.scale(1.0 / ctx.cli.batch_size as f32);
                    grad_norm = DecoderTransformer::clip_grads_adamw_subset(
                        &mut grads,
                        ctx.cli.grad_clip,
                    );
                    ctx.model.step_muon_hybrid(&grads, &muon_cfg, &adam);
                }
                ctx.step += 1;
            }

            let avg_loss = batch_loss / ctx.cli.batch_size as f32;
            total_nats += avg_loss * ctx.cli.seq_len as f32;
            total_tokens += ctx.cli.seq_len;

            let bpb_running =
                bits_per_byte(total_nats, (total_tokens as f32 * ctx.comp) as usize);
            #[cfg(feature = "viz")]
            {
                let elapsed_running = total_start.elapsed().as_secs_f64();
                publish_snapshot(
                    &ctx,
                    epoch,
                    avg_loss,
                    bpb_running,
                    lr_t,
                    grad_norm,
                    elapsed_running,
                    last_cache.as_ref(),
                    &grads,
                    &last_input,
                    &last_target,
                    &last_logits,
                );
            }

            if (chunk_idx + 1).is_multiple_of(report_every) {
                let bpb = bpb_running;
                let elapsed = epoch_start.elapsed().as_secs_f64();
                println!(
                    "  step {:>7}  chunk {:>5}/{:<5}  loss {:.3}  bpb {:.3}  lr {:.2e}  gn {:.2}  {:>6.1}s",
                    ctx.step,
                    chunk_idx + 1,
                    n_chunks_per_stream,
                    avg_loss,
                    bpb,
                    lr_t,
                    grad_norm,
                    elapsed
                );
                if let Some(m) = metrics.as_mut() {
                    m.step(epoch, ctx.step, avg_loss, bpb, lr_t, grad_norm, elapsed);
                }
            }
        }

        // If GPU backend is active, snapshot weights back to CPU model
        // before validation + checkpoint.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = ctx.gpu_backend.as_ref() {
            ctx.model = gpu.to_cpu();
        }

        let train_bpb = bits_per_byte(total_nats, (total_tokens as f32 * ctx.comp) as usize);
        let eval_start = Instant::now();
        #[cfg(feature = "gpu")]
        let val_bpb = if let Some(gpu) = ctx.gpu_backend.as_ref() {
            evaluate_gpu(gpu, val_toks, ctx.cli.seq_len, ctx.comp)
        } else {
            evaluate(&ctx.model, val_toks, ctx.cli.seq_len, ctx.comp)
        };
        #[cfg(not(feature = "gpu"))]
        let val_bpb = evaluate(&ctx.model, val_toks, ctx.cli.seq_len, ctx.comp);
        let eval_s = eval_start.elapsed().as_secs_f64();
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        println!(
            "  done   train bpb {train_bpb:.3}  val bpb {val_bpb:.3}  {epoch_elapsed:.1}s (eval {eval_s:.1}s)"
        );
        if let Some(m) = metrics.as_mut() {
            m.epoch_done(epoch, train_bpb, val_bpb, epoch_elapsed, eval_s);
        }

        if let Err(e) = checkpoint::save(&ctx.cli.checkpoint, &ctx.model, epoch + 1, ctx.step) {
            eprintln!("⚠ checkpoint save failed: {e}");
        } else {
            println!("  checkpoint → {}", ctx.cli.checkpoint.display());
            if let Some(m) = metrics.as_mut() {
                m.checkpoint(epoch + 1, ctx.step, &ctx.cli.checkpoint);
            }
        }

        // Sample generation.
        let seed_tokens: Vec<u32> = val_toks.iter().take(8).copied().collect();
        let sample = generate(
            &ctx.model,
            &seed_tokens,
            ctx.cli.sample_len,
            ctx.cli.sample_temp,
            &mut rng,
        );
        let seed_text = ctx.bpe.decode(&seed_tokens);
        let sample_text = ctx.bpe.decode(&sample);
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

    #[cfg(feature = "viz")]
    if let Some(snap) = ctx.snapshot.as_ref() {
        if let Ok(mut s) = snap.lock() {
            s.finished = true;
        }
    }
}

#[cfg(feature = "viz")]
#[allow(clippy::too_many_arguments)]
fn publish_snapshot(
    ctx: &TrainCtx,
    epoch: u32,
    loss: f32,
    bpb: f32,
    lr: f32,
    grad_norm: f32,
    elapsed_s: f64,
    cache: Option<&ForwardCache>,
    grads: &ModelGrads,
    last_input: &[u32],
    last_target: &[u32],
    last_logits: &[f32],
) {
    let Some(snap) = ctx.snapshot.as_ref() else {
        return;
    };
    let Ok(mut s) = snap.try_lock() else {
        return;
    };

    s.step = ctx.step;
    s.epoch = epoch;
    s.loss = loss;
    s.bpb = bpb;
    s.lr = lr;
    s.grad_norm = grad_norm;
    s.elapsed_s = elapsed_s;
    s.step_tick = s.step_tick.wrapping_add(1);
    s.push_history(ctx.step, loss, bpb);

    // Layer health: read from per-stage caches when available, else
    // fall back to grad norms only.
    let n_layer = ctx.model.config.n_layer;
    let total_stages = n_layer + 2;
    if s.layers.len() != total_stages {
        s.layers = aurora::viz::TrainingSnapshot::initial(
            n_layer,
            s.total_epochs,
            s.backend.clone(),
            s.params,
        )
        .layers;
    }

    let embed_g = aurora::viz::l2_norm(&grads.embed.d_w_tok)
        .hypot(aurora::viz::l2_norm(&grads.embed.d_w_pos));
    s.layers[0].grad_norm = embed_g;
    if let Some(c) = cache {
        if let Some(blk0) = c.blocks.first() {
            let denom = blk0.x_in.len().max(1) as f32;
            s.layers[0].act_norm = aurora::viz::l2_norm(&blk0.x_in) / denom.sqrt();
            s.layers[0].act_max_abs = aurora::viz::max_abs(&blk0.x_in);
            s.layers[0].health =
                aurora::viz::health_from_norm(s.layers[0].act_norm, s.layers[0].act_max_abs);
        }
    }

    // Two-pass: bars are relative to the max layer-grad so even small
    // absolute gradients still use the full bar range.
    let mut raw: Vec<(f32, f32)> = Vec::with_capacity(grads.blocks.len());
    let mut max_g: f32 = 1e-6;
    for blk in &grads.blocks {
        let mha_g = (aurora::viz::l2_norm(&blk.d_mha.d_w_q.d_w).powi(2)
            + aurora::viz::l2_norm(&blk.d_mha.d_w_k.d_w).powi(2)
            + aurora::viz::l2_norm(&blk.d_mha.d_w_v.d_w).powi(2)
            + aurora::viz::l2_norm(&blk.d_mha.d_w_o.d_w).powi(2))
        .sqrt();
        let ffn_g = aurora::viz::l2_norm(&blk.d_ffn.d_w_fc.d_w)
            .hypot(aurora::viz::l2_norm(&blk.d_ffn.d_w_proj.d_w));
        max_g = max_g.max(mha_g).max(ffn_g);
        raw.push((mha_g, ffn_g));
    }

    for (i, (mha_g, ffn_g)) in raw.iter().copied().enumerate() {
        let stage = i + 1;
        s.layers[stage].grad_norm = (mha_g * mha_g + ffn_g * ffn_g).sqrt();
        // Per-step relative bars: 0..1 with the largest layer at 1.0,
        // plus a 0.18 floor so even the quietest layer shows a sliver.
        s.layers[stage].attn_bar = (0.18 + 0.82 * mha_g / max_g).clamp(0.0, 1.0);
        s.layers[stage].ffn_bar = (0.18 + 0.82 * ffn_g / max_g).clamp(0.0, 1.0);

        if let Some(c) = cache {
            if let Some(bc) = c.blocks.get(i) {
                let denom = bc.post_attn.len().max(1) as f32;
                s.layers[stage].act_norm = aurora::viz::l2_norm(&bc.post_attn) / denom.sqrt();
                s.layers[stage].act_max_abs = aurora::viz::max_abs(&bc.post_attn);
                s.layers[stage].health = aurora::viz::health_from_norm(
                    s.layers[stage].act_norm,
                    s.layers[stage].act_max_abs,
                );
            }
        }
    }

    // LM head stage = final LN + tied head; the tied head has no grad of its own.
    let head_stage = n_layer + 1;
    s.layers[head_stage].grad_norm = aurora::viz::l2_norm(&grads.final_ln.d_gamma)
        .hypot(aurora::viz::l2_norm(&grads.final_ln.d_beta));
    if let Some(c) = cache {
        let denom = c.h_final.len().max(1) as f32;
        s.layers[head_stage].act_norm = aurora::viz::l2_norm(&c.h_final) / denom.sqrt();
        s.layers[head_stage].act_max_abs = aurora::viz::max_abs(&c.h_final);
        s.layers[head_stage].health = aurora::viz::health_from_norm(
            s.layers[head_stage].act_norm,
            s.layers[head_stage].act_max_abs,
        );
    }

    // Input / prediction preview.
    if !last_input.is_empty() {
        let take = last_input.len().min(16);
        let preview_tokens: Vec<u32> = last_input.iter().rev().take(take).rev().copied().collect();
        let bytes = ctx.bpe.decode(&preview_tokens);
        s.input_preview = String::from_utf8_lossy(&bytes).into_owned();
    }
    if !last_logits.is_empty() && !last_target.is_empty() {
        let v = ctx.model.config.vocab;
        let t = last_input.len();
        if t > 0 && last_logits.len() >= t * v {
            let last_row = &last_logits[(t - 1) * v..t * v];
            let mut argmax = 0_u32;
            let mut best = f32::NEG_INFINITY;
            for (idx, &val) in last_row.iter().enumerate() {
                if val > best {
                    best = val;
                    argmax = idx as u32;
                }
            }
            let bytes = ctx.bpe.decode(&[argmax]);
            s.pred_preview = String::from_utf8_lossy(&bytes).into_owned();
            s.pred_correct = Some(argmax == *last_target.last().unwrap_or(&u32::MAX));
        }
    }
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

/// Window starts for overlapping-window eval (stride = seq_len/2).
/// Window `wi` scores positions `scored_from(wi)..seq_len`, so every
/// scored token sees at least seq_len/2 context. Non-overlapping
/// windows score early positions with near-zero context and inflate
/// BPB by ~0.05–0.15.
fn eval_windows(val_len: usize, seq_len: usize) -> (Vec<usize>, usize) {
    let stride = (seq_len / 2).max(1);
    let mut starts = Vec::new();
    let mut p = 0;
    while p + seq_len + 1 < val_len {
        starts.push(p);
        p += stride;
    }
    (starts, seq_len - stride)
}

/// Compute val BPB by sliding a window of length `seq_len` over `val_toks`.
/// Windows are scored in parallel (forward is `&self`); per-row scoring
/// uses logsumexp directly — no per-position softmax allocation.
fn evaluate(model: &DecoderTransformer, val_toks: &[u32], seq_len: usize, comp: f32) -> f32 {
    use rayon::prelude::*;
    if val_toks.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let (starts, scored_tail) = eval_windows(val_toks.len(), seq_len);
    let v = model.config.vocab;
    let (total_nats, total_tokens) = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &p)| {
            let scored_from = if wi == 0 { 0 } else { scored_tail };
            let input = &val_toks[p..p + seq_len];
            let target = &val_toks[p + 1..p + seq_len + 1];
            let (logits, _) = model.forward(input);
            let mut nats = 0.0_f32;
            for ti in scored_from..seq_len {
                let row = &logits[ti * v..(ti + 1) * v];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                let logp =
                    (row[target[ti] as usize] - max - sum.ln()).max(-27.631_021);
                nats += -logp;
            }
            (nats, seq_len - scored_from)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}

/// GPU equivalent of `evaluate`. Same overlapping-window scoring, but
/// windows run in batches of `gpu.max_batch` — one command encoder /
/// submit / readback per batch instead of per window, so the host↔device
/// round-trip (a full device stall) is paid B× less often.
#[cfg(feature = "gpu")]
fn evaluate_gpu(
    gpu: &DecoderTransformerGpu,
    val_toks: &[u32],
    seq_len: usize,
    comp: f32,
) -> f32 {
    if val_toks.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let (starts, scored_tail) = eval_windows(val_toks.len(), seq_len);
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    for (ci, chunk) in starts.chunks(gpu.max_batch).enumerate() {
        let inputs: Vec<&[u32]> =
            chunk.iter().map(|&p| &val_toks[p..p + seq_len]).collect();
        let tgts: Vec<&[u32]> =
            chunk.iter().map(|&p| &val_toks[p + 1..p + seq_len + 1]).collect();
        let nlls = gpu.eval_step_batch(&inputs, &tgts);
        for wi in 0..chunk.len() {
            let scored_from = if ci == 0 && wi == 0 { 0 } else { scored_tail };
            total_nats += nlls[wi * seq_len + scored_from..(wi + 1) * seq_len]
                .iter()
                .sum::<f32>();
            total_tokens += seq_len - scored_from;
        }
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
        " model        : d={} L={} h={} dff={} ctx={} pos={}",
        cli.d_model, cli.n_layer, cli.n_head, cli.d_ff, cli.seq_len,
        if cli.rope { "rope" } else { "learned" }
    );
    println!(
        " optim        : muon hybrid | matrix lr {:.2e}, momentum {}, wd {} | embed/LN adamw lr {:.2e} → {:.2e}",
        cli.muon_lr,
        cli.muon_momentum,
        cli.muon_wd,
        cli.lr,
        cli.min_lr,
    );
    println!(
        "              : warmup {} step, adamw wd {}, clip {}",
        cli.warmup_steps, cli.weight_decay, cli.grad_clip
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
            "{{\"event\":\"run\",\"corpus\":\"{}\",\"corpus_bytes\":{corpus_bytes},\"tokens\":{n_tokens},\"vocab\":{actual_vocab},\"d_model\":{},\"n_layer\":{},\"n_head\":{},\"d_ff\":{},\"seq_len\":{},\"batch_size\":{},\"epochs\":{},\"start_epoch\":{start_epoch},\"lr\":{:.6e},\"min_lr\":{:.6e},\"warmup_steps\":{},\"weight_decay\":{},\"grad_clip\":{},\"seed\":{},\"backend\":\"{backend}\",\"params\":{params},\"optimizer\":\"muon\",\"muon_lr\":{:.6e},\"muon_momentum\":{},\"muon_wd\":{},\"rope\":{}}}\n",
            cli.corpus.display(),
            cli.d_model, cli.n_layer, cli.n_head, cli.d_ff, cli.seq_len,
            cli.batch_size, cli.epochs, cli.lr, cli.min_lr, cli.warmup_steps,
            cli.weight_decay, cli.grad_clip, cli.seed,
            cli.muon_lr, cli.muon_momentum, cli.muon_wd, cli.rope,
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

    fn epoch_done(&mut self, epoch: u32, train_bpb: f32, val_bpb: f32, elapsed_s: f64, eval_s: f64) {
        let line = format!(
            "{{\"event\":\"epoch_done\",\"epoch\":{epoch},\"train_bpb\":{},\"val_bpb\":{},\"elapsed_s\":{:.3},\"eval_s\":{:.3}}}\n",
            json_f32(train_bpb), json_f32(val_bpb), elapsed_s, eval_s
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
