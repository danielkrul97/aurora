#![cfg(feature = "gpu")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::large_enum_variant
)]

//! Sprint 28: Aurora's primary training example.
//!
//! Combines every speed + quality feature shipped so far:
//! * GPU 2-layer megakernel (Sprint 25)
//! * BPE tokenization (Sprint 26)
//! * Token-level GPU forward/backward (Sprint 28)
//! * Checkpointing — auto-saves weights at end of every epoch and
//!   auto-resumes if a checkpoint with matching architecture exists.
//!
//! Designed for overnight runs on enwik8/enwik9. Use BPE for quality,
//! GPU megakernel for speed, and checkpoints so you don't have to
//! restart from scratch after a power blip or system update.
//!
//! Usage:
//!   cargo run --release --features gpu --example aurora_train -- \
//!     --corpus data/enwik8.txt \
//!     --vocab 4096 \
//!     --hidden 512 \
//!     --epochs 8 \
//!     --chunk 64 \
//!     --lr 0.002 \
//!     --checkpoint data/aurora.ckpt
//!
//! All flags optional; defaults shown above.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::apoptosis::{ApoptosisConfig, ApoptosisMode};
use aurora::bpe::BpeTokenizer;
use aurora::checkpoint::{self, Checkpoint};
use aurora::gpu::snn_gated_gpu_multi::MultiLayerGatedSnnGpu;
use aurora::gpu::GpuContext;
use aurora::optimizer::AdamConfig;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use rand::rngs::StdRng;
use rand::SeedableRng;

const GRAD_CLIP: f32 = 1.0;
const SAMPLE_LEN: usize = 200;
const SAMPLE_TEMP: f32 = 0.8;
const BATCH_SIZE: usize = 16;

#[derive(Debug, Clone)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    hidden: usize,
    epochs: usize,
    chunk: usize,
    lr: f32,
    checkpoint: PathBuf,
    bpe_path: Option<PathBuf>,
    metrics: Option<PathBuf>,
    /// Sprint 39: opt out of GPU and run on CPU. Required for `--sdsa`
    /// until the GPU SDSA port lands. Only useful for small configs
    /// (hidden ≤ 256 typically); CPU is ~10× slower than GPU at scale.
    cpu: bool,
    /// Sprint 39: enable Spike-Driven Self-Attention block.
    /// CPU-only path (Sprint 38 wiring); GPU port = Sprint 40+.
    sdsa: bool,
    /// Sprint 40: enable AdLIF adaptation current. CPU-only.
    adlif: bool,
    /// Sprint 41: stack depth. Default 2 (matches GPU megakernel
    /// constraint). CPU supports any value.
    n_layers: usize,
    /// Sprint 42: apoptosis mode. `off` (default), `activity`
    /// (kill neurons with EMA firing rate < threshold), or `random`
    /// (one-shot kill of fixed counts per layer — control for the
    /// activity arm of the experiment).
    apoptosis_mode: String,
    apop_warmup: usize,
    apop_ema: f32,
    apop_threshold: f32,
    apop_check: usize,
    /// Comma-separated per-layer kill counts for `--apoptosis random`.
    apop_counts: Option<String>,
    /// Sprint 43 trophic mode: kill if EMA < frac × layer_mean.
    apop_trophic_frac: f32,
    apop_trophic_max: usize,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut corpus: PathBuf = "data/enwik8.txt".into();
        let mut vocab: usize = 4096;
        let mut hidden: usize = 512;
        let mut epochs: usize = 8;
        let mut chunk: usize = 64;
        let mut lr: f32 = 0.002;
        let mut checkpoint: Option<PathBuf> = None;
        let mut bpe_path: Option<PathBuf> = None;
        let mut metrics: Option<PathBuf> = None;
        let mut cpu: bool = false;
        let mut sdsa: bool = false;
        let mut adlif: bool = false;
        let mut n_layers: usize = 2;
        let mut apoptosis_mode: String = "off".to_string();
        let mut apop_warmup: usize = 200;
        let mut apop_ema: f32 = 0.05;
        let mut apop_threshold: f32 = 0.005;
        let mut apop_check: usize = 100;
        let mut apop_counts: Option<String> = None;
        let mut apop_trophic_frac: f32 = 0.1;
        let mut apop_trophic_max: usize = 5;

        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            let next = || {
                args.get(i + 1)
                    .ok_or_else(|| format!("missing value for {arg}"))
            };
            match arg.as_str() {
                "--corpus" => {
                    corpus = next()?.into();
                    i += 2;
                }
                "--vocab" => {
                    vocab = next()?.parse().map_err(|e| format!("--vocab: {e}"))?;
                    i += 2;
                }
                "--hidden" => {
                    hidden = next()?.parse().map_err(|e| format!("--hidden: {e}"))?;
                    i += 2;
                }
                "--epochs" => {
                    epochs = next()?.parse().map_err(|e| format!("--epochs: {e}"))?;
                    i += 2;
                }
                "--chunk" => {
                    chunk = next()?.parse().map_err(|e| format!("--chunk: {e}"))?;
                    i += 2;
                }
                "--lr" => {
                    lr = next()?.parse().map_err(|e| format!("--lr: {e}"))?;
                    i += 2;
                }
                "--checkpoint" => {
                    checkpoint = Some(next()?.into());
                    i += 2;
                }
                "--bpe" => {
                    bpe_path = Some(next()?.into());
                    i += 2;
                }
                "--metrics" => {
                    metrics = Some(next()?.into());
                    i += 2;
                }
                "--cpu" => {
                    cpu = true;
                    i += 1;
                }
                "--sdsa" => {
                    sdsa = true;
                    i += 1;
                }
                "--adlif" => {
                    adlif = true;
                    i += 1;
                }
                "--n-layers" => {
                    n_layers =
                        next()?.parse().map_err(|e| format!("--n-layers: {e}"))?;
                    i += 2;
                }
                "--apoptosis" => {
                    apoptosis_mode = next()?.clone();
                    i += 2;
                }
                "--apoptosis-warmup" => {
                    apop_warmup = next()?.parse().map_err(|e| format!("--apoptosis-warmup: {e}"))?;
                    i += 2;
                }
                "--apoptosis-ema" => {
                    apop_ema = next()?.parse().map_err(|e| format!("--apoptosis-ema: {e}"))?;
                    i += 2;
                }
                "--apoptosis-threshold" => {
                    apop_threshold = next()?.parse().map_err(|e| format!("--apoptosis-threshold: {e}"))?;
                    i += 2;
                }
                "--apoptosis-check" => {
                    apop_check = next()?.parse().map_err(|e| format!("--apoptosis-check: {e}"))?;
                    i += 2;
                }
                "--apoptosis-counts" => {
                    apop_counts = Some(next()?.clone());
                    i += 2;
                }
                "--apoptosis-trophic-frac" => {
                    apop_trophic_frac = next()?.parse().map_err(|e| format!("--apoptosis-trophic-frac: {e}"))?;
                    i += 2;
                }
                "--apoptosis-trophic-max" => {
                    apop_trophic_max = next()?.parse().map_err(|e| format!("--apoptosis-trophic-max: {e}"))?;
                    i += 2;
                }
                "-h" | "--help" => {
                    return Err("see file header for usage".into());
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }

        let checkpoint = checkpoint.unwrap_or_else(|| {
            let stem = corpus
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("corpus");
            PathBuf::from(format!("data/{stem}_h{hidden}_v{vocab}.ckpt"))
        });

        if sdsa && !cpu {
            return Err(
                "--sdsa currently requires --cpu (GPU SDSA port = Sprint 40+)".into(),
            );
        }
        if adlif && !cpu {
            return Err(
                "--adlif currently requires --cpu (GPU AdLIF port = Sprint 41+)".into(),
            );
        }
        if n_layers != 2 && !cpu {
            return Err(format!(
                "--n-layers {n_layers} requires --cpu \
                (GPU megakernel hardcoded for 2 layers; deeper GPU = Sprint 42+)"
            ));
        }
        if n_layers == 0 {
            return Err("--n-layers must be ≥ 1".into());
        }

        match apoptosis_mode.as_str() {
            "off" | "activity" | "random" | "trophic" => {}
            other => {
                return Err(format!(
                    "--apoptosis must be off|activity|trophic|random, got '{other}'"
                ))
            }
        }
        if apoptosis_mode != "off" && !cpu {
            return Err("--apoptosis requires --cpu (Sprint 42 scope)".into());
        }
        if apoptosis_mode == "random" && apop_counts.is_none() {
            return Err("--apoptosis random requires --apoptosis-counts X,Y,…".into());
        }

        Ok(Self {
            corpus,
            vocab,
            hidden,
            epochs,
            chunk,
            lr,
            checkpoint,
            bpe_path,
            metrics,
            cpu,
            sdsa,
            adlif,
            n_layers,
            apoptosis_mode,
            apop_warmup,
            apop_ema,
            apop_threshold,
            apop_check,
            apop_counts,
            apop_trophic_frac,
            apop_trophic_max,
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

    if cli.hidden > 1024 {
        eprintln!("hidden must be ≤ 1024 (megakernel workgroup memory budget)");
        return ExitCode::FAILURE;
    }

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

    let t0 = Instant::now();
    let tokens = bpe.encode(&corpus);
    let encode_s = t0.elapsed().as_secs_f64();
    let comp = (corpus.len() as f32) / (tokens.len() as f32);

    let n_train = (tokens.len() * 9) / 10;
    let train_toks = &tokens[..n_train];
    let val_toks = &tokens[n_train..];

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let (mut net_init, start_epoch) =
        load_or_init_net(&cli.checkpoint, actual_vocab, cli.hidden, cli.n_layers, &mut rng);
    let resumed = start_epoch > 0;

    if cli.sdsa {
        net_init.enable_sdsa(&mut rng);
    }
    if cli.adlif {
        net_init.enable_adlif();
    }
    let apop_config_opt = match build_apoptosis_config(&cli, cli.n_layers) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("apoptosis config error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(cfg) = apop_config_opt.clone() {
        net_init.enable_apoptosis(cfg);
    }

    enum Backend {
        Gpu(MultiLayerGatedSnnGpu, String),
        Cpu(MultiLayerGatedSnn, String),
    }
    let mut backend = if cli.cpu {
        Backend::Cpu(net_init, "CPU (rayon)".to_string())
    } else {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("GPU init failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let name = ctx.adapter_info().name;
        Backend::Gpu(MultiLayerGatedSnnGpu::from_cpu(&net_init, ctx), name)
    };
    let backend_name = match &backend {
        Backend::Gpu(_, n) => n.clone(),
        Backend::Cpu(_, n) => n.clone(),
    };

    let adam = AdamConfig {
        lr: cli.lr,
        ..AdamConfig::default()
    };

    // Sprint 36: structured metrics output. Defaults to
    // `logs/<ckpt_stem>_<unix_ts>.jsonl`; pass `--metrics path.jsonl`
    // to override or `/dev/null` to disable.
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
    if let Some(m) = metrics.as_mut() {
        m.run_meta(
            &cli,
            corpus.len(),
            tokens.len(),
            actual_vocab,
            start_epoch,
            &backend_name,
        );
    }

    print_header(
        &cli,
        corpus.len(),
        tokens.len(),
        actual_vocab,
        n_train,
        val_toks.len(),
        encode_s,
        start_epoch,
        resumed,
        &backend_name,
    );

    let total_start = Instant::now();
    let mut best_val = f32::INFINITY;
    let h = cli.hidden;
    type StateTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    for epoch in start_epoch..cli.epochs as u32 {
        let epoch_start = Instant::now();
        epoch_open(epoch);

        let stream_len = train_toks.len() / BATCH_SIZE;
        let n_chunks_per_stream = stream_len.saturating_sub(1) / cli.chunk;
        let report_every = (n_chunks_per_stream / 50).max(1);
        let mut total_nats = 0.0_f32;
        let mut total_tokens = 0_usize;

        // Per-backend state allocation. CPU uses `Vec<GatedState>`,
        // GPU uses a flat 4-tuple (l0_h_v_after, l0_h_spike, l1_h_v_after, l1_h_spike).
        let mut gpu_states: Vec<StateTuple> = Vec::new();
        let mut cpu_states: Vec<Vec<aurora::snn_gated::GatedState>> = Vec::new();
        match &backend {
            Backend::Gpu(_, _) => {
                gpu_states = (0..BATCH_SIZE)
                    .map(|_| (vec![0.0; h], vec![0.0; h], vec![0.0; h], vec![0.0; h]))
                    .collect();
            }
            Backend::Cpu(net, _) => {
                cpu_states = (0..BATCH_SIZE).map(|_| net.fresh_states()).collect();
            }
        }

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
            let loss = match &mut backend {
                Backend::Gpu(net_gpu, _) => {
                    net_gpu.zero_grad_gpu();
                    let l = net_gpu.forward_backward_batch_gpu_multi_fused_tokens(
                        &inputs_owned,
                        &targets_owned,
                        &mut gpu_states,
                    );
                    net_gpu.clip_grads_gpu(GRAD_CLIP);
                    net_gpu.step_adam_gpu(&adam);
                    l
                }
                Backend::Cpu(net, _) => {
                    net.zero_grad();
                    let (l, rates_opt) = if net.apoptosis.is_some() {
                        let (loss, rates) = net
                            .forward_backward_batch_tokens_stateful_with_rates(
                                &inputs_owned,
                                &targets_owned,
                                &mut cpu_states,
                            );
                        (loss, Some(rates))
                    } else {
                        let loss = net.forward_backward_batch_tokens_stateful(
                            &inputs_owned,
                            &targets_owned,
                            &mut cpu_states,
                        );
                        (loss, None)
                    };
                    // Sprint 43: trophic signal must be read from raw
                    // grads before clip_grads scales them.
                    let trophic_opt = match net
                        .apoptosis
                        .as_ref()
                        .map(|a| a.config.mode.clone())
                    {
                        Some(aurora::apoptosis::ApoptosisMode::Trophic { .. }) => {
                            Some(net.trophic_signal_from_grads())
                        }
                        _ => None,
                    };
                    net.clip_grads(GRAD_CLIP);
                    net.step_adam(&adam);
                    if let Some(apop) = net.apoptosis.as_mut() {
                        let signal = trophic_opt
                            .or(rates_opt)
                            .expect("apoptosis on requires either rates or trophic");
                        apop.observe_rates(&signal);
                        apop.tick();
                        let kills = apop.maybe_kill(&mut rng);
                        if !kills.is_empty() {
                            if let Some(m) = metrics.as_mut() {
                                for k in &kills {
                                    m.apoptosis_kill(epoch, k);
                                }
                            }
                        }
                        if !kills.is_empty()
                            || apop.chunks_seen.is_multiple_of(apop.config.check_interval)
                        {
                            let dead: Vec<usize> = (0..apop.dead_mask.len())
                                .map(|li| apop.dead_count(li))
                                .collect();
                            if let Some(m) = metrics.as_mut() {
                                m.apoptosis_state(
                                    epoch,
                                    apop.chunks_seen,
                                    &dead,
                                );
                            }
                        }
                    }
                    l
                }
            };
            total_nats += loss;
            total_tokens += cli.chunk * BATCH_SIZE;

            if (chunk_idx + 1).is_multiple_of(report_every) {
                let frac = (chunk_idx + 1) as f64 / n_chunks_per_stream as f64;
                let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
                let elapsed = epoch_start.elapsed().as_secs_f64();
                epoch_progress(frac, bpb, elapsed);
                if let Some(m) = metrics.as_mut() {
                    m.chunk(epoch, chunk_idx + 1, bpb, elapsed);
                }
            }
        }

        let train_bpb =
            bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
        let net_snapshot = match &backend {
            Backend::Gpu(net_gpu, _) => net_gpu.to_cpu(),
            Backend::Cpu(net, _) => net.clone(),
        };
        let (val_bpb, rates, mem) = evaluate(&net_snapshot, val_toks, cli.chunk, comp);
        if val_bpb < best_val {
            best_val = val_bpb;
        }
        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        epoch_done(train_bpb, val_bpb, epoch_elapsed);
        let alphas: Vec<f32> = net_snapshot.layers.iter().map(|l| l.alpha).collect();
        epoch_diagnostics(&rates, &alphas, &mem);

        if let Some(m) = metrics.as_mut() {
            let v_means: Vec<f32> = mem.iter().map(|(m, _)| *m).collect();
            let v_peaks: Vec<f32> = mem.iter().map(|(_, p)| *p).collect();
            m.epoch_done(
                epoch,
                train_bpb,
                val_bpb,
                &rates,
                &alphas,
                &v_means,
                &v_peaks,
                epoch_elapsed,
            );
        }

        if let Err(e) = checkpoint::save(&cli.checkpoint, &net_snapshot, epoch + 1) {
            eprintln!("⚠ checkpoint save failed: {e}");
        } else {
            println!(
                "{}",
                line(&format!(" checkpoint saved → {}", cli.checkpoint.display()))
            );
            if let Some(m) = metrics.as_mut() {
                m.checkpoint(epoch + 1, &cli.checkpoint);
            }
        }
        epoch_close();

        let seed_tokens: Vec<u32> = val_toks.iter().take(16).copied().collect();
        let sample_tokens = generate(
            &net_snapshot,
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

    let val_bytes_est = corpus.len().saturating_sub(n_train * corpus.len() / tokens.len());
    print_footer(best_val, total_start.elapsed().as_secs_f64(), val_bytes_est);
    ExitCode::SUCCESS
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

fn build_apoptosis_config(cli: &Cli, n_layers: usize) -> Result<Option<ApoptosisConfig>, String> {
    let mode = match cli.apoptosis_mode.as_str() {
        "off" => return Ok(None),
        "activity" => ApoptosisMode::Activity,
        "trophic" => ApoptosisMode::Trophic {
            frac: cli.apop_trophic_frac,
            max_per_check: cli.apop_trophic_max,
        },
        "random" => {
            let s = cli
                .apop_counts
                .as_deref()
                .ok_or("--apoptosis random needs --apoptosis-counts")?;
            let counts: Vec<usize> = s
                .split(',')
                .map(|p| p.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("--apoptosis-counts parse: {e}"))?;
            if counts.len() != n_layers {
                return Err(format!(
                    "--apoptosis-counts has {} entries but --n-layers={n_layers}",
                    counts.len()
                ));
            }
            ApoptosisMode::RandomFixed {
                per_layer_counts: counts,
            }
        }
        other => return Err(format!("unknown mode '{other}'")),
    };
    Ok(Some(ApoptosisConfig {
        mode,
        warmup_chunks: cli.apop_warmup,
        ema_alpha: cli.apop_ema,
        kill_threshold: cli.apop_threshold,
        check_interval: cli.apop_check,
    }))
}

fn load_or_init_net(
    path: &Path,
    vocab: usize,
    hidden: usize,
    n_layers: usize,
    rng: &mut StdRng,
) -> (MultiLayerGatedSnn, u32) {
    if path.exists() {
        match checkpoint::load(path) {
            Ok(Checkpoint { net, epoch }) => {
                let arch_ok = net.n_input == vocab
                    && net.n_output == vocab
                    && net.layers.len() == n_layers
                    && net.layers.iter().all(|l| l.n_hidden == hidden);
                if arch_ok {
                    println!(
                        "resumed from {} (completed epochs: {epoch})",
                        path.display()
                    );
                    return (net, epoch);
                }
                eprintln!(
                    "checkpoint at {} has different architecture; starting fresh",
                    path.display()
                );
            }
            Err(e) => {
                eprintln!("checkpoint load failed: {e}; starting fresh");
            }
        }
    }
    let hidden_dims: Vec<usize> = vec![hidden; n_layers];
    let net = MultiLayerGatedSnn::new(
        vocab,
        &hidden_dims,
        vocab,
        SpikeFn::default(),
        rng,
    );
    (net, 0)
}

fn evaluate(
    net: &MultiLayerGatedSnn,
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
    // Stateful eval: carry hidden state across val chunks so the
    // firing-rate measurement reflects accumulated dynamics, not just
    // the first 64 timesteps from a cold start.
    let mut states = net.fresh_states();
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut rate_acc = vec![0.0_f32; net.layers.len()];
    let mut mem_acc: Vec<(f32, f32)> = vec![(0.0, f32::NEG_INFINITY); net.layers.len()];
    let mut chunks_seen = 0_usize;
    let mut pos = 0;
    while pos + chunk < val_toks.len() {
        let inputs = &val_toks[pos..pos + chunk];
        let targets = &val_toks[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_tokens_stateful(inputs, &mut states);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            let max = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp =
                max + logits_t.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            total_nats += log_sum_exp - logits_t[target as usize];
        }
        for (acc, r) in rate_acc.iter_mut().zip(cache.firing_rates()) {
            *acc += r;
        }
        for (acc, (m, p)) in mem_acc.iter_mut().zip(cache.membrane_stats()) {
            acc.0 += m;
            if p > acc.1 {
                acc.1 = p;
            }
        }
        chunks_seen += 1;
        total_tokens += chunk;
        pos += chunk;
    }
    let mean_rates: Vec<f32> = rate_acc
        .into_iter()
        .map(|s| if chunks_seen == 0 { 0.0 } else { s / chunks_seen as f32 })
        .collect();
    let mean_mem: Vec<(f32, f32)> = mem_acc
        .into_iter()
        .map(|(m, p)| {
            (
                if chunks_seen == 0 { 0.0 } else { m / chunks_seen as f32 },
                p,
            )
        })
        .collect();
    let bpb = bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize);
    (bpb, mean_rates, mean_mem)
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
    cli: &Cli,
    corpus_bytes: usize,
    n_tokens: usize,
    vocab: usize,
    n_train: usize,
    n_val: usize,
    encode_s: f64,
    start_epoch: u32,
    resumed: bool,
    backend: &str,
) {
    let bar_h = "═".repeat(BOX_WIDTH);
    println!("╔{bar_h}╗");
    let title = if cli.sdsa {
        "Aurora · 2-layer gated SNN · BPE · SDSA"
    } else {
        "Aurora · 2-layer gated SNN · BPE"
    };
    println!("{}", center(title));
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
        "{vocab} → ({hidden} × {n_layers} layers) → {vocab}",
        hidden = cli.hidden,
        n_layers = cli.n_layers,
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
        kv(
            "epochs",
            &format!("{start_epoch} → {} ({})", cli.epochs, if resumed { "resumed" } else { "fresh" })
        )
    );
    println!("{}", kv("checkpoint", &cli.checkpoint.display().to_string()));
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

fn epoch_open(epoch: u32) {
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
        " epoch done — train bpb {train:>5.3}  val bpb {val:>5.3}  ({elapsed:>5.1}s)"
    );
    println!("{}", line(&body));
}

/// Sprint 30/31: per-layer health diagnostics. Healthy rates land in
/// 0.05–0.20; if `mem` peak stays well below θ=1.0, the layer is dead
/// (gradient signal blocked at the surrogate).
fn epoch_diagnostics(rates: &[f32], alphas: &[f32], mem: &[(f32, f32)]) {
    let rates_s = rates
        .iter()
        .map(|r| format!("{r:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    let alphas_s = alphas
        .iter()
        .map(|a| format!("{a:.2}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mem_s = mem
        .iter()
        .map(|(m, p)| format!("{m:+.3}/{p:+.2}"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = format!(" rates [{rates_s}]  α [{alphas_s}]");
    println!("{}", line(&body));
    let body2 = format!(" v mean/peak [{mem_s}]");
    println!("{}", line(&body2));
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

// =====================================================================
// Sprint 36: structured metrics writer (JSON Lines).
//
// One JSON object per line. Schema is hand-rolled to avoid a serde_json
// dep — strings are static (no escaping needed beyond `\n` in samples)
// and numbers are float-formatted with `{}` (NaN/Inf become "NaN"/"inf"
// strings, deliberately invalid JSON so analysis script flags them).
// =====================================================================

use std::fs::File;
use std::io::{BufWriter, Write};

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
        start_epoch: u32,
        gpu_name: &str,
    ) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"run","corpus":{corpus:?},"corpus_bytes":{corpus_bytes},"n_tokens":{n_tokens},"vocab":{actual_vocab},"hidden":{hidden},"chunk":{chunk},"lr":{lr},"epochs":{epochs},"start_epoch":{start_epoch},"checkpoint":{ckpt:?},"gpu":{gpu:?}}}"#,
            corpus = cli.corpus.display().to_string(),
            hidden = cli.hidden,
            chunk = cli.chunk,
            lr = format_finite(cli.lr),
            epochs = cli.epochs,
            ckpt = cli.checkpoint.display().to_string(),
            gpu = gpu_name,
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
        // Flush so `tail -f` reflects training progress with sub-second
        // lag and SIGTERM-killed runs leave a usable JSONL trail.
        let _ = self.out.flush();
    }

    #[allow(clippy::too_many_arguments)]
    fn epoch_done(
        &mut self,
        epoch: u32,
        train_bpb: f32,
        val_bpb: f32,
        rates: &[f32],
        alphas: &[f32],
        v_means: &[f32],
        v_peaks: &[f32],
        elapsed_s: f64,
    ) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"epoch_done","epoch":{epoch},"train_bpb":{train},"val_bpb":{val},"rates":{rates},"alphas":{alphas},"v_means":{v_means},"v_peaks":{v_peaks},"elapsed_s":{elapsed:.3}}}"#,
            train = format_finite(train_bpb),
            val = format_finite(val_bpb),
            rates = format_array(rates),
            alphas = format_array(alphas),
            v_means = format_array(v_means),
            v_peaks = format_array(v_peaks),
            elapsed = elapsed_s,
        );
        let _ = self.out.flush();
    }

    fn apoptosis_kill(&mut self, epoch: u32, ev: &aurora::apoptosis::KillEvent) {
        let _ = writeln!(
            self.out,
            r#"{{"event":"apoptosis_kill","epoch":{epoch},"chunk":{chunk},"layer":{layer},"neuron":{neuron},"rate":{rate}}}"#,
            chunk = ev.chunk,
            layer = ev.layer,
            neuron = ev.neuron,
            rate = format_finite(ev.rate_at_death),
        );
        let _ = self.out.flush();
    }

    fn apoptosis_state(&mut self, epoch: u32, chunk: usize, dead_per_layer: &[usize]) {
        let mut s = String::from("[");
        for (i, d) in dead_per_layer.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&d.to_string());
        }
        s.push(']');
        let _ = writeln!(
            self.out,
            r#"{{"event":"apoptosis_state","epoch":{epoch},"chunk":{chunk},"dead":{dead}}}"#,
            dead = s,
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
        // Deliberately emit invalid JSON token so the parser fails
        // loudly. Typical SNN training failure: NaN loss after runaway.
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
