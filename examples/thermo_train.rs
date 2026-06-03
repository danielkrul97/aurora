#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::too_many_lines)]

//! Thermodynamic-LM training entry point (Fáze T0/T0.5 — see
//! `docs/13_thermo_lm_plan.md`).
//!
//! Trains the Level-B gas-substrate LM by rollout BPTT and reports val
//! bits-per-byte against the bigram baseline (the §6 kill-gate). With
//! `--viz` (requires `--features viz`) it opens a live Bevy window showing
//! the gas field's Temperature/Density/Energy/|Velocity| heatmaps, a
//! train-bpb curve with the bigram baseline, and the input/prediction.
//!
//! Usage:
//!   cargo run --release --features viz --example thermo_train -- \
//!     --corpus data/tiny_200k.txt --vocab 256 --depth 4 --grid 16 \
//!     --k-steps 4 --dt 0.1 --pool 1 --seq-len 16 --batch-size 24 \
//!     --steps 1500 --lr 2.5e-3 --viz

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::optimizer::{AdamWConfig, MuonConfig, MuonScratch};
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::model::{ThermoGrads, ThermoLM};
use aurora::transformer::cosine_warmup;
use aurora::transformer::loss::softmax;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[cfg(feature = "viz")]
use aurora::thermo::field::{CH_T, CH_UX, CH_UY, CH_UZ};

#[derive(Clone)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    depth: usize,
    grid: usize,
    channels: usize,
    conv_hidden: usize,
    k_steps: usize,
    pool: usize,
    dt: f32,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    weight_decay: f32,
    grad_clip: f32,
    seed: u64,
    eval_every: usize,
    eval_windows: usize,
    sample_len: usize,
    sample_temp: f32,
    viz: bool,
    readout: String,
    modes: usize,
    no_conserve: bool,
    no_diffusion: bool,
    k_start: usize,
    k_warmup: usize,
    gpu: bool,
    optimizer: String,
    muon_lr: f32,
    muon_momentum: f32,
    muon_ns: usize,
    anneal_tmax: f32,
    anneal_alpha: f32,
    t_init: f32,
    inject_every: bool,
    no_unigram_bias: bool,
    aux_weight: f32,
    delta_readout: bool,
    vortex_beta: f32,
    vortex_swirl: f32,
    inject_ring: f32,
    read_horizon: f32,
    sink_radius: f32,
    sink_strength: f32,
    diff_k0: f32,
}

impl Cli {
    /// Features the GPU rollout does not implement — force CPU when any is on.
    fn cpu_only_features(&self) -> bool {
        self.aux_weight > 0.0
            || self.delta_readout
            || self.vortex_beta != 0.0
            || self.inject_ring > 0.0
            || self.read_horizon > 0.0
            || (self.sink_radius > 0.0 && self.sink_strength > 0.0)
    }
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut c = Self {
            corpus: "data/tiny_200k.txt".into(),
            vocab: 256,
            depth: 1,
            grid: 16,
            channels: 8,
            conv_hidden: 24,
            k_steps: 4,
            pool: 1,
            dt: 0.1,
            seq_len: 16,
            batch_size: 24,
            steps: 1500,
            lr: 2.5e-3,
            min_lr: 3e-4,
            warmup_steps: 60,
            weight_decay: 0.01,
            grad_clip: 1.0,
            seed: 0xC0FFEE,
            eval_every: 50,
            eval_windows: 256,
            sample_len: 128,
            sample_temp: 0.8,
            viz: false,
            readout: "moments".into(),
            modes: 4,
            no_conserve: false,
            no_diffusion: false,
            k_start: 2,
            k_warmup: 0,
            gpu: false,
            optimizer: "muon".into(),
            muon_lr: 0.04,
            muon_momentum: 0.95,
            muon_ns: 5,
            anneal_tmax: 1.0,
            anneal_alpha: 1.0,
            t_init: 1.0,
            inject_every: false,
            no_unigram_bias: false,
            aux_weight: 0.0,
            delta_readout: false,
            vortex_beta: 0.0,
            vortex_swirl: 1.0,
            inject_ring: 0.0,
            read_horizon: 0.0,
            sink_radius: 0.0,
            sink_strength: 0.0,
            diff_k0: 0.0,
        };
        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let val = || args.get(i + 1).cloned().ok_or_else(|| format!("missing value for {}", args[i]));
            let mut step = 2;
            match args[i].as_str() {
                "--corpus" => c.corpus = val()?.into(),
                "--vocab" => c.vocab = val()?.parse().map_err(|e| format!("{e}"))?,
                "--depth" => c.depth = val()?.parse().map_err(|e| format!("{e}"))?,
                "--grid" => c.grid = val()?.parse().map_err(|e| format!("{e}"))?,
                "--channels" => c.channels = val()?.parse().map_err(|e| format!("{e}"))?,
                "--conv-hidden" => c.conv_hidden = val()?.parse().map_err(|e| format!("{e}"))?,
                "--k-steps" => c.k_steps = val()?.parse().map_err(|e| format!("{e}"))?,
                "--pool" => c.pool = val()?.parse().map_err(|e| format!("{e}"))?,
                "--dt" => c.dt = val()?.parse().map_err(|e| format!("{e}"))?,
                "--seq-len" => c.seq_len = val()?.parse().map_err(|e| format!("{e}"))?,
                "--batch-size" => c.batch_size = val()?.parse().map_err(|e| format!("{e}"))?,
                "--steps" => c.steps = val()?.parse().map_err(|e| format!("{e}"))?,
                "--lr" => c.lr = val()?.parse().map_err(|e| format!("{e}"))?,
                "--min-lr" => c.min_lr = val()?.parse().map_err(|e| format!("{e}"))?,
                "--warmup-steps" => c.warmup_steps = val()?.parse().map_err(|e| format!("{e}"))?,
                "--weight-decay" => c.weight_decay = val()?.parse().map_err(|e| format!("{e}"))?,
                "--grad-clip" => c.grad_clip = val()?.parse().map_err(|e| format!("{e}"))?,
                "--seed" => c.seed = val()?.parse().map_err(|e| format!("{e}"))?,
                "--eval-every" => c.eval_every = val()?.parse().map_err(|e| format!("{e}"))?,
                "--eval-windows" => c.eval_windows = val()?.parse().map_err(|e| format!("{e}"))?,
                "--sample-len" => c.sample_len = val()?.parse().map_err(|e| format!("{e}"))?,
                "--sample-temp" => c.sample_temp = val()?.parse().map_err(|e| format!("{e}"))?,
                "--readout" => c.readout = val()?,
                "--modes" => c.modes = val()?.parse().map_err(|e| format!("{e}"))?,
                "--k-start" => c.k_start = val()?.parse().map_err(|e| format!("{e}"))?,
                "--k-warmup" => c.k_warmup = val()?.parse().map_err(|e| format!("{e}"))?,
                "--optimizer" => c.optimizer = val()?,
                "--muon-lr" => c.muon_lr = val()?.parse().map_err(|e| format!("{e}"))?,
                "--muon-momentum" => c.muon_momentum = val()?.parse().map_err(|e| format!("{e}"))?,
                "--muon-ns" => c.muon_ns = val()?.parse().map_err(|e| format!("{e}"))?,
                "--aux-weight" => c.aux_weight = val()?.parse().map_err(|e| format!("{e}"))?,
                "--anneal-tmax" => c.anneal_tmax = val()?.parse().map_err(|e| format!("{e}"))?,
                "--anneal-alpha" => c.anneal_alpha = val()?.parse().map_err(|e| format!("{e}"))?,
                "--t-init" => c.t_init = val()?.parse().map_err(|e| format!("{e}"))?,
                "--vortex-beta" => c.vortex_beta = val()?.parse().map_err(|e| format!("{e}"))?,
                "--vortex-swirl" => c.vortex_swirl = val()?.parse().map_err(|e| format!("{e}"))?,
                "--inject-ring" => c.inject_ring = val()?.parse().map_err(|e| format!("{e}"))?,
                "--read-horizon" => c.read_horizon = val()?.parse().map_err(|e| format!("{e}"))?,
                "--sink-radius" => c.sink_radius = val()?.parse().map_err(|e| format!("{e}"))?,
                "--sink-strength" => c.sink_strength = val()?.parse().map_err(|e| format!("{e}"))?,
                "--diff-k0" => c.diff_k0 = val()?.parse().map_err(|e| format!("{e}"))?,
                "--viz" => {
                    c.viz = true;
                    step = 1;
                }
                "--no-conserve" => {
                    c.no_conserve = true;
                    step = 1;
                }
                "--no-diffusion" => {
                    c.no_diffusion = true;
                    step = 1;
                }
                "--gpu" => {
                    c.gpu = true;
                    step = 1;
                }
                "--inject-every" => {
                    c.inject_every = true;
                    step = 1;
                }
                "--no-unigram-bias" => {
                    c.no_unigram_bias = true;
                    step = 1;
                }
                "--delta-readout" => {
                    c.delta_readout = true;
                    step = 1;
                }
                other => return Err(format!("unknown flag {other}")),
            }
            i += step;
        }
        if !matches!(c.readout.as_str(), "moments" | "spectral" | "both") {
            return Err(format!("--readout must be moments|spectral|both, got {}", c.readout));
        }
        if !matches!(c.optimizer.as_str(), "adamw" | "muon") {
            return Err(format!("--optimizer must be adamw|muon, got {}", c.optimizer));
        }
        let uses_moments = matches!(c.readout.as_str(), "moments" | "both");
        if uses_moments && c.grid % c.pool != 0 {
            return Err(format!("grid {} not divisible by pool {}", c.grid, c.pool));
        }
        if uses_moments && c.depth % c.pool != 0 {
            return Err(format!("depth {} not divisible by pool {}", c.depth, c.pool));
        }
        Ok(c)
    }

    fn config(&self, vocab: usize) -> ThermoConfig {
        let (readout_moments, readout_spectral) = match self.readout.as_str() {
            "spectral" => (false, true),
            "both" => (true, true),
            _ => (true, false),
        };
        ThermoConfig {
            vocab,
            d: self.depth,
            h: self.grid,
            w: self.grid,
            c: self.channels,
            conv_hidden: self.conv_hidden,
            conserved: if self.no_conserve { Vec::new() } else { vec![CH_RHO, CH_E] },
            k_steps: self.k_steps,
            pool: self.pool,
            dt: self.dt,
            t_init: self.t_init,
            anneal_tmax: self.anneal_tmax,
            anneal_alpha: self.anneal_alpha,
            readout_moments,
            readout_spectral,
            readout_modes: self.modes,
            use_diffusion: !self.no_diffusion,
        }
    }
}

struct TrainCtx {
    cli: Cli,
    cfg: ThermoConfig,
    model: ThermoLM,
    tokens: Vec<u32>,
    n_train: usize,
    comp: f32,
    bpe: BpeTokenizer,
    bigram: f32,
    #[cfg(feature = "viz")]
    snapshot: Option<aurora::viz::ThermoSnapshotArc>,
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

    let bpe = load_or_train_bpe(&corpus, cli.vocab, &cli.corpus);
    let vocab = bpe.vocab_size();
    let tokens = bpe.encode(&corpus);
    let comp = corpus_bytes as f32 / tokens.len() as f32;
    drop(corpus);

    let n_train = tokens.len() * 9 / 10;
    let cfg = cli.config(vocab);
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);
    model.inject_every = cli.inject_every;
    // Start the readout at the marginal token distribution (loss ≈ unigram
    // entropy, not ln V) so capacity goes to structure, not the prior. The
    // GPU path picks this up for free — it uploads `model.readout.bias`.
    if !cli.no_unigram_bias {
        model.readout.bias = unigram_log_bias(&tokens[..n_train], vocab);
    }
    model.aux_weight = cli.aux_weight;
    model.delta_readout = cli.delta_readout;
    model.vortex_beta = cli.vortex_beta;
    model.vortex_swirl = cli.vortex_swirl;
    model.inject_ring_radius = cli.inject_ring;
    model.readout_horizon = cli.read_horizon;
    model.sink_radius = cli.sink_radius;
    model.sink_strength = cli.sink_strength;
    model.rebuild_experimental();
    // Stronger initial diffusion (the cheap-lever stability control vs vortex).
    if cli.diff_k0 > 0.0 {
        model.step.diffusion.k0 = vec![cli.diff_k0; cfg.c];
    }
    let bigram = bigram_bpb(&tokens[..n_train], &tokens[n_train..], vocab, comp);

    println!("════════════════════════════════════════════════════════════");
    println!(" Thermo-LM train · gas substrate · {} params", model.param_count());
    println!("════════════════════════════════════════════════════════════");
    println!(" corpus     : {} ({corpus_bytes} bytes)", cli.corpus.display());
    println!(" tokens     : {} (vocab {vocab}, comp {comp:.2} B/tok)", tokens.len());
    println!(" grid       : {}×{}×{}×{}  conv_hidden {}  K {}  dt {}  pool {}",
        cli.depth, cli.grid, cli.grid, cli.channels, cli.conv_hidden, cli.k_steps, cli.dt, cli.pool);
    let ew_disp = if cli.eval_windows == 0 { "all".into() } else { cli.eval_windows.to_string() };
    println!(" train/loop : seq {} batch {} steps {} lr {:.1e}→{:.1e} warmup {}  eval/{} w{}",
        cli.seq_len, cli.batch_size, cli.steps, cli.lr, cli.min_lr, cli.warmup_steps, cli.eval_every, ew_disp);
    if cli.optimizer == "muon" {
        println!(" optimizer  : muon (matrices, lr {:.1e}) + adamw (gates/biases/inject)", cli.muon_lr);
    } else {
        println!(" optimizer  : adamw");
    }
    println!(" T1 opts    : readout {} (modes {}) | conserve {} | diffusion {} | K {}→{} (warmup {})",
        cli.readout, cli.modes, !cli.no_conserve, !cli.no_diffusion,
        if cli.k_warmup > 0 { cli.k_start } else { cli.k_steps }, cli.k_steps, cli.k_warmup);
    if cli.aux_weight > 0.0 {
        println!(" deep-sup   : aux_weight {:.2} on intermediate sub-step readouts (CPU-only)", cli.aux_weight);
    }
    if cli.delta_readout {
        println!(" delta-ro   : readout on (field_after − field_before)  (CPU-only)");
    }
    if cli.cpu_only_features() && (cli.vortex_beta != 0.0 || cli.inject_ring > 0.0 || cli.read_horizon > 0.0 || cli.sink_radius > 0.0) {
        println!(
            " vortex     : β {:.2} swirl {:.2} | ring r{:.1} | read-horizon r{:.1} | sink r{:.1}×{:.2}  (CPU-only)",
            cli.vortex_beta, cli.vortex_swirl, cli.inject_ring, cli.read_horizon, cli.sink_radius, cli.sink_strength
        );
    }
    println!(" BASELINE   : bigram val bpb {bigram:.4}   ← must beat this");
    println!("════════════════════════════════════════════════════════════");

    #[cfg(not(feature = "viz"))]
    if cli.viz {
        eprintln!("⚠ --viz given but binary built without `--features viz`; running headless");
    }

    #[cfg(feature = "viz")]
    let snapshot = if cli.viz {
        Some(aurora::viz::new_thermo_snapshot(
            cfg.h,
            cfg.w,
            cfg.k_steps,
            cfg.dt,
            model.param_count(),
            cli.steps as u64,
            bigram,
            // Per quantity: central z-slice + max-over-z projection (3D viz).
            &["Temp z½", "Temp maxZ", "Density z½", "Density maxZ",
              "Energy z½", "Energy maxZ", "|Vel| z½", "|Vel| maxZ"],
        ))
    } else {
        None
    };
    #[cfg(feature = "viz")]
    if let Some(s) = snapshot.as_ref() {
        if let Ok(mut g) = s.lock() {
            g.vortex_beta = cli.vortex_beta;
            g.vortex_swirl = cli.vortex_swirl;
            g.inject_ring = cli.inject_ring;
            g.read_horizon = cli.read_horizon;
            g.sink_radius = cli.sink_radius;
        }
    }

    let ctx = TrainCtx {
        cli,
        cfg,
        model,
        tokens,
        n_train,
        comp,
        bpe,
        bigram,
        #[cfg(feature = "viz")]
        snapshot: {
            #[cfg(feature = "viz")]
            {
                snapshot.as_ref().map(std::sync::Arc::clone)
            }
        },
    };

    #[cfg(feature = "viz")]
    if let Some(snap) = snapshot {
        let handle = std::thread::spawn(move || run_training(ctx));
        aurora::viz::run_thermo_window(snap);
        let _ = handle.join();
        return ExitCode::SUCCESS;
    }

    run_training(ctx);
    ExitCode::SUCCESS
}

fn run_training(ctx: TrainCtx) {
    #[cfg(feature = "gpu")]
    {
        if ctx.cli.gpu && ctx.cli.cpu_only_features() {
            eprintln!("⚠ --aux-weight / --delta-readout / --vortex-* / --inject-ring / --read-horizon / --sink-* are CPU-only; the GPU rollout ignores them — running on CPU");
        } else if ctx.cli.gpu {
            match aurora::gpu::GpuContext::new() {
                Ok(gctx) => {
                    println!(" GPU          : {}", gctx.adapter_info().name);
                    return run_training_gpu(ctx, gctx);
                }
                Err(e) => eprintln!("⚠ GPU init failed: {e}; falling back to CPU"),
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    if ctx.cli.gpu {
        eprintln!("⚠ --gpu given but binary built without `--features gpu`; running on CPU");
    }
    run_training_cpu(ctx);
}

fn run_training_cpu(mut ctx: TrainCtx) {
    let (train_toks, val_toks) = ctx.tokens.split_at(ctx.n_train);
    let cli = &ctx.cli;
    let cfg = &ctx.cfg;

    let stream_len = train_toks.len() / cli.batch_size;
    let n_windows = stream_len.saturating_sub(1) / cli.seq_len.max(1);
    if n_windows == 0 {
        eprintln!("corpus too small for batch×seq_len");
        return;
    }

    let mut grads = ThermoGrads::zeros(cfg);
    let mut batch_grads: Vec<ThermoGrads> = (0..cli.batch_size).map(|_| ThermoGrads::zeros(cfg)).collect();
    let adam_base = AdamWConfig { weight_decay: cli.weight_decay, ..AdamWConfig::default() };
    let use_muon = cli.optimizer == "muon";
    let mut muon_scratch = MuonScratch::new();
    let start = Instant::now();
    let mut best_val = f32::INFINITY;
    let mut last_val = f32::INFINITY;

    for step in 0..cli.steps {
        let lr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.lr, cli.min_lr);
        let adam = AdamWConfig { lr, ..adam_base };
        let window = step % n_windows;

        // K-curriculum: ramp k_start → k_steps over k_warmup steps (§5.4).
        let k = if cli.k_warmup == 0 {
            cli.k_steps
        } else {
            let frac = (step as f32 / cli.k_warmup as f32).min(1.0);
            ((cli.k_start as f32 + (cli.k_steps as f32 - cli.k_start as f32) * frac).round() as usize).max(1)
        };
        ctx.model.config.k_steps = k;

        let model_ref = &ctx.model;
        let batch_loss: f32 = batch_grads
            .par_iter_mut()
            .enumerate()
            .map(|(b, g)| {
                g.zero();
                let p = b * stream_len + window * cli.seq_len;
                let input = &train_toks[p..p + cli.seq_len];
                let target = &train_toks[p + 1..p + cli.seq_len + 1];
                let cache = model_ref.forward(input);
                model_ref.backward(&cache, target, g)
            })
            .sum();

        grads.zero();
        for g in &batch_grads {
            grads.add_assign(g);
        }
        grads.scale(1.0 / cli.batch_size as f32);
        let gn = ThermoLM::clip_grads(&mut grads, cli.grad_clip);
        if use_muon {
            let mlr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.muon_lr, cli.muon_lr * 0.1);
            let mcfg = MuonConfig { lr: mlr, momentum: cli.muon_momentum, weight_decay: cli.weight_decay, ns_steps: cli.muon_ns, ..MuonConfig::default() };
            ctx.model.step_muon_hybrid(&grads, &adam, &mcfg, &mut muon_scratch);
        } else {
            ctx.model.step_adamw(&grads, &adam);
        }

        let avg_loss = batch_loss / cli.batch_size as f32;
        let train_bpb = bits_per_byte(avg_loss * cli.seq_len as f32, (cli.seq_len as f32 * ctx.comp) as usize);

        let do_eval = step == 0 || (step + 1) % cli.eval_every == 0 || step + 1 == cli.steps;
        if do_eval {
            last_val = evaluate(&ctx.model, val_toks, cli.seq_len, ctx.comp, cli.eval_windows);
            best_val = best_val.min(last_val);
            println!(
                "  step {:>4}/{:<4}  loss {:.3}  train_bpb {:.3}  val_bpb {:.3}  lr {:.1e}  gn {:.2}  {:>5.1}s",
                step + 1, cli.steps, avg_loss, train_bpb, last_val, lr, gn, start.elapsed().as_secs_f64()
            );
        }

        #[cfg(feature = "viz")]
        if let Some(snap) = ctx.snapshot.as_ref() {
            let sample = &train_toks[0..cli.seq_len];
            let sample_target = train_toks[cli.seq_len];
            publish(snap, &ctx.model, cfg, &ctx.bpe, sample, sample_target,
                (step + 1) as u64, cli.steps as u64, avg_loss, train_bpb,
                last_val, best_val, lr, gn, start.elapsed().as_secs_f64());
        }
    }

    let final_val = evaluate(&ctx.model, val_toks, cli.seq_len, ctx.comp, 0);
    best_val = best_val.min(final_val);
    println!("════════════════════════════════════════════════════════════");
    println!(" RESULT     : thermo best val bpb {best_val:.4}  vs  bigram {:.4}", ctx.bigram);
    if best_val < ctx.bigram {
        println!(" ✅ KILL-GATE PASSED — thermo beats bigram by {:.4} bpb", ctx.bigram - best_val);
    } else {
        println!(" ❌ KILL-GATE FAILED — does not beat bigram (Δ {:.4})", best_val - ctx.bigram);
    }
    println!("════════════════════════════════════════════════════════════");

    let mut rng = StdRng::seed_from_u64(ctx.cli.seed ^ 0xABCD);
    let seed: Vec<u32> = val_toks.iter().take(8).copied().collect();
    let cont = generate(&ctx.model, &seed, ctx.cli.sample_len, ctx.cli.sample_temp, &mut rng);
    println!(" seed: {:?}", String::from_utf8_lossy(&ctx.bpe.decode(&seed)));
    println!(" cont: {:?}", String::from_utf8_lossy(&ctx.bpe.decode(&cont)));

    #[cfg(feature = "viz")]
    if let Some(snap) = ctx.snapshot.as_ref() {
        if let Ok(mut s) = snap.lock() {
            s.finished = true;
        }
    }
}

#[cfg(feature = "gpu")]
fn run_training_gpu(mut ctx: TrainCtx, gctx: aurora::gpu::GpuContext) {
    use aurora::gpu::ThermoGpu;
    let (train_toks, val_toks) = ctx.tokens.split_at(ctx.n_train);
    let seq = ctx.cli.seq_len;
    let b = ctx.cli.batch_size.max(1);
    // Decorrelated streams, mirroring the CPU batch layout.
    let stream_len = train_toks.len() / b;
    let n_windows = stream_len.saturating_sub(1) / seq.max(1);
    if n_windows == 0 {
        eprintln!("corpus too small for batch×seq_len");
        return;
    }
    let mut gpu = ThermoGpu::new(&gctx, &ctx.model, seq, b);
    gpu.inject_every = ctx.cli.inject_every;
    let adam_base = AdamWConfig { weight_decay: ctx.cli.weight_decay, ..AdamWConfig::default() };
    let use_muon = ctx.cli.optimizer == "muon";
    let muon_base = MuonConfig { momentum: ctx.cli.muon_momentum, weight_decay: ctx.cli.weight_decay, ns_steps: ctx.cli.muon_ns, ..MuonConfig::default() };
    let start = Instant::now();
    let mut best_val = f32::INFINITY;
    let mut last_val = f32::INFINITY;

    // Flattened [T, B] window matrices (row = ti*B + bi), refilled per step.
    let mut toks = vec![0u32; seq * b];
    let mut tgts = vec![0u32; seq * b];

    for step in 0..ctx.cli.steps {
        let lr = cosine_warmup(step, ctx.cli.warmup_steps, ctx.cli.steps.max(1), ctx.cli.lr, ctx.cli.min_lr);
        let adam = AdamWConfig { lr, ..adam_base };
        let window = step % n_windows;
        for bi in 0..b {
            let base = bi * stream_len + window * seq;
            for ti in 0..seq {
                toks[ti * b + bi] = train_toks[base + ti];
                tgts[ti * b + bi] = train_toks[base + ti + 1];
            }
        }

        let do_eval = step == 0 || (step + 1) % ctx.cli.eval_every == 0 || step + 1 == ctx.cli.steps;
        // Loss readback and `to_cpu()` both block on the GPU, so do them only
        // on logging steps — per-step sync is what starved the device.
        let loss = if use_muon {
            let mlr = cosine_warmup(step, ctx.cli.warmup_steps, ctx.cli.steps.max(1), ctx.cli.muon_lr, ctx.cli.muon_lr * 0.1);
            let mcfg = MuonConfig { lr: mlr, ..muon_base };
            gpu.train_step_muon(&toks, &tgts, &adam, &mcfg, do_eval)
        } else {
            gpu.train_step(&toks, &tgts, &adam, do_eval)
        };

        if do_eval {
            // On-device eval: no param download, no CPU rollout (H1).
            let (nats, ntok) = gpu.eval_nats(val_toks, ctx.cli.eval_windows);
            last_val = bits_per_byte(nats as f32, (ntok as f32 * ctx.comp) as usize);
            best_val = best_val.min(last_val);
            let loss = loss.unwrap_or(f32::NAN);
            let train_bpb = bits_per_byte(loss * seq as f32, (seq as f32 * ctx.comp) as usize);
            println!(
                "  step {:>4}/{:<4}  loss {:.3}  train_bpb {:.3}  val_bpb {:.3}  lr {:.1e}  {:>5.1}s  [GPU B={}]",
                step + 1, ctx.cli.steps, loss, train_bpb, last_val, lr, start.elapsed().as_secs_f64(), b
            );

            #[cfg(feature = "viz")]
            if let Some(snap) = ctx.snapshot.as_ref() {
                ctx.model = gpu.to_cpu();
                let cfg = ctx.cfg.clone();
                let sample = &train_toks[0..seq];
                let sample_target = train_toks[seq];
                publish(snap, &ctx.model, &cfg, &ctx.bpe, sample, sample_target,
                    (step + 1) as u64, ctx.cli.steps as u64, loss, train_bpb, last_val, best_val, lr, 0.0, start.elapsed().as_secs_f64());
            }
        }
    }

    let (nats, ntok) = gpu.eval_nats(val_toks, 0);
    let final_val = bits_per_byte(nats as f32, (ntok as f32 * ctx.comp) as usize);
    best_val = best_val.min(final_val);
    ctx.model = gpu.to_cpu(); // snapshot for the sample generation below
    println!("════════════════════════════════════════════════════════════");
    println!(" RESULT (GPU) : thermo best val bpb {best_val:.4}  vs  bigram {:.4}", ctx.bigram);
    if best_val < ctx.bigram {
        println!(" ✅ KILL-GATE PASSED — beats bigram by {:.4} bpb", ctx.bigram - best_val);
    } else {
        println!(" ❌ does not beat bigram (Δ {:.4})", best_val - ctx.bigram);
    }
    println!("════════════════════════════════════════════════════════════");

    let mut rng = StdRng::seed_from_u64(ctx.cli.seed ^ 0xABCD);
    let seed: Vec<u32> = val_toks.iter().take(8).copied().collect();
    let cont = generate(&ctx.model, &seed, ctx.cli.sample_len, ctx.cli.sample_temp, &mut rng);
    println!(" cont: {:?}", String::from_utf8_lossy(&ctx.bpe.decode(&cont)));

    #[cfg(feature = "viz")]
    if let Some(snap) = ctx.snapshot.as_ref() {
        if let Ok(mut s) = snap.lock() {
            s.finished = true;
        }
    }
}

#[cfg(feature = "viz")]
#[allow(clippy::too_many_arguments)]
fn publish(
    snap: &aurora::viz::ThermoSnapshotArc,
    model: &ThermoLM,
    cfg: &ThermoConfig,
    bpe: &BpeTokenizer,
    sample_input: &[u32],
    sample_target: u32,
    step: u64,
    total: u64,
    loss: f32,
    train_bpb: f32,
    val_bpb: f32,
    best_val: f32,
    lr: f32,
    gn: f32,
    elapsed: f64,
) {
    let (logits, field) = model.forward_field(sample_input);
    let channels = extract_channels(&field, cfg);
    let v = cfg.vocab;
    let last = &logits[(sample_input.len() - 1) * v..sample_input.len() * v];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i) as u32;

    let Ok(mut s) = snap.try_lock() else { return };
    s.step = step;
    s.total_steps = total;
    s.loss = loss;
    s.train_bpb = train_bpb;
    s.val_bpb = val_bpb;
    s.best_val = best_val;
    s.lr = lr;
    s.grad_norm = gn;
    s.elapsed_s = elapsed;
    s.channels = channels;
    s.input_preview = String::from_utf8_lossy(&bpe.decode(sample_input)).into_owned();
    s.pred_preview = String::from_utf8_lossy(&bpe.decode(&[argmax])).into_owned();
    s.pred_correct = Some(argmax == sample_target);
    s.push_bpb(step, train_bpb);
    s.step_tick = s.step_tick.wrapping_add(1);
}

/// Reduce the `D×H×W` field to per-quantity `H×W` views for the 2D-texture
/// heatmaps: the central z-slice and the max-over-z projection (the §6 "3D
/// řezy" choice). `D=1` makes both views identical to the single plane.
#[cfg(feature = "viz")]
fn extract_channels(field: &[f32], cfg: &ThermoConfig) -> Vec<aurora::viz::ThermoChannel> {
    let (d, h, w, c) = (cfg.d, cfg.h, cfg.w, cfg.c);
    let hw = h * w;
    let zc = d / 2;
    let mk = |name: &str, vals: Vec<f32>| {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &vals {
            if x.is_finite() {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        aurora::viz::ThermoChannel { name: name.to_string(), data: vals, vmin: lo, vmax: hi }
    };
    let mut out = Vec::with_capacity(8);
    let mut views = |name: &str, val: &dyn Fn(usize) -> f32| {
        let mut slice = vec![0.0_f32; hw];
        let mut proj = vec![f32::NEG_INFINITY; hw];
        for z in 0..d {
            let base = z * hw;
            for p in 0..hw {
                let v = val(base + p);
                if z == zc {
                    slice[p] = v;
                }
                proj[p] = proj[p].max(v);
            }
        }
        out.push(mk(&format!("{name} z½"), slice));
        out.push(mk(&format!("{name} maxZ"), proj));
    };
    views("Temp", &|cell| field[cell * c + CH_T]);
    views("Density", &|cell| field[cell * c + CH_RHO]);
    views("Energy", &|cell| field[cell * c + CH_E]);
    views("|Vel|", &|cell| {
        (field[cell * c + CH_UX].powi(2)
            + field[cell * c + CH_UY].powi(2)
            + field[cell * c + CH_UZ].powi(2))
        .sqrt()
    });
    out
}

/// Add-1 smoothed unigram log-prior over `train`, one entry per vocab id.
/// Used to warm-start the readout bias at the marginal distribution.
fn unigram_log_bias(train: &[u32], vocab: usize) -> Vec<f32> {
    let mut counts = vec![0u64; vocab];
    for &t in train {
        counts[t as usize] += 1;
    }
    let denom = train.len() as f64 + vocab as f64;
    counts
        .iter()
        .map(|&c| (((c as f64) + 1.0) / denom).ln() as f32)
        .collect()
}

/// Add-1 smoothed bigram model trained on `train`, evaluated on `val`.
fn bigram_bpb(train: &[u32], val: &[u32], vocab: usize, comp: f32) -> f32 {
    let mut counts = vec![0u32; vocab * vocab];
    let mut row_tot = vec![0u32; vocab];
    for w in train.windows(2) {
        counts[w[0] as usize * vocab + w[1] as usize] += 1;
        row_tot[w[0] as usize] += 1;
    }
    let mut nats = 0.0_f32;
    for w in val.windows(2) {
        let (prev, next) = (w[0] as usize, w[1] as usize);
        let p = (counts[prev * vocab + next] as f32 + 1.0) / (row_tot[prev] as f32 + vocab as f32);
        nats += -p.ln();
    }
    bits_per_byte(nats, ((val.len() - 1) as f32 * comp) as usize)
}

fn evaluate(model: &ThermoLM, val_toks: &[u32], seq_len: usize, comp: f32, max_windows: usize) -> f32 {
    if val_toks.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let v = model.config.vocab;
    let all: Vec<usize> = (0..)
        .map(|i| i * seq_len)
        .take_while(|&p| p + seq_len + 1 < val_toks.len())
        .collect();
    // Cap CPU cost: evenly stride-sample max_windows across the val set (0 = full).
    let windows: Vec<usize> = if max_windows == 0 || max_windows >= all.len() {
        all
    } else {
        (0..max_windows).map(|i| all[i * all.len() / max_windows]).collect()
    };
    let (nats, toks): (f32, usize) = windows
        .par_iter()
        .map(|&p| {
            let input = &val_toks[p..p + seq_len];
            let target = &val_toks[p + 1..p + seq_len + 1];
            let logits = model.forward(input).logits;
            let mut n = 0.0_f32;
            for ti in 0..seq_len {
                let probs = softmax(&logits[ti * v..(ti + 1) * v]);
                n += -probs[target[ti] as usize].max(1e-12).ln();
            }
            (n, seq_len)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    bits_per_byte(nats, (toks as f32 * comp) as usize)
}

fn generate(model: &ThermoLM, seed: &[u32], n: usize, temp: f32, rng: &mut impl rand::Rng) -> Vec<u32> {
    let mut tokens = seed.to_vec();
    let v = model.config.vocab;
    let max_ctx = 64;
    for _ in 0..n {
        let start = tokens.len().saturating_sub(max_ctx);
        let ctx = &tokens[start..];
        let logits = model.forward(ctx).logits;
        let last = &logits[(ctx.len() - 1) * v..ctx.len() * v];
        tokens.push(sample_token(last, temp, rng));
    }
    tokens[seed.len()..].to_vec()
}

fn load_or_train_bpe(corpus: &[u8], vocab: usize, corpus_path: &Path) -> BpeTokenizer {
    let stem = corpus_path.file_stem().and_then(|s| s.to_str()).unwrap_or("corpus");
    let path = PathBuf::from(format!("data/{stem}_{vocab}.bpe"));
    if let Ok(t) = BpeTokenizer::load(&path) {
        if t.vocab_size() == vocab {
            return t;
        }
    }
    let t = BpeTokenizer::train(corpus, vocab);
    let _ = t.save(&path);
    t
}
