#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::too_many_lines)]

//! Spacetime-LM trainer (the §21 non-traditional-geometry experiment).
//!
//! Sequence laid out as the field axis; processed in PARALLEL (no recurrent
//! carry) — embed-at-time → K causal dilated conv steps → per-position readout.
//! BPTT is only K deep. CPU-only. A/B against the recurrent grid/tree models.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::optimizer::{AdamWConfig, MuonConfig, MuonScratch};
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use aurora::thermo::spacetime_model::{StConfig, StGrads, StLM};
use aurora::transformer::cosine_warmup;
use aurora::transformer::loss::softmax;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[cfg(all(feature = "gpu", feature = "viz"))]
type GpuVizSnap = Option<aurora::viz::ThermoSnapshotArc>;
#[cfg(all(feature = "gpu", not(feature = "viz")))]
type GpuVizSnap = Option<()>;

#[derive(Clone)]
struct Cli {
    corpus: PathBuf,
    vocab: usize,
    channels: usize,
    conv_hidden: usize,
    k_steps: usize,
    taps: usize,
    dilation_base: usize,
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
    muon_lr: f32,
    viz: bool,
    gpu: bool,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut c = Self {
            corpus: "data/tiny.txt".into(),
            vocab: 256,
            channels: 128,
            conv_hidden: 256,
            k_steps: 6,
            taps: 2,
            dilation_base: 2,
            dt: 1.0,
            seq_len: 64,
            batch_size: 24,
            steps: 350,
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
            muon_lr: 0.04,
            viz: false,
            gpu: false,
        };
        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let val = || args.get(i + 1).cloned().ok_or_else(|| format!("missing value for {}", args[i]));
            let mut stp = 2;
            match args[i].as_str() {
                "--corpus" => c.corpus = val()?.into(),
                "--vocab" => c.vocab = val()?.parse().map_err(|e| format!("{e}"))?,
                "--channels" => c.channels = val()?.parse().map_err(|e| format!("{e}"))?,
                "--conv-hidden" => c.conv_hidden = val()?.parse().map_err(|e| format!("{e}"))?,
                "--k-steps" => c.k_steps = val()?.parse().map_err(|e| format!("{e}"))?,
                "--taps" => c.taps = val()?.parse().map_err(|e| format!("{e}"))?,
                "--dilation-base" => c.dilation_base = val()?.parse().map_err(|e| format!("{e}"))?,
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
                "--muon-lr" => c.muon_lr = val()?.parse().map_err(|e| format!("{e}"))?,
                "--viz" => {
                    c.viz = true;
                    stp = 1;
                }
                "--gpu" => {
                    c.gpu = true;
                    stp = 1;
                }
                other => return Err(format!("unknown flag {other}")),
            }
            i += stp;
        }
        Ok(c)
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
    let corpus_bytes = corpus.len();
    let bpe = load_or_train_bpe(&corpus, cli.vocab, &cli.corpus);
    let vocab = bpe.vocab_size();
    let tokens = bpe.encode(&corpus);
    let comp = corpus_bytes as f32 / tokens.len() as f32;
    drop(corpus);

    let n_train = tokens.len() * 9 / 10;
    let cfg = StConfig {
        vocab,
        c: cli.channels,
        conv_hidden: cli.conv_hidden,
        k_steps: cli.k_steps,
        taps: cli.taps,
        dilation_base: cli.dilation_base,
        dt: cli.dt,
    };
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut model = StLM::new(cfg, &mut rng);
    model.ro_bias = unigram_log_bias(&tokens[..n_train], vocab);
    let bigram = bigram_bpb(&tokens[..n_train], &tokens[n_train..], vocab, comp);
    let max_reach = (0..cli.k_steps).map(|k| cli.dilation_base.checked_pow(k as u32).unwrap_or(usize::MAX).min(cli.seq_len)).sum::<usize>();

    println!("════════════════════════════════════════════════════════════");
    println!(" Spacetime-LM (parallel, causal) · {} params", model.param_count());
    println!("════════════════════════════════════════════════════════════");
    println!(" corpus     : {} ({corpus_bytes} bytes)", cli.corpus.display());
    println!(" tokens     : {} (vocab {vocab}, comp {comp:.2} B/tok)", tokens.len());
    println!(" model      : c {} conv_hidden {} K {} taps {} dil_base {} dt {}  (temporal reach ~{})",
        cli.channels, cli.conv_hidden, cli.k_steps, cli.taps, cli.dilation_base, cli.dt, max_reach);
    println!(" train/loop : seq {} batch {} steps {} lr {:.1e}→{:.1e}  muon-lr {:.1e}  eval/{}  (BPTT depth = K = {})",
        cli.seq_len, cli.batch_size, cli.steps, cli.lr, cli.min_lr, cli.muon_lr, cli.eval_every, cli.k_steps);
    println!(" BASELINE   : bigram val bpb {bigram:.4}   ← must beat this");
    println!("════════════════════════════════════════════════════════════");

    #[cfg(feature = "gpu")]
    if cli.gpu {
        match aurora::gpu::GpuContext::new() {
            Ok(gctx) => {
                println!(" GPU        : {}", gctx.adapter_info().name);
                #[cfg(feature = "viz")]
                if cli.viz {
                    let snapshot = aurora::viz::new_thermo_snapshot(
                        cli.channels, cli.seq_len, cli.k_steps, cli.dt, model.param_count(),
                        cli.steps as u64, bigram,
                        &["Field after embed (chan×time)", "Field final (chan×time)"],
                    );
                    let snap2 = std::sync::Arc::clone(&snapshot);
                    let handle = std::thread::spawn(move || run_training_gpu(cli, model, tokens, n_train, comp, bpe, bigram, gctx, Some(snap2)));
                    aurora::viz::run_thermo_window(snapshot);
                    let _ = handle.join();
                    return ExitCode::SUCCESS;
                }
                run_training_gpu(cli, model, tokens, n_train, comp, bpe, bigram, gctx, None);
                return ExitCode::SUCCESS;
            }
            Err(e) => eprintln!("⚠ GPU init failed: {e}; falling back to CPU"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    if cli.gpu {
        eprintln!("⚠ --gpu given but built without --features gpu; running on CPU");
    }

    #[cfg(feature = "viz")]
    let snapshot = if cli.viz {
        Some(aurora::viz::new_thermo_snapshot(
            cli.channels,
            cli.seq_len,
            cli.k_steps,
            cli.dt,
            model.param_count(),
            cli.steps as u64,
            bigram,
            &["Field after embed (chan×time)", "Field final (chan×time)"],
        ))
    } else {
        None
    };
    #[cfg(not(feature = "viz"))]
    if cli.viz {
        eprintln!("⚠ --viz given but built without --features viz; running headless");
    }

    let ctx = Ctx {
        cli,
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

struct Ctx {
    cli: Cli,
    model: StLM,
    tokens: Vec<u32>,
    n_train: usize,
    comp: f32,
    bpe: BpeTokenizer,
    bigram: f32,
    #[cfg(feature = "viz")]
    snapshot: Option<aurora::viz::ThermoSnapshotArc>,
}

fn run_training(mut ctx: Ctx) {
    let cli = ctx.cli.clone();
    let (comp, bigram) = (ctx.comp, ctx.bigram);
    let (train_toks, val_toks) = ctx.tokens.split_at(ctx.n_train);
    let stream_len = train_toks.len() / cli.batch_size;
    let n_windows = stream_len.saturating_sub(1) / cli.seq_len.max(1);
    if n_windows == 0 {
        eprintln!("corpus too small for batch×seq_len");
        return;
    }

    let mut grads = StGrads::zeros(&ctx.model);
    let mut batch_grads: Vec<StGrads> = (0..cli.batch_size).map(|_| StGrads::zeros(&ctx.model)).collect();
    let adam_base = AdamWConfig { weight_decay: cli.weight_decay, ..AdamWConfig::default() };
    let mut muon_scratch = MuonScratch::new();
    let start = Instant::now();
    let mut best_val = f32::INFINITY;
    let mut last_val = f32::INFINITY;

    for step in 0..cli.steps {
        let lr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.lr, cli.min_lr);
        let adam = AdamWConfig { lr, ..adam_base };
        let window = step % n_windows;

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
                model_ref.backward(&cache, input, target, g)
            })
            .sum();

        grads.zero();
        for g in &batch_grads {
            grads.add_assign(g);
        }
        grads.scale(1.0 / cli.batch_size as f32);
        let gn = StLM::clip_grads(&mut grads, cli.grad_clip);
        let mlr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.muon_lr, cli.muon_lr * 0.1);
        let mcfg = MuonConfig { lr: mlr, weight_decay: cli.weight_decay, ..MuonConfig::default() };
        ctx.model.step_muon_hybrid(&grads, &adam, &mcfg, &mut muon_scratch);

        let avg_loss = batch_loss / cli.batch_size as f32;
        let train_bpb = bits_per_byte(avg_loss * cli.seq_len as f32, (cli.seq_len as f32 * comp) as usize);
        let do_eval = step == 0 || (step + 1) % cli.eval_every == 0 || step + 1 == cli.steps;
        if do_eval {
            last_val = evaluate(&ctx.model, val_toks, cli.seq_len, comp, cli.eval_windows);
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
            publish_st(snap, &ctx.model, &cli, &ctx.bpe, sample, sample_target,
                (step + 1) as u64, cli.steps as u64, avg_loss, train_bpb, last_val, best_val, lr, gn,
                start.elapsed().as_secs_f64());
        }
    }

    let final_val = evaluate(&ctx.model, val_toks, cli.seq_len, comp, 0);
    best_val = best_val.min(final_val);
    println!("════════════════════════════════════════════════════════════");
    println!(" RESULT     : spacetime best val bpb {best_val:.4}  vs  bigram {:.4}", bigram);
    if best_val < bigram {
        println!(" ✅ beats bigram by {:.4} bpb", bigram - best_val);
    } else {
        println!(" ❌ does not beat bigram (Δ {:.4})", best_val - bigram);
    }
    println!("════════════════════════════════════════════════════════════");

    let mut grng = StdRng::seed_from_u64(cli.seed ^ 0xABCD);
    let seed: Vec<u32> = val_toks.iter().take(16).copied().collect();
    let cont = generate(&ctx.model, &seed, cli.sample_len, cli.sample_temp, &mut grng);
    println!(" cont: {:?}", String::from_utf8_lossy(&ctx.bpe.decode(&cont)));

    #[cfg(feature = "viz")]
    if let Some(snap) = ctx.snapshot.as_ref() {
        if let Ok(mut s) = snap.lock() {
            s.finished = true;
        }
    }
}

/// Publish the spacetime field as a `[C rows × L cols]` heatmap (channel ×
/// time): you watch each time-column build causally from its past.
#[cfg(feature = "viz")]
#[allow(clippy::too_many_arguments)]
fn publish_st(
    snap: &aurora::viz::ThermoSnapshotArc,
    model: &StLM,
    cli: &Cli,
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
    let cache = model.forward(sample_input);
    let (l, c) = (sample_input.len(), cli.channels);
    let to_panel = |field: &[f32], name: &str| {
        let mut data = vec![0.0_f32; c * l];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for t in 0..l {
            for ci in 0..c {
                let v = field[t * c + ci];
                data[ci * l + t] = v;
                if v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        aurora::viz::ThermoChannel { name: name.to_string(), data, vmin: lo, vmax: hi }
    };
    let after_embed = if cache.caches.is_empty() {
        to_panel(&cache.final_field, "Field after embed (chan×time)")
    } else {
        to_panel(&cache.caches[0].s, "Field after embed (chan×time)")
    };
    let final_ch = to_panel(&cache.final_field, "Field final (chan×time)");

    let v = cli.vocab;
    let last = &cache.logits[(l - 1) * v..l * v];
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
    s.channels = vec![after_embed, final_ch];
    s.input_preview = String::from_utf8_lossy(&bpe.decode(sample_input)).into_owned();
    s.pred_preview = String::from_utf8_lossy(&bpe.decode(&[argmax])).into_owned();
    s.pred_correct = Some(argmax == sample_target);
    s.push_bpb(step, train_bpb);
    s.step_tick = s.step_tick.wrapping_add(1);
}

#[cfg(feature = "gpu")]
fn run_training_gpu(cli: Cli, model: StLM, tokens: Vec<u32>, n_train: usize, comp: f32, bpe: BpeTokenizer, bigram: f32, gctx: aurora::gpu::GpuContext, snapshot: GpuVizSnap) {
    use aurora::gpu::StLmGpu;
    #[cfg(not(feature = "viz"))]
    let _ = &snapshot;
    let (train_toks, val_toks) = tokens.split_at(n_train);
    let (seq, b) = (cli.seq_len, cli.batch_size.max(1));
    let stream_len = train_toks.len() / b;
    let n_windows = stream_len.saturating_sub(1) / seq.max(1);
    if n_windows == 0 {
        eprintln!("corpus too small for batch×seq_len");
        return;
    }
    let mut gpu = StLmGpu::new(&gctx, &model, seq, b);
    let adam_base = AdamWConfig { weight_decay: cli.weight_decay, ..AdamWConfig::default() };
    let muon_base = MuonConfig { weight_decay: cli.weight_decay, ..MuonConfig::default() };
    let start = Instant::now();
    let mut best_val = f32::INFINITY;
    // Per-window-contiguous [B*L] window matrices (row = bi*seq + ti).
    let mut toks = vec![0u32; seq * b];
    let mut tgts = vec![0u32; seq * b];

    for step in 0..cli.steps {
        let lr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.lr, cli.min_lr);
        let mlr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.muon_lr, cli.muon_lr * 0.1);
        let adam = AdamWConfig { lr, ..adam_base };
        let mcfg = MuonConfig { lr: mlr, ..muon_base };
        let window = step % n_windows;
        for bi in 0..b {
            let base = bi * stream_len + window * seq;
            for ti in 0..seq {
                toks[bi * seq + ti] = train_toks[base + ti];
                tgts[bi * seq + ti] = train_toks[base + ti + 1];
            }
        }
        let do_eval = step == 0 || (step + 1) % cli.eval_every == 0 || step + 1 == cli.steps;
        let loss = gpu.train_step_muon(&toks, &tgts, &adam, &mcfg, do_eval);
        if do_eval {
            let (nats, ntok) = gpu.eval_nats(val_toks, cli.eval_windows);
            let val = bits_per_byte(nats as f32, (ntok as f32 * comp) as usize);
            best_val = best_val.min(val);
            let loss = loss.unwrap_or(f32::NAN);
            let train_bpb = bits_per_byte(loss * seq as f32, (seq as f32 * comp) as usize);
            println!(
                "  step {:>4}/{:<4}  loss {:.3}  train_bpb {:.3}  val_bpb {:.3}  lr {:.1e}  {:>5.1}s  [GPU B={}]",
                step + 1, cli.steps, loss, train_bpb, val, lr, start.elapsed().as_secs_f64(), b
            );
            #[cfg(feature = "viz")]
            if let Some(snap) = snapshot.as_ref() {
                let model_cpu = gpu.to_cpu();
                let s_in = &train_toks[0..seq];
                let s_tgt = train_toks[seq];
                publish_st(snap, &model_cpu, &cli, &bpe, s_in, s_tgt, (step + 1) as u64, cli.steps as u64, loss, train_bpb, val, best_val, lr, 0.0, start.elapsed().as_secs_f64());
            }
        }
    }

    let (nats, ntok) = gpu.eval_nats(val_toks, 0);
    let final_val = bits_per_byte(nats as f32, (ntok as f32 * comp) as usize);
    best_val = best_val.min(final_val);
    let model = gpu.to_cpu();
    println!("════════════════════════════════════════════════════════════");
    println!(" RESULT (GPU): spacetime best val bpb {best_val:.4}  vs  bigram {:.4}", bigram);
    if best_val < bigram {
        println!(" ✅ beats bigram by {:.4} bpb", bigram - best_val);
    } else {
        println!(" ❌ does not beat bigram (Δ {:.4})", best_val - bigram);
    }
    println!("════════════════════════════════════════════════════════════");
    let mut grng = StdRng::seed_from_u64(cli.seed ^ 0xABCD);
    let seed: Vec<u32> = val_toks.iter().take(16).copied().collect();
    let cont = generate(&model, &seed, cli.sample_len, cli.sample_temp, &mut grng);
    println!(" cont: {:?}", String::from_utf8_lossy(&bpe.decode(&cont)));

    #[cfg(feature = "viz")]
    if let Some(snap) = snapshot.as_ref() {
        if let Ok(mut s) = snap.lock() {
            s.finished = true;
        }
    }
}

fn unigram_log_bias(train: &[u32], vocab: usize) -> Vec<f32> {
    let mut counts = vec![0u64; vocab];
    for &t in train {
        counts[t as usize] += 1;
    }
    let denom = train.len() as f64 + vocab as f64;
    counts.iter().map(|&c| (((c as f64) + 1.0) / denom).ln() as f32).collect()
}

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

fn evaluate(model: &StLM, val_toks: &[u32], seq_len: usize, comp: f32, max_windows: usize) -> f32 {
    if val_toks.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let v = model.cfg.vocab;
    let all: Vec<usize> = (0..)
        .map(|i| i * seq_len)
        .take_while(|&p| p + seq_len + 1 < val_toks.len())
        .collect();
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

fn generate(model: &StLM, seed: &[u32], n: usize, temp: f32, rng: &mut impl rand::Rng) -> Vec<u32> {
    let mut tokens = seed.to_vec();
    let v = model.cfg.vocab;
    let max_ctx = 96;
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
