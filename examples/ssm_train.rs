#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::too_many_lines)]

//! Selective-SSM LM trainer (the §22 experiment — deep-research top pick).
//!
//! embed → N residual selective-SSM blocks (input-dependent forget gate +
//! linear scan) → per-position readout. Parallel over the window; CPU-only.
//! A/B against the static spacetime model (the 2.78 BPB winner it must beat).

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::optimizer::{AdamWConfig, MuonConfig, MuonScratch};
use aurora::text_io::{bits_per_byte, load_corpus, sample_token};
use aurora::thermo::ssm_model::{SsmConfig, SsmGrads, SsmLM};
use aurora::transformer::cosine_warmup;
use aurora::transformer::loss::softmax;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

struct Cli {
    corpus: PathBuf,
    vocab: usize,
    channels: usize,
    layers: usize,
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
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut c = Self {
            corpus: "data/tiny.txt".into(),
            vocab: 256,
            channels: 128,
            layers: 2,
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
        };
        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let val = || args.get(i + 1).cloned().ok_or_else(|| format!("missing value for {}", args[i]));
            match args[i].as_str() {
                "--corpus" => c.corpus = val()?.into(),
                "--vocab" => c.vocab = val()?.parse().map_err(|e| format!("{e}"))?,
                "--channels" => c.channels = val()?.parse().map_err(|e| format!("{e}"))?,
                "--layers" => c.layers = val()?.parse().map_err(|e| format!("{e}"))?,
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
                other => return Err(format!("unknown flag {other}")),
            }
            i += 2;
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
    let cfg = SsmConfig { vocab, c: cli.channels, n_layers: cli.layers };
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut model = SsmLM::new(cfg, &mut rng);
    model.ro_bias = unigram_log_bias(&tokens[..n_train], vocab);
    let bigram = bigram_bpb(&tokens[..n_train], &tokens[n_train..], vocab, comp);

    println!("════════════════════════════════════════════════════════════");
    println!(" Selective-SSM LM · {} layers · c {} · {} params", cli.layers, cli.channels, model.param_count());
    println!("════════════════════════════════════════════════════════════");
    println!(" corpus     : {} ({corpus_bytes} bytes)", cli.corpus.display());
    println!(" tokens     : {} (vocab {vocab}, comp {comp:.2} B/tok)", tokens.len());
    println!(" train/loop : seq {} batch {} steps {} lr {:.1e}→{:.1e}  muon-lr {:.1e}  eval/{}",
        cli.seq_len, cli.batch_size, cli.steps, cli.lr, cli.min_lr, cli.muon_lr, cli.eval_every);
    println!(" BASELINE   : bigram val bpb {bigram:.4}   ← must beat this (and spacetime 2.78)");
    println!("════════════════════════════════════════════════════════════");

    let (train_toks, val_toks) = tokens.split_at(n_train);
    let stream_len = train_toks.len() / cli.batch_size;
    let n_windows = stream_len.saturating_sub(1) / cli.seq_len.max(1);
    if n_windows == 0 {
        eprintln!("corpus too small for batch×seq_len");
        return ExitCode::FAILURE;
    }

    let mut grads = SsmGrads::zeros(&model);
    let mut batch_grads: Vec<SsmGrads> = (0..cli.batch_size).map(|_| SsmGrads::zeros(&model)).collect();
    let adam_base = AdamWConfig { weight_decay: cli.weight_decay, ..AdamWConfig::default() };
    let mut muon_scratch = MuonScratch::new();
    let start = Instant::now();
    let mut best_val = f32::INFINITY;

    for step in 0..cli.steps {
        let lr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.lr, cli.min_lr);
        let adam = AdamWConfig { lr, ..adam_base };
        let window = step % n_windows;

        let model_ref = &model;
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
        let gn = SsmLM::clip_grads(&mut grads, cli.grad_clip);
        let mlr = cosine_warmup(step, cli.warmup_steps, cli.steps.max(1), cli.muon_lr, cli.muon_lr * 0.1);
        let mcfg = MuonConfig { lr: mlr, weight_decay: cli.weight_decay, ..MuonConfig::default() };
        model.step_muon_hybrid(&grads, &adam, &mcfg, &mut muon_scratch);

        let avg_loss = batch_loss / cli.batch_size as f32;
        let do_eval = step == 0 || (step + 1) % cli.eval_every == 0 || step + 1 == cli.steps;
        if do_eval {
            let train_bpb = bits_per_byte(avg_loss * cli.seq_len as f32, (cli.seq_len as f32 * comp) as usize);
            let val = evaluate(&model, val_toks, cli.seq_len, comp, cli.eval_windows);
            best_val = best_val.min(val);
            println!(
                "  step {:>4}/{:<4}  loss {:.3}  train_bpb {:.3}  val_bpb {:.3}  lr {:.1e}  gn {:.2}  {:>5.1}s",
                step + 1, cli.steps, avg_loss, train_bpb, val, lr, gn, start.elapsed().as_secs_f64()
            );
        }
    }

    let final_val = evaluate(&model, val_toks, cli.seq_len, comp, 0);
    best_val = best_val.min(final_val);
    println!("════════════════════════════════════════════════════════════");
    println!(" RESULT     : ssm best val bpb {best_val:.4}  vs  bigram {:.4}", bigram);
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
    ExitCode::SUCCESS
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

fn evaluate(model: &SsmLM, val_toks: &[u32], seq_len: usize, comp: f32, max_windows: usize) -> f32 {
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

fn generate(model: &SsmLM, seed: &[u32], n: usize, temp: f32, rng: &mut impl rand::Rng) -> Vec<u32> {
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
