//! Standalone checkpoint evaluation — sliding-window val BPB without
//! training a single step. Decouples eval protocol (window length) from
//! the training run that produced the checkpoint:
//!
//!   cargo run --release --features gpu --example eval_ckpt -- \
//!     --checkpoint data/foo.ckpt --corpus data/enwik8.txt \
//!     --bpe data/enwik8_10m_4096.bpe --seq-len 256 --gpu

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
#[cfg(feature = "gpu")]
use aurora::gpu::{DecoderTransformerGpu, GpuContext};
use aurora::text_io::{bits_per_byte, load_corpus};
use aurora::transformer::DecoderTransformer;

fn main() -> ExitCode {
    let mut ckpt_path: Option<PathBuf> = None;
    let mut corpus_path: Option<PathBuf> = None;
    let mut bpe_path: Option<PathBuf> = None;
    let mut seq_len: usize = 256;
    let mut gpu = false;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                ckpt_path = args.get(i + 1).map(Into::into);
                i += 2;
            }
            "--corpus" => {
                corpus_path = args.get(i + 1).map(Into::into);
                i += 2;
            }
            "--bpe" => {
                bpe_path = args.get(i + 1).map(Into::into);
                i += 2;
            }
            "--seq-len" => {
                seq_len = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(256);
                i += 2;
            }
            "--gpu" => {
                gpu = true;
                i += 1;
            }
            other => {
                eprintln!("unknown flag {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (Some(ckpt_path), Some(corpus_path), Some(bpe_path)) =
        (ckpt_path, corpus_path, bpe_path)
    else {
        eprintln!("usage: --checkpoint C --corpus F --bpe B [--seq-len N] [--gpu]");
        return ExitCode::FAILURE;
    };

    let ckpt = match checkpoint::load(&ckpt_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = ckpt.model;
    if seq_len > model.config.seq_len_max {
        eprintln!(
            "--seq-len {seq_len} > model seq_len_max {} — unsupported",
            model.config.seq_len_max
        );
        return ExitCode::FAILURE;
    }

    let corpus = load_corpus(&corpus_path).expect("corpus");
    let bpe = BpeTokenizer::load(&bpe_path).expect("bpe");
    let tokens = bpe.encode(&corpus);
    let comp = corpus.len() as f32 / tokens.len() as f32;
    let n_train = (tokens.len() * 9) / 10;
    let val_toks = &tokens[n_train..];
    println!(
        "ckpt {} (epoch {}, step {}) — val {} tokens, eval seq_len {seq_len}",
        ckpt_path.display(),
        ckpt.epoch,
        ckpt.step,
        val_toks.len()
    );

    let t0 = Instant::now();
    #[cfg(feature = "gpu")]
    let bpb = if gpu {
        match GpuContext::new() {
            Ok(ctx) => {
                let g = DecoderTransformerGpu::from_cpu(&model, ctx, 16);
                evaluate_gpu(&g, val_toks, seq_len, comp)
            }
            Err(e) => {
                eprintln!("GPU init failed ({e}); CPU fallback");
                evaluate_cpu(&model, val_toks, seq_len, comp)
            }
        }
    } else {
        evaluate_cpu(&model, val_toks, seq_len, comp)
    };
    #[cfg(not(feature = "gpu"))]
    let bpb = {
        let _ = gpu;
        evaluate_cpu(&model, val_toks, seq_len, comp)
    };

    println!("val bpb {bpb:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

/// Overlapping windows, stride = seq_len/2 — identical protocol to the
/// trainer's `evaluate` (window 0 scores everything, later windows only
/// their suffix).
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

fn evaluate_cpu(model: &DecoderTransformer, val_toks: &[u32], seq_len: usize, comp: f32) -> f32 {
    use rayon::prelude::*;
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
                let logp = (row[target[ti] as usize] - max - sum.ln()).max(-27.631_021);
                nats += -logp;
            }
            (nats, seq_len - scored_from)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}

#[cfg(feature = "gpu")]
fn evaluate_gpu(
    gpu: &DecoderTransformerGpu,
    val_toks: &[u32],
    seq_len: usize,
    comp: f32,
) -> f32 {
    let (starts, scored_tail) = eval_windows(val_toks.len(), seq_len);
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    for (ci, chunk) in starts.chunks(gpu.max_batch).enumerate() {
        let inputs: Vec<&[u32]> = chunk.iter().map(|&p| &val_toks[p..p + seq_len]).collect();
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
