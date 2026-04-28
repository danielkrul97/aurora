//! Sprint 28: end-to-end LM quality regression tests.
//!
//! These run as part of `cargo test --release` and catch silent
//! quality/perf regressions that wouldn't show up in unit tests.
//! Targets are loose (true BPB is much lower than the assertion) so
//! these don't get flaky on minor numerical drift, but tight enough
//! to flag e.g. a broken backward pass that lets training stall.

#![allow(clippy::cast_precision_loss)]

use std::path::Path;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedState;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, load_corpus};
use rand::rngs::StdRng;
use rand::SeedableRng;

const GRAD_CLIP: f32 = 1.0;

fn corpus_or_skip(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("skipping: corpus {path} not present");
        return None;
    }
    load_corpus(path).ok()
}

/// Byte-level CPU multi-layer regression. Uses a small corpus + tiny
/// model so the test runs in a few seconds. Asserts val BPB drops
/// well below random (8.0) within one epoch — typical observed value
/// is ~5.0 on this config, so the 6.0 threshold has headroom but still
/// catches a stalled trainer.
#[test]
fn cpu_byte_lm_converges_under_one_epoch() {
    let Some(full) = corpus_or_skip("data/tiny_200k.txt") else { return };
    let corpus = &full[..50_000.min(full.len())];

    let mut rng = StdRng::seed_from_u64(0x000A_170A);
    let mut net = MultiLayerGatedSnn::new(
        256,
        &[64, 64],
        256,
        SpikeFn::default(),
        &mut rng,
    );
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    let n_train = (corpus.len() * 9) / 10;
    let train = &corpus[..n_train];
    let val = &corpus[n_train..];

    let chunk = 32;
    let batch = 8;
    let stream_len = train.len() / batch;
    let n_chunks = stream_len.saturating_sub(1) / chunk;
    let mut states: Vec<Vec<GatedState>> =
        (0..batch).map(|_| net.fresh_states()).collect();

    for ci in 0..n_chunks {
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(batch);
        let mut targets: Vec<&[u8]> = Vec::with_capacity(batch);
        for b in 0..batch {
            let p = b * stream_len + ci * chunk;
            inputs.push(&train[p..p + chunk]);
            targets.push(&train[p + 1..p + chunk + 1]);
        }
        net.forward_backward_batch_stateful(&inputs, &targets, &mut states);
        net.clip_grads(GRAD_CLIP);
        net.step_adam(&adam);
    }

    let val_bpb = eval_byte(&net, val, chunk);
    eprintln!("cpu_byte_lm val BPB = {val_bpb:.3}");
    assert!(
        val_bpb < 6.0,
        "byte-level LM stalled: val BPB {val_bpb:.3} ≥ 6.0 (random=8.0)"
    );
    assert!(val_bpb.is_finite(), "BPB is non-finite — backward broken");
}

/// CPU + BPE regression. Same shape as the byte test, but the corpus
/// is BPE-tokenized first; loss is normalized back to BPB so it's
/// comparable to the byte-level baseline.
#[test]
fn cpu_bpe_lm_converges_under_one_epoch() {
    let Some(full) = corpus_or_skip("data/tiny_200k.txt") else { return };
    let corpus = &full[..50_000.min(full.len())];

    let bpe = BpeTokenizer::train(corpus, 384);
    let tokens = bpe.encode(corpus);
    let comp = (corpus.len() as f32) / (tokens.len() as f32);
    assert!(comp > 1.5, "BPE compression unexpectedly low: {comp:.2}×");

    let vocab = bpe.vocab_size();
    let mut rng = StdRng::seed_from_u64(0xBEE);
    let mut net = MultiLayerGatedSnn::new(
        vocab,
        &[64, 64],
        vocab,
        SpikeFn::default(),
        &mut rng,
    );
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    let n_train = (tokens.len() * 9) / 10;
    let train = &tokens[..n_train];
    let val = &tokens[n_train..];

    let chunk = 32;
    let batch = 8;
    let stream_len = train.len() / batch;
    let n_chunks = stream_len.saturating_sub(1) / chunk;
    let mut states: Vec<Vec<GatedState>> =
        (0..batch).map(|_| net.fresh_states()).collect();

    for ci in 0..n_chunks {
        let mut inputs: Vec<&[u32]> = Vec::with_capacity(batch);
        let mut targets: Vec<&[u32]> = Vec::with_capacity(batch);
        for b in 0..batch {
            let p = b * stream_len + ci * chunk;
            inputs.push(&train[p..p + chunk]);
            targets.push(&train[p + 1..p + chunk + 1]);
        }
        net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(GRAD_CLIP);
        net.step_adam(&adam);
    }

    let val_bpb = eval_token(&net, val, chunk, comp);
    eprintln!("cpu_bpe_lm val BPB = {val_bpb:.3} (comp {comp:.2}×)");
    assert!(
        val_bpb < 6.0,
        "BPE LM stalled: val BPB {val_bpb:.3} ≥ 6.0 (random=8.0)"
    );
    assert!(val_bpb.is_finite(), "BPB is non-finite — backward broken");
}

/// Checkpoint round-trip: train briefly, save, load into a fresh
/// network, and confirm the loaded network reproduces the original's
/// validation loss bit-for-bit.
#[test]
fn checkpoint_round_trip_preserves_loss() {
    let Some(full) = corpus_or_skip("data/tiny_200k.txt") else { return };
    let corpus = &full[..20_000.min(full.len())];

    let mut rng = StdRng::seed_from_u64(0xC4FE);
    let mut net = MultiLayerGatedSnn::new(
        256,
        &[32, 32],
        256,
        SpikeFn::default(),
        &mut rng,
    );
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };
    let chunk = 32;
    let batch = 4;
    let n_train = (corpus.len() * 9) / 10;
    let train = &corpus[..n_train];
    let val = &corpus[n_train..];
    let stream_len = train.len() / batch;
    let n_chunks = stream_len.saturating_sub(1) / chunk;
    let mut states: Vec<Vec<GatedState>> =
        (0..batch).map(|_| net.fresh_states()).collect();

    for ci in 0..n_chunks.min(20) {
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(batch);
        let mut targets: Vec<&[u8]> = Vec::with_capacity(batch);
        for b in 0..batch {
            let p = b * stream_len + ci * chunk;
            inputs.push(&train[p..p + chunk]);
            targets.push(&train[p + 1..p + chunk + 1]);
        }
        net.forward_backward_batch_stateful(&inputs, &targets, &mut states);
        net.clip_grads(GRAD_CLIP);
        net.step_adam(&adam);
    }

    let bpb_before = eval_byte(&net, val, chunk);
    let tmp = std::env::temp_dir().join("aurora_checkpoint_test.ckpt");
    checkpoint::save(&tmp, &net, 1).expect("checkpoint save");
    let loaded = checkpoint::load(&tmp).expect("checkpoint load");
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(loaded.epoch, 1);
    let bpb_after = eval_byte(&loaded.net, val, chunk);
    assert!(
        (bpb_before - bpb_after).abs() < 1e-5,
        "checkpoint round-trip changed val BPB: {bpb_before:.6} → {bpb_after:.6}"
    );
}

fn eval_byte(net: &MultiLayerGatedSnn, val: &[u8], chunk: usize) -> f32 {
    if val.len() < chunk + 1 {
        return f32::INFINITY;
    }
    let mut total_nats = 0.0_f32;
    let mut total_bytes = 0_usize;
    let mut pos = 0;
    while pos + chunk < val.len() {
        let inputs = &val[pos..pos + chunk];
        let targets = &val[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_bytes(inputs);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            let max = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp =
                max + logits_t.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            total_nats += log_sum_exp - logits_t[target as usize];
        }
        total_bytes += chunk;
        pos += chunk;
    }
    bits_per_byte(total_nats, total_bytes)
}

fn eval_token(net: &MultiLayerGatedSnn, val: &[u32], chunk: usize, comp: f32) -> f32 {
    if val.len() < chunk + 1 {
        return f32::INFINITY;
    }
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut pos = 0;
    while pos + chunk < val.len() {
        let inputs = &val[pos..pos + chunk];
        let targets = &val[pos + 1..pos + chunk + 1];
        let cache = net.forward_lm_tokens(inputs);
        for (logits_t, &target) in cache.logits.iter().zip(targets.iter()) {
            let max = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp =
                max + logits_t.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            total_nats += log_sum_exp - logits_t[target as usize];
        }
        total_tokens += chunk;
        pos += chunk;
    }
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}
