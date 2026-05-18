//! Sprint 54: end-to-end LM quality regression tests for the CPU
//! transformer. Catches silent training stalls or backward bugs.
//!
//! Thresholds are loose (true BPB is much lower) so these don't flake
//! on minor numerical drift, but tight enough to catch a broken
//! training step where loss never decreases.

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use std::path::Path;

use aurora::bpe::BpeTokenizer;
use aurora::checkpoint;
use aurora::optimizer::AdamWConfig;
use aurora::text_io::{bits_per_byte, load_corpus};
use aurora::transformer::loss::softmax;
use aurora::transformer::{
    cosine_warmup, DecoderTransformer, ModelGrads, TransformerConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn corpus_or_skip(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("skipping: corpus {path} not present");
        return None;
    }
    load_corpus(path).ok()
}

/// Tiny byte-level training run: confirm val BPB falls well below
/// random (8.0) within ~1 epoch. Expected ~5–6 on this config.
#[test]
fn cpu_transformer_byte_lm_converges_under_one_epoch() {
    let Some(full) = corpus_or_skip("data/tiny_200k.txt") else {
        return;
    };
    let corpus = &full[..50_000.min(full.len())];

    let mut rng = StdRng::seed_from_u64(0xA51A);
    let cfg = TransformerConfig {
        vocab: 256,
        d_model: 64,
        n_layer: 2,
        n_head: 2,
        d_ff: 128,
        seq_len_max: 64,
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut grads = ModelGrads::zeros(&cfg);

    let tokens: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();
    let n_train = (tokens.len() * 9) / 10;
    let train = &tokens[..n_train];
    let val = &tokens[n_train..];

    let seq_len = 32;
    let n_chunks = (train.len().saturating_sub(1)) / seq_len;
    let mut step = 0u64;

    for chunk_idx in 0..n_chunks {
        let lr = cosine_warmup(step as usize, 50, n_chunks, 3e-3, 1e-4);
        let adam = AdamWConfig {
            lr,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        };
        let p = chunk_idx * seq_len;
        if p + seq_len + 1 > train.len() {
            break;
        }
        let input = &train[p..p + seq_len];
        let target = &train[p + 1..p + seq_len + 1];

        let (_logits, cache) = model.forward(input);
        grads.zero();
        let _ = model.backward(&cache, target, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);
        step += 1;
    }

    let val_bpb = eval_bpb(&model, val, seq_len, 1.0);
    assert!(val_bpb < 6.0, "byte transformer val BPB {val_bpb} ≥ 6.0");
}

#[test]
fn cpu_transformer_bpe_lm_converges_under_one_epoch() {
    let Some(corpus) = corpus_or_skip("data/tiny_200k.txt") else {
        return;
    };
    let corpus = &corpus[..50_000.min(corpus.len())];

    let bpe = BpeTokenizer::train(corpus, 384);
    let tokens = bpe.encode(corpus);
    let comp = corpus.len() as f32 / tokens.len() as f32;

    let actual_vocab = bpe.vocab_size();
    let mut rng = StdRng::seed_from_u64(0xB7E5);
    let cfg = TransformerConfig {
        vocab: actual_vocab,
        d_model: 64,
        n_layer: 2,
        n_head: 2,
        d_ff: 128,
        seq_len_max: 32,
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut grads = ModelGrads::zeros(&cfg);

    let n_train = (tokens.len() * 9) / 10;
    let train = &tokens[..n_train];
    let val = &tokens[n_train..];
    let seq_len = 32;
    let n_chunks = (train.len().saturating_sub(1)) / seq_len;
    let mut step = 0u64;

    for chunk_idx in 0..n_chunks {
        let lr = cosine_warmup(step as usize, 50, n_chunks, 3e-3, 1e-4);
        let adam = AdamWConfig {
            lr,
            weight_decay: 0.0,
            ..AdamWConfig::default()
        };
        let p = chunk_idx * seq_len;
        if p + seq_len + 1 > train.len() {
            break;
        }
        let input = &train[p..p + seq_len];
        let target = &train[p + 1..p + seq_len + 1];

        let (_logits, cache) = model.forward(input);
        grads.zero();
        let _ = model.backward(&cache, target, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);
        step += 1;
    }

    let val_bpb = eval_bpb(&model, val, seq_len, comp);
    assert!(val_bpb < 6.0, "BPE transformer val BPB {val_bpb} ≥ 6.0");
}

#[test]
fn checkpoint_v3_round_trip_preserves_loss() {
    let mut rng = StdRng::seed_from_u64(0x99);
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = (0..16u32).collect();
    let input = &tokens[..15];
    let target = &tokens[1..];

    let (_logits_before, cache_before) = model.forward(input);
    let mut grads_before = ModelGrads::zeros(&cfg);
    let loss_before = model.clone().backward(&cache_before, target, &mut grads_before);

    let tmp = std::env::temp_dir().join("aurora_ckpt_v3_roundtrip.bin");
    checkpoint::save(&tmp, &model, 0, model.step).expect("save");

    let loaded = checkpoint::load(&tmp).expect("load");
    let (_logits_after, cache_after) = loaded.model.forward(input);
    let mut grads_after = ModelGrads::zeros(&cfg);
    let loss_after = loaded
        .model
        .clone()
        .backward(&cache_after, target, &mut grads_after);

    let diff = (loss_before - loss_after).abs();
    assert!(
        diff < 1e-5,
        "loss before {loss_before}, after {loss_after}, diff {diff}"
    );

    // Spot-check params are identical.
    for i in 0..16 {
        assert!((model.embed.w_tok[i] - loaded.model.embed.w_tok[i]).abs() < 1e-6);
    }
    for (b1, b2) in model.blocks.iter().zip(loaded.model.blocks.iter()) {
        for i in 0..16 {
            assert!((b1.mha.w_q.w[i] - b2.mha.w_q.w[i]).abs() < 1e-6);
        }
    }

    std::fs::remove_file(&tmp).ok();
}

fn eval_bpb(model: &DecoderTransformer, val: &[u32], seq_len: usize, comp: f32) -> f32 {
    if val.len() <= seq_len + 1 {
        return f32::INFINITY;
    }
    let v = model.config.vocab;
    let mut total_nats = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut p = 0;
    while p + seq_len + 1 < val.len() {
        let input = &val[p..p + seq_len];
        let target = &val[p + 1..p + seq_len + 1];
        let (logits, _) = model.forward(input);
        for ti in 0..seq_len {
            let row = &logits[ti * v..(ti + 1) * v];
            let probs = softmax(row);
            let p_t = probs[target[ti] as usize].max(1e-12);
            total_nats += -p_t.ln();
            total_tokens += 1;
        }
        p += seq_len;
    }
    bits_per_byte(total_nats, (total_tokens as f32 * comp) as usize)
}
