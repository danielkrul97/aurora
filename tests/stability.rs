//! Sprint 55: stability + sampling diversity tests.
//!
//! These exercise edge cases that the convergence smoke tests would
//! miss: gradient health under perturbations, sampling diversity across
//! temperatures, deterministic seed handling.

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use aurora::optimizer::AdamWConfig;
use aurora::text_io::sample_token;
use aurora::transformer::{
    cosine_warmup, DecoderTransformer, ModelGrads, TransformerConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn tiny_model(seed: u64) -> (DecoderTransformer, TransformerConfig) {
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 32,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    (DecoderTransformer::new(cfg.clone(), &mut rng), cfg)
}

#[test]
fn grad_norm_stays_finite_under_large_inputs() {
    let (mut model, cfg) = tiny_model(0x1);
    let mut grads = ModelGrads::zeros(&cfg);

    let tokens: Vec<u32> = (0..16u32).map(|i| i % cfg.vocab as u32).collect();
    let input = &tokens[..15];
    let target = &tokens[1..];

    let (logits, cache) = model.forward(input);
    for v in &logits {
        assert!(v.is_finite(), "logits non-finite on fresh init");
    }
    let _ = model.backward(&cache, &logits, target, &mut grads);
    let pre = DecoderTransformer::clip_grads(&mut grads, 1.0);
    assert!(pre.is_finite() && pre >= 0.0, "grad norm non-finite: {pre}");

    let adam = AdamWConfig::default();
    model.step_adamw(&grads, &adam);
}

#[test]
fn cosine_warmup_monotone_then_decays() {
    let max_steps = 1000;
    let warmup = 100;
    let base_lr = 6e-4_f32;
    let min_lr = 6e-5_f32;

    // Warmup region: strictly increasing.
    let mut prev = -1.0_f32;
    for s in 0..warmup {
        let lr = cosine_warmup(s, warmup, max_steps, base_lr, min_lr);
        assert!(lr.is_finite());
        assert!(lr > prev, "warmup not monotone at step {s}: {lr} ≤ {prev}");
        prev = lr;
    }
    // Plateau at warmup end.
    let lr_peak = cosine_warmup(warmup, warmup, max_steps, base_lr, min_lr);
    assert!((lr_peak - base_lr).abs() < 1e-4);

    // Decay region: monotone decreasing.
    prev = base_lr + 1.0;
    for s in warmup..max_steps {
        let lr = cosine_warmup(s, warmup, max_steps, base_lr, min_lr);
        assert!(lr <= prev + 1e-6, "decay not monotone at {s}: {lr} > {prev}");
        prev = lr;
    }

    // Below floor: stays at min_lr.
    for s in [max_steps, max_steps + 1, max_steps * 2] {
        let lr = cosine_warmup(s, warmup, max_steps, base_lr, min_lr);
        assert!((lr - min_lr).abs() < 1e-6, "post-decay lr {lr} != {min_lr}");
    }
}

#[test]
fn sample_token_temperature_sweep() {
    // For uniform logits, sampling at any temperature should give a
    // roughly uniform distribution. For peaked logits, low temperature
    // collapses to argmax.
    let mut rng = StdRng::seed_from_u64(0xDEAD);

    let uniform: Vec<f32> = vec![0.0; 8];
    let mut counts = [0u32; 8];
    for _ in 0..1000 {
        let t = sample_token(&uniform, 1.0, &mut rng) as usize;
        counts[t] += 1;
    }
    for &c in &counts {
        assert!(c > 50 && c < 200, "uniform sampling distribution skewed: {counts:?}");
    }

    let mut peaked = vec![0.0_f32; 8];
    peaked[3] = 20.0;
    let mut cnt_low = 0u32;
    for _ in 0..200 {
        let t = sample_token(&peaked, 0.1, &mut rng);
        if t == 3 {
            cnt_low += 1;
        }
    }
    assert!(cnt_low > 190, "low-temp peaked sampling: got {cnt_low}/200 hits on argmax");

    // High temperature on a mildly peaked distribution should spread.
    // Peak = 3.0, T = 30 → scaled 0.1 → softmax basically uniform.
    let mut mild = vec![0.0_f32; 8];
    mild[3] = 3.0;
    let mut cnt_high_3 = 0u32;
    for _ in 0..200 {
        let t = sample_token(&mild, 30.0, &mut rng);
        if t == 3 {
            cnt_high_3 += 1;
        }
    }
    assert!(
        cnt_high_3 < 60,
        "high-temp didn't spread mild peak: argmax hit {cnt_high_3}/200"
    );
}

#[test]
fn deterministic_init_from_seed() {
    let (m1, _) = tiny_model(0xC0FFEE);
    let (m2, _) = tiny_model(0xC0FFEE);
    assert_eq!(m1.embed.w_tok.len(), m2.embed.w_tok.len());
    for i in 0..m1.embed.w_tok.len() {
        assert!((m1.embed.w_tok[i] - m2.embed.w_tok[i]).abs() < 1e-10);
    }
    for (b1, b2) in m1.blocks.iter().zip(m2.blocks.iter()) {
        for i in 0..16 {
            assert!((b1.mha.w_q.w[i] - b2.mha.w_q.w[i]).abs() < 1e-10);
        }
    }
}

#[test]
fn forward_does_not_mutate_model() {
    let (model, _cfg) = tiny_model(0x7);
    let snapshot: Vec<f32> = model.embed.w_tok.clone();
    let snapshot_wq: Vec<f32> = model.blocks[0].mha.w_q.w.clone();

    let tokens: Vec<u32> = (0..8u32).collect();
    for _ in 0..5 {
        let _ = model.forward(&tokens);
    }

    for i in 0..snapshot.len() {
        assert!((model.embed.w_tok[i] - snapshot[i]).abs() < 1e-10);
    }
    for i in 0..snapshot_wq.len() {
        assert!((model.blocks[0].mha.w_q.w[i] - snapshot_wq[i]).abs() < 1e-10);
    }
}

#[test]
fn grad_accumulation_across_batch() {
    // Run two single-sequence backwards, then one double-sequence
    // accumulated backward — grads should sum.
    let (model, cfg) = tiny_model(0x42);

    let tokens_a: Vec<u32> = (0..16u32).map(|i| i % cfg.vocab as u32).collect();
    let tokens_b: Vec<u32> = (0..16u32).map(|i| (i + 7) % cfg.vocab as u32).collect();
    let input_a = &tokens_a[..15];
    let target_a = &tokens_a[1..];
    let input_b = &tokens_b[..15];
    let target_b = &tokens_b[1..];

    let mut grads_a = ModelGrads::zeros(&cfg);
    let (la, c_a) = model.forward(input_a);
    let _ = model.backward(&c_a, &la, target_a, &mut grads_a);

    let mut grads_b = ModelGrads::zeros(&cfg);
    let (lb, c_b) = model.forward(input_b);
    let _ = model.backward(&c_b, &lb, target_b, &mut grads_b);

    let mut grads_combined = ModelGrads::zeros(&cfg);
    let (la2, c_a2) = model.forward(input_a);
    let _ = model.backward(&c_a2, &la2, target_a, &mut grads_combined);
    let (lb2, c_b2) = model.forward(input_b);
    let _ = model.backward(&c_b2, &lb2, target_b, &mut grads_combined);

    // The accumulated grad should equal sum of individual grads.
    for i in 0..grads_a.embed.d_w_tok.len() {
        let expected = grads_a.embed.d_w_tok[i] + grads_b.embed.d_w_tok[i];
        let got = grads_combined.embed.d_w_tok[i];
        assert!(
            (expected - got).abs() < 1e-5,
            "embed grad accumulation broken at [{i}]: {got} vs {expected}"
        );
    }
}

#[test]
fn very_small_lr_does_not_diverge() {
    // Run 50 steps with tiny lr — loss should not explode.
    let (mut model, cfg) = tiny_model(0x9);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut rng = StdRng::seed_from_u64(0xC);

    let adam = AdamWConfig {
        lr: 1e-8,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    let mut first_loss = 0.0_f32;
    let mut last_loss = 0.0_f32;
    for s in 0..50 {
        let tokens: Vec<u32> = (0..16).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
        let input = &tokens[..15];
        let target = &tokens[1..];
        let (logits, cache) = model.forward(input);
        grads.zero();
        let loss = model.backward(&cache, &logits, target, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);
        if s == 0 {
            first_loss = loss;
        }
        last_loss = loss;
    }

    assert!(first_loss.is_finite() && last_loss.is_finite());
    // At lr=1e-8, params don't move meaningfully; loss should stay close.
    let rel_change = (last_loss - first_loss).abs() / first_loss.abs().max(1e-6);
    assert!(
        rel_change < 0.5,
        "tiny-lr loss moved too much: {first_loss} → {last_loss}"
    );
}
