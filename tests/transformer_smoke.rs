//! Sprint 53: end-to-end CPU transformer smoke tests.
//!
//! - copy task convergence: model learns to echo previous token
//! - single training step does not NaN/Inf

#![allow(clippy::needless_range_loop, clippy::cast_precision_loss)]

use aurora::optimizer::AdamWConfig;
use aurora::transformer::loss::argmax;
use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn cpu_transformer_converges_on_copy_task() {
    // Toy task: sequence is `[t0, t0, t1, t1, t2, t2, ...]` — predict
    // the previous token, which after one step is the same as current.
    // Random vocab=8 with seed-driven sequences.
    let mut rng = StdRng::seed_from_u64(0x5071);
    let cfg = TransformerConfig {
        vocab: 8,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let adam = AdamWConfig {
        lr: 1e-2,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    let mut grads = ModelGrads::zeros(&cfg);

    let mut first_loss: f32 = 0.0;
    let mut last_loss: f32 = 0.0;
    let steps = 200;

    for step in 0..steps {
        // Build a length-12 sequence by repeating each random token twice.
        let half: Vec<u32> = (0..6).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
        let mut tokens: Vec<u32> = Vec::with_capacity(12);
        for &t in &half {
            tokens.push(t);
            tokens.push(t);
        }
        // Input: tokens[..11], target: tokens[1..] — model predicts
        // next, which equals previous token half the time. Trivially
        // learnable.
        let input: Vec<u32> = tokens[..11].to_vec();
        let target: Vec<u32> = tokens[1..].to_vec();

        let (_logits, cache) = model.forward(&input);
        grads.zero();
        let loss = model.backward(&cache, &target, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);

        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
    }

    let baseline = (cfg.vocab as f32).ln();
    assert!(
        last_loss < first_loss * 0.7,
        "loss did not decrease: first={first_loss}, last={last_loss}, baseline={baseline}"
    );
    assert!(
        last_loss < baseline * 0.9,
        "loss {last_loss} above 0.9× random baseline {baseline}"
    );
}

#[test]
fn cpu_transformer_step_does_not_nan() {
    let mut rng = StdRng::seed_from_u64(0xABCD);
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 32,
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = (0..16).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let input = &tokens[..15];
    let target = &tokens[1..];

    let (logits, cache) = model.forward(input);
    for &v in &logits {
        assert!(v.is_finite(), "logits contain NaN/Inf");
    }
    let mut grads = ModelGrads::zeros(&cfg);
    let loss = model.backward(&cache, target, &mut grads);
    assert!(loss.is_finite(), "loss is NaN/Inf");

    let pre_clip_norm = DecoderTransformer::clip_grads(&mut grads, 1.0);
    assert!(pre_clip_norm.is_finite(), "grad norm is NaN/Inf");

    let adam = AdamWConfig::default();
    model.step_adamw(&grads, &adam);

    // Spot-check a few params for finiteness.
    for v in model.embed.w_tok.iter().take(64) {
        assert!(v.is_finite(), "tok embed has NaN/Inf after step");
    }
    for blk in &model.blocks {
        for v in blk.mha.w_q.w.iter().take(16) {
            assert!(v.is_finite(), "W_q has NaN/Inf after step");
        }
        for v in &blk.ln1.gamma {
            assert!(v.is_finite(), "LN gamma has NaN/Inf after step");
        }
    }
}

#[test]
fn cpu_transformer_argmax_sanity() {
    // After convergence on a trivial repeat task the model should
    // greedily emit each input as its own next token (since
    // tokens[t+1] == tokens[t] for even t).
    let mut rng = StdRng::seed_from_u64(0x1234);
    let cfg = TransformerConfig {
        vocab: 4,
        d_model: 16,
        n_layer: 1,
        n_head: 1,
        d_ff: 32,
        seq_len_max: 8,
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let adam = AdamWConfig {
        lr: 3e-2,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut grads = ModelGrads::zeros(&cfg);

    // 100 steps of trivial "predict same token" task.
    for _ in 0..100 {
        let tok = rng.random_range(0..cfg.vocab as u32);
        let input: Vec<u32> = vec![tok; 4];
        let target: Vec<u32> = vec![tok; 4];
        let (_, cache) = model.forward(&input);
        grads.zero();
        let _ = model.backward(&cache, &target, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);
    }

    // After training, predicting token 2 from input `[2, 2, 2, 2]` should
    // produce 2 at the final position with high probability.
    let input: Vec<u32> = vec![2u32, 2, 2, 2];
    let (logits, _) = model.forward(&input);
    let v = cfg.vocab;
    let final_logits = &logits[3 * v..4 * v];
    let pred = argmax(final_logits);
    assert_eq!(pred, 2, "model should learn identity prediction, got {pred}");
}
