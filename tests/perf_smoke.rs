//! Phase 0: coarse throughput smoke. NOT a micro-benchmark — it exists
//! to fail loudly on a *catastrophic* perf regression (lost
//! vectorization, an accidental super-linear blowup, a path that used to
//! be parallel going serial) without flaking on normal hardware
//! variance. The ceiling is ~100× the real per-step time on a modern
//! core, so only a true algorithmic regression trips it.

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use aurora::optimizer::AdamWConfig;
use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn cpu_train_step_throughput_smoke() {
    let mut rng = StdRng::seed_from_u64(0x5A1E);
    let cfg = TransformerConfig {
        vocab: 256,
        d_model: 64,
        n_layer: 2,
        n_head: 2,
        d_ff: 128,
        seq_len_max: 64,
        rope: false,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut grads = ModelGrads::zeros(&cfg);
    let adam = AdamWConfig { lr: 1e-3, weight_decay: 0.0, ..AdamWConfig::default() };

    let seq_len = 32;
    let toks: Vec<u32> = (0..=seq_len as u32).map(|i| (i * 7 + 3) % cfg.vocab as u32).collect();
    let input = &toks[..seq_len];
    let target = &toks[1..=seq_len];

    let mut one_step = |model: &mut DecoderTransformer, grads: &mut ModelGrads| -> f32 {
        let (logits, cache) = model.forward(input);
        grads.zero();
        let loss = model.backward(&cache, &logits, target, grads);
        DecoderTransformer::clip_grads(grads, 1.0);
        model.step_adamw(grads, &adam);
        loss
    };

    for _ in 0..3 {
        one_step(&mut model, &mut grads);
    }

    let n = 30;
    let t0 = Instant::now();
    let mut last_loss = 0.0_f32;
    for _ in 0..n {
        last_loss = one_step(&mut model, &mut grads);
    }
    let ms_per_step = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    eprintln!("perf_smoke: tiny d64/L2/T32 = {ms_per_step:.3} ms/step");

    assert!(last_loss.is_finite(), "loss diverged: {last_loss}");
    assert!(
        ms_per_step > 0.0 && ms_per_step < 250.0,
        "tiny CPU step {ms_per_step:.2} ms outside sane range — catastrophic perf regression?"
    );
}
