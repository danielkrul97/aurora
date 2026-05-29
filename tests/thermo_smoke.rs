//! End-to-end CPU thermo-LM smoke tests.
//!
//! - the model can drive training loss down (gradients + optimizer wired
//!   correctly): overfit a small fixed window
//! - a single training step does not NaN/Inf

#![allow(clippy::needless_range_loop)]

use aurora::optimizer::AdamWConfig;
use aurora::thermo::field::{ThermoConfig, CH_RHO, CH_E};
use aurora::thermo::model::{ThermoGrads, ThermoLM};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn small_cfg() -> ThermoConfig {
    ThermoConfig {
        vocab: 6,
        h: 8,
        w: 8,
        c: 6,
        conv_hidden: 12,
        conserved: vec![CH_RHO, CH_E],
        k_steps: 3,
        pool: 2,
        dt: 1.0,
        t_init: 1.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: true,
        readout_modes: 2,
        use_diffusion: true,
    }
}

#[test]
fn thermo_overfits_small_window() {
    let mut rng = StdRng::seed_from_u64(0x5A1D);
    let cfg = small_cfg();
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);

    // Fixed window; predict the next token (teacher forced).
    let seq: Vec<u32> = vec![1, 3, 0, 5, 2, 4, 1, 3, 0];
    let tokens = &seq[..8];
    let targets = &seq[1..9];

    let adam = AdamWConfig {
        lr: 5e-3,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };
    let mut grads = ThermoGrads::zeros(&cfg);
    let mut first = 0.0_f32;
    let mut last = 0.0_f32;
    for step in 0..400 {
        grads.zero();
        let cache = model.forward(tokens);
        let loss = model.backward(&cache, targets, &mut grads);
        ThermoLM::clip_grads(&mut grads, 1.0);
        model.step_adamw(&grads, &adam);
        if step == 0 {
            first = loss;
        }
        last = loss;
        if step % 50 == 0 {
            println!("step {step:>3} loss {loss:.4}");
        }
    }
    println!("first {first:.4}  last {last:.4}");
    assert!(last.is_finite(), "loss went non-finite");
    assert!(last < first - 0.3, "no learning: {first:.4} -> {last:.4}");
}

#[test]
fn checkpoint_round_trip() {
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let cfg = small_cfg();
    let model = ThermoLM::new(cfg, &mut rng);
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 0];
    let before = model.forward(&tokens).logits;

    let path = std::env::temp_dir().join("thermo_ckpt_round_trip.bin");
    aurora::thermo::checkpoint::save(&path, &model, 3, 42).expect("save");
    let ck = aurora::thermo::checkpoint::load(&path).expect("load");
    assert_eq!(ck.epoch, 3);
    assert_eq!(ck.step, 42);

    let after = ck.model.forward(&tokens).logits;
    for (a, b) in before.iter().zip(&after) {
        assert!((a - b).abs() < 1e-6, "logit mismatch after round-trip");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn thermo_step_does_not_nan() {
    let mut rng = StdRng::seed_from_u64(7);
    let cfg = small_cfg();
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);
    let tokens: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
    let targets: Vec<u32> = vec![1, 2, 3, 4, 5, 0];
    let mut grads = ThermoGrads::zeros(&cfg);
    let cache = model.forward(&tokens);
    assert!(cache.logits.iter().all(|v| v.is_finite()));
    let loss = model.backward(&cache, &targets, &mut grads);
    assert!(loss.is_finite());
    assert!(grads.l2_norm().is_finite());
}
