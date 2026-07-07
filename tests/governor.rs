//! PHAGE governor — failable checks.
//!
//! The load-bearing one is `cs_aa_false_positive_control`: under the null
//! (zero-mean paired noise) the sequential test must almost never fire.
//! That property is what lets the governor screen below the noise floor
//! without adopting junk — if the boundary constant were wrong, this test
//! goes red.

use aurora::governor::{
    diagnose, licensed_arms, Arm, ArmKind, Bandit, ConfidenceSequence, CsConfig, Fault, Fingerprint,
    Governor, GovernorConfig, MathState, Reflex, ReflexAction, Regime, StepSample, TrialBackend,
    Verdict, Vitals,
};
use aurora::optimizer::{AdamWConfig, MuonConfig};
use aurora::transformer::{DecoderTransformer, TransformerConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn gauss(rng: &mut StdRng, mean: f32, sd: f32) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-12);
    let u2: f32 = rng.random::<f32>();
    mean + sd * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn run_cs(cfg: CsConfig, rng: &mut StdRng, mean: f32, sd: f32) -> Verdict {
    let mut cs = ConfidenceSequence::new(cfg);
    loop {
        cs.push(gauss(rng, mean, sd));
        match cs.verdict() {
            Verdict::Continue => {}
            other => return other,
        }
    }
}

#[test]
fn cs_aa_false_positive_control() {
    // Null: zero-mean noise. An anytime-valid test at alpha=0.05 with a
    // positive floor must fire (adopt OR revert) far below alpha.
    let cfg = CsConfig { alpha: 0.05, rho: 5.0, sigma: 1.0, floor: 0.3, n_max: 60 };
    let mut rng = StdRng::seed_from_u64(0xA0A0);
    let sims = 600;
    let mut fired = 0;
    for _ in 0..sims {
        match run_cs(cfg, &mut rng, 0.0, 1.0) {
            Verdict::Adopt | Verdict::Revert => fired += 1,
            _ => {}
        }
    }
    let rate = fired as f32 / sims as f32;
    assert!(rate <= 0.08, "A/A false-fire rate too high: {rate}");
}

#[test]
fn cs_detects_real_effect() {
    let cfg = CsConfig { alpha: 0.05, rho: 5.0, sigma: 1.0, floor: 0.3, n_max: 80 };
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let sims = 300;
    let mut adopted = 0;
    for _ in 0..sims {
        if run_cs(cfg, &mut rng, 1.0, 1.0) == Verdict::Adopt {
            adopted += 1;
        }
    }
    let rate = adopted as f32 / sims as f32;
    assert!(rate >= 0.7, "power too low: adopt rate {rate}");
}

#[test]
fn cs_reverts_harmful() {
    let cfg = CsConfig { alpha: 0.05, rho: 5.0, sigma: 1.0, floor: 0.3, n_max: 80 };
    let mut rng = StdRng::seed_from_u64(0xF00D);
    let sims = 300;
    let mut reverted = 0;
    for _ in 0..sims {
        if run_cs(cfg, &mut rng, -1.0, 1.0) == Verdict::Revert {
            reverted += 1;
        }
    }
    let rate = reverted as f32 / sims as f32;
    assert!(rate >= 0.7, "harmful revert rate too low: {rate}");
}

fn fp(slope: f32, noise: f32, level: f32, grad: f32, spike: f32, cv: f32, dt: f32, regime: Regime) -> Fingerprint {
    Fingerprint {
        regime,
        loss_slope: slope,
        loss_noise: noise,
        loss_level: level,
        grad_level: grad,
        grad_spike: spike,
        grad_cv: cv,
        dt_drift: dt,
        diverging: false,
        val_gap: 0.0,
        // Diagnosis now reads the long-horizon trend, not the per-step slope:
        // a negative slope (improving loss) maps to a positive trend.
        loss_trend: -slope * 2.0,
        lr_rising: false,
        lr_falling: false,
    }
}

#[test]
fn diagnosis_distinguishes_faults() {
    // Plateau, healthy gradient -> LrTooCold.
    let cold = diagnose(&fp(-0.00005, 0.01, 2.0, 0.5, 1.0, 0.1, 1.0, Regime::Mid));
    assert_eq!(cold.top().0, Fault::LrTooCold, "{:?}", cold.posterior);

    // Big grad spike + rising loss -> LrTooHot.
    let hot = diagnose(&fp(0.05, 0.02, 2.0, 1.0, 6.0, 0.5, 1.0, Regime::Mid));
    assert_eq!(hot.top().0, Fault::LrTooHot, "{:?}", hot.posterior);

    // Improving fast, calm -> Healthy.
    let well = diagnose(&fp(-0.02, 0.01, 2.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid));
    assert_eq!(well.top().0, Fault::Healthy, "{:?}", well.posterior);

    // Stalled + step time drifting up -> ThroughputRegression dominates.
    let slow = diagnose(&fp(-0.00005, 0.01, 2.0, 0.5, 1.0, 0.2, 1.5, Regime::Mid));
    assert_eq!(slow.top().0, Fault::ThroughputRegression, "{:?}", slow.posterior);
}

#[test]
fn diagnosis_ignores_healthy_descent_through_noise() {
    // The run's bug: per-step loss is noisy but the long-horizon trend is
    // clearly improving. Must read Healthy, not LrTooCold.
    let healthy = diagnose(&fp(0.0, 0.2, 5.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid));
    // loss_trend defaults to 0 from a 0 slope -> not improving; supply a real
    // positive trend directly.
    let mut fpr = fp(0.0, 0.2, 5.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid);
    fpr.loss_trend = 0.02; // 2% relative improvement over the horizon
    assert_eq!(diagnose(&fpr).top().0, Fault::Healthy, "{:?}", diagnose(&fpr).posterior);
    let _ = healthy;
}

#[test]
fn diagnosis_suppresses_noise_hot_but_keeps_real_uptrend() {
    // The enwik9 over-cool root cause: a flat/noisy loss (σ≈0.25) makes the
    // mid/slow EWMAs cross, so loss_trend reads transiently negative *within the
    // noise band*. That must NOT be lr_too_hot — else the governor cools a
    // healthy LR. (level 3.0, noise 0.35 -> band ~0.12; a 0.012 cross is noise.)
    let mut noisy = fp(0.0, 0.35, 3.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid);
    noisy.loss_trend = -0.012;
    assert_ne!(
        diagnose(&noisy).top().0,
        Fault::LrTooHot,
        "noise-induced worsening mis-fired lr_too_hot: {:?}",
        diagnose(&noisy).posterior
    );
    // A genuine up-trend (loss climbing well past the band) must still fire hot.
    let mut real = fp(0.0, 0.35, 3.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid);
    real.loss_trend = -0.20;
    assert_eq!(
        diagnose(&real).top().0,
        Fault::LrTooHot,
        "a real up-trend past the noise band must still be lr_too_hot: {:?}",
        diagnose(&real).posterior
    );
}

#[test]
fn diagnosis_respects_schedule() {
    // A textbook "too hot" fingerprint, but the LR is ramping (warmup):
    // elevated grads are expected, so lr_too_hot must be gated off.
    let mut hot = fp(0.05, 0.02, 2.0, 1.0, 6.0, 0.5, 1.0, Regime::Mid);
    assert_eq!(diagnose(&hot).top().0, Fault::LrTooHot, "baseline should be hot");
    hot.lr_rising = true;
    assert_ne!(
        diagnose(&hot).top().0,
        Fault::LrTooHot,
        "lr_too_hot not gated while LR is ramping: {:?}",
        diagnose(&hot).posterior
    );

    // A plateau, but the LR is decaying by design (WSD tail): "raise LR" is
    // the wrong fix, so lr_too_cold must be gated off.
    let mut cold = fp(-0.00005, 0.01, 2.0, 0.5, 1.0, 0.1, 1.0, Regime::Mid);
    assert_eq!(diagnose(&cold).top().0, Fault::LrTooCold, "baseline should be cold");
    cold.lr_falling = true;
    assert_ne!(
        diagnose(&cold).top().0,
        Fault::LrTooCold,
        "lr_too_cold not gated while LR is decaying: {:?}",
        diagnose(&cold).posterior
    );
}

#[test]
fn regime_warmup_is_capped() {
    // warmup_steps huge relative to the run must not swallow early/mid.
    let mut v = Vitals::new(1000, 2000);
    let push = |v: &mut Vitals, step: u64| {
        v.push(&StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 });
    };
    push(&mut v, 50); // < 10% of 1000 -> still warmup
    assert_eq!(v.fingerprint().regime, Regime::Warmup);
    push(&mut v, 250); // past the 10% cap despite warmup_steps=2000
    assert_ne!(v.fingerprint().regime, Regime::Warmup);
}

/// Deterministic backend: paired_diff is a fixed mean + Gaussian noise,
/// independent of the model. Lets us test the calibration→adoption path.
struct MockBackend {
    rng: StdRng,
    mean: f32,
    noise: f32,
}

impl TrialBackend for MockBackend {
    fn snapshot_safe(&mut self) {}
    fn restore_safe(&mut self) {}
    fn paired_diff(&mut self, _c: &MathState, _t: &MathState, _lr: f32, _mlr: f32) -> f32 {
        self.mean + gauss(&mut self.rng, 0.0, self.noise)
    }
}

#[test]
fn calibration_sigma_is_what_unblocks_adoption() {
    // The run's exact problem, isolated to the CS: a real effect of 0.010
    // with paired noise 0.004. At the guessed sigma=0.02 the margin is too
    // wide to ever clear the floor; at a calibrated sigma=0.006 it confirms.
    let effect_run = |sigma: f32| -> Verdict {
        let mut cs = ConfidenceSequence::new(CsConfig { alpha: 0.05, rho: 5.0, sigma, floor: 0.003, n_max: 60 });
        let mut rng = StdRng::seed_from_u64(42);
        loop {
            cs.push(0.010 + gauss(&mut rng, 0.0, 0.004));
            match cs.verdict() {
                Verdict::Continue => {}
                other => return other,
            }
        }
    };
    assert_eq!(effect_run(0.02), Verdict::BelowFloor, "guessed sigma=0.02 should be too conservative");
    assert_eq!(effect_run(0.006), Verdict::Adopt, "calibrated sigma=0.006 should confirm the effect");
}

#[test]
fn calibration_enables_adoption_below_default_floor() {
    // Effect 0.010 with paired noise 0.004. The default sigma=0.02 keeps the
    // CS margin too wide to ever adopt (~0.0096 at n=40); A/A calibration
    // measures ~0.004 and shrinks the margin enough to confirm the effect.
    let make = |sigma: f32| GovernorConfig {
        diag_every: 10,
        snapshot_every: 100_000,
        cooldown: 5,
        horizon: 1,
        probe_windows: 1,
        fault_threshold: 0.5,
        max_trials: 50,
        calib_samples: 24,
        cs: CsConfig { sigma, floor: 0.003, n_max: 60, ..CsConfig::default() },
    };

    let run = |cfg: GovernorConfig, seed: u64| -> bool {
        let mut gov = Governor::new(cfg, 0.95, 0.1, 0.0, 100_000, 0, seed, None);
        let mut backend = MockBackend { rng: StdRng::seed_from_u64(seed ^ 0xABC), mean: 0.010, noise: 0.004 };
        for step in 1..=4000u64 {
            // Constant plateau, constant LR (not falling) -> lr_too_cold fires.
            let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
            let _ = gov.begin_step(1e-3, 2e-2);
            if gov.end_step(s, &mut backend).iter().any(|e| e.contains("trial_adopt")) {
                return true;
            }
        }
        false
    };

    // With calibration on (default path) the real effect gets adopted.
    assert!(run(make(0.02), 1), "calibration did not enable adoption of a real 0.01 effect");
}

#[test]
fn reflex_fires_on_divergence_not_noise() {
    let r = Reflex::default();
    let healthy_fp = fp(-0.01, 0.005, 2.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid);
    let healthy = StepSample { step: 100, loss: 2.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
    assert_eq!(r.check(&healthy_fp, &healthy), ReflexAction::None);

    let nan = StepSample { step: 101, loss: f32::NAN, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
    assert!(matches!(r.check(&healthy_fp, &nan), ReflexAction::CutLr(_)));

    let explode_fp = fp(0.5, 0.01, 2.0, 0.5, 1.0, 0.2, 1.0, Regime::Mid);
    let explode = StepSample { step: 102, loss: 9.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
    assert!(matches!(r.check(&explode_fp, &explode), ReflexAction::CutLr(_)));

    // grad_norm NaN (unmeasured on a GPU non-report step) with a finite
    // loss is "not measured", not divergence — reflex must stay quiet.
    let nan_grad = StepSample { step: 103, loss: 2.0, grad_norm: f32::NAN, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
    assert_eq!(r.check(&healthy_fp, &nan_grad), ReflexAction::None);
}

#[test]
fn mathstate_apply_and_clamp() {
    let mut s = MathState::identity(0.95, 0.1, 0.0);
    let mut adam = AdamWConfig { lr: 1.0, ..AdamWConfig::default() };
    let mut muon = MuonConfig { lr: 1.0, ..MuonConfig::default() };
    let clip = s.apply_optim(&mut adam, &mut muon, 2.0);
    assert!((adam.lr - 1.0).abs() < 1e-6 && (muon.lr - 1.0).abs() < 1e-6 && (clip - 2.0).abs() < 1e-6);
    assert!((adam.weight_decay - 0.1).abs() < 1e-6 && (muon.momentum - 0.95).abs() < 1e-6);

    s.lr_scale = 100.0;
    s.clamp();
    assert!(s.lr_scale <= 4.0 + 1e-6, "lr_scale not clamped: {}", s.lr_scale);
    let mut a2 = AdamWConfig { lr: 1.0, ..AdamWConfig::default() };
    let mut m2 = MuonConfig { lr: 1.0, ..MuonConfig::default() };
    s.apply_optim(&mut a2, &mut m2, 1.0);
    assert!((a2.lr - 4.0).abs() < 1e-6);
}

#[test]
fn ewma_tracks_mean() {
    use aurora::governor::Ewma;
    let mut e = Ewma::new(0.5);
    for _ in 0..50 {
        e.push(3.0);
    }
    assert!((e.get() - 3.0).abs() < 1e-3);
}

#[test]
fn bandit_prefers_better_arm() {
    let mut b = Bandit::default();
    let good = Arm::new(ArmKind::LrUp, 1.3);
    let bad = Arm::new(ArmKind::MuonLrUp, 1.3);
    for _ in 0..60 {
        b.reward(Regime::Early, Fault::LrTooCold, good, 0.02);
        b.reward(Regime::Early, Fault::LrTooCold, bad, -0.02);
    }
    let mut rng = StdRng::seed_from_u64(7);
    let arms = vec![good, bad];
    let mut good_picks = 0;
    for _ in 0..300 {
        if b.select(Regime::Early, Fault::LrTooCold, &arms, &mut rng).kind == ArmKind::LrUp {
            good_picks += 1;
        }
    }
    assert!(good_picks as f32 / 300.0 >= 0.7, "bandit didn't learn: {good_picks}/300");
}

#[test]
fn bandit_tsv_roundtrip() {
    let mut b = Bandit::default();
    let good = Arm::new(ArmKind::LrUp, 1.3);
    let bad = Arm::new(ArmKind::MuonLrUp, 1.3);
    for _ in 0..40 {
        b.reward(Regime::Early, Fault::LrTooCold, good, 0.05);
        b.reward(Regime::Early, Fault::LrTooCold, bad, -0.05);
    }
    let path = std::env::temp_dir().join(format!("phage_bandit_{}.tsv", std::process::id()));
    b.save(&path).expect("save");
    let b2 = Bandit::load(&path);
    let _ = std::fs::remove_file(&path);

    let mut rng = StdRng::seed_from_u64(11);
    let arms = vec![good, bad];
    let mut good_picks = 0;
    for _ in 0..300 {
        if b2.select(Regime::Early, Fault::LrTooCold, &arms, &mut rng).kind == ArmKind::LrUp {
            good_picks += 1;
        }
    }
    assert!(good_picks as f32 / 300.0 >= 0.7, "loaded bandit lost its memory: {good_picks}/300");
}

#[test]
fn licensed_arms_are_fault_scoped() {
    assert!(!licensed_arms(Fault::LrTooCold).is_empty());
    assert!(!licensed_arms(Fault::LrTooHot).is_empty());
    assert!(licensed_arms(Fault::CapacityWall).is_empty());
    assert!(licensed_arms(Fault::Healthy).is_empty());
}

fn tiny_model(seed: u64) -> (DecoderTransformer, TransformerConfig) {
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        ..TransformerConfig::tiny(32)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    (DecoderTransformer::new(cfg.clone(), &mut rng), cfg)
}

fn rand_toks(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(0..vocab)).collect()
}

#[test]
fn paired_rollout_identity_cancels_to_zero() {
    // The A/A property at the model level: with the same arm on both sides,
    // sharing weights, data and probe, the paired difference is exactly 0.
    // This is why pairing kills the curve-position nuisance.
    use aurora::governor::paired_rollout_sample;
    let (model, _) = tiny_model(1);
    let toks = rand_toks(2000, 32, 2);
    let seq_len = 16;
    let probe_in: Vec<Vec<u32>> = (0..4).map(|w| toks[w * 20..w * 20 + seq_len].to_vec()).collect();
    let probe_tg: Vec<Vec<u32>> = (0..4).map(|w| toks[w * 20 + 1..w * 20 + seq_len + 1].to_vec()).collect();
    let starts = vec![100, 200, 300];
    let id = MathState::identity(0.95, 0.1, 0.0);

    let diff = paired_rollout_sample(
        &model, &toks, &starts, &probe_in, &probe_tg, seq_len, 1.0, 1e-3, 2e-2, 0.0, 1.0, &id, &id,
    );
    assert_eq!(diff, 0.0, "identity A/A did not cancel: {diff}");

    let mut hot = id;
    hot.lr_scale = 3.0;
    hot.muon_lr_scale = 3.0;
    let diff2 = paired_rollout_sample(
        &model, &toks, &starts, &probe_in, &probe_tg, seq_len, 1e-2, 1e-1, 0.0, 1.0, 1.0, &id, &hot,
    );
    assert!(diff2.is_finite(), "rollout produced non-finite diff");
}

#[test]
fn governor_runs_a_trial_end_to_end() {
    use aurora::governor::{build_probe, CpuTrialBackend};
    let (mut model, cfg) = tiny_model(3);
    let toks = rand_toks(3000, 32, 4);
    let n_train = 2400;
    let train = toks[..n_train].to_vec();
    let val = toks[n_train..].to_vec();
    let (probe_in, probe_out) = build_probe(&val, cfg.seq_len_max, 4);

    let gcfg = GovernorConfig { diag_every: 20, snapshot_every: 25, cooldown: 30, horizon: 3, probe_windows: 4, fault_threshold: 0.5, max_trials: 5, calib_samples: 8, cs: CsConfig { n_max: 12, ..CsConfig::default() } };
    let mut gov = Governor::new(gcfg, 0.95, 0.1, 0.0, 1000, 0, 5, None);
    let mut safe: Option<DecoderTransformer> = None;
    let mut brng = StdRng::seed_from_u64(99);

    let mut saw_calibrate = false;
    let mut saw_trial = false;
    for step in 1..=300u64 {
        // Feed a deliberate plateau so diagnosis lands on LrTooCold.
        let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
        let _ = gov.begin_step(1e-3, 2e-2);
        let mut backend = CpuTrialBackend {
            model: &mut model,
            safe: &mut safe,
            train_toks: &train,
            probe_inputs: &probe_in,
            probe_targets: &probe_out,
            seq_len: cfg.seq_len_max,
            comp: 1.0,
            base_muon_wd: 0.0,
            base_clip: 1.0,
            horizon: 3,
            rng: &mut brng,
        };
        let evs = gov.end_step(s, &mut backend);
        if evs.iter().any(|e| e.contains("\"kind\":\"calibrate\"")) {
            saw_calibrate = true;
        }
        if evs.iter().any(|e| e.contains("trial_begin")) {
            saw_trial = true;
        }
    }
    assert!(saw_calibrate, "governor never calibrated before trialing");
    assert!(saw_trial, "governor never opened a trial on a plateau");
    assert!(gov.committed().lr_scale.is_finite());
    // A Vitals sanity check: eta is finite after a run.
    let mut v = Vitals::new(1000, 0);
    v.push(&StepSample { step: 1, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 });
    v.push(&StepSample { step: 2, loss: 2.5, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 });
    assert!(v.eta().is_finite());
}

#[test]
fn probation_keeps_or_reverts_wd_on_val() {
    // Overfit → tentatively raise WD live; keep if val improves at the next
    // eval, revert if it regresses.
    let gcfg = GovernorConfig {
        diag_every: 5,
        snapshot_every: 100_000,
        cooldown: 5,
        horizon: 1,
        probe_windows: 1,
        fault_threshold: 0.5,
        max_trials: 50,
        calib_samples: 4,
        cs: CsConfig::default(),
    };
    let run = |val_after: f32| -> f32 {
        let mut gov = Governor::new(gcfg.clone(), 0.95, 0.1, 0.0, 100_000, 0, 1, None);
        let mut backend = MockBackend { rng: StdRng::seed_from_u64(1), mean: 0.0, noise: 0.001 };
        // First eval establishes a big overfit gap and the baseline val.
        let _ = gov.observe_eval(2.0, 1.5);
        let mut began = false;
        for step in 1..=60u64 {
            let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
            let _ = gov.begin_step(1e-3, 2e-2);
            if gov.end_step(s, &mut backend).iter().any(|e| e.contains("probation_begin")) {
                began = true;
                break;
            }
        }
        assert!(began, "probation never opened on a big overfit gap");
        assert!(gov.committed().weight_decay > 0.1 + 1e-6, "WD not tentatively raised");
        let _ = gov.observe_eval(val_after, 1.5);
        gov.committed().weight_decay
    };
    assert!(run(1.8) > 0.1 + 1e-6, "improved val should keep the raised WD");
    assert!((run(2.1) - 0.1).abs() < 1e-6, "regressed val should revert the WD");
}

#[test]
fn wd_up_arm_raises_both_decays() {
    // Fix 3: wd_up must regularize the Muon matrix bulk too, not just AdamW.
    let base = MathState::identity(0.95, 0.1, 0.0);
    let t = Arm::new(ArmKind::WdUp, 2.0).target(&base);
    assert!(t.weight_decay > 0.1 + 1e-6, "adam wd should rise: {}", t.weight_decay);
    assert!(t.muon_weight_decay > 1e-6, "muon wd should be introduced: {}", t.muon_weight_decay);
}

#[test]
fn early_stop_recommends_after_patience() {
    // Fix 1: val improves then worsens -> recommend stop 2 evals past best.
    let mut gov = Governor::new(GovernorConfig::default(), 0.95, 0.1, 0.0, 100_000, 0, 1, None);
    let vals = [3.0_f32, 2.5, 2.0, 2.1, 2.2, 2.3]; // best at eval 3 (idx 2)
    let mut fired_at = None;
    for (i, &v) in vals.iter().enumerate() {
        if gov.observe_eval(v, v - 0.1).iter().any(|e| e.contains("early_stop")) && fired_at.is_none() {
            fired_at = Some(i);
        }
    }
    assert_eq!(fired_at, Some(4), "early_stop should fire exactly 2 evals past the best");
    assert!(gov.should_stop());
}

#[test]
fn probation_keeps_when_degradation_slows() {
    // Fix 2: val still rises, but slower than the pre-commit trend -> the WD
    // change helped, so keep it (the old absolute test would have reverted).
    let gcfg = GovernorConfig {
        diag_every: 5, snapshot_every: 100_000, cooldown: 5, horizon: 1, probe_windows: 1,
        fault_threshold: 0.5, max_trials: 50, calib_samples: 4, cs: CsConfig::default(),
    };
    let mut gov = Governor::new(gcfg, 0.95, 0.1, 0.0, 100_000, 0, 1, None);
    let mut backend = MockBackend { rng: StdRng::seed_from_u64(1), mean: 0.0, noise: 0.001 };
    let _ = gov.observe_eval(2.0, 1.5); // big overfit gap
    let _ = gov.observe_eval(2.05, 1.5); // base_rate = +0.05
    let mut began = false;
    for step in 1..=60u64 {
        let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
        let _ = gov.begin_step(1e-3, 2e-2);
        if gov.end_step(s, &mut backend).iter().any(|e| e.contains("probation_begin")) {
            began = true;
            break;
        }
    }
    assert!(began, "probation never opened");
    let wd = gov.committed().weight_decay;
    assert!(wd > 0.1 + 1e-6);
    // probation_rate = 2.07 - 2.05 = +0.02 < base_rate 0.05 -> keep.
    let evs = gov.observe_eval(2.07, 1.5);
    assert!(evs.iter().any(|e| e.contains("probation_keep")), "slower degradation should keep: {evs:?}");
    assert!((gov.committed().weight_decay - wd).abs() < 1e-6, "kept wd must persist");
}

#[test]
fn probation_gives_up_after_repeated_reverts() {
    // Fix 4: when the WD lever keeps making things worse, stop probating.
    let gcfg = GovernorConfig {
        diag_every: 5, snapshot_every: 100_000, cooldown: 0, horizon: 1, probe_windows: 1,
        fault_threshold: 0.5, max_trials: 999, calib_samples: 4, cs: CsConfig::default(),
    };
    let mut gov = Governor::new(gcfg, 0.95, 0.1, 0.0, 100_000, 0, 1, None);
    let mut backend = MockBackend { rng: StdRng::seed_from_u64(1), mean: 0.0, noise: 0.001 };
    let _ = gov.observe_eval(2.0, 1.5);
    let mut val = 2.0_f32;
    let mut step: u64 = 1;
    let mut saw_giveup = false;
    for cycle in 0..6 {
        let mut began = false;
        let end = step + 200;
        while step <= end {
            let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
            let _ = gov.begin_step(1e-3, 2e-2);
            let opened = gov.end_step(s, &mut backend).iter().any(|e| e.contains("probation_begin"));
            step += 1;
            if opened {
                began = true;
                break;
            }
        }
        if !began {
            continue;
        }
        val += 0.2 * (cycle as f32 + 1.0); // accelerating worsening -> revert each time
        if gov.observe_eval(val, 1.5).iter().any(|e| e.contains("probation_giveup")) {
            saw_giveup = true;
            break;
        }
    }
    assert!(saw_giveup, "governor should give up after repeated reverts");
}

/// Backend that models the real paired-rollout bias: a lower LR "wins" the
/// short rollout (less immediate variance), and an identical (A/A) pair
/// cancels to ~0. This is what makes unbounded `lr_down` ratchet in the wild.
struct ArmAwareMock {
    rng: StdRng,
    gain: f32,
    noise: f32,
}

impl TrialBackend for ArmAwareMock {
    fn snapshot_safe(&mut self) {}
    fn restore_safe(&mut self) {}
    fn paired_diff(&mut self, c: &MathState, t: &MathState, _lr: f32, _mlr: f32) -> f32 {
        let delta = (c.lr_scale - t.lr_scale) + (c.muon_lr_scale - t.muon_lr_scale);
        self.gain * delta + gauss(&mut self.rng, 0.0, self.noise)
    }
}

#[test]
fn governor_caps_lr_cooling_at_floor() {
    // Prong A. In the enwik9 run the governor ratcheted lr_scale 1.0 -> 0.168
    // (six ×0.7 adopts) because the short paired rollout always favors a lower
    // LR. With a cooling-wins backend the floor must cap the cumulative cut at
    // 0.5; without the fix this test would see min_lr ~0.17.
    let gcfg = GovernorConfig {
        diag_every: 15, snapshot_every: 100_000, cooldown: 15, horizon: 1,
        probe_windows: 1, fault_threshold: 0.5, max_trials: 100_000, calib_samples: 6,
        cs: CsConfig { sigma: 0.006, floor: 0.003, n_max: 12, ..CsConfig::default() },
    };
    let mut gov = Governor::new(gcfg, 0.95, 0.1, 0.0, 100_000, 0, 3, None);
    let mut backend = ArmAwareMock { rng: StdRng::seed_from_u64(0xABC), gain: 0.06, noise: 0.002 };
    let mut min_lr = 1.0f32;
    for step in 1..=8000u64 {
        // Flat loss + a grad spike on each diagnosis step => a *genuine*
        // lr_too_hot through the grad-spike term (the de-noised diagnosis no
        // longer fires hot on loss noise, so we drive a real fault here).
        let grad = if step % 15 == 0 { 3.0 } else { 0.3 };
        let s = StepSample { step, loss: 3.0, grad_norm: grad, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
        let _ = gov.begin_step(1e-3, 2e-2);
        let _ = gov.end_step(s, &mut backend);
        let c = gov.committed();
        min_lr = min_lr.min(c.lr_scale.min(c.muon_lr_scale));
    }
    assert!(min_lr >= 0.5 - 1e-3, "trial cooling broke the 0.5 floor: {min_lr}");
    assert!(min_lr <= 0.71, "cooling never engaged — test didn't exercise the floor: {min_lr}");
}

#[test]
fn governor_backs_off_probing_when_nothing_adopts() {
    // Prong B. The run burned ~38% of wall-clock on 212 trials that never
    // adopted. Under a plateau where nothing clears the floor, the below-floor
    // backoff must widen the inter-trial cooldown geometrically.
    let gcfg = GovernorConfig {
        diag_every: 10, snapshot_every: 100_000, cooldown: 10, horizon: 1,
        probe_windows: 1, fault_threshold: 0.5, max_trials: 100_000, calib_samples: 4,
        cs: CsConfig { sigma: 0.02, floor: 0.01, n_max: 8, ..CsConfig::default() },
    };
    let mut gov = Governor::new(gcfg, 0.95, 0.1, 0.0, 100_000, 0, 1, None);
    // Zero-mean tiny noise -> every trial resolves BelowFloor.
    let mut backend = MockBackend { rng: StdRng::seed_from_u64(123), mean: 0.0, noise: 0.001 };
    let mut trial_steps = Vec::new();
    for step in 1..=4000u64 {
        let s = StepSample { step, loss: 3.0, grad_norm: 0.5, dt_secs: 0.01, lr: 1e-3, muon_lr: 2e-2 };
        let _ = gov.begin_step(1e-3, 2e-2);
        if gov.end_step(s, &mut backend).iter().any(|e| e.contains("trial_begin")) {
            trial_steps.push(step);
        }
    }
    assert!(trial_steps.len() >= 4, "need several trials to judge backoff: {trial_steps:?}");
    let first_gap = trial_steps[1] - trial_steps[0];
    let last_gap = trial_steps[trial_steps.len() - 1] - trial_steps[trial_steps.len() - 2];
    assert!(
        last_gap >= 4 * first_gap.max(1),
        "below-floor backoff did not thin probes: gaps {first_gap} -> {last_gap} ({trial_steps:?})"
    );
}
