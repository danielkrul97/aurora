#![allow(
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::module_name_repetitions
)]

//! PHAGE — Probe-Hardened Adaptive Governor of Efficiency.
//!
//! A runtime training controller for Aurora. It watches one quantity —
//! efficiency `η = −Δloss / Δwall`, speed and quality fused into one —
//! and when learning stalls it (1) diagnoses the most likely cause, (2)
//! proposes a live math change from a fault-licensed arm set chosen by a
//! contextual bandit, and (3) commits that change only after a paired
//! probe-rollout sequential test shows the gain clears the noise floor.
//! Divergence is handled by a separate reflex that rolls back and cuts the
//! LR without waiting for a trial.
//!
//! The core here is pure decision logic: it never touches a model.
//! Snapshotting and the paired rollout are delegated to a [`TrialBackend`],
//! so the same governor drives the CPU model and the on-device GPU model.
//! Everything actuates through the Math State Vector ([`MathState`]), whose
//! fields Aurora reads fresh each step — runtime math change, no recompile.

mod backend;
mod diagnosis;
mod pharmacy;
mod reflex;
mod state;
mod trial;
mod verifier;
mod vitals;

pub use backend::{build_probe, TrialBackend};
pub use diagnosis::{diagnose, Diagnosis, Fault};
pub use pharmacy::{licensed_arms, Arm, ArmKind, Bandit};
pub use reflex::{Reflex, ReflexAction};
pub use state::MathState;
pub use trial::{paired_rollout_sample, CpuTrialBackend, Trial};
pub use verifier::{ConfidenceSequence, CsConfig, Verdict};
pub use vitals::{Ewma, Fingerprint, Regime, StepSample, Vitals};

use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct GovernorConfig {
    /// Run diagnosis (and maybe open a trial) every N steps when idle.
    pub diag_every: u64,
    /// Refresh the reflex rollback snapshot every N idle steps.
    pub snapshot_every: u64,
    /// Steps to wait after a trial before considering another.
    pub cooldown: u64,
    /// Shadow micro-steps per paired sample (consumed by the backend).
    pub horizon: usize,
    /// Fixed held-out probe windows (consumed by the backend).
    pub probe_windows: usize,
    /// Posterior mass an actionable fault needs before a trial opens.
    pub fault_threshold: f32,
    /// Hard cap on total trials over a run.
    pub max_trials: u32,
    /// Paired samples for the one-time A/A noise calibration that sets the
    /// CS σ from the real rollout/probe noise instead of a guessed default.
    pub calib_samples: usize,
    pub cs: CsConfig,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            diag_every: 25,
            snapshot_every: 50,
            cooldown: 40,
            horizon: 4,
            probe_windows: 6,
            fault_threshold: 0.5,
            max_trials: 100_000,
            calib_samples: 16,
            cs: CsConfig::default(),
        }
    }
}

/// One-time A/A noise calibration. Runs a real-magnitude probe arm and
/// keeps a Welford accumulator of the paired diffs; the *spread* around the
/// (discarded) mean is the rollout/probe noise scale that real trials face —
/// the σ the sequential test needs, measured rather than guessed.
struct Calib {
    treatment: MathState,
    target_n: usize,
    n: usize,
    mean: f64,
    m2: f64,
}

impl Calib {
    fn new(treatment: MathState, target_n: usize) -> Self {
        Self { treatment, target_n, n: 0, mean: 0.0, m2: 0.0 }
    }

    fn push(&mut self, x: f32) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let d = f64::from(x) - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (f64::from(x) - self.mean);
    }

    fn std(&self) -> f32 {
        if self.n >= 2 {
            (self.m2 / (self.n - 1) as f64).sqrt() as f32
        } else {
            0.0
        }
    }
}

enum GovState {
    Idle,
    Calibrating(Calib),
    Trialing(Trial),
}

pub struct Governor {
    cfg: GovernorConfig,
    committed: MathState,
    vitals: Vitals,
    reflex: Reflex,
    bandit: Bandit,
    rng: StdRng,
    state: GovState,
    last_lr: f32,
    last_muon_lr: f32,
    cooldown_until: u64,
    trials_done: u32,
    snapshotted_once: bool,
    calibrated: bool,
    mem_path: Option<PathBuf>,
}

impl Governor {
    pub fn new(
        cfg: GovernorConfig,
        base_momentum: f32,
        base_weight_decay: f32,
        total_steps: u64,
        warmup_steps: u64,
        seed: u64,
        mem_path: Option<PathBuf>,
    ) -> Self {
        let bandit = mem_path.as_ref().map_or_else(Bandit::default, |p| Bandit::load(p));
        Self {
            committed: MathState::identity(base_momentum, base_weight_decay),
            vitals: Vitals::new(total_steps, warmup_steps),
            reflex: Reflex::default(),
            bandit,
            rng: StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
            state: GovState::Idle,
            last_lr: 0.0,
            last_muon_lr: 0.0,
            cooldown_until: 0,
            trials_done: 0,
            snapshotted_once: false,
            calibrated: false,
            mem_path,
            cfg,
        }
    }

    /// Record the step's scheduled LRs (so rollouts use realistic rates)
    /// and hand back the MSV to apply onto this step's configs.
    pub fn begin_step(&mut self, sched_lr: f32, sched_muon_lr: f32) -> &MathState {
        self.last_lr = sched_lr;
        self.last_muon_lr = sched_muon_lr;
        &self.committed
    }

    pub fn committed(&self) -> &MathState {
        &self.committed
    }

    pub fn eta(&self) -> f32 {
        self.vitals.eta()
    }

    pub fn set_val_gap(&mut self, gap: f32) {
        self.vitals.set_val_gap(gap);
    }

    pub fn save_memory(&self) {
        if let Some(p) = self.mem_path.as_ref() {
            let _ = self.bandit.save(p);
        }
    }

    /// Observe one completed training step. May roll back on divergence,
    /// advance an active trial by one paired sample, or open a new trial.
    /// Returns JSONL governor events for the host to log.
    pub fn end_step(&mut self, s: StepSample, backend: &mut dyn TrialBackend) -> Vec<String> {
        let mut events = Vec::new();
        self.vitals.push(&s);
        let fp = self.vitals.fingerprint();

        if let ReflexAction::CutLr(c) = self.reflex.check(&fp, &s) {
            if self.snapshotted_once {
                backend.restore_safe();
            }
            self.committed.lr_scale *= c;
            self.committed.muon_lr_scale *= c;
            self.committed.clamp();
            self.cooldown_until = s.step + self.cfg.cooldown;
            self.state = GovState::Idle;
            events.push(self.ev(s.step, "reflex", fp.regime, "divergence", "cut_lr", c, 0.0, 0));
            return events;
        }

        if matches!(self.state, GovState::Idle)
            && (!self.snapshotted_once
                || (self.cfg.snapshot_every > 0 && s.step % self.cfg.snapshot_every == 0))
        {
            backend.snapshot_safe();
            self.snapshotted_once = true;
        }

        match &self.state {
            GovState::Idle => {
                let due = self.cfg.diag_every > 0 && s.step % self.cfg.diag_every == 0;
                if due && s.step >= self.cooldown_until && self.trials_done < self.cfg.max_trials {
                    let diag = diagnose(&fp);
                    let (fault, p) = diag.top();
                    if fault.is_actionable() && p >= self.cfg.fault_threshold {
                        let arms = licensed_arms(fault);
                        if !arms.is_empty() && self.calibrated {
                            let arm = self.bandit.select(fp.regime, fault, &arms, &mut self.rng);
                            let mut treatment = arm.target(&self.committed);
                            treatment.clamp();
                            let cs = ConfidenceSequence::new(self.cfg.cs);
                            let trial = Trial::new(
                                fault, arm, fp.regime, self.committed, treatment, cs, s.step,
                            );
                            events.push(self.ev(
                                s.step, "trial_begin", fp.regime, fault.label(), arm.label(), 0.0,
                                0.0, 0,
                            ));
                            self.state = GovState::Trialing(trial);
                        } else if !arms.is_empty() {
                            // Calibrate the noise floor once before the first
                            // real trial — measures the true paired σ.
                            let calib_arm = Arm::new(ArmKind::LrUp, 1.3);
                            let mut treatment = calib_arm.target(&self.committed);
                            treatment.clamp();
                            events.push(self.ev(
                                s.step, "calibrate_begin", fp.regime, "noise", calib_arm.label(),
                                0.0, 0.0, 0,
                            ));
                            self.state =
                                GovState::Calibrating(Calib::new(treatment, self.cfg.calib_samples));
                        }
                    } else if p >= self.cfg.fault_threshold && fault != Fault::Healthy {
                        events.push(self.ev(
                            s.step, "diagnose", fp.regime, fault.label(), "report_only", 0.0, 0.0, 0,
                        ));
                    }
                }
            }
            GovState::Calibrating(_) => {
                let treatment = match &self.state {
                    GovState::Calibrating(c) => c.treatment,
                    _ => unreachable!(),
                };
                let diff =
                    backend.paired_diff(&self.committed, &treatment, self.last_lr, self.last_muon_lr);
                let done = match &mut self.state {
                    GovState::Calibrating(c) => {
                        c.push(diff);
                        c.n >= c.target_n
                    }
                    _ => unreachable!(),
                };
                if done {
                    let (std, n) = match &self.state {
                        GovState::Calibrating(c) => (c.std(), c.n),
                        _ => unreachable!(),
                    };
                    // σ = measured spread, inflated for safety and floored so a
                    // degenerate (≈0) measurement can't make the test reckless.
                    let sigma = (std * 1.5).max(0.001);
                    self.cfg.cs.sigma = sigma;
                    self.calibrated = true;
                    self.cooldown_until = s.step + self.cfg.cooldown;
                    events.push(self.ev(s.step, "calibrate", fp.regime, "noise", "lr_up", sigma, std, n));
                    self.state = GovState::Idle;
                }
            }
            GovState::Trialing(_) => {
                let (control, treatment) = match &self.state {
                    GovState::Trialing(t) => (t.control, t.treatment),
                    _ => unreachable!(),
                };
                let diff =
                    backend.paired_diff(&control, &treatment, self.last_lr, self.last_muon_lr);
                let (verdict, fault, arm, regime, effect, margin, n) = match &mut self.state {
                    GovState::Trialing(t) => {
                        let v = t.observe_diff(diff);
                        (v, t.fault, t.arm, t.regime, t.effect(), t.cs.margin() as f32, t.cs.n())
                    }
                    _ => unreachable!(),
                };
                match verdict {
                    Verdict::Continue => {}
                    Verdict::Adopt => {
                        self.committed = treatment;
                        self.committed.clamp();
                        self.bandit.reward(regime, fault, arm, effect);
                        self.cooldown_until = s.step + self.cfg.cooldown;
                        self.trials_done += 1;
                        events.push(self.ev(
                            s.step, "trial_adopt", regime, fault.label(), arm.label(), effect,
                            margin, n,
                        ));
                        self.state = GovState::Idle;
                    }
                    Verdict::Revert => {
                        self.bandit.reward(regime, fault, arm, effect);
                        self.cooldown_until = s.step + self.cfg.cooldown;
                        self.trials_done += 1;
                        events.push(self.ev(
                            s.step, "trial_revert", regime, fault.label(), arm.label(), effect,
                            margin, n,
                        ));
                        self.state = GovState::Idle;
                    }
                    Verdict::BelowFloor => {
                        self.bandit.reward(regime, fault, arm, 0.0);
                        self.cooldown_until = s.step + self.cfg.cooldown;
                        self.trials_done += 1;
                        events.push(self.ev(
                            s.step, "trial_belowfloor", regime, fault.label(), arm.label(), effect,
                            margin, n,
                        ));
                        self.state = GovState::Idle;
                    }
                }
            }
        }
        events
    }

    fn ev(
        &self,
        step: u64,
        kind: &str,
        regime: Regime,
        fault: &str,
        arm: &str,
        effect: f32,
        margin: f32,
        n: usize,
    ) -> String {
        format!(
            "{{\"event\":\"governor\",\"step\":{step},\"kind\":\"{kind}\",\"regime\":\"{}\",\"fault\":\"{fault}\",\"arm\":\"{arm}\",\"eta\":{:.6},\"effect\":{:.6},\"margin\":{:.6},\"n\":{n},\"lr_scale\":{:.4},\"muon_lr_scale\":{:.4},\"wd\":{:.4},\"clip_scale\":{:.4}}}",
            regime.label(),
            self.vitals.eta(),
            effect,
            margin,
            self.committed.lr_scale,
            self.committed.muon_lr_scale,
            self.committed.weight_decay,
            self.committed.grad_clip_scale,
        )
    }
}
