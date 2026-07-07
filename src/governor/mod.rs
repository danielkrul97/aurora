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

/// Evals past the best val before the early-stop recommendation fires.
const EARLY_STOP_PATIENCE: u32 = 2;
/// Consecutive probation reverts before the governor gives up on the slow
/// lever and falls back to report-only (the human must intervene).
const GIVE_UP_REVERTS: u32 = 3;
/// Slack on the val-degradation rate when deciding to keep a probation —
/// keep unless the change made the trajectory clearly worse than its trend.
const PROBATION_RATE_MARGIN: f32 = 0.01;
/// Floor on trial-driven LR/Muon-LR cooling. The short paired rollout is
/// biased toward lower LR (less immediate loss variance), so without a bound
/// `lr_down` ratchets the LR down against a tuned schedule. A deeper cut is
/// clamped to an A/A probe (→ BelowFloor → backoff) instead of ratcheting.
/// Genuine divergence still cuts harder via the reflex, which bypasses this.
const MIN_TRIAL_LR_SCALE: f32 = 0.5;
/// After this many consecutive below-floor trials the inter-trial cooldown
/// grows as `cooldown · 2^min(streak, cap)`, so a converged governor stops
/// burning probe rollouts. Any adopt / revert / divergence / regime change
/// resets the streak back to dense probing.
const QUIET_BACKOFF_CAP: u32 = 6;

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

/// Tentative commit of a slow-acting lever (weight decay for overfitting),
/// whose effect a short paired rollout can't see. It is applied live and
/// kept only if val BPB doesn't regress at the next eval — a *guarded direct
/// action*, not a rigorous A/B. The keep/revert is a safety net (did val get
/// worse?), not a counterfactual efficacy test, so it is weaker than the
/// fast-lever sequential test; documented as such.
struct Probation {
    fault: Fault,
    arm: Arm,
    regime: Regime,
    pre_msv: MathState,
    baseline_val: f32,
    /// Per-eval val change just before the commit — the trajectory the
    /// change has to beat (rate-based keep, not absolute).
    baseline_rate: f32,
}

enum GovState {
    Idle,
    Calibrating(Calib),
    Trialing(Trial),
    Probation(Probation),
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
    last_step: u64,
    last_val: f32,
    prev_val: f32,
    best_val: f32,
    best_eval: u32,
    eval_count: u32,
    probation_reverts: u32,
    overfit_givenup: bool,
    quiet_streak: u32,
    last_regime: Regime,
    mem_path: Option<PathBuf>,
}

impl Governor {
    pub fn new(
        cfg: GovernorConfig,
        base_momentum: f32,
        base_weight_decay: f32,
        base_muon_wd: f32,
        total_steps: u64,
        warmup_steps: u64,
        seed: u64,
        mem_path: Option<PathBuf>,
    ) -> Self {
        let bandit = mem_path.as_ref().map_or_else(Bandit::default, |p| Bandit::load(p));
        Self {
            committed: MathState::identity(base_momentum, base_weight_decay, base_muon_wd),
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
            last_step: 0,
            last_val: f32::INFINITY,
            prev_val: f32::INFINITY,
            best_val: f32::INFINITY,
            best_eval: 0,
            eval_count: 0,
            probation_reverts: 0,
            overfit_givenup: false,
            quiet_streak: 0,
            last_regime: Regime::Warmup,
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

    /// Observe an epoch eval. Updates the overfit signal and resolves an
    /// active slow-lever probation: keep the tentative commit unless val BPB
    /// regressed vs the pre-commit baseline, in which case revert it.
    pub fn observe_eval(&mut self, val_bpb: f32, train_bpb: f32) -> Vec<String> {
        let mut events = Vec::new();
        self.vitals.set_val_gap(val_bpb - train_bpb);
        self.eval_count += 1;

        if matches!(self.state, GovState::Probation(_)) {
            let (fault, arm, regime, pre, baseline, base_rate) = match &self.state {
                GovState::Probation(p) => {
                    (p.fault, p.arm, p.regime, p.pre_msv, p.baseline_val, p.baseline_rate)
                }
                _ => unreachable!(),
            };
            // Rate-based, not absolute: during monotonic overfitting val rises
            // no matter what, so keep unless the change made the trajectory
            // clearly *worse than it was already heading*.
            let probation_rate = val_bpb - baseline;
            let worsened = probation_rate > base_rate + PROBATION_RATE_MARGIN;
            let kind = if worsened {
                self.committed = pre;
                self.probation_reverts += 1;
                "probation_revert"
            } else {
                self.probation_reverts = 0;
                "probation_keep"
            };
            self.bandit.reward(regime, fault, arm, -probation_rate);
            self.cooldown_until = self.last_step + self.cfg.cooldown;
            events.push(self.ev(self.last_step, kind, regime, fault.label(), arm.label(), val_bpb, base_rate, 0));
            self.state = GovState::Idle;
            if self.probation_reverts >= GIVE_UP_REVERTS && !self.overfit_givenup {
                self.overfit_givenup = true;
                events.push(self.ev(self.last_step, "probation_giveup", regime, fault.label(), arm.label(), val_bpb, 0.0, 0));
            }
        }

        // Early-stop recommendation. The governor can't un-train, but it sees
        // the val trajectory: if we are N evals past the best, recommend
        // stopping loudly (mid-run overfitting is the human's call).
        if val_bpb < self.best_val {
            self.best_val = val_bpb;
            self.best_eval = self.eval_count;
        }
        let since_best = self.eval_count - self.best_eval;
        if since_best >= EARLY_STOP_PATIENCE {
            events.push(self.ev_earlystop(val_bpb, since_best));
        }

        self.prev_val = self.last_val;
        self.last_val = val_bpb;
        events
    }

    /// True once the run is `EARLY_STOP_PATIENCE` evals past its best val —
    /// the host may break the epoch loop on this if it wants auto-stop.
    pub fn should_stop(&self) -> bool {
        self.best_eval > 0 && self.eval_count.saturating_sub(self.best_eval) >= EARLY_STOP_PATIENCE
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
        self.last_step = s.step;
        let fp = self.vitals.fingerprint();
        // A regime change shifts the bandit's context (and the schedule phase),
        // so resume dense probing rather than carrying a stale quiet streak.
        if fp.regime != self.last_regime {
            self.last_regime = fp.regime;
            self.quiet_streak = 0;
        }

        if let ReflexAction::CutLr(c) = self.reflex.check(&fp, &s) {
            if self.snapshotted_once {
                backend.restore_safe();
            }
            self.committed.lr_scale *= c;
            self.committed.muon_lr_scale *= c;
            self.committed.clamp();
            self.quiet_streak = 0;
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
                        if arms.is_empty() {
                            // nothing licensed — fall through
                        } else if fault.is_slow() {
                            // Slow lever (weight decay) — a paired rollout can't
                            // see its effect, so tentatively commit it live and
                            // verify at the next eval. Once we've given up
                            // (repeated reverts), the early-stop reco carries
                            // the signal instead.
                            if !self.overfit_givenup {
                                let arm =
                                    self.bandit.select(fp.regime, fault, &arms, &mut self.rng);
                                let pre = self.committed;
                                self.committed = arm.target(&self.committed);
                                self.committed.clamp();
                                let base_rate = if self.prev_val.is_finite() {
                                    self.last_val - self.prev_val
                                } else {
                                    0.0
                                };
                                events.push(self.ev(
                                    s.step, "probation_begin", fp.regime, fault.label(),
                                    arm.label(), self.last_val, base_rate, 0,
                                ));
                                self.state = GovState::Probation(Probation {
                                    fault,
                                    arm,
                                    regime: fp.regime,
                                    pre_msv: pre,
                                    baseline_val: self.last_val,
                                    baseline_rate: base_rate,
                                });
                            }
                        } else if self.calibrated {
                            let arm = self.bandit.select(fp.regime, fault, &arms, &mut self.rng);
                            let mut treatment = arm.target(&self.committed);
                            treatment.clamp();
                            // Bound trial-driven cooling: a would-be cut past the
                            // floor (or past an already-cut level) collapses to an
                            // A/A probe, so it reads BelowFloor instead of ratcheting
                            // the LR down on the rollout's short-horizon bias.
                            let floor_lr = MIN_TRIAL_LR_SCALE.min(self.committed.lr_scale);
                            let floor_mlr = MIN_TRIAL_LR_SCALE.min(self.committed.muon_lr_scale);
                            treatment.lr_scale = treatment.lr_scale.max(floor_lr);
                            treatment.muon_lr_scale = treatment.muon_lr_scale.max(floor_mlr);
                            let cs = ConfidenceSequence::new(self.cfg.cs);
                            let trial = Trial::new(
                                fault, arm, fp.regime, self.committed, treatment, cs, s.step,
                            );
                            events.push(self.ev(
                                s.step, "trial_begin", fp.regime, fault.label(), arm.label(), 0.0,
                                0.0, 0,
                            ));
                            self.state = GovState::Trialing(trial);
                        } else {
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
                        self.quiet_streak = 0;
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
                        self.quiet_streak = 0;
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
                        self.quiet_streak = self.quiet_streak.saturating_add(1);
                        let mult = 1u64 << self.quiet_streak.min(QUIET_BACKOFF_CAP);
                        self.cooldown_until = s.step + self.cfg.cooldown * mult;
                        self.trials_done += 1;
                        events.push(self.ev(
                            s.step, "trial_belowfloor", regime, fault.label(), arm.label(), effect,
                            margin, n,
                        ));
                        self.state = GovState::Idle;
                    }
                }
            }
            GovState::Probation(_) => {}
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

    fn ev_earlystop(&self, val: f32, since_best: u32) -> String {
        format!(
            "{{\"event\":\"governor\",\"step\":{},\"kind\":\"early_stop\",\"regime\":\"{}\",\"fault\":\"overfit\",\"arm\":\"stop\",\"eta\":{:.6},\"val\":{:.4},\"best_val\":{:.4},\"best_epoch\":{},\"evals_past_best\":{since_best},\"wd\":{:.4},\"muon_wd\":{:.4}}}",
            self.last_step,
            self.vitals.regime().label(),
            self.vitals.eta(),
            val,
            self.best_val,
            self.best_eval.saturating_sub(1),
            self.committed.weight_decay,
            self.committed.muon_weight_decay,
        )
    }
}
