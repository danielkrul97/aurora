#![allow(clippy::must_use_candidate, clippy::suboptimal_flops)]

//! Differential diagnosis. From the symptom fingerprint we score a fixed
//! taxonomy of training faults and normalize to a posterior. The governor
//! only opens a trial when an *actionable* fault holds enough posterior
//! mass — and even then the verifier, not this module, decides whether the
//! proposed fix actually helps. Coarse diagnosis is therefore safe: its
//! only job is to narrow the arm set the bandit explores.

use crate::governor::vitals::{Fingerprint, Regime};

/// Multiple of the loss-noise band (`loss_noise / loss_level`) an up-trend must
/// clear before it counts toward `lr_too_hot`. Below it, the mid/slow loss-EWMA
/// cross is just minibatch noise — treating it as worsening cools a healthy LR
/// (the enwik9 run, where σ≈0.25 noise dwarfed the per-step signal).
const HOT_TREND_NOISE_K: f32 = 0.4;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    Healthy,
    LrTooHot,
    LrTooCold,
    GradNoiseHigh,
    ThroughputRegression,
    OverfitOnset,
    DepthInstability,
    CapacityWall,
}

impl Fault {
    pub const ALL: [Fault; 8] = [
        Fault::Healthy,
        Fault::LrTooHot,
        Fault::LrTooCold,
        Fault::GradNoiseHigh,
        Fault::ThroughputRegression,
        Fault::OverfitOnset,
        Fault::DepthInstability,
        Fault::CapacityWall,
    ];

    pub fn idx(self) -> u8 {
        Self::ALL.iter().position(|&f| f == self).unwrap_or(0) as u8
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::LrTooHot => "lr_too_hot",
            Self::LrTooCold => "lr_too_cold",
            Self::GradNoiseHigh => "grad_noise_high",
            Self::ThroughputRegression => "throughput_regression",
            Self::OverfitOnset => "overfit_onset",
            Self::DepthInstability => "depth_instability",
            Self::CapacityWall => "capacity_wall",
        }
    }

    /// Faults the governor can try to treat with a live scalar change.
    /// The rest are report-only (throughput, depth) or out of scope
    /// (capacity wall needs a structural change).
    pub fn is_actionable(self) -> bool {
        matches!(
            self,
            Self::LrTooHot | Self::LrTooCold | Self::GradNoiseHigh | Self::OverfitOnset
        )
    }

    /// Slow-acting faults whose fix (e.g. weight decay for overfitting) can't
    /// be confirmed by a short paired rollout — they go through the
    /// tentative-commit + val-watch probation path instead.
    pub fn is_slow(self) -> bool {
        matches!(self, Self::OverfitOnset)
    }
}

#[derive(Clone, Debug)]
pub struct Diagnosis {
    /// Faults with posterior mass, sorted descending.
    pub posterior: Vec<(Fault, f32)>,
}

impl Diagnosis {
    pub fn top(&self) -> (Fault, f32) {
        self.posterior.first().copied().unwrap_or((Fault::Healthy, 0.0))
    }

    /// True when the top two faults are too close to call apart — the cue
    /// to run a discriminating probe before committing to one of them.
    pub fn is_confusable(&self) -> bool {
        if self.posterior.len() < 2 {
            return false;
        }
        let a = self.posterior[0].1;
        let b = self.posterior[1].1;
        b > 1e-6 && a < 1.25 * b
    }
}

pub fn diagnose(fp: &Fingerprint) -> Diagnosis {
    let eps = 1e-6;
    // Use the long-horizon trend, not the per-step slope: per-step improvement
    // (~0.003 nats) is dwarfed by batch noise (~0.2), so the slope reads ~0 and
    // healthy descent looks like a plateau. `trend_ref` = the relative drop
    // that counts as clearly improving (0.4 % between the mid/slow EWMAs).
    let trend_ref = 0.004;
    let improving = (fp.loss_trend / trend_ref).max(0.0);
    // Worsening must clear the loss's own noise band, else healthy-but-noisy
    // descent (mid/slow EWMA crossing on minibatch noise) mis-reads as
    // lr_too_hot. A real up-trend clears it by a wide margin; a true grad spike
    // still fires hot through its own term below.
    let trend_band = (fp.loss_noise / fp.loss_level.max(1e-6)).max(0.0);
    let worsening = ((-fp.loss_trend - HOT_TREND_NOISE_K * trend_band) / trend_ref).max(0.0);
    let stalled = 1.0 / (1.0 + 4.0 * improving);
    let not_warmup = f32::from(fp.regime != Regime::Warmup);
    let alive = f32::from(fp.grad_level > 1e-4);
    let grad_ok = (-(fp.grad_spike - 1.0).max(0.0).powi(2)).exp();
    let late = f32::from(fp.regime == Regime::Late);

    // Schedule-aware gating: "too hot" is a false alarm while the LR ramps
    // (warmup grads are naturally large), and "too cold → raise LR" is the
    // wrong fix while the LR decays by design (WSD tail). Hard-gate both —
    // a genuine warmup explosion is still caught by the reflex (loss
    // divergence), which doesn't go through diagnosis.
    let hot_sched = f32::from(!fp.lr_rising);
    let cold_sched = f32::from(!fp.lr_falling);

    // Cap healthy so a fast-improving train loss can't drown a real
    // structural fault (overfit / throughput), which lives on a different
    // axis — you can be improving train loss *and* overfitting.
    let healthy = improving.clamp(0.25, 3.0);
    let hot = ((fp.grad_spike - 2.5).max(0.0) * 1.5 + worsening * 2.0) * hot_sched;
    let cold = stalled * grad_ok * not_warmup * alive * 2.0 * cold_sched;
    let noise = (fp.grad_cv - 0.5).max(0.0) * 2.0;
    let through = (fp.dt_drift - 1.1).max(0.0) * 8.0;
    let overfit = (fp.val_gap - 0.05).max(0.0) * 12.0;
    let depth = 0.0;
    let wall = stalled * not_warmup * alive * late * 0.8;

    let raw = [
        (Fault::Healthy, healthy),
        (Fault::LrTooHot, hot),
        (Fault::LrTooCold, cold),
        (Fault::GradNoiseHigh, noise),
        (Fault::ThroughputRegression, through),
        (Fault::OverfitOnset, overfit),
        (Fault::DepthInstability, depth),
        (Fault::CapacityWall, wall),
    ];
    let sum: f32 = raw.iter().map(|(_, s)| s.max(0.0)).sum::<f32>().max(eps);
    let mut posterior: Vec<(Fault, f32)> =
        raw.iter().map(|(f, s)| (*f, s.max(0.0) / sum)).collect();
    posterior.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Diagnosis { posterior }
}
