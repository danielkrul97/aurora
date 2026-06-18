#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::must_use_candidate,
    clippy::suboptimal_flops
)]

//! Anytime-valid sequential test on a stream of paired A/B differences.
//!
//! Each `push`ed value is one paired probe difference `bpb_control −
//! bpb_treatment` (positive = the candidate arm is better). The
//! normal-mixture confidence sequence gives a time-uniform interval for
//! the mean effect: it is valid under optional stopping, so the governor
//! can peek after every sample and stop the instant the interval clears
//! the noise floor — without inflating the false-adopt rate.
//!
//! Boundary (Robbins normal mixture, sub-Gaussian scale σ, mixing
//! parameter ρ): for the running mean `m_n`,
//!
//! ```text
//! margin(n) = (σ / n) · sqrt( 2 (n + ρ) · ln( sqrt((n + ρ)/ρ) / α ) )
//! ```
//!
//! and `P(∃ n : |m_n − μ| ≥ margin(n)) ≤ α`. Adopt only when the whole
//! interval sits above `+floor`; revert when it sits below `−floor`.

#[derive(Copy, Clone, Debug)]
pub struct CsConfig {
    /// Time-uniform error level (two-sided). 0.05 = 95% confidence seq.
    pub alpha: f32,
    /// Mixture parameter; the boundary is tightest near n ≈ 1/ρ samples.
    pub rho: f32,
    /// Sub-Gaussian scale of the paired difference. A calibrated upper
    /// bound — the boundary widens further if the empirical std exceeds it.
    pub sigma: f32,
    /// Effect must exceed this magnitude to count (the noise floor, in
    /// the same units as the paired difference — BPB).
    pub floor: f32,
    /// Give up (BelowFloor) once this many samples fail to clear the floor.
    pub n_max: usize,
}

impl Default for CsConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            rho: 5.0,
            sigma: 0.02,
            floor: 0.003,
            n_max: 40,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    Adopt,
    Revert,
    BelowFloor,
}

#[derive(Clone, Debug)]
pub struct ConfidenceSequence {
    cfg: CsConfig,
    n: usize,
    /// Total push attempts incl. non-finite ones, so an all-NaN trial
    /// still reaches n_max instead of spinning forever.
    pushed: usize,
    sum: f64,
    sumsq: f64,
}

impl ConfidenceSequence {
    pub fn new(cfg: CsConfig) -> Self {
        Self {
            cfg,
            n: 0,
            pushed: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    pub fn push(&mut self, x: f32) {
        self.pushed += 1;
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        self.sum += f64::from(x);
        self.sumsq += f64::from(x) * f64::from(x);
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    /// Max of the configured sub-Gaussian bound and the empirical std.
    /// Using the larger of the two only ever *widens* the interval, so it
    /// preserves the one-sided false-adopt guarantee while adapting to a
    /// noisier-than-assumed signal.
    fn effective_sigma(&self) -> f64 {
        let cfg_sigma = f64::from(self.cfg.sigma).max(1e-9);
        if self.n < 2 {
            return cfg_sigma;
        }
        let n = self.n as f64;
        let mean = self.sum / n;
        let var = (self.sumsq / n - mean * mean).max(0.0);
        var.sqrt().max(cfg_sigma)
    }

    pub fn margin(&self) -> f64 {
        if self.n == 0 {
            return f64::INFINITY;
        }
        let n = self.n as f64;
        let rho = f64::from(self.cfg.rho).max(1e-6);
        let sigma = self.effective_sigma();
        let alpha = f64::from(self.cfg.alpha).clamp(1e-6, 0.5);
        // ln( sqrt((n+ρ)/ρ) / α ) = 0.5·ln((n+ρ)/ρ) − ln α  (always > 0).
        let ln_term = (0.5 * ((n + rho) / rho).ln() - alpha.ln()).max(1e-9);
        (sigma / n) * (2.0 * (n + rho) * ln_term).sqrt()
    }

    pub fn verdict(&self) -> Verdict {
        if self.n > 0 {
            let mean = self.mean();
            let margin = self.margin();
            let floor = f64::from(self.cfg.floor).max(0.0);
            if mean - margin > floor {
                return Verdict::Adopt;
            }
            if mean + margin < -floor {
                return Verdict::Revert;
            }
        }
        if self.pushed >= self.cfg.n_max {
            return Verdict::BelowFloor;
        }
        Verdict::Continue
    }
}
