#![allow(
    clippy::cast_precision_loss,
    clippy::must_use_candidate,
    clippy::suboptimal_flops
)]

//! Afferent bus: cheap O(1)-per-step signals the governor reasons over.
//! Everything is an exponential moving average at one of two timescales,
//! so the structure is fixed-size regardless of run length. The derived
//! `Fingerprint` is the symptom vector handed to differential diagnosis.

#[derive(Clone, Copy, Debug)]
pub struct Ewma {
    value: f32,
    alpha: f32,
    inited: bool,
}

impl Ewma {
    pub fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha,
            inited: false,
        }
    }

    pub fn push(&mut self, x: f32) {
        if !x.is_finite() {
            return;
        }
        if self.inited {
            self.value += self.alpha * (x - self.value);
        } else {
            self.value = x;
            self.inited = true;
        }
    }

    pub fn get(&self) -> f32 {
        self.value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepSample {
    pub step: u64,
    /// Mean cross-entropy this step, in nats.
    pub loss: f32,
    /// AdamW-subset pre-clip gradient norm.
    pub grad_norm: f32,
    /// Wall-clock for this step, seconds.
    pub dt_secs: f32,
    /// Effective AdamW LR applied this step (post-governor).
    pub lr: f32,
    pub muon_lr: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Warmup,
    Early,
    Mid,
    Late,
}

impl Regime {
    pub fn idx(self) -> u8 {
        match self {
            Self::Warmup => 0,
            Self::Early => 1,
            Self::Mid => 2,
            Self::Late => 3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Early => "early",
            Self::Mid => "mid",
            Self::Late => "late",
        }
    }
}

/// Snapshot of derived symptoms at one step. Pure data — diagnosis and
/// reflex read it, nothing else.
#[derive(Clone, Copy, Debug)]
pub struct Fingerprint {
    pub regime: Regime,
    /// EWMA of Δloss per step (negative = improving).
    pub loss_slope: f32,
    /// Std of Δloss (step-to-step loss noise).
    pub loss_noise: f32,
    /// Long-horizon loss level.
    pub loss_level: f32,
    /// Long-horizon grad-norm level.
    pub grad_level: f32,
    /// Latest grad norm / long-horizon grad norm (spike detector).
    pub grad_spike: f32,
    /// Grad-norm coefficient of variation.
    pub grad_cv: f32,
    /// Short/long dt ratio (throughput drift; >1 = slowing down).
    pub dt_drift: f32,
    /// NaN/Inf or a loss explosion happened this step.
    pub diverging: bool,
    /// val_bpb − train_bpb at the last epoch eval (overfit signal).
    pub val_gap: f32,
    /// Relative loss improvement over a long horizon: `(slow_ewma −
    /// mid_ewma) / slow_ewma`, positive = still learning. Robust to per-step
    /// batch noise (which buries the tiny per-step slope), so it separates a
    /// real plateau from healthy-descent-through-noise.
    pub loss_trend: f32,
    /// Scheduled LR is currently ramping up (warmup) — elevated grads are
    /// expected, so "too hot" is a false alarm here.
    pub lr_rising: bool,
    /// Scheduled LR is currently decaying (e.g. WSD tail) — the LR is meant
    /// to drop, so "too cold → raise LR" is the wrong fix here.
    pub lr_falling: bool,
}

pub struct Vitals {
    loss_fast: Ewma,
    loss_mid: Ewma,
    loss_slow: Ewma,
    dloss: Ewma,
    dloss_sq: Ewma,
    grad_slow: Ewma,
    grad_sq: Ewma,
    dt_fast: Ewma,
    dt_slow: Ewma,
    lr_slow: Ewma,
    prev_loss: Option<f32>,
    last: Option<StepSample>,
    /// Last finite grad norm. On GPU the norm is only read back on report
    /// steps (a non-report step returns NaN to avoid a device stall), so
    /// `StepSample::grad_norm` may be NaN — "not measured", not divergence.
    last_grad: f32,
    val_gap: f32,
    total_steps: u64,
    warmup_steps: u64,
}

impl Vitals {
    pub fn new(total_steps: u64, warmup_steps: u64) -> Self {
        Self {
            loss_fast: Ewma::new(0.3),
            loss_mid: Ewma::new(0.05),
            loss_slow: Ewma::new(0.01),
            dloss: Ewma::new(0.1),
            dloss_sq: Ewma::new(0.1),
            grad_slow: Ewma::new(0.03),
            grad_sq: Ewma::new(0.03),
            dt_fast: Ewma::new(0.2),
            dt_slow: Ewma::new(0.02),
            lr_slow: Ewma::new(0.02),
            prev_loss: None,
            last: None,
            last_grad: 0.0,
            val_gap: 0.0,
            total_steps: total_steps.max(1),
            warmup_steps,
        }
    }

    pub fn push(&mut self, s: &StepSample) {
        if let Some(prev) = self.prev_loss {
            let d = s.loss - prev;
            self.dloss.push(d);
            self.dloss_sq.push(d * d);
        }
        if s.loss.is_finite() {
            self.loss_fast.push(s.loss);
            self.loss_mid.push(s.loss);
            self.loss_slow.push(s.loss);
            self.prev_loss = Some(s.loss);
        }
        if s.grad_norm.is_finite() {
            self.grad_slow.push(s.grad_norm);
            self.grad_sq.push(s.grad_norm * s.grad_norm);
            self.last_grad = s.grad_norm;
        }
        self.dt_fast.push(s.dt_secs);
        self.dt_slow.push(s.dt_secs);
        if s.lr.is_finite() {
            self.lr_slow.push(s.lr);
        }
        self.last = Some(*s);
    }

    pub fn set_val_gap(&mut self, gap: f32) {
        self.val_gap = gap;
    }

    pub fn regime(&self) -> Regime {
        let step = self.last.map_or(0, |s| s.step);
        // Cap the warmup regime at 10% of the run — warmup_steps can be most
        // of a short run (the user's run had warmup 2000 of 2564), which
        // would otherwise swallow early/mid and degenerate the bandit context.
        let warmup_cap = self.warmup_steps.min(self.total_steps / 10);
        if step < warmup_cap {
            return Regime::Warmup;
        }
        let frac = step as f32 / self.total_steps as f32;
        if frac < 0.3 {
            Regime::Early
        } else if frac < 0.7 {
            Regime::Mid
        } else {
            Regime::Late
        }
    }

    /// Efficiency: nats of loss improvement per wall-clock second. The one
    /// number the governor ultimately maximizes (speed × quality).
    pub fn eta(&self) -> f32 {
        let dt = self.dt_slow.get().max(1e-6);
        -self.loss_slope() / dt
    }

    fn loss_slope(&self) -> f32 {
        self.dloss.get()
    }

    fn loss_noise(&self) -> f32 {
        let mean = self.dloss.get();
        (self.dloss_sq.get() - mean * mean).max(0.0).sqrt()
    }

    fn grad_cv(&self) -> f32 {
        let mean = self.grad_slow.get().max(1e-9);
        let var = (self.grad_sq.get() - mean * mean).max(0.0);
        var.sqrt() / mean
    }

    pub fn fingerprint(&self) -> Fingerprint {
        let last = self.last;
        let grad_now = self.last_grad;
        let grad_level = self.grad_slow.get();
        let grad_spike = if grad_level > 1e-9 {
            grad_now / grad_level
        } else {
            1.0
        };
        // Divergence keys on loss only — it is computed every step on both
        // backends, whereas grad_norm may be unmeasured (NaN) on GPU.
        let diverging = last.is_some_and(|s| !s.loss.is_finite());
        // LR direction from the schedule: current LR vs its own slow EWMA.
        // Monotone within a phase, so the sign is clean; the lag only blurs
        // the few steps around a phase change.
        let lr_now = last.map_or(0.0, |s| s.lr);
        let lr_ref = self.lr_slow.get();
        let lr_rising = lr_ref > 1e-12 && lr_now > lr_ref * 1.005;
        let lr_falling = lr_ref > 1e-12 && lr_now < lr_ref * 0.995;
        let slow = self.loss_slow.get();
        let loss_trend = (slow - self.loss_mid.get()) / slow.max(1e-6);
        Fingerprint {
            regime: self.regime(),
            loss_slope: self.loss_slope(),
            loss_noise: self.loss_noise(),
            loss_level: self.loss_slow.get(),
            grad_level,
            grad_spike,
            grad_cv: self.grad_cv(),
            dt_drift: self.dt_fast.get() / self.dt_slow.get().max(1e-9),
            diverging,
            val_gap: self.val_gap,
            loss_trend,
            lr_rising,
            lr_falling,
        }
    }
}
