#![allow(clippy::must_use_candidate)]

//! The reflex arc: the only part of the governor that acts without a
//! trial, because divergence is unambiguous and waiting for a sequential
//! test would be too late. It fires on NaN/Inf or a genuine loss
//! explosion, and stays quiet through ordinary stochastic spikes — a
//! rollback throws away real progress, so the bar to trip is high.

use crate::governor::vitals::{Fingerprint, StepSample};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ReflexAction {
    None,
    /// Restore the last safe snapshot and multiply the LR scales by this.
    CutLr(f32),
}

#[derive(Copy, Clone, Debug)]
pub struct Reflex {
    /// Loss must exceed `level + jump_sigmas·noise` to count as explosion.
    jump_sigmas: f32,
    /// Grad spike ratio above which, combined with a worsening loss, trips.
    spike_ratio: f32,
    hard_cut: f32,
    soft_cut: f32,
}

impl Default for Reflex {
    fn default() -> Self {
        Self {
            jump_sigmas: 8.0,
            spike_ratio: 8.0,
            hard_cut: 0.3,
            soft_cut: 0.5,
        }
    }
}

impl Reflex {
    pub fn check(&self, fp: &Fingerprint, s: &StepSample) -> ReflexAction {
        // Only a non-finite loss is divergence. grad_norm may be NaN on GPU
        // non-report steps (skipped readback), which is "not measured".
        if !s.loss.is_finite() {
            return ReflexAction::CutLr(self.hard_cut);
        }
        let worsening = fp.loss_slope > 0.0;
        let exploded = fp.loss_noise > 0.0
            && s.loss > fp.loss_level + self.jump_sigmas * fp.loss_noise
            && worsening;
        let spiking = fp.grad_spike > self.spike_ratio && worsening;
        if exploded || spiking {
            ReflexAction::CutLr(self.soft_cut)
        } else {
            ReflexAction::None
        }
    }
}
