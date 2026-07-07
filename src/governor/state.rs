#![allow(clippy::must_use_candidate)]

//! The Math State Vector (MSV): the small bundle of live scalars the
//! training step reads each iteration. Every field maps to a value that
//! Aurora already rebuilds per step (`AdamWConfig`/`MuonConfig` at the top
//! of the loop) or reads each forward (`softcap`), so changing the MSV
//! changes the math with no reallocation and no recompile.

use crate::optimizer::{AdamWConfig, MuonConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MathState {
    /// Multiplier on the scheduled AdamW learning rate.
    pub lr_scale: f32,
    /// Multiplier on the scheduled Muon learning rate.
    pub muon_lr_scale: f32,
    /// Absolute Muon momentum.
    pub momentum: f32,
    /// Absolute AdamW weight decay (embed/head/LN groups).
    pub weight_decay: f32,
    /// Absolute Muon weight decay (the matrix bulk — Q/K/V/O, FC, proj). The
    /// strong regularizer for overfitting; AdamW-group WD alone only touches
    /// embeddings.
    pub muon_weight_decay: f32,
    /// Multiplier on the AdamW-subset gradient-clip threshold.
    pub grad_clip_scale: f32,
    /// Absolute logit softcap; `< 0` means "leave the model's value".
    pub softcap: f32,
}

impl MathState {
    /// Identity state: scalars at 1.0, momentum/decay at the run's base.
    pub fn identity(base_momentum: f32, base_weight_decay: f32, base_muon_wd: f32) -> Self {
        Self {
            lr_scale: 1.0,
            muon_lr_scale: 1.0,
            momentum: base_momentum,
            weight_decay: base_weight_decay,
            muon_weight_decay: base_muon_wd,
            grad_clip_scale: 1.0,
            softcap: -1.0,
        }
    }

    /// Apply the MSV onto this step's freshly-built configs and return the
    /// effective grad-clip threshold. The configs already carry the
    /// scheduled LR; we scale on top of it.
    pub fn apply_optim(
        &self,
        adam: &mut AdamWConfig,
        muon: &mut MuonConfig,
        base_clip: f32,
    ) -> f32 {
        adam.lr *= self.lr_scale;
        adam.weight_decay = self.weight_decay;
        muon.lr *= self.muon_lr_scale;
        muon.momentum = self.momentum;
        muon.weight_decay = self.muon_weight_decay;
        base_clip * self.grad_clip_scale
    }

    /// Keep every field inside a sane band so repeated adopts can't drift
    /// the run into an absurd regime.
    pub fn clamp(&mut self) {
        self.lr_scale = self.lr_scale.clamp(0.1, 4.0);
        self.muon_lr_scale = self.muon_lr_scale.clamp(0.1, 4.0);
        self.momentum = self.momentum.clamp(0.5, 0.999);
        self.weight_decay = self.weight_decay.clamp(0.0, 1.0);
        self.muon_weight_decay = self.muon_weight_decay.clamp(0.0, 1.0);
        self.grad_clip_scale = self.grad_clip_scale.clamp(0.25, 4.0);
        if self.softcap >= 0.0 {
            self.softcap = self.softcap.clamp(1.0, 60.0);
        }
    }
}
