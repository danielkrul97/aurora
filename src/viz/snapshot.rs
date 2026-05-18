//! Snapshot shared with the Bevy thread via `Arc<Mutex<_>>` — the
//! training thread writes and the render thread `try_lock`s, so a
//! contested frame is dropped rather than stalling either side.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const LOSS_HISTORY_CAP: usize = 512;

#[derive(Clone, Debug)]
pub struct TrainingSnapshot {
    pub step: u64,
    pub epoch: u32,
    pub total_epochs: u32,
    pub loss: f32,
    pub bpb: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub elapsed_s: f64,
    pub backend: String,
    pub params: usize,

    pub layers: Vec<LayerStats>,

    pub input_preview: String,
    pub pred_preview: String,
    pub pred_correct: Option<bool>,

    pub loss_history: VecDeque<(u64, f32)>,
    pub bpb_history: VecDeque<(u64, f32)>,

    pub step_tick: u64,
    pub finished: bool,
}

/// One entry per pipeline stage in display order: embed, blocks `0..n_layer`, lm_head.
#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    pub label: String,
    pub kind: LayerKind,
    pub act_norm: f32,
    pub act_max_abs: f32,
    pub attn_bar: f32,
    pub ffn_bar: f32,
    pub grad_norm: f32,
    pub health: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerKind {
    #[default]
    Embed,
    Block,
    LmHead,
}

impl TrainingSnapshot {
    #[must_use]
    pub fn initial(
        n_layer: usize,
        total_epochs: u32,
        backend: String,
        params: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(n_layer + 2);
        layers.push(LayerStats {
            label: "EMBED".into(),
            kind: LayerKind::Embed,
            health: 1.0,
            ..Default::default()
        });
        for i in 0..n_layer {
            layers.push(LayerStats {
                label: format!("BLOCK {}", i + 1),
                kind: LayerKind::Block,
                health: 1.0,
                ..Default::default()
            });
        }
        layers.push(LayerStats {
            label: "LM HEAD".into(),
            kind: LayerKind::LmHead,
            health: 1.0,
            ..Default::default()
        });

        Self {
            step: 0,
            epoch: 0,
            total_epochs,
            loss: 0.0,
            bpb: 0.0,
            lr: 0.0,
            grad_norm: 0.0,
            elapsed_s: 0.0,
            backend,
            params,
            layers,
            input_preview: String::new(),
            pred_preview: String::new(),
            pred_correct: None,
            loss_history: VecDeque::with_capacity(LOSS_HISTORY_CAP),
            bpb_history: VecDeque::with_capacity(LOSS_HISTORY_CAP),
            step_tick: 0,
            finished: false,
        }
    }

    pub fn push_history(&mut self, step: u64, loss: f32, bpb: f32) {
        if self.loss_history.len() == LOSS_HISTORY_CAP {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back((step, loss));
        if self.bpb_history.len() == LOSS_HISTORY_CAP {
            self.bpb_history.pop_front();
        }
        self.bpb_history.push_back((step, bpb));
    }
}

pub type SnapshotArc = Arc<Mutex<TrainingSnapshot>>;

#[must_use]
pub fn new_snapshot(
    n_layer: usize,
    total_epochs: u32,
    backend: String,
    params: usize,
) -> SnapshotArc {
    Arc::new(Mutex::new(TrainingSnapshot::initial(
        n_layer,
        total_epochs,
        backend,
        params,
    )))
}

#[must_use]
pub fn l2_norm(xs: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for &v in xs {
        s += v * v;
    }
    s.sqrt()
}

#[must_use]
pub fn max_abs(xs: &[f32]) -> f32 {
    let mut m = 0.0_f32;
    for &v in xs {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// 1.0 = healthy mean activation norm; 0.0 = NaN/Inf or saturated.
/// Bell-shaped over `norm/50`, peak around 0.05..0.6.
#[must_use]
pub fn health_from_norm(norm: f32, max_abs_val: f32) -> f32 {
    if !norm.is_finite() || !max_abs_val.is_finite() {
        return 0.0;
    }
    let n = (norm / 50.0).clamp(0.0, 1.0);
    let bell = 1.0 - (2.0 * n - 0.5).abs().min(1.0);
    bell.clamp(0.0, 1.0)
}
