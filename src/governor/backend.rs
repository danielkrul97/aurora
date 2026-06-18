#![allow(clippy::must_use_candidate)]

//! Backend abstraction for the governor's trial machinery. The governor
//! core is pure decision logic — it never touches a model. Snapshotting
//! for reflex rollback and the paired probe-rollout that feeds the
//! sequential test are delegated to a `TrialBackend`, so the same governor
//! drives both the CPU model and the on-device GPU model.

use crate::governor::state::MathState;

pub trait TrialBackend {
    /// Capture the current weights + optimizer state as the reflex
    /// rollback point.
    fn snapshot_safe(&mut self);

    /// Restore weights + optimizer state from the last `snapshot_safe`.
    fn restore_safe(&mut self);

    /// Run one paired sample — control vs treatment rolled from the same
    /// current weights over the same data, scored on the same probe — and
    /// return `bpb_control − bpb_treatment` (positive = treatment better).
    /// `sched_lr`/`sched_muon_lr` are this step's scheduled base rates the
    /// rollout scales by the arm MSV.
    fn paired_diff(
        &mut self,
        control: &MathState,
        treatment: &MathState,
        sched_lr: f32,
        sched_muon_lr: f32,
    ) -> f32;
}

/// Sample `count` fixed probe windows evenly across a held-out token slice.
/// Both backends score every arm on this same set, so the paired
/// difference isolates the arm effect.
pub fn build_probe(toks: &[u32], seq_len: usize, count: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    if count == 0 || toks.len() <= seq_len + 1 {
        return (inputs, targets);
    }
    let usable = toks.len() - (seq_len + 1);
    let stride = (usable / count).max(1);
    let mut p = 0;
    while p <= usable && inputs.len() < count {
        inputs.push(toks[p..p + seq_len].to_vec());
        targets.push(toks[p + 1..p + seq_len + 1].to_vec());
        p += stride;
    }
    (inputs, targets)
}
