#![allow(
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

//! The paired-rollout sampler — how the governor sees through the noise
//! floor. From one snapshot `W` we roll the control MSV and the candidate
//! MSV over the *same* batches, then score both on the *same* fixed probe
//! set. Sharing `W` and the data makes "where we are on the curve" cancel
//! in the difference `bpb_control − bpb_treatment`, so its variance is far
//! below that of raw BPB. The difference is one observation for the
//! sequential test; rounds at different `W` supply the rest.

use rand::rngs::StdRng;
use rand::Rng;

use crate::governor::backend::TrialBackend;
use crate::governor::diagnosis::Fault;
use crate::governor::pharmacy::Arm;
use crate::governor::state::MathState;
use crate::governor::verifier::{ConfidenceSequence, Verdict};
use crate::governor::vitals::Regime;
use crate::optimizer::{AdamWConfig, MuonConfig};
use crate::text_io::bits_per_byte;
use crate::transformer::{DecoderTransformer, ModelGrads};

/// One paired probe difference. Positive = the treatment reached a lower
/// probe BPB than the control after the same short rollout.
pub fn paired_rollout_sample(
    base: &DecoderTransformer,
    train_toks: &[u32],
    rollout_starts: &[usize],
    probe_inputs: &[Vec<u32>],
    probe_targets: &[Vec<u32>],
    seq_len: usize,
    comp: f32,
    sched_adam_lr: f32,
    sched_muon_lr: f32,
    base_muon_wd: f32,
    base_clip: f32,
    control: &MathState,
    treatment: &MathState,
) -> f32 {
    let bpb_c = rollout_and_probe(
        base, train_toks, rollout_starts, probe_inputs, probe_targets, seq_len, comp,
        sched_adam_lr, sched_muon_lr, base_muon_wd, base_clip, control,
    );
    let bpb_t = rollout_and_probe(
        base, train_toks, rollout_starts, probe_inputs, probe_targets, seq_len, comp,
        sched_adam_lr, sched_muon_lr, base_muon_wd, base_clip, treatment,
    );
    bpb_c - bpb_t
}

fn rollout_and_probe(
    base: &DecoderTransformer,
    train_toks: &[u32],
    rollout_starts: &[usize],
    probe_inputs: &[Vec<u32>],
    probe_targets: &[Vec<u32>],
    seq_len: usize,
    comp: f32,
    sched_adam_lr: f32,
    sched_muon_lr: f32,
    base_muon_wd: f32,
    base_clip: f32,
    msv: &MathState,
) -> f32 {
    let mut m = base.clone();
    let mut grads = ModelGrads::zeros(&m.config);
    for &p in rollout_starts {
        if p + seq_len + 1 > train_toks.len() {
            continue;
        }
        let input = &train_toks[p..p + seq_len];
        let target = &train_toks[p + 1..p + seq_len + 1];
        grads.zero();
        let (logits, cache) = m.forward(input);
        m.backward(&cache, &logits, target, &mut grads);
        let mut adam = AdamWConfig { lr: sched_adam_lr, ..AdamWConfig::default() };
        let mut muon = MuonConfig {
            lr: sched_muon_lr,
            weight_decay: base_muon_wd,
            ..MuonConfig::default()
        };
        let clip = msv.apply_optim(&mut adam, &mut muon, base_clip);
        DecoderTransformer::clip_grads_adamw_subset(&mut grads, clip);
        m.step_muon_hybrid(&grads, &muon, &adam);
    }
    probe_bpb(&m, probe_inputs, probe_targets, seq_len, comp)
}

/// BPB over a fixed probe set, scoring only the context-rich tail of each
/// window (mirrors the overlapping-window eval's `scored_tail`).
fn probe_bpb(
    model: &DecoderTransformer,
    inputs: &[Vec<u32>],
    targets: &[Vec<u32>],
    seq_len: usize,
    comp: f32,
) -> f32 {
    let v = model.config.vocab;
    let scored_from = seq_len.saturating_sub(seq_len / 2).min(seq_len.saturating_sub(1));
    let mut nats = 0.0_f32;
    let mut tokens = 0usize;
    for (input, target) in inputs.iter().zip(targets) {
        let (logits, _) = model.forward(input);
        for ti in scored_from..seq_len {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            let logp = (row[target[ti] as usize] - max - sum.ln()).max(-27.631_021);
            nats += -logp;
            tokens += 1;
        }
    }
    bits_per_byte(nats, (tokens as f32 * comp) as usize)
}

/// A trial in flight: the proposed arm, the frozen control, and the
/// sequential test accumulating paired differences across rounds.
pub struct Trial {
    pub fault: Fault,
    pub arm: Arm,
    pub regime: Regime,
    pub control: MathState,
    pub treatment: MathState,
    pub cs: ConfidenceSequence,
    pub started_step: u64,
}

impl Trial {
    pub fn new(
        fault: Fault,
        arm: Arm,
        regime: Regime,
        control: MathState,
        treatment: MathState,
        cs: ConfidenceSequence,
        started_step: u64,
    ) -> Self {
        Self {
            fault,
            arm,
            regime,
            control,
            treatment,
            cs,
            started_step,
        }
    }

    pub fn observe_diff(&mut self, diff: f32) -> Verdict {
        self.cs.push(diff);
        self.cs.verdict()
    }

    pub fn effect(&self) -> f32 {
        self.cs.mean() as f32
    }
}

/// CPU trial backend: snapshots via `Clone`, rolls on the live model
/// through the public forward/backward/step API.
pub struct CpuTrialBackend<'a> {
    pub model: &'a mut DecoderTransformer,
    pub safe: &'a mut Option<DecoderTransformer>,
    pub train_toks: &'a [u32],
    pub probe_inputs: &'a [Vec<u32>],
    pub probe_targets: &'a [Vec<u32>],
    pub seq_len: usize,
    pub comp: f32,
    pub base_muon_wd: f32,
    pub base_clip: f32,
    pub horizon: usize,
    pub rng: &'a mut StdRng,
}

impl TrialBackend for CpuTrialBackend<'_> {
    fn snapshot_safe(&mut self) {
        match self.safe.as_mut() {
            Some(s) => s.clone_from(self.model),
            None => *self.safe = Some(self.model.clone()),
        }
    }

    fn restore_safe(&mut self) {
        if let Some(s) = self.safe.as_ref() {
            self.model.clone_from(s);
        }
    }

    fn paired_diff(
        &mut self,
        control: &MathState,
        treatment: &MathState,
        sched_lr: f32,
        sched_muon_lr: f32,
    ) -> f32 {
        let max_start = self.train_toks.len().saturating_sub(self.seq_len + 1);
        if max_start == 0 {
            return 0.0;
        }
        let starts: Vec<usize> = (0..self.horizon)
            .map(|_| self.rng.random_range(0..max_start))
            .collect();
        paired_rollout_sample(
            self.model,
            self.train_toks,
            &starts,
            self.probe_inputs,
            self.probe_targets,
            self.seq_len,
            self.comp,
            sched_lr,
            sched_muon_lr,
            self.base_muon_wd,
            self.base_clip,
            control,
            treatment,
        )
    }
}
