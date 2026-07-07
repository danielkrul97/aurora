#![allow(
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops
)]

//! The pharmacy: the catalog of interventions, the map from a diagnosed
//! fault to the arms licensed to treat it, and the contextual bandit that
//! learns — per (regime, fault) — which arm actually pays off. Reward is
//! the verifier's measured effect (BPB improvement), so the bandit is
//! trained on confirmed gains, never on raw correlation.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

use rand::rngs::StdRng;
use rand::Rng;

use crate::governor::diagnosis::Fault;
use crate::governor::state::MathState;
use crate::governor::vitals::Regime;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmKind {
    Identity,
    LrUp,
    LrDown,
    MuonLrUp,
    MuonLrDown,
    ClipUp,
    ClipDown,
    WdUp,
    WdDown,
    MomentumUp,
    MomentumDown,
}

impl ArmKind {
    pub const ALL: [ArmKind; 11] = [
        ArmKind::Identity,
        ArmKind::LrUp,
        ArmKind::LrDown,
        ArmKind::MuonLrUp,
        ArmKind::MuonLrDown,
        ArmKind::ClipUp,
        ArmKind::ClipDown,
        ArmKind::WdUp,
        ArmKind::WdDown,
        ArmKind::MomentumUp,
        ArmKind::MomentumDown,
    ];

    pub fn idx(self) -> u8 {
        Self::ALL.iter().position(|&k| k == self).unwrap_or(0) as u8
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::LrUp => "lr_up",
            Self::LrDown => "lr_down",
            Self::MuonLrUp => "muon_lr_up",
            Self::MuonLrDown => "muon_lr_down",
            Self::ClipUp => "clip_up",
            Self::ClipDown => "clip_down",
            Self::WdUp => "wd_up",
            Self::WdDown => "wd_down",
            Self::MomentumUp => "momentum_up",
            Self::MomentumDown => "momentum_down",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Arm {
    pub kind: ArmKind,
    pub factor: f32,
}

impl Arm {
    pub fn new(kind: ArmKind, factor: f32) -> Self {
        Self { kind, factor }
    }

    pub fn label(self) -> &'static str {
        self.kind.label()
    }

    /// Produce the candidate MSV by applying this arm to the control.
    pub fn target(self, control: &MathState) -> MathState {
        let mut t = *control;
        match self.kind {
            ArmKind::Identity => {}
            ArmKind::LrUp | ArmKind::LrDown => {
                t.lr_scale *= self.factor;
                t.muon_lr_scale *= self.factor;
            }
            ArmKind::MuonLrUp | ArmKind::MuonLrDown => t.muon_lr_scale *= self.factor,
            ArmKind::ClipUp | ArmKind::ClipDown => t.grad_clip_scale *= self.factor,
            ArmKind::WdUp => {
                // Scale both decays so the matrix bulk (Muon) is regularized,
                // not just the AdamW embed/head group. Introduce a base value
                // when a decay starts at ~0 (Muon WD is often 0).
                t.weight_decay = if t.weight_decay < 1e-6 { 0.1 } else { t.weight_decay * self.factor };
                t.muon_weight_decay =
                    if t.muon_weight_decay < 1e-6 { 0.05 } else { t.muon_weight_decay * self.factor };
            }
            ArmKind::WdDown => {
                t.weight_decay *= self.factor;
                t.muon_weight_decay *= self.factor;
            }
            // Move the momentum gap (1−m) closer to / further from 1.
            ArmKind::MomentumUp => t.momentum = 1.0 - (1.0 - t.momentum) / self.factor,
            ArmKind::MomentumDown => t.momentum = 1.0 - (1.0 - t.momentum) * self.factor,
        }
        t.clamp();
        t
    }
}

/// Arms allowed to treat a given fault. Diagnosis narrows the bandit's
/// choice to this set; an empty set means "observe/report only".
pub fn licensed_arms(fault: Fault) -> Vec<Arm> {
    match fault {
        Fault::LrTooHot => vec![Arm::new(ArmKind::LrDown, 0.7), Arm::new(ArmKind::ClipDown, 0.7)],
        Fault::LrTooCold => vec![Arm::new(ArmKind::LrUp, 1.3), Arm::new(ArmKind::MuonLrUp, 1.3)],
        Fault::GradNoiseHigh => {
            vec![Arm::new(ArmKind::LrDown, 0.8), Arm::new(ArmKind::MomentumUp, 1.5)]
        }
        Fault::OverfitOnset => vec![Arm::new(ArmKind::WdUp, 2.0)],
        _ => vec![],
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct GaussReward {
    n: u32,
    mean: f32,
    m2: f32,
}

impl GaussReward {
    fn update(&mut self, r: f32) {
        self.n += 1;
        let delta = r - self.mean;
        self.mean += delta / self.n as f32;
        self.m2 += delta * (r - self.mean);
    }

    fn var(&self) -> f32 {
        if self.n >= 2 {
            self.m2 / (self.n - 1) as f32
        } else {
            0.0
        }
    }
}

/// Gaussian Thompson sampling over a contextual reward table keyed by
/// (regime, fault, arm). Cold arms draw from a broad prior, so the bandit
/// explores before it exploits.
pub struct Bandit {
    table: HashMap<(u8, u8, u8), GaussReward>,
    prior_sd: f32,
}

impl Default for Bandit {
    fn default() -> Self {
        Self {
            table: HashMap::new(),
            prior_sd: 0.03,
        }
    }
}

impl Bandit {
    fn sample_gauss(rng: &mut StdRng, mean: f32, sd: f32) -> f32 {
        let u1: f32 = rng.random::<f32>().max(1e-12);
        let u2: f32 = rng.random::<f32>();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        mean + sd * z
    }

    /// Thompson-sample a posterior reward for each candidate and return the
    /// arm with the highest draw.
    pub fn select(&self, regime: Regime, fault: Fault, arms: &[Arm], rng: &mut StdRng) -> Arm {
        let mut best = arms[0];
        let mut best_theta = f32::NEG_INFINITY;
        for &arm in arms {
            let key = (regime.idx(), fault.idx(), arm.kind.idx());
            let theta = match self.table.get(&key) {
                None => Self::sample_gauss(rng, 0.0, self.prior_sd),
                Some(stat) if stat.n == 1 => Self::sample_gauss(rng, stat.mean, self.prior_sd),
                Some(stat) => {
                    let sd = (stat.var() / stat.n as f32 + 1e-12).sqrt();
                    Self::sample_gauss(rng, stat.mean, sd)
                }
            };
            if theta > best_theta {
                best_theta = theta;
                best = arm;
            }
        }
        best
    }

    pub fn reward(&mut self, regime: Regime, fault: Fault, arm: Arm, r: f32) {
        let key = (regime.idx(), fault.idx(), arm.kind.idx());
        self.table.entry(key).or_default().update(r);
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut buf = String::new();
        for ((r, f, a), stat) in &self.table {
            buf.push_str(&format!(
                "{r} {f} {a} {} {} {}\n",
                stat.n, stat.mean, stat.m2
            ));
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(buf.as_bytes())
    }

    /// Best-effort load; malformed rows are skipped, an absent file yields
    /// an empty bandit. Persistence must never crash a run.
    pub fn load(path: &Path) -> Self {
        let mut bandit = Self::default();
        let Ok(text) = std::fs::read_to_string(path) else {
            return bandit;
        };
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 6 {
                continue;
            }
            let (Ok(r), Ok(fa), Ok(a), Ok(n), Ok(mean), Ok(m2)) = (
                f[0].parse::<u8>(),
                f[1].parse::<u8>(),
                f[2].parse::<u8>(),
                f[3].parse::<u32>(),
                f[4].parse::<f32>(),
                f[5].parse::<f32>(),
            ) else {
                continue;
            };
            bandit.table.insert((r, fa, a), GaussReward { n, mean, m2 });
        }
        bandit
    }
}
