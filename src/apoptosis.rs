//! Sprint 42: activity-dependent neuron pruning ("apoptosis").
//!
//! Per-neuron firing-rate EMA. After a warmup period, neurons whose EMA
//! falls below `kill_threshold` get permanently masked (forward spike →
//! 0, backward gradient → 0). Mirrors the biological "trophic
//! withdrawal" pattern: neurons that don't carry signal lose support
//! and die.
//!
//! Three modes:
//! - `Off`            — no pruning (baseline).
//! - `Activity`       — kill any neuron with EMA < threshold at each
//!                      check interval after warmup.
//! - `RandomFixed`    — kill a fixed number of random neurons per layer
//!                      in a single shot at warmup_chunks. Control: same
//!                      capacity reduction, no selectivity. Caller is
//!                      expected to copy the counts from a prior
//!                      `Activity` run for fair A/B.
//!
//! State is **not** serialized in the checkpoint (Sprint 42 scope: 1-epoch
//! experiments from scratch). If the mechanism shows promise, add to
//! `AURORA03` in a follow-up sprint.

use rand::prelude::IndexedRandom;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub enum ApoptosisMode {
    Off,
    Activity,
    /// Sprint 43: trophic-factor analogue. Signal = |∂L/∂W_outgoing|_1
    /// per neuron — measures how much downstream wants to update the
    /// neuron's outgoing connections. Kill if EMA < `frac × layer_mean`,
    /// at most `max_per_check` per layer per check.
    Trophic { frac: f32, max_per_check: usize },
    RandomFixed { per_layer_counts: Vec<usize> },
}

impl Eq for ApoptosisMode {}

#[derive(Clone, Debug)]
pub struct ApoptosisConfig {
    pub mode: ApoptosisMode,
    pub warmup_chunks: usize,
    pub ema_alpha: f32,
    pub kill_threshold: f32,
    pub check_interval: usize,
}

impl Default for ApoptosisConfig {
    fn default() -> Self {
        Self {
            mode: ApoptosisMode::Off,
            warmup_chunks: 200,
            ema_alpha: 0.05,
            kill_threshold: 0.005,
            check_interval: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KillEvent {
    pub chunk: usize,
    pub layer: usize,
    pub neuron: usize,
    pub rate_at_death: f32,
}

#[derive(Clone, Debug)]
pub struct ApoptosisState {
    pub config: ApoptosisConfig,
    /// EMA of per-neuron observed signal — firing rate for `Activity`
    /// mode, |∂L/∂W_outgoing|_1 for `Trophic` mode. One row per layer.
    pub ema_rate: Vec<Vec<f32>>,
    /// Permanent dead mask, one row per layer. Once `true` for a
    /// neuron, the network treats its spike as 0 and zeros gradients
    /// through it.
    pub dead_mask: Vec<Vec<bool>>,
    /// Number of training chunks observed so far. Drives warmup +
    /// check-interval gating.
    pub chunks_seen: usize,
    /// `true` once `RandomFixed` mode has executed its one-shot kill.
    random_fired: bool,
}

impl ApoptosisState {
    #[must_use]
    pub fn new(config: ApoptosisConfig, layer_sizes: &[usize]) -> Self {
        Self {
            config,
            ema_rate: layer_sizes.iter().map(|&n| vec![0.0; n]).collect(),
            dead_mask: layer_sizes.iter().map(|&n| vec![false; n]).collect(),
            chunks_seen: 0,
            random_fired: false,
        }
    }

    #[must_use]
    pub fn dead_count(&self, layer: usize) -> usize {
        self.dead_mask[layer].iter().filter(|d| **d).count()
    }

    #[must_use]
    pub fn alive_count(&self, layer: usize) -> usize {
        self.dead_mask[layer].len() - self.dead_count(layer)
    }

    /// Update EMA from per-layer per-neuron rates measured on the most
    /// recent chunk(s). `rates[l][i]` is the mean spike fraction over
    /// time/batch for neuron `i` in layer `l`.
    pub fn observe_rates(&mut self, rates: &[Vec<f32>]) {
        debug_assert_eq!(rates.len(), self.ema_rate.len());
        let alpha = self.config.ema_alpha;
        for (ema_row, rate_row) in self.ema_rate.iter_mut().zip(rates.iter()) {
            debug_assert_eq!(ema_row.len(), rate_row.len());
            for (e, &r) in ema_row.iter_mut().zip(rate_row.iter()) {
                *e = (1.0 - alpha) * *e + alpha * r;
            }
        }
    }

    /// Increment the chunk counter. Call once per training chunk.
    pub fn tick(&mut self) {
        self.chunks_seen += 1;
    }

    /// Decide whether to kill any neurons this chunk; if so, mutate
    /// `dead_mask` and return the new kill events. Returns empty vec
    /// in the common case (warmup not done, off cycle, or no candidates).
    pub fn maybe_kill<R: Rng>(&mut self, rng: &mut R) -> Vec<KillEvent> {
        if self.chunks_seen < self.config.warmup_chunks {
            return Vec::new();
        }
        match self.config.mode.clone() {
            ApoptosisMode::Off => Vec::new(),
            ApoptosisMode::Activity => self.kill_activity(),
            ApoptosisMode::Trophic { frac, max_per_check } => {
                self.kill_trophic(frac, max_per_check)
            }
            ApoptosisMode::RandomFixed { .. } => self.kill_random(rng),
        }
    }

    fn kill_activity(&mut self) -> Vec<KillEvent> {
        if !self.chunks_seen.is_multiple_of(self.config.check_interval) {
            return Vec::new();
        }
        let mut events = Vec::new();
        let threshold = self.config.kill_threshold;
        for (l, ema_row) in self.ema_rate.iter().enumerate() {
            for (i, &rate) in ema_row.iter().enumerate() {
                if !self.dead_mask[l][i] && rate < threshold {
                    self.dead_mask[l][i] = true;
                    events.push(KillEvent {
                        chunk: self.chunks_seen,
                        layer: l,
                        neuron: i,
                        rate_at_death: rate,
                    });
                }
            }
        }
        events
    }

    /// Trophic kill: signal is `|∂L/∂W_outgoing|_1`. Threshold scales
    /// with the layer's *current alive* mean to be invariant to overall
    /// gradient magnitude (which drifts during training). A neuron is
    /// "starving" if its trophic supply is far below typical.
    /// `max_per_check` caps kills per layer per check — avoids the
    /// snowball failure observed in Sprint 43.
    fn kill_trophic(&mut self, frac: f32, max_per_check: usize) -> Vec<KillEvent> {
        if !self.chunks_seen.is_multiple_of(self.config.check_interval) {
            return Vec::new();
        }
        let mut events = Vec::new();
        for (l, ema_row) in self.ema_rate.iter().enumerate() {
            let alive: Vec<usize> = (0..ema_row.len())
                .filter(|&i| !self.dead_mask[l][i])
                .collect();
            if alive.is_empty() {
                continue;
            }
            // Use median instead of mean — gradient magnitudes have a
            // heavy-tailed distribution where mean is dominated by the
            // top few neurons. Median gives a stable per-layer scale.
            let mut signals: Vec<f32> = alive.iter().map(|&i| ema_row[i]).collect();
            signals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = signals[signals.len() / 2];
            let threshold = frac * median;
            let mut candidates: Vec<(usize, f32)> = alive
                .iter()
                .map(|&i| (i, ema_row[i]))
                .filter(|(_, v)| *v < threshold)
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, signal) in candidates.into_iter().take(max_per_check) {
                self.dead_mask[l][i] = true;
                events.push(KillEvent {
                    chunk: self.chunks_seen,
                    layer: l,
                    neuron: i,
                    rate_at_death: signal,
                });
            }
        }
        events
    }

    fn kill_random<R: Rng>(&mut self, rng: &mut R) -> Vec<KillEvent> {
        if self.random_fired {
            return Vec::new();
        }
        let counts = match &self.config.mode {
            ApoptosisMode::RandomFixed { per_layer_counts } => per_layer_counts.clone(),
            _ => return Vec::new(),
        };
        debug_assert_eq!(counts.len(), self.dead_mask.len());
        let mut events = Vec::new();
        for (l, &want) in counts.iter().enumerate() {
            let alive: Vec<usize> = (0..self.dead_mask[l].len())
                .filter(|&i| !self.dead_mask[l][i])
                .collect();
            let take = want.min(alive.len());
            let chosen: Vec<usize> = alive.choose_multiple(rng, take).copied().collect();
            for i in chosen {
                self.dead_mask[l][i] = true;
                let rate = self.ema_rate[l][i];
                events.push(KillEvent {
                    chunk: self.chunks_seen,
                    layer: l,
                    neuron: i,
                    rate_at_death: rate,
                });
            }
        }
        self.random_fired = true;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ema_converges_to_steady_input() {
        let mut s = ApoptosisState::new(
            ApoptosisConfig {
                mode: ApoptosisMode::Activity,
                ema_alpha: 0.5,
                ..Default::default()
            },
            &[4],
        );
        let rates = vec![vec![0.2, 0.0, 0.5, 0.0]];
        for _ in 0..40 {
            s.observe_rates(&rates);
        }
        for (e, r) in s.ema_rate[0].iter().zip(rates[0].iter()) {
            assert!((e - r).abs() < 1e-3, "ema={e} target={r}");
        }
    }

    #[test]
    fn activity_kills_low_rate_after_warmup() {
        let mut s = ApoptosisState::new(
            ApoptosisConfig {
                mode: ApoptosisMode::Activity,
                warmup_chunks: 5,
                ema_alpha: 1.0, // immediate snap
                kill_threshold: 0.01,
                check_interval: 1,
            },
            &[4],
        );
        let rates = vec![vec![0.0, 0.5, 0.0, 0.5]];
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..4 {
            s.tick();
            s.observe_rates(&rates);
            assert!(s.maybe_kill(&mut rng).is_empty(), "no kills before warmup");
        }
        s.tick(); // chunks_seen = 5 → warmup met, check fires
        s.observe_rates(&rates);
        let killed = s.maybe_kill(&mut rng);
        assert_eq!(killed.len(), 2);
        assert!(s.dead_mask[0][0]);
        assert!(s.dead_mask[0][2]);
        assert!(!s.dead_mask[0][1]);
        assert!(!s.dead_mask[0][3]);
    }

    #[test]
    fn random_fixed_kills_exact_count_once() {
        let mut s = ApoptosisState::new(
            ApoptosisConfig {
                mode: ApoptosisMode::RandomFixed {
                    per_layer_counts: vec![3],
                },
                warmup_chunks: 2,
                ..Default::default()
            },
            &[10],
        );
        let mut rng = StdRng::seed_from_u64(7);
        s.tick();
        s.tick();
        let first = s.maybe_kill(&mut rng);
        assert_eq!(first.len(), 3);
        assert_eq!(s.dead_count(0), 3);
        s.tick();
        let second = s.maybe_kill(&mut rng);
        assert!(second.is_empty(), "RandomFixed fires only once");
        assert_eq!(s.dead_count(0), 3);
    }

    #[test]
    fn random_fixed_clamps_to_alive_pool() {
        let mut s = ApoptosisState::new(
            ApoptosisConfig {
                mode: ApoptosisMode::RandomFixed {
                    per_layer_counts: vec![20],
                },
                warmup_chunks: 1,
                ..Default::default()
            },
            &[5],
        );
        let mut rng = StdRng::seed_from_u64(7);
        s.tick();
        let killed = s.maybe_kill(&mut rng);
        assert_eq!(killed.len(), 5);
        assert_eq!(s.dead_count(0), 5);
    }
}
