//! Synapse with pair-based STDP. Targets either a specific dendrite on the
//! post-synaptic neuron (typical case) or the soma directly.
//!
//! `on_pre_spike` / `on_post_spike` return the STDP delta but do **not**
//! mutate `weight`. The caller (Network) decides whether to apply it
//! directly to `weight` (classical STDP) or accumulate into `eligibility`
//! (three-factor R-STDP, committed later via the modulator).

use crate::stdp::{stdp_delta, StdpParams};
use crate::types::{NeuronId, Time, Weight};

#[derive(Copy, Clone, Debug)]
pub struct Synapse {
    pub weight: Weight,
    pub pre: NeuronId,
    pub post_neuron: NeuronId,
    pub post_dendrite: Option<usize>,
    pub delay: Time,
    pub last_pre_t: Option<Time>,
    pub last_post_t: Option<Time>,
    pub eligibility: f64,
    /// When `false`, STDP deltas are ignored — the synapse keeps its
    /// initial weight regardless of pre/post activity. Used for
    /// architectural connections (e.g., lateral inhibition) that should
    /// not erode under reward-modulated learning.
    pub plastic: bool,
}

impl Synapse {
    #[must_use]
    pub const fn new(
        pre: NeuronId,
        post_neuron: NeuronId,
        post_dendrite: Option<usize>,
        weight: Weight,
        delay: Time,
    ) -> Self {
        Self {
            weight,
            pre,
            post_neuron,
            post_dendrite,
            delay,
            last_pre_t: None,
            last_post_t: None,
            eligibility: 0.0,
            plastic: true,
        }
    }

    pub fn on_pre_spike(&mut self, t: Time, params: &StdpParams) -> Weight {
        self.last_pre_t = Some(t);
        if let Some(t_post) = self.last_post_t {
            return stdp_delta(t_post - t, params);
        }
        0.0
    }

    pub fn on_post_spike(&mut self, t: Time, params: &StdpParams) -> Weight {
        self.last_post_t = Some(t);
        if let Some(t_pre) = self.last_pre_t {
            return stdp_delta(t - t_pre, params);
        }
        0.0
    }

    pub fn apply_stdp_delta(&mut self, dw: Weight, three_factor: bool) {
        if !self.plastic {
            return;
        }
        if three_factor {
            self.eligibility += dw;
        } else {
            self.weight += dw;
        }
    }
}
