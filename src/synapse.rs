//! Synapse with pair-based STDP. Updates run lazily on pre/post spike events.

use crate::stdp::{stdp_delta, StdpParams};
use crate::types::{NeuronId, Time, Weight};

#[derive(Copy, Clone, Debug)]
pub struct Synapse {
    pub weight: Weight,
    pub pre: NeuronId,
    pub post: NeuronId,
    pub delay: Time,
    pub last_pre_t: Option<Time>,
    pub last_post_t: Option<Time>,
}

impl Synapse {
    #[must_use]
    pub const fn new(pre: NeuronId, post: NeuronId, weight: Weight, delay: Time) -> Self {
        Self {
            weight,
            pre,
            post,
            delay,
            last_pre_t: None,
            last_post_t: None,
        }
    }

    /// Pre-synaptic spike at time `t`. If a post-spike preceded it, applies
    /// depression (`Δt < 0`). Returns the applied Δw (zero if no prior post).
    pub fn on_pre_spike(&mut self, t: Time, params: &StdpParams) -> Weight {
        self.last_pre_t = Some(t);
        if let Some(t_post) = self.last_post_t {
            let dw = stdp_delta(t_post - t, params);
            self.weight += dw;
            return dw;
        }
        0.0
    }

    /// Post-synaptic spike at time `t`. If a pre-spike preceded it, applies
    /// potentiation (`Δt > 0`). Returns the applied Δw (zero if no prior pre).
    pub fn on_post_spike(&mut self, t: Time, params: &StdpParams) -> Weight {
        self.last_post_t = Some(t);
        if let Some(t_pre) = self.last_pre_t {
            let dw = stdp_delta(t - t_pre, params);
            self.weight += dw;
            return dw;
        }
        0.0
    }
}
