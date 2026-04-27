//! Synapse with pair-based STDP. Targets either a specific dendrite on the
//! post-synaptic neuron (typical case) or the soma directly.

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
        }
    }

    pub fn on_pre_spike(&mut self, t: Time, params: &StdpParams) -> Weight {
        self.last_pre_t = Some(t);
        if let Some(t_post) = self.last_post_t {
            let dw = stdp_delta(t_post - t, params);
            self.weight += dw;
            return dw;
        }
        0.0
    }

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
