//! Discrete-time network: synapses deliver, neurons integrate, STDP updates.

use crate::lif::{LifParams, Neuron};
use crate::stdp::StdpParams;
use crate::synapse::Synapse;
use crate::types::{NeuronId, Spike, SynapseId, Time, Volts, Weight};

#[derive(Copy, Clone, Debug)]
struct Pending {
    deliver_at: Time,
    target: usize,
    weight: Volts,
}

#[derive(Clone, Debug, Default)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub stdp_params: StdpParams,
    pub now: Time,
    pending: Vec<Pending>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    external_input: Vec<Volts>,
}

impl Network {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_neuron(&mut self, params: LifParams) -> NeuronId {
        let id = NeuronId(self.neurons.len());
        self.neurons.push(Neuron::new(id, params));
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.external_input.push(0.0);
        id
    }

    pub fn add_synapse(
        &mut self,
        pre: NeuronId,
        post: NeuronId,
        weight: Weight,
        delay: Time,
    ) -> SynapseId {
        let idx = self.synapses.len();
        self.synapses.push(Synapse::new(pre, post, weight, delay));
        self.outgoing[pre.0].push(idx);
        self.incoming[post.0].push(idx);
        SynapseId(idx)
    }

    pub fn set_external_input(&mut self, id: NeuronId, input: Volts) {
        self.external_input[id.0] = input;
    }

    /// Advance the network by `dt`. Returns spikes emitted this step.
    pub fn step(&mut self, dt: Time) -> Vec<Spike> {
        self.now += dt;

        let mut inputs = self.external_input.clone();

        let mut still_pending = Vec::with_capacity(self.pending.len());
        for p in self.pending.drain(..) {
            if p.deliver_at <= self.now {
                inputs[p.target] += p.weight;
            } else {
                still_pending.push(p);
            }
        }
        self.pending = still_pending;

        let mut spikes = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(spike) = neuron.step(dt, inputs[i]) {
                spikes.push(spike);
            }
        }

        for spike in &spikes {
            let i = spike.source.0;
            for &syn_idx in &self.outgoing[i] {
                let syn = &mut self.synapses[syn_idx];
                let _ = syn.on_pre_spike(self.now, &self.stdp_params);
                self.pending.push(Pending {
                    deliver_at: self.now + syn.delay,
                    target: syn.post.0,
                    weight: syn.weight,
                });
            }
            for &syn_idx in &self.incoming[i] {
                let syn = &mut self.synapses[syn_idx];
                let _ = syn.on_post_spike(self.now, &self.stdp_params);
            }
        }

        spikes
    }
}
