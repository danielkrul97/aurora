//! Discrete-time network: pending synaptic deliveries inject charge into
//! their target compartment, neurons tick, STDP updates fire on every spike.
//!
//! Two-factor mode (default): STDP delta is applied directly to `weight`.
//! Three-factor mode (when both `eligibility_params` and `modulator` are
//! `Some`): STDP delta accumulates in `synapse.eligibility`; per-step,
//! `weight += η · M · e · dt` and `e` decays exponentially.

use crate::dendrite::DendriteParams;
use crate::eligibility::EligibilityParams;
use crate::lif::LifParams;
use crate::neuromodulator::Modulator;
use crate::neuron::Neuron;
use crate::stdp::StdpParams;
use crate::synapse::Synapse;
use crate::types::{NeuronId, Spike, SynapseId, Time, Volts, Weight};

#[derive(Copy, Clone, Debug)]
struct Pending {
    deliver_at: Time,
    target_neuron: usize,
    target_dendrite: Option<usize>,
    weight: Volts,
}

#[derive(Clone, Debug, Default)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub stdp_params: StdpParams,
    pub eligibility_params: Option<EligibilityParams>,
    pub modulator: Option<Modulator>,
    /// When true, dendrite-targeted synapses route post-STDP through the
    /// dendrite's own spikes (not the soma). Breaks the credit-assignment
    /// symmetry that otherwise blocks R-STDP from learning XOR-like tasks.
    pub dendrite_local_stdp: bool,
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

    pub fn add_neuron(&mut self, soma_params: LifParams) -> NeuronId {
        let id = NeuronId(self.neurons.len());
        self.neurons.push(Neuron::new(id, soma_params));
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.external_input.push(0.0);
        id
    }

    /// # Panics
    /// Panics if `neuron` is out of bounds.
    pub fn add_dendrite(&mut self, neuron: NeuronId, params: DendriteParams) -> usize {
        self.neurons[neuron.0].add_dendrite(params)
    }

    pub fn add_synapse(
        &mut self,
        pre: NeuronId,
        post_neuron: NeuronId,
        post_dendrite: Option<usize>,
        weight: Weight,
        delay: Time,
    ) -> SynapseId {
        let idx = self.synapses.len();
        self.synapses
            .push(Synapse::new(pre, post_neuron, post_dendrite, weight, delay));
        self.outgoing[pre.0].push(idx);
        self.incoming[post_neuron.0].push(idx);
        SynapseId(idx)
    }

    pub fn set_external_input(&mut self, id: NeuronId, input: Volts) {
        self.external_input[id.0] = input;
    }

    pub fn schedule_input(
        &mut self,
        t: Time,
        neuron: NeuronId,
        dendrite: Option<usize>,
        amplitude: Volts,
    ) {
        self.pending.push(Pending {
            deliver_at: t,
            target_neuron: neuron.0,
            target_dendrite: dendrite,
            weight: amplitude,
        });
    }

    pub fn reward(&mut self, amount: f64) {
        if let Some(ref mut m) = self.modulator {
            m.reward(self.now, amount);
        }
    }

    pub fn step(&mut self, dt: Time) -> Vec<Spike> {
        self.now += dt;

        if let Some(ref mut m) = self.modulator {
            m.evolve_to(self.now);
        }

        let now = self.now;
        let mut write = 0;
        for read in 0..self.pending.len() {
            let p = self.pending[read];
            if p.deliver_at <= now {
                match p.target_dendrite {
                    Some(d) => self.neurons[p.target_neuron].inject_dendrite(d, p.weight),
                    None => self.neurons[p.target_neuron].inject_soma(p.weight),
                }
            } else {
                self.pending[write] = p;
                write += 1;
            }
        }
        self.pending.truncate(write);

        let mut spikes = Vec::new();
        let mut dendrite_spikes: Vec<(usize, usize)> = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let step = neuron.step(self.now, dt, self.external_input[i]);
            if let Some(spike) = step.soma_spike {
                spikes.push(spike);
            }
            for d_idx in step.dendrite_spikes {
                dendrite_spikes.push((i, d_idx));
            }
        }

        let three_factor = self.eligibility_params.is_some() && self.modulator.is_some();
        let dendrite_local = self.dendrite_local_stdp;

        for spike in &spikes {
            let i = spike.source.0;
            for &syn_idx in &self.outgoing[i] {
                let syn = &mut self.synapses[syn_idx];
                let dw = syn.on_pre_spike(self.now, &self.stdp_params);
                syn.apply_stdp_delta(dw, three_factor);
                self.pending.push(Pending {
                    deliver_at: self.now + syn.delay,
                    target_neuron: syn.post_neuron.0,
                    target_dendrite: syn.post_dendrite,
                    weight: syn.weight,
                });
            }
            for &syn_idx in &self.incoming[i] {
                let syn = &mut self.synapses[syn_idx];
                if dendrite_local && syn.post_dendrite.is_some() {
                    continue;
                }
                let dw = syn.on_post_spike(self.now, &self.stdp_params);
                syn.apply_stdp_delta(dw, three_factor);
            }
        }

        if dendrite_local {
            for &(n_idx, d_idx) in &dendrite_spikes {
                for &syn_idx in &self.incoming[n_idx] {
                    let syn = &mut self.synapses[syn_idx];
                    if syn.post_dendrite != Some(d_idx) {
                        continue;
                    }
                    let dw = syn.on_post_spike(self.now, &self.stdp_params);
                    syn.apply_stdp_delta(dw, three_factor);
                }
            }
        }

        if let (Some(e_params), Some(m)) = (self.eligibility_params, self.modulator) {
            let decay = (-dt / e_params.tau).exp();
            let weight_factor = e_params.learning_rate * m.value * dt;
            for syn in &mut self.synapses {
                syn.weight += weight_factor * syn.eligibility;
                syn.eligibility *= decay;
            }
        }

        spikes
    }
}
