//! Event-driven network engine.
//!
//! Same neuron + synapse + STDP primitives as `Network`, but the scheduler
//! advances V analytically between events instead of stepping a fixed `dt`.
//! Cost scales with event count (≈ spike count) rather than `T/dt`.

use crate::neuron::Neuron;
use crate::scheduler::{Event, EventKind, EventQueue};
use crate::stdp::StdpParams;
use crate::synapse::Synapse;
use crate::types::{NeuronId, Spike, SynapseId, Time, Volts, Weight};
use crate::{dendrite::DendriteParams, lif::LifParams};

#[derive(Clone, Debug, Default)]
pub struct EventNetwork {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub stdp_params: StdpParams,
    pub now: Time,
    queue: EventQueue,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    seq: u64,
}

impl EventNetwork {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_neuron(&mut self, soma_params: LifParams) -> NeuronId {
        let id = NeuronId(self.neurons.len());
        self.neurons.push(Neuron::new(id, soma_params));
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
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

    pub fn schedule_input(
        &mut self,
        t: Time,
        neuron: NeuronId,
        dendrite: Option<usize>,
        amplitude: Volts,
    ) {
        self.seq += 1;
        self.queue.push(Event {
            t,
            seq: self.seq,
            kind: EventKind::Pulse {
                neuron,
                dendrite,
                amplitude,
            },
        });
    }

    /// Process every event with `t ≤ t_end`. Returns spikes emitted in
    /// processing order. `now` advances to `t_end` after the loop.
    pub fn run_until(&mut self, t_end: Time) -> Vec<Spike> {
        let mut spikes_out = Vec::new();
        while let Some(ev) = self.queue.peek() {
            if ev.t > t_end {
                break;
            }
            let ev = self.queue.pop().unwrap_or_else(|| unreachable!());
            self.now = ev.t;
            let EventKind::Pulse {
                neuron,
                dendrite,
                amplitude,
            } = ev.kind;
            if let Some(spike) = self.deliver(neuron, dendrite, amplitude) {
                spikes_out.push(spike);
                self.propagate_spike(spike);
            }
        }
        self.now = t_end;
        spikes_out
    }

    fn deliver(
        &mut self,
        neuron_id: NeuronId,
        dendrite: Option<usize>,
        amplitude: Volts,
    ) -> Option<Spike> {
        let now = self.now;
        let neuron = &mut self.neurons[neuron_id.0];

        if let Some(ref mut h) = neuron.homeo {
            let delta = h.evolve_to(now, false);
            neuron.soma.params.v_threshold += delta;
        }

        let fired = if let Some(d_idx) = dendrite {
            let dend = &mut neuron.dendrites[d_idx];
            dend.evolve_to(now);
            dend.inject(amplitude);
            let boost = dend.check_threshold();
            if boost == 0.0 {
                false
            } else {
                neuron.soma.evolve_to(now, 0.0);
                neuron.soma.inject(boost);
                neuron.soma.check_threshold(now)
            }
        } else {
            neuron.soma.evolve_to(now, 0.0);
            neuron.soma.inject(amplitude);
            neuron.soma.check_threshold(now)
        };

        if fired {
            if let Some(ref mut h) = neuron.homeo {
                h.spike_trace += 1.0;
            }
            Some(Spike {
                t: now,
                source: neuron_id,
            })
        } else {
            None
        }
    }

    fn propagate_spike(&mut self, spike: Spike) {
        let i = spike.source.0;

        let n_in = self.incoming[i].len();
        for k in 0..n_in {
            let syn_idx = self.incoming[i][k];
            let dw = self.synapses[syn_idx].on_post_spike(spike.t, &self.stdp_params);
            self.synapses[syn_idx].weight += dw;
        }

        let n_out = self.outgoing[i].len();
        for k in 0..n_out {
            let syn_idx = self.outgoing[i][k];
            let syn = &mut self.synapses[syn_idx];
            let dw = syn.on_pre_spike(spike.t, &self.stdp_params);
            syn.weight += dw;
            let deliver_at = spike.t + syn.delay;
            let target_neuron = syn.post_neuron;
            let target_dendrite = syn.post_dendrite;
            let amplitude = syn.weight;
            self.seq += 1;
            self.queue.push(Event {
                t: deliver_at,
                seq: self.seq,
                kind: EventKind::Pulse {
                    neuron: target_neuron,
                    dendrite: target_dendrite,
                    amplitude,
                },
            });
        }
    }
}
