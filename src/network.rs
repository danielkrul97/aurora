//! Discrete-time network.
//!
//! Two-factor mode: STDP delta applies directly to `weight`.
//! Three-factor mode (`eligibility_params` + a modulator are `Some`):
//! STDP accumulates into `synapse.eligibility`; each step
//! `weight += η · M · e · dt` and `e` decays exponentially.

use rayon::prelude::*;

use crate::dendrite::DendriteParams;
use crate::eligibility::EligibilityParams;
use crate::lif::LifParams;
use crate::neuromodulator::Modulator;
use crate::neuron::Neuron;
use crate::stdp::StdpParams;
use crate::synapse::Synapse;
use crate::types::{NeuronId, Spike, SynapseId, Time, Volts, Weight};

const M_EPSILON: f64 = 1e-4;
const PAR_THRESHOLD: usize = 2_000;

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
    /// When set for a post-neuron, synapses targeting it commit weight
    /// against this modulator instead of the global one — per-output
    /// reward attribution for R-STDP credit assignment.
    pub per_neuron_modulators: Vec<Option<Modulator>>,
    /// Routes post-STDP through dendrite spikes (not soma). Breaks the
    /// credit-assignment symmetry that blocks R-STDP from XOR-like tasks.
    pub dendrite_local_stdp: bool,
    /// Inclusive `(min, max)` clamp on plastic weights after every commit.
    /// Non-plastic synapses (e.g., lateral inhibition) are exempt.
    pub weight_clamp: Option<(Weight, Weight)>,
    pub now: Time,
    pending: Vec<Pending>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    external_input: Vec<Volts>,
    t_last_eligibility_flush: Time,
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
        self.per_neuron_modulators.push(None);
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

    /// # Panics
    /// Panics if `neuron` is out of bounds.
    pub fn enable_neuron_modulator(
        &mut self,
        neuron: NeuronId,
        params: crate::neuromodulator::ModulatorParams,
    ) {
        self.per_neuron_modulators[neuron.0] = Some(Modulator::new(params));
    }

    /// No-op when `neuron` has no per-neuron modulator installed.
    ///
    /// # Panics
    /// Panics if `neuron` is out of bounds.
    pub fn reward_neuron(&mut self, neuron: NeuronId, amount: f64) {
        if let Some(ref mut m) = self.per_neuron_modulators[neuron.0] {
            m.reward(self.now, amount);
        }
    }

    /// # Panics
    /// Panics if `syn` is out of bounds.
    pub fn set_plastic(&mut self, syn: SynapseId, plastic: bool) {
        self.synapses[syn.0].plastic = plastic;
    }

    /// Bring eligibility values up to `self.now` for diagnostic reads —
    /// the lazy-decay batch is otherwise deferred to the next active step.
    pub fn flush_eligibility_to_now(&mut self) {
        if let Some(e_params) = self.eligibility_params {
            self.flush_eligibility(e_params.tau);
        }
    }

    fn flush_eligibility(&mut self, tau_e: Time) {
        let dt_stale = self.now - self.t_last_eligibility_flush;
        if dt_stale <= 0.0 {
            return;
        }
        let factor = (-dt_stale / tau_e).exp();
        if (factor - 1.0).abs() < f64::EPSILON {
            self.t_last_eligibility_flush = self.now;
            return;
        }
        if self.synapses.len() >= PAR_THRESHOLD {
            self.synapses
                .par_iter_mut()
                .for_each(|s| s.eligibility *= factor);
        } else {
            for s in &mut self.synapses {
                s.eligibility *= factor;
            }
        }
        self.t_last_eligibility_flush = self.now;
    }

    #[allow(clippy::too_many_lines)]
    pub fn step(&mut self, dt: Time) -> Vec<Spike> {
        self.now += dt;

        if let Some(ref mut m) = self.modulator {
            m.evolve_to(self.now);
        }
        for m in self.per_neuron_modulators.iter_mut().flatten() {
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

        let any_per_neuron_modulator = self.per_neuron_modulators.iter().any(Option::is_some);
        let three_factor = self.eligibility_params.is_some()
            && (self.modulator.is_some() || any_per_neuron_modulator);
        let dendrite_local = self.dendrite_local_stdp;
        let stdp_modifies_eligibility =
            three_factor && (!spikes.is_empty() || (dendrite_local && !dendrite_spikes.is_empty()));

        if let Some(e_params) = self.eligibility_params {
            if stdp_modifies_eligibility {
                self.flush_eligibility(e_params.tau);
            }
        }

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

        if let Some(e_params) = self.eligibility_params {
            if !three_factor {
                return spikes;
            }
            let global_m = self.modulator.as_ref().map(|m| m.value);
            let per_neuron = &self.per_neuron_modulators[..];
            let m_for = |post_idx: usize| -> f64 {
                per_neuron[post_idx]
                    .as_ref()
                    .map(|m| m.value)
                    .or(global_m)
                    .unwrap_or(0.0)
            };
            let global_active = global_m.is_some_and(|v| v.abs() > M_EPSILON);
            let any_per_neuron_active = per_neuron
                .iter()
                .any(|slot| slot.as_ref().is_some_and(|m| m.value.abs() > M_EPSILON));
            let m_active = global_active || any_per_neuron_active;

            if !stdp_modifies_eligibility && !m_active {
                return spikes;
            }

            let lr = e_params.learning_rate;
            let per_step_decay = (-dt / e_params.tau).exp();
            let flush_factor = if stdp_modifies_eligibility {
                1.0
            } else {
                let dt_stale = self.now - self.t_last_eligibility_flush;
                if dt_stale > 0.0 {
                    (-dt_stale / e_params.tau).exp()
                } else {
                    1.0
                }
            };
            let par = self.synapses.len() >= PAR_THRESHOLD;
            let clamp = self.weight_clamp;
            let apply_clamp = |w: &mut Weight, plastic: bool| {
                if plastic {
                    if let Some((lo, hi)) = clamp {
                        *w = w.clamp(lo, hi);
                    }
                }
            };
            if m_active {
                let commit = |syn: &mut Synapse| {
                    syn.eligibility *= flush_factor;
                    let m_val = m_for(syn.post_neuron.0);
                    if m_val.abs() > M_EPSILON {
                        syn.weight += lr * m_val * dt * syn.eligibility;
                        apply_clamp(&mut syn.weight, syn.plastic);
                    }
                    syn.eligibility *= per_step_decay;
                };
                if par {
                    self.synapses.par_iter_mut().for_each(commit);
                } else {
                    self.synapses.iter_mut().for_each(commit);
                }
            } else {
                let decay_only = |syn: &mut Synapse| syn.eligibility *= per_step_decay;
                if par {
                    self.synapses.par_iter_mut().for_each(decay_only);
                } else {
                    self.synapses.iter_mut().for_each(decay_only);
                }
            }
            self.t_last_eligibility_flush = self.now + dt;
        }

        spikes
    }
}
