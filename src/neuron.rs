//! Compound neuron: soma + dendrites + optional homeostasis.

use crate::dendrite::{Dendrite, DendriteParams};
use crate::homeostasis::Homeostasis;
use crate::lif::{Lif, LifParams};
use crate::types::{NeuronId, Spike, Time, Volts};

#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: NeuronId,
    pub soma: Lif,
    pub dendrites: Vec<Dendrite>,
    pub homeo: Option<Homeostasis>,
}

impl Neuron {
    #[must_use]
    pub const fn new(id: NeuronId, soma_params: LifParams) -> Self {
        Self {
            id,
            soma: Lif::new(soma_params),
            dendrites: Vec::new(),
            homeo: None,
        }
    }

    pub fn add_dendrite(&mut self, params: DendriteParams) -> usize {
        let idx = self.dendrites.len();
        self.dendrites.push(Dendrite::new(params));
        idx
    }

    pub const fn enable_homeostasis(&mut self, homeo: Homeostasis) {
        self.homeo = Some(homeo);
    }

    pub fn inject_soma(&mut self, v: Volts) {
        self.soma.inject(v);
    }

    /// # Panics
    /// Panics if `idx >= self.dendrites.len()`.
    pub fn inject_dendrite(&mut self, idx: usize, v: Volts) {
        self.dendrites[idx].inject(v);
    }

    /// Dendrites tick before the soma so their boosts land in the same step.
    pub fn step(&mut self, now: Time, dt: Time, soma_dc: Volts) -> Option<Spike> {
        for d in &mut self.dendrites {
            let boost = d.tick(dt);
            if boost != 0.0 {
                self.soma.inject(boost);
            }
        }

        let fired = self.soma.tick(now, dt, soma_dc);

        if let Some(ref mut h) = self.homeo {
            self.soma.params.v_threshold += h.step(dt, fired);
        }

        if fired {
            Some(Spike {
                t: now,
                source: self.id,
            })
        } else {
            None
        }
    }
}
