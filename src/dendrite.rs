//! Active dendritic compartment with local threshold and fixed-amplitude
//! boost on crossing.

use crate::types::{Time, Volts};

#[derive(Copy, Clone, Debug)]
pub struct DendriteParams {
    pub tau: Time,
    pub v_rest: Volts,
    pub v_reset: Volts,
    pub threshold: Volts,
    pub boost: Volts,
}

impl Default for DendriteParams {
    fn default() -> Self {
        Self {
            tau: 0.010,
            v_rest: 0.0,
            v_reset: 0.0,
            threshold: 0.015,
            boost: 0.025,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Dendrite {
    pub params: DendriteParams,
    pub v_local: Volts,
    pub t_last_eval: Time,
}

impl Dendrite {
    #[must_use]
    pub const fn new(params: DendriteParams) -> Self {
        Self {
            params,
            v_local: params.v_rest,
            t_last_eval: 0.0,
        }
    }

    pub fn inject(&mut self, v: Volts) {
        self.v_local += v;
    }

    pub fn evolve_to(&mut self, t: Time) {
        if t <= self.t_last_eval {
            self.t_last_eval = t;
            return;
        }
        let dt = t - self.t_last_eval;
        self.v_local = (self.v_local - self.params.v_rest)
            .mul_add((-dt / self.params.tau).exp(), self.params.v_rest);
        self.t_last_eval = t;
    }

    pub fn check_threshold(&mut self) -> Volts {
        if self.v_local >= self.params.threshold {
            self.v_local = self.params.v_reset;
            self.params.boost
        } else {
            0.0
        }
    }

    pub fn tick(&mut self, dt: Time) -> Volts {
        self.evolve_to(self.t_last_eval + dt);
        self.check_threshold()
    }
}
