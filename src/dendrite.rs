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
}

impl Dendrite {
    #[must_use]
    pub const fn new(params: DendriteParams) -> Self {
        Self {
            params,
            v_local: params.v_rest,
        }
    }

    pub fn inject(&mut self, v: Volts) {
        self.v_local += v;
    }

    pub fn tick(&mut self, dt: Time) -> Volts {
        self.v_local = (self.v_local - self.params.v_rest)
            .mul_add((-dt / self.params.tau).exp(), self.params.v_rest);

        if self.v_local >= self.params.threshold {
            self.v_local = self.params.v_reset;
            self.params.boost
        } else {
            0.0
        }
    }
}
