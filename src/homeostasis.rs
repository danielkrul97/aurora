//! Slow per-neuron threshold adaptation driven by an exponential spike
//! trace; high activity raises θ, low activity lowers it.

use crate::types::{Time, Volts};

#[derive(Copy, Clone, Debug)]
pub struct HomeoParams {
    pub tau: Time,
    pub target_rate: f64,
    pub eta: f64,
}

impl Default for HomeoParams {
    fn default() -> Self {
        Self {
            tau: 1.0,
            target_rate: 3.0,
            eta: 0.001,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Homeostasis {
    pub params: HomeoParams,
    pub spike_trace: f64,
}

impl Homeostasis {
    #[must_use]
    pub const fn new(params: HomeoParams) -> Self {
        Self {
            params,
            spike_trace: 0.0,
        }
    }

    pub fn step(&mut self, dt: Time, fired: bool) -> Volts {
        self.spike_trace *= (-dt / self.params.tau).exp();
        if fired {
            self.spike_trace += 1.0;
        }
        let rate_est = self.spike_trace / self.params.tau;
        self.params.eta * (rate_est - self.params.target_rate) * dt
    }

    #[must_use]
    pub fn rate_estimate(&self) -> f64 {
        self.spike_trace / self.params.tau
    }
}
