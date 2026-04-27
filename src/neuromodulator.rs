//! Global neuromodulator (think dopamine). `reward(t, amount)` raises the
//! signal; it decays exponentially toward `baseline` between events.

use crate::types::Time;

#[derive(Copy, Clone, Debug)]
pub struct ModulatorParams {
    pub decay_tau: Time,
    pub baseline: f64,
}

impl Default for ModulatorParams {
    fn default() -> Self {
        Self {
            decay_tau: 0.200,
            baseline: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Modulator {
    pub params: ModulatorParams,
    pub value: f64,
    pub t_last_eval: Time,
}

impl Modulator {
    #[must_use]
    pub const fn new(params: ModulatorParams) -> Self {
        Self {
            params,
            value: params.baseline,
            t_last_eval: 0.0,
        }
    }

    pub fn reward(&mut self, t: Time, amount: f64) {
        self.evolve_to(t);
        self.value += amount;
    }

    pub fn evolve_to(&mut self, t: Time) {
        if t <= self.t_last_eval {
            self.t_last_eval = t;
            return;
        }
        let dt = t - self.t_last_eval;
        let decay = (-dt / self.params.decay_tau).exp();
        self.value = (self.value - self.params.baseline).mul_add(decay, self.params.baseline);
        self.t_last_eval = t;
    }

    pub fn step(&mut self, dt: Time) -> f64 {
        self.evolve_to(self.t_last_eval + dt);
        self.value
    }
}
