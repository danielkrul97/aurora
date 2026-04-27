//! Leaky Integrate-and-Fire neuron with exact exponential integration.
//!
//! Dynamics: `τ_m · dV/dt = -(V - V_rest) + input`. Between steps the input
//! is treated as piecewise constant, so we integrate the ODE analytically:
//! `V(t+dt) = V_eq + (V(t) - V_eq) · exp(-dt/τ_m)` with `V_eq = V_rest + input`.
//! This avoids the O(dt) bias of explicit Euler.

use crate::types::{NeuronId, Spike, Time, Volts};

#[derive(Copy, Clone, Debug)]
pub struct LifParams {
    pub v_rest: Volts,
    pub v_reset: Volts,
    pub v_threshold: Volts,
    pub tau_m: Time,
    pub tau_ref: Time,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            v_rest: -0.070,
            v_reset: -0.070,
            v_threshold: -0.050,
            tau_m: 0.020,
            tau_ref: 0.005,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Neuron {
    pub id: NeuronId,
    pub params: LifParams,
    pub v: Volts,
    pub last_spike_t: Option<Time>,
    pub now: Time,
}

impl Neuron {
    #[must_use]
    pub const fn new(id: NeuronId, params: LifParams) -> Self {
        let v = params.v_rest;
        Self {
            id,
            params,
            v,
            last_spike_t: None,
            now: 0.0,
        }
    }

    pub fn step(&mut self, dt: Time, input: Volts) -> Option<Spike> {
        self.now += dt;

        if let Some(t_last) = self.last_spike_t {
            if self.now - t_last < self.params.tau_ref {
                self.v = self.params.v_reset;
                return None;
            }
        }

        let v_eq = self.params.v_rest + input;
        self.v = (self.v - v_eq).mul_add((-dt / self.params.tau_m).exp(), v_eq);

        if self.v >= self.params.v_threshold {
            self.v = self.params.v_reset;
            self.last_spike_t = Some(self.now);
            Some(Spike {
                t: self.now,
                source: self.id,
            })
        } else {
            None
        }
    }
}
