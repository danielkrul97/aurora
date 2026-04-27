//! Leaky integrate-and-fire soma. Exact exponential integration between
//! steps avoids the O(dt) bias of explicit Euler.

use crate::types::{Time, Volts};

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
pub struct Lif {
    pub params: LifParams,
    pub v: Volts,
    pub last_spike_t: Option<Time>,
    pub t_last_eval: Time,
}

impl Lif {
    #[must_use]
    pub const fn new(params: LifParams) -> Self {
        Self {
            params,
            v: params.v_rest,
            last_spike_t: None,
            t_last_eval: 0.0,
        }
    }

    pub fn inject(&mut self, v: Volts) {
        self.v += v;
    }

    /// Analytically advance V to absolute time `t` with sustained input `dc`.
    /// Refractory window of the last spike clamps V to `v_reset`.
    pub fn evolve_to(&mut self, t: Time, dc: Volts) {
        if t <= self.t_last_eval {
            self.t_last_eval = t;
            return;
        }
        if let Some(t_last) = self.last_spike_t {
            if t < t_last + self.params.tau_ref {
                self.v = self.params.v_reset;
                self.t_last_eval = t;
                return;
            }
        }
        let dt = t - self.t_last_eval;
        let v_eq = self.params.v_rest + dc;
        self.v = (self.v - v_eq).mul_add((-dt / self.params.tau_m).exp(), v_eq);
        self.t_last_eval = t;
    }

    /// Refractory injects collapse V back to `v_reset` and never fire.
    pub fn check_threshold(&mut self, now: Time) -> bool {
        if let Some(t_last) = self.last_spike_t {
            if now < t_last + self.params.tau_ref {
                self.v = self.params.v_reset;
                return false;
            }
        }
        if self.v >= self.params.v_threshold {
            self.v = self.params.v_reset;
            self.last_spike_t = Some(now);
            true
        } else {
            false
        }
    }

    pub fn tick(&mut self, now: Time, dc: Volts) -> bool {
        self.evolve_to(now, dc);
        self.check_threshold(now)
    }
}
