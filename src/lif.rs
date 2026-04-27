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
}

impl Lif {
    #[must_use]
    pub const fn new(params: LifParams) -> Self {
        Self {
            params,
            v: params.v_rest,
            last_spike_t: None,
        }
    }

    pub fn inject(&mut self, v: Volts) {
        self.v += v;
    }

    pub fn tick(&mut self, now: Time, dt: Time, dc: Volts) -> bool {
        if let Some(t_last) = self.last_spike_t {
            if now - t_last < self.params.tau_ref {
                self.v = self.params.v_reset;
                return false;
            }
        }

        let v_eq = self.params.v_rest + dc;
        self.v = (self.v - v_eq).mul_add((-dt / self.params.tau_m).exp(), v_eq);

        if self.v >= self.params.v_threshold {
            self.v = self.params.v_reset;
            self.last_spike_t = Some(now);
            true
        } else {
            false
        }
    }
}
