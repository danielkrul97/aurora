//! Spike-Timing Dependent Plasticity rule.
//!
//! `Δw = +A⁺·exp(-Δt/τ⁺)` for `Δt > 0` (pre fired before post → potentiation).
//! `Δw = −A⁻·exp(+Δt/τ⁻)` for `Δt < 0` (pre fired after post → depression).

use crate::types::{Time, Weight};

#[derive(Copy, Clone, Debug)]
pub struct StdpParams {
    pub a_plus: Weight,
    pub a_minus: Weight,
    pub tau_plus: Time,
    pub tau_minus: Time,
}

impl Default for StdpParams {
    /// Bi & Poo (1998) classical values.
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 0.020,
            tau_minus: 0.020,
        }
    }
}

/// Compute Δw given `dt = t_post − t_pre`.
#[must_use]
pub fn stdp_delta(dt_post_minus_pre: Time, params: &StdpParams) -> Weight {
    if dt_post_minus_pre > 0.0 {
        params.a_plus * (-dt_post_minus_pre / params.tau_plus).exp()
    } else {
        -params.a_minus * (dt_post_minus_pre / params.tau_minus).exp()
    }
}
