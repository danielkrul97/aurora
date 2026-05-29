//! Thermodynamic LM — Aurora research line B (see `docs/13_thermo_lm_plan.md`).
//!
//! A Level-B continuous gas substrate as a language model: a learned
//! local operator `N_θ` evolves an `H×W×C` field; conserved channels move
//! by antisymmetric edge fluxes (mass/energy conserved by construction),
//! every channel diffuses under a temperature-gated Laplacian, tokens
//! inject localized forcing, and a moment-pooling readout maps the field
//! to logits. Trained by backprop-through-time over a K-step-per-token
//! rollout.
//!
//! Same hand-written forward/backward primitive style as
//! `crate::transformer`: each op exposes `*Weights` / `*Grads` / `*Cache`
//! structs plus `_forward` / `_backward` functions, validated by
//! finite-difference gradient checks in `tests/thermo_primitives.rs`.

pub mod checkpoint;
pub mod conv;
pub mod diffusion;
pub mod field;
pub mod flux;
pub mod inject;
pub mod model;
pub mod readout;
pub mod step;

pub use conv::{conv_backward, conv_forward, ConvCache, ConvGrads, ConvWeights};
pub use diffusion::{
    diffusion_backward, diffusion_forward, DiffusionCache, DiffusionGrads, DiffusionWeights,
};
pub use flux::{flux_div_backward, flux_div_forward};
pub use inject::{inject_backward, inject_forward, InjectCache, InjectGrads, InjectWeights};
pub use model::{anneal_at, RolloutCache, ThermoGrads, ThermoLM};
pub use readout::{readout_backward, readout_forward, ReadoutCache, ReadoutGrads, ReadoutWeights};
pub use step::{step_backward, step_forward, StepCache, StepGrads, StepWeights};
pub use field::{idx, init_field, ThermoConfig, CH_E, CH_RHO, CH_T, CH_UX, CH_UY, N_PHYS};
