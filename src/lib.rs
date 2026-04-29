//! Aurora — biologically-inspired language model.
//!
//! Single architecture: closed-form continuous-time (CfC) Liquid Neural
//! Network cell (Hasani et al., Nature MI 2022, arXiv:2106.13898).
//! Originally inspired by C. elegans nervous system. Smooth activations
//! (sigmoid, tanh, exp) — no surrogate gradient — and bounded state by
//! construction.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod bpe;
pub mod checkpoint;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod lnn;
pub mod loss;
pub mod optimizer;
pub mod text_io;
