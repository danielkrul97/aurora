//! Aurora — bipolar Hyperdimensional Computing (HDC / VSA) language model.
//!
//! Single architecture: position-binding bipolar HDC encoder
//! (Kanerva 2009, MAP) feeding a trainable linear readout. The encoder
//! is gradient-free w.r.t. the codebook (frozen ±1) for the current
//! sprint range; the readout learns by Adam against cross-entropy on
//! BPE tokens.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod bpe;
pub mod checkpoint;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hdc;
pub mod loss;
pub mod optimizer;
pub mod text_io;
