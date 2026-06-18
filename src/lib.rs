//! Aurora — decoder-only transformer LM, from-scratch Rust.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod attention;
pub mod bpe;
pub mod checkpoint;
pub mod governor;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod linalg;
pub mod optimizer;
pub mod text_io;
pub mod transformer;
#[cfg(feature = "viz")]
pub mod viz;
