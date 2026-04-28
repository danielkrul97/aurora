//! Aurora — biologically-inspired spiking neural network.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod attention;
pub mod bpe;
pub mod dendrite;
pub mod eligibility;
pub mod encoding;
pub mod engine;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod homeostasis;
pub mod lif;
pub mod linalg;
pub mod mnist;
pub mod network;
pub mod neuromodulator;
pub mod neuron;
pub mod optimizer;
pub mod readout;
pub mod scheduler;
pub mod snn_dense;
pub mod snn_gated;
pub mod snn_gated_multi;
pub mod snn_loss;
pub mod snn_recurrent;
pub mod stdp;
pub mod surrogate;
pub mod synapse;
pub mod text_io;
pub mod topology;
pub mod types;
