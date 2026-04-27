//! Aurora — biologically-inspired spiking neural network.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod dendrite;
pub mod eligibility;
pub mod engine;
pub mod homeostasis;
pub mod lif;
pub mod network;
pub mod neuromodulator;
pub mod neuron;
pub mod scheduler;
pub mod stdp;
pub mod synapse;
pub mod types;
