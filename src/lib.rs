//! Aurora — biologically-inspired spiking neural network.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod dendrite;
pub mod homeostasis;
pub mod lif;
pub mod network;
pub mod neuron;
pub mod stdp;
pub mod synapse;
pub mod types;
