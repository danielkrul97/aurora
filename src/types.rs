//! Core scalar and identifier types shared across the network.

pub type Time = f64;
pub type Volts = f64;
pub type Weight = f64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NeuronId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SynapseId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Spike {
    pub t: Time,
    pub source: NeuronId,
}
