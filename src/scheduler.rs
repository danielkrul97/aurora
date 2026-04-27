//! Event queue for the asynchronous engine.
//!
//! Min-heap over `(t, seq)`: events are processed in time order, ties broken
//! by insertion sequence so traversal is deterministic across runs.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::types::{NeuronId, Time, Volts};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EventKind {
    /// Deliver `amplitude` to the addressed compartment. Both synaptic
    /// deliveries and external pulses use this — they're indistinguishable
    /// at the receiving end.
    Pulse {
        neuron: NeuronId,
        dendrite: Option<usize>,
        amplitude: Volts,
    },
}

#[derive(Copy, Clone, Debug)]
pub struct Event {
    pub t: Time,
    pub seq: u64,
    pub kind: EventKind,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && self.seq == other.seq
    }
}

impl Eq for Event {}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .t
            .partial_cmp(&self.t)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Default)]
pub struct EventQueue {
    heap: BinaryHeap<Event>,
}

impl EventQueue {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, ev: Event) {
        self.heap.push(ev);
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop()
    }

    #[must_use]
    pub fn peek(&self) -> Option<&Event> {
        self.heap.peek()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}
