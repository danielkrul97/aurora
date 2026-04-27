//! Rate-based readout: count spikes per output neuron in a time window,
//! predict class by argmax. Ties broken by lowest neuron index.

use crate::types::{NeuronId, Spike, Time};

#[must_use]
pub fn count_spikes_per_neuron(
    spikes: &[Spike],
    target_neurons: &[NeuronId],
    window_start: Time,
    window_end: Time,
) -> Vec<usize> {
    let mut counts = vec![0_usize; target_neurons.len()];
    for s in spikes {
        if s.t < window_start || s.t > window_end {
            continue;
        }
        if let Some(idx) = target_neurons.iter().position(|n| *n == s.source) {
            counts[idx] += 1;
        }
    }
    counts
}

#[must_use]
pub fn argmax(counts: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (i, &c) in counts.iter().enumerate() {
        match best {
            None => best = Some((i, c)),
            Some((_, bc)) if c > bc => best = Some((i, c)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

#[must_use]
pub fn predict_class(
    spikes: &[Spike],
    output_neurons: &[NeuronId],
    window_start: Time,
    window_end: Time,
) -> Option<usize> {
    let counts = count_spikes_per_neuron(spikes, output_neurons, window_start, window_end);
    let total: usize = counts.iter().sum();
    if total == 0 {
        return None;
    }
    argmax(&counts)
}
