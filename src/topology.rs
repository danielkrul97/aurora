//! Builders for layered feedforward networks.

use rand::Rng;

use crate::lif::LifParams;
use crate::network::Network;
use crate::types::{NeuronId, Time, Weight};

/// All-to-all lateral inhibition within a layer: every neuron sends a
/// fixed inhibitory synapse of weight `-strength` to every other. The
/// connections are marked non-plastic so STDP cannot erode them.
///
/// # Panics
/// Panics if `strength <= 0`.
pub fn add_lateral_inhibition(
    net: &mut Network,
    layer: &[NeuronId],
    strength: Weight,
    delay: Time,
) {
    assert!(
        strength > 0.0,
        "strength is the magnitude; pass a positive value"
    );
    for &pre in layer {
        for &post in layer {
            if pre == post {
                continue;
            }
            let s = net.add_synapse(pre, post, None, -strength, delay);
            net.set_plastic(s, false);
        }
    }
}

/// Construct a feedforward network from `layer_sizes` (e.g., `[784, 100, 10]`).
///
/// Adjacent layers are fully connected with random weights uniform in
/// `weight_range` and a fixed `delay`. Returns the network and per-layer
/// neuron-id lists.
pub fn build_feedforward<R: Rng>(
    layer_sizes: &[usize],
    weight_range: (Weight, Weight),
    delay: Time,
    rng: &mut R,
) -> (Network, Vec<Vec<NeuronId>>) {
    let mut net = Network::new();
    let mut layers: Vec<Vec<NeuronId>> = Vec::with_capacity(layer_sizes.len());
    for &size in layer_sizes {
        let layer: Vec<NeuronId> = (0..size)
            .map(|_| net.add_neuron(LifParams::default()))
            .collect();
        layers.push(layer);
    }
    let (lo, hi) = weight_range;
    let span = hi - lo;
    for w in layers.windows(2) {
        for &pre in &w[0] {
            for &post in &w[1] {
                let weight = span.mul_add(rng.random::<f64>(), lo);
                net.add_synapse(pre, post, None, weight, delay);
            }
        }
    }
    (net, layers)
}
