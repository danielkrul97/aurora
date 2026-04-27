//! Rate coding: pixel intensity in [0,1] maps to a Poisson spike train
//! whose mean rate is `intensity · max_rate_hz`. Output spike times are
//! relative to the start of the encoding window.

use rand::Rng;

use crate::types::Time;

/// Generate Poisson spike times in `[0, duration)` with mean rate `rate_hz`.
/// Returns an empty vec if `rate_hz <= 0`.
pub fn poisson_spike_train(rate_hz: f64, duration: Time, rng: &mut impl Rng) -> Vec<Time> {
    let mut out = Vec::new();
    if rate_hz <= 0.0 {
        return out;
    }
    let mean_isi = 1.0 / rate_hz;
    let mut t = -mean_isi * (1.0 - rng.random::<f64>()).ln();
    loop {
        if t >= duration {
            break;
        }
        out.push(t);
        t += -mean_isi * (1.0 - rng.random::<f64>()).ln();
    }
    out
}

/// Per-pixel Poisson encoding. Returns `Vec<Vec<Time>>` indexed by pixel,
/// where each inner vec is the spike times for that input neuron.
pub fn encode_image_rate(
    image: &[f64],
    duration: Time,
    max_rate_hz: f64,
    rng: &mut impl Rng,
) -> Vec<Vec<Time>> {
    image
        .iter()
        .map(|&p| poisson_spike_train(p.clamp(0.0, 1.0) * max_rate_hz, duration, rng))
        .collect()
}
