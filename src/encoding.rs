//! Rate coding helpers.
//!
//! `encode_image_rate` returns Poisson spike-time lists per pixel for the
//! event-driven engine. `rate_encode_bernoulli` returns a per-timestep
//! Bernoulli mask for the surrogate-gradient SNN — different output shape,
//! kept separate on purpose.

use rand::Rng;

use crate::types::Time;

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

pub fn rate_encode_bernoulli(
    image: &[f32],
    t_steps: usize,
    max_rate: f32,
    rng: &mut impl Rng,
) -> Vec<Vec<f32>> {
    (0..t_steps)
        .map(|_| {
            image
                .iter()
                .map(|&p| {
                    if rng.random::<f32>() < p * max_rate {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}
