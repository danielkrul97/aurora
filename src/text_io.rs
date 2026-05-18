#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

//! Token-level corpus I/O and sampling.

use std::fs;
use std::path::Path;

/// Read raw bytes from a file. No filtering, no normalization — keeps
/// the corpus exactly as-is so bits-per-byte numbers are comparable to
/// canonical char-LM benchmarks.
///
/// # Errors
/// Returns the underlying I/O error if the file cannot be read.
pub fn load_corpus<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>> {
    fs::read(path)
}

/// Sample a token id from `softmax(logits / temperature)`.
///
/// `T → 0` is greedy argmax; `T = 1` is the model's own distribution.
pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl rand::Rng) -> u32 {
    let n = logits.len();
    debug_assert!(n > 0);
    let inv_t = 1.0 / temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|&v| v * inv_t).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let r: f32 = rng.random::<f32>() * sum;
    let mut acc = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        acc += e;
        if r <= acc {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Average per-byte cross-entropy in **bits** ("bits per byte" / BPB).
/// Random uniform = 8.0 bits/byte; perfect prediction = 0.0.
#[must_use]
pub fn bits_per_byte(total_nats: f32, n_bytes: usize) -> f32 {
    if n_bytes == 0 {
        return 0.0;
    }
    total_nats / (n_bytes as f32) / std::f32::consts::LN_2
}
