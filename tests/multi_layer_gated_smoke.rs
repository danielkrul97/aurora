#![allow(clippy::cast_precision_loss)]

//! Sprint 23 smoke test: multi-layer gated SNN.
//!
//! 1. `n_layers=1` should match single-layer `GatedRecurrentSnn`
//!    behavior on the same task.
//! 2. `n_layers=2` should compress the 8-byte cycle below uniform
//!    baseline (loss decreases over epochs).

use aurora::optimizer::AdamConfig;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, VOCAB};
use rand::rngs::StdRng;
use rand::SeedableRng;

const CHUNK: usize = 32;

#[test]
fn multi_layer_n1_compresses_repeating_corpus() {
    let pattern: Vec<u8> = b"abcabcab".to_vec();
    let mut corpus = Vec::with_capacity(512);
    for _ in 0..64 {
        corpus.extend_from_slice(&pattern);
    }

    let mut rng = StdRng::seed_from_u64(0x10);
    let mut net = MultiLayerGatedSnn::new(VOCAB, &[64], VOCAB, SpikeFn::default(), &mut rng);
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    let mut last_bpb = f32::INFINITY;
    for epoch in 0..6 {
        let mut total_nats = 0.0_f32;
        let mut total_bytes = 0_usize;
        let mut pos = 0;
        while pos + CHUNK < corpus.len() {
            let inputs = &corpus[pos..pos + CHUNK];
            let targets = &corpus[pos + 1..pos + CHUNK + 1];
            let cache = net.forward_lm_bytes(inputs);
            net.zero_grad();
            let loss = net.backward_lm_bytes(&cache, inputs, targets);
            net.clip_grads(1.0);
            net.step_adam(&adam);
            total_nats += loss;
            total_bytes += CHUNK;
            pos += CHUNK;
        }
        last_bpb = bits_per_byte(total_nats, total_bytes);
        eprintln!("[n=1] epoch {epoch} bpb {last_bpb:.3}");
    }

    assert!(
        last_bpb < 4.0,
        "1-layer multi must compress 8-byte cycle below 4.0 bpb, got {last_bpb:.3}"
    );
}

#[test]
fn multi_layer_n2_compresses_repeating_corpus() {
    let pattern: Vec<u8> = b"abcabcab".to_vec();
    let mut corpus = Vec::with_capacity(512);
    for _ in 0..64 {
        corpus.extend_from_slice(&pattern);
    }

    let mut rng = StdRng::seed_from_u64(0x20);
    let mut net = MultiLayerGatedSnn::new(VOCAB, &[64, 64], VOCAB, SpikeFn::default(), &mut rng);
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    let mut first_bpb = f32::INFINITY;
    let mut last_bpb = f32::INFINITY;
    for epoch in 0..6 {
        let mut total_nats = 0.0_f32;
        let mut total_bytes = 0_usize;
        let mut pos = 0;
        while pos + CHUNK < corpus.len() {
            let inputs = &corpus[pos..pos + CHUNK];
            let targets = &corpus[pos + 1..pos + CHUNK + 1];
            let cache = net.forward_lm_bytes(inputs);
            net.zero_grad();
            let loss = net.backward_lm_bytes(&cache, inputs, targets);
            net.clip_grads(1.0);
            net.step_adam(&adam);
            total_nats += loss;
            total_bytes += CHUNK;
            pos += CHUNK;
        }
        last_bpb = bits_per_byte(total_nats, total_bytes);
        if epoch == 0 {
            first_bpb = last_bpb;
        }
        eprintln!("[n=2] epoch {epoch} bpb {last_bpb:.3}");
    }

    assert!(
        last_bpb < first_bpb * 0.85,
        "2-layer multi must reduce bpb over training: first={first_bpb:.3} last={last_bpb:.3}"
    );
    assert!(
        last_bpb < 6.0,
        "2-layer multi must beat uniform (8.0) on 8-byte cycle, got {last_bpb:.3}"
    );
}
