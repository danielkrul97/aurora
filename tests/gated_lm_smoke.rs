#![allow(clippy::cast_precision_loss)]

//! Sprint 11 smoke test: gated recurrent SNN must compress the same
//! 8-byte cycle as `char_lm_smoke` to under uniform-random baseline.
//! This is the smoke version — long-range memory benefits show up on
//! tiny-shakespeare in `examples/char_lm_gated`.

use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{bits_per_byte, VOCAB};
use rand::rngs::StdRng;
use rand::SeedableRng;

const N_HIDDEN: usize = 64;
const CHUNK: usize = 32;

#[test]
fn gated_lm_compresses_repeating_corpus() {
    let pattern: Vec<u8> = b"abcabcab".to_vec();
    let mut corpus = Vec::with_capacity(512);
    for _ in 0..64 {
        corpus.extend_from_slice(&pattern);
    }

    let mut rng = StdRng::seed_from_u64(0x10);
    let mut net = GatedRecurrentSnn::new(VOCAB, N_HIDDEN, VOCAB, SpikeFn::default(), &mut rng);
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
        eprintln!("epoch {epoch} bpb {last_bpb:.3}");
    }

    assert!(
        last_bpb < 4.0,
        "gated LM must compress 8-byte cycle below 4.0 bpb (uniform=8.0), got {last_bpb:.3}"
    );
}
