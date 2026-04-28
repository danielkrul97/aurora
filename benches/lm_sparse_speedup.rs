//! Measures dense vs sparse LM throughput on a synthetic byte stream.
//! Both paths produce bit-identical results (see tests/sparse_lm_parity);
//! this bench just confirms the speedup is real.

use std::time::Instant;

use aurora::optimizer::AdamConfig;
use aurora::snn_recurrent::RecurrentSurrogateSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{one_hot, VOCAB};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_HIDDEN: usize = 256;
const CHUNK: usize = 64;
const N_CHUNKS: usize = 200;

fn main() {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let bytes: Vec<u8> = (0..CHUNK * (N_CHUNKS + 1))
        .map(|_| rng.random::<u8>())
        .collect();
    let adam = AdamConfig::default();

    let mut net_dense = RecurrentSurrogateSnn::new(
        VOCAB,
        N_HIDDEN,
        VOCAB,
        SpikeFn::default(),
        5.0,
        1.0,
        &mut StdRng::seed_from_u64(0xCAFE_F00D),
    );
    let mut net_sparse = RecurrentSurrogateSnn::new(
        VOCAB,
        N_HIDDEN,
        VOCAB,
        SpikeFn::default(),
        5.0,
        1.0,
        &mut StdRng::seed_from_u64(0xCAFE_F00D),
    );

    // Dense path
    let dense_start = Instant::now();
    for c in 0..N_CHUNKS {
        let pos = c * CHUNK;
        let inputs: Vec<Vec<f32>> = (0..CHUNK).map(|t| one_hot(bytes[pos + t])).collect();
        let targets: Vec<u8> = (0..CHUNK).map(|t| bytes[pos + t + 1]).collect();
        let cache = net_dense.forward_lm(&inputs);
        net_dense.zero_grad();
        let _ = net_dense.backward_lm(&cache, &inputs, &targets);
        net_dense.step_adam(&adam);
    }
    let dense_s = dense_start.elapsed().as_secs_f64();

    // Sparse path
    let sparse_start = Instant::now();
    for c in 0..N_CHUNKS {
        let pos = c * CHUNK;
        let inputs: &[u8] = &bytes[pos..pos + CHUNK];
        let targets: &[u8] = &bytes[pos + 1..pos + CHUNK + 1];
        let cache = net_sparse.forward_lm_bytes(inputs);
        net_sparse.zero_grad();
        let _ = net_sparse.backward_lm_bytes(&cache, inputs, targets);
        net_sparse.step_adam(&adam);
    }
    let sparse_s = sparse_start.elapsed().as_secs_f64();

    let speedup = dense_s / sparse_s;
    println!(
        "{N_CHUNKS} chunks × {CHUNK} bytes ({} → {N_HIDDEN} → {}):",
        VOCAB, VOCAB
    );
    println!("  dense  : {dense_s:.2}s");
    println!("  sparse : {sparse_s:.2}s");
    println!("  speedup: {speedup:.2}×");
}
