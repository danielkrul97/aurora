//! Quick wall-time bench for MultiLayerGatedSnn forward+backward.
//! Run via `cargo run --release --bench multi_layer_perf` (harness=false).

use std::time::Instant;

use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedState;
use aurora::snn_gated_multi::MultiLayerGatedSnn;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let bytes: Vec<u8> = (0..200_000).map(|_| rng.random::<u8>()).collect();

    let mut net = MultiLayerGatedSnn::new(
        256,
        &[128, 128],
        256,
        SpikeFn::default(),
        &mut StdRng::seed_from_u64(0xCAFE_F00D),
    );
    let adam = AdamConfig {
        lr: 0.005,
        ..AdamConfig::default()
    };

    let chunk = 64;
    let batch = 8;
    let stream_len = bytes.len() / batch;
    let n_chunks = 100;
    let mut states: Vec<Vec<GatedState>> = (0..batch).map(|_| net.fresh_states()).collect();

    // Warm-up
    for ci in 0..5 {
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(batch);
        let mut targets: Vec<&[u8]> = Vec::with_capacity(batch);
        for b in 0..batch {
            let p = b * stream_len + ci * chunk;
            inputs.push(&bytes[p..p + chunk]);
            targets.push(&bytes[p + 1..p + chunk + 1]);
        }
        net.forward_backward_batch_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
    }

    let start = Instant::now();
    let mut total_loss = 0.0_f32;
    for ci in 0..n_chunks {
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(batch);
        let mut targets: Vec<&[u8]> = Vec::with_capacity(batch);
        for b in 0..batch {
            let p = b * stream_len + (ci + 5) * chunk;
            inputs.push(&bytes[p..p + chunk]);
            targets.push(&bytes[p + 1..p + chunk + 1]);
        }
        total_loss += net.forward_backward_batch_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&adam);
    }
    let elapsed = start.elapsed().as_secs_f64();

    println!(
        "{n_chunks} chunks × {batch} batch × {chunk} steps (256→128→128→256):"
    );
    println!("  wall-time : {elapsed:.3}s");
    println!("  per chunk : {:.2}ms", elapsed * 1000.0 / n_chunks as f64);
    println!("  loss      : {total_loss:.2}");
}
