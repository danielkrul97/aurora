#![cfg(feature = "gpu")]

//! Measures GPU vs CPU end-to-end forward+backward+Adam throughput on
//! the gated recurrent SNN. Both paths use identical hyperparameters
//! and identical initial weights; only the execution backend differs.
//!
//! Run with:
//!   cargo bench --features gpu --bench lm_gpu_speedup

use std::time::Instant;

use aurora::gpu::{GatedRecurrentSnnGpu, GpuContext};
use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::VOCAB;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CHUNK: usize = 64;
const N_CHUNKS: usize = 50;

fn main() {
    for &n_hidden in &[128_usize, 256, 512] {
        run_one(n_hidden);
    }
}

fn run_one(n_hidden: usize) {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let bytes: Vec<u8> = (0..CHUNK * (N_CHUNKS + 1))
        .map(|_| rng.random::<u8>())
        .collect();
    let adam = AdamConfig::default();
    let clip = 1.5_f32;

    // Both backends initialized from identical weights.
    let mut weight_rng = StdRng::seed_from_u64(0xCAFE_F00D);
    let net_init = GatedRecurrentSnn::new(VOCAB, n_hidden, VOCAB, SpikeFn::default(), &mut weight_rng);
    let mut net_cpu = net_init.clone();

    // CPU run.
    let cpu_start = Instant::now();
    for c in 0..N_CHUNKS {
        let pos = c * CHUNK;
        let inputs: &[u8] = &bytes[pos..pos + CHUNK];
        let targets: &[u8] = &bytes[pos + 1..pos + CHUNK + 1];
        let cache = net_cpu.forward_lm_bytes(inputs);
        net_cpu.zero_grad();
        let _ = net_cpu.backward_lm_bytes(&cache, inputs, targets);
        net_cpu.clip_grads(clip);
        net_cpu.step_adam(&adam);
    }
    let cpu_s = cpu_start.elapsed().as_secs_f64();

    // GPU run.
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available: {e}");
            return;
        }
    };
    let mut net_gpu = GatedRecurrentSnnGpu::from_cpu(&net_init, ctx);
    let gpu_start = Instant::now();
    for c in 0..N_CHUNKS {
        let pos = c * CHUNK;
        let inputs: &[u8] = &bytes[pos..pos + CHUNK];
        let targets: &[u8] = &bytes[pos + 1..pos + CHUNK + 1];
        net_gpu.zero_grad_gpu();
        let _ = net_gpu.forward_backward_lm_bytes_gpu_stateless(inputs, targets);
        net_gpu.clip_grads_gpu(clip);
        net_gpu.step_adam_gpu(&adam);
    }
    let gpu_s = gpu_start.elapsed().as_secs_f64();

    let speedup = cpu_s / gpu_s;
    println!(
        "{N_CHUNKS} chunks × {CHUNK} bytes ({} → {n_hidden} → {}):",
        VOCAB, VOCAB
    );
    println!("  cpu : {cpu_s:.2}s ({:.1} chunks/s)", N_CHUNKS as f64 / cpu_s);
    println!("  gpu : {gpu_s:.2}s ({:.1} chunks/s)", N_CHUNKS as f64 / gpu_s);
    println!("  speedup: {speedup:.2}×\n");
}
