//! Per-step latency micro-bench for the GPU transformer. Run with:
//!   cargo run --release --features gpu --example gpu_perf_bench

#![cfg(feature = "gpu")]

use std::time::Instant;

use aurora::gpu::{DecoderTransformerGpu, GpuContext};
use aurora::optimizer::AdamWConfig;
use aurora::transformer::{DecoderTransformer, TransformerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn run_config(label: &str, cfg: TransformerConfig, seq_len: usize, warmup: usize, iters: usize) {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{label}] skip — no GPU: {e}");
            return;
        }
    };
    eprintln!("[{label}] adapter: {}", ctx.adapter_info().name);
    eprintln!(
        "[{label}] cfg: vocab={} d={} n_layer={} n_head={} d_ff={} T={}",
        cfg.vocab, cfg.d_model, cfg.n_layer, cfg.n_head, cfg.d_ff, seq_len
    );

    let mut rng = StdRng::seed_from_u64(0xC1A0);
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu = DecoderTransformerGpu::from_cpu(&cpu, ctx);
    let adam = AdamWConfig::default();

    let tokens: Vec<u32> = (0..seq_len as u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    // Warm-up (skip first runs to avoid pipeline / driver caching noise).
    for _ in 0..warmup {
        gpu.zero_grad();
        let _ = gpu.train_step(&tokens, &targets);
        gpu.clip_grads(1.0);
        gpu.step_adamw(&adam);
    }

    let mut t_zero = 0u128;
    let mut t_train = 0u128;
    let mut t_clip = 0u128;
    let mut t_adam = 0u128;
    let mut loss_acc = 0.0_f32;
    for _ in 0..iters {
        let s = Instant::now();
        gpu.zero_grad();
        t_zero += s.elapsed().as_nanos();
        let s = Instant::now();
        let loss = gpu.train_step(&tokens, &targets);
        t_train += s.elapsed().as_nanos();
        loss_acc += loss;
        let s = Instant::now();
        gpu.clip_grads(1.0);
        t_clip += s.elapsed().as_nanos();
        let s = Instant::now();
        gpu.step_adamw(&adam);
        t_adam += s.elapsed().as_nanos();
    }
    let total = t_zero + t_train + t_clip + t_adam;
    let ms = |ns: u128| (ns as f64) / 1_000_000.0 / (iters as f64);
    eprintln!(
        "[{label}] per-step: zero={:.2}ms train={:.2}ms clip={:.2}ms adam={:.2}ms TOTAL={:.2}ms (mean loss={:.4})",
        ms(t_zero),
        ms(t_train),
        ms(t_clip),
        ms(t_adam),
        ms(total),
        loss_acc / iters as f32,
    );
}

fn main() {
    // Tiny config — fits memory tight on GTX 1650 even unoptimized.
    run_config(
        "tiny",
        TransformerConfig {
            vocab: 256,
            d_model: 64,
            n_layer: 2,
            n_head: 4,
            d_ff: 128,
            seq_len_max: 64,
        },
        32,
        5,
        20,
    );
    run_config(
        "nano",
        TransformerConfig {
            vocab: 4096,
            d_model: 384,
            n_layer: 6,
            n_head: 6,
            d_ff: 1536,
            seq_len_max: 256,
        },
        128,
        5,
        20,
    );
}
