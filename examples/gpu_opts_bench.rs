//! Backend-options micro-bench: baseline vs --f16 vs --flash vs combo,
//! production-shaped steps (train_step_batch + clip subset + Muon
//! hybrid). In-process so every variant sees the same thermal state.
//!
//!   cargo run --release --features gpu --example gpu_opts_bench

#![cfg(feature = "gpu")]

use std::time::Instant;

use aurora::gpu::{DecoderTransformerGpu, GpuContext, GpuOpts};
use aurora::optimizer::{AdamWConfig, MuonConfig};
use aurora::transformer::{DecoderTransformer, TransformerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bench_variant(
    ctx: &GpuContext,
    cfg: &TransformerConfig,
    opts: GpuOpts,
    label: &str,
    batch: usize,
    warmup: usize,
    iters: usize,
) {
    let mut rng = StdRng::seed_from_u64(0xC1A0);
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu = DecoderTransformerGpu::from_cpu_with_opts(&cpu, ctx.clone(), batch, opts);
    let adam = AdamWConfig::default();
    let muon = MuonConfig::default();

    let t = cfg.seq_len_max;
    let seqs: Vec<Vec<u32>> = (0..batch)
        .map(|s| (0..t as u32).map(|i| (i * 7 + s as u32) % cfg.vocab as u32).collect())
        .collect();
    let tgts: Vec<Vec<u32>> = seqs
        .iter()
        .map(|s| s.iter().map(|&x| (x + 1) % cfg.vocab as u32).collect())
        .collect();
    let inputs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let tgt_refs: Vec<&[u32]> = tgts.iter().map(Vec::as_slice).collect();

    for _ in 0..warmup {
        gpu.zero_grad();
        let _ = gpu.train_step_batch(&inputs, &tgt_refs);
        gpu.clip_grads_adamw_subset(1.0 * batch as f32, false);
        gpu.step_muon_hybrid(&muon, &adam);
    }

    let mut loss_last = 0.0_f32;
    let s = Instant::now();
    for _ in 0..iters {
        gpu.zero_grad();
        loss_last = gpu.train_step_batch(&inputs, &tgt_refs) / batch as f32;
        gpu.clip_grads_adamw_subset(1.0 * batch as f32, false);
        gpu.step_muon_hybrid(&muon, &adam);
    }
    let ms = s.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let toks = (batch * t) as f64 / (ms / 1000.0);
    // 6·N·D model FLOPs per token (params from the CPU mirror).
    let n_params = cpu.param_count() as f64;
    let tflops = 6.0 * n_params * toks / 1e12;
    println!(
        "[{label}] {ms:>8.1} ms/step  {toks:>9.0} tok/s  ~{tflops:.3} TFLOPS  (loss {loss_last:.3})"
    );
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    println!(
        "adapter: {} (shader_f16: {})",
        ctx.adapter_info().name,
        ctx.shader_f16
    );

    let variants: [(&str, GpuOpts); 4] = [
        ("base      ", GpuOpts { flash: false, f16_weights: false }),
        ("f16       ", GpuOpts { flash: false, f16_weights: true }),
        ("flash     ", GpuOpts { flash: true, f16_weights: false }),
        ("flash+f16 ", GpuOpts { flash: true, f16_weights: true }),
    ];

    let nano = TransformerConfig::nano(4096);
    println!("\n── nano d384 L6, T=256, B=16 ──");
    for (label, opts) in variants {
        bench_variant(&ctx, &nano, opts, label, 16, 3, 10);
    }

    let mut nano_1k = TransformerConfig::nano(4096);
    nano_1k.seq_len_max = 1024;
    println!("\n── nano d384 L6, T=1024, B=2 ──");
    for (label, opts) in variants {
        bench_variant(&ctx, &nano_1k, opts, label, 2, 2, 6);
    }
}
