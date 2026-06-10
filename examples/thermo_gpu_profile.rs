#![allow(clippy::cast_precision_loss)]

//! GPU thermo profiler. Resolves the perf regime by measuring per-step wall
//! time vs batch size (flat in B = dispatch/overhead-bound; linear in B =
//! compute-bound), plus the forward/backward split and the cost of the
//! per-step loss readback (host sync). Requires `--features gpu` + a GPU.
//!
//! `read_mean_loss()` doubles as a full queue drain (its buffer download
//! polls the device to completion), so it is the timing barrier.

use std::time::Instant;

use aurora::gpu::{GpuContext, ThermoGpu};
use aurora::optimizer::AdamWConfig;
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::model::ThermoLM;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cfg() -> ThermoConfig {
    ThermoConfig {
        vocab: 256,
        d: 1,
        h: 16,
        w: 16,
        c: 8,
        conv_hidden: 24,
        conserved: vec![CH_RHO, CH_E],
        k_steps: 4,
        pool: 1,
        dt: 0.1,
        t_init: 1.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: false,
        readout_modes: 4,
        use_diffusion: true,
    }
}

fn toks(seq: usize, b: usize, vocab: usize) -> Vec<u32> {
    (0..seq * b).map(|i| (i as u32 * 37 + 11) % vocab as u32).collect()
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    println!("GPU: {}", ctx.adapter_info().name);
    let c = cfg();
    let seq = 16;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let model = ThermoLM::new(c.clone(), &mut rng);
    let adam = AdamWConfig::default();
    println!(
        "cfg: grid {0}×{0}×{1} convH {2} K {3} pool {4} seq {5} · params {6}",
        c.h, c.c, c.conv_hidden, c.k_steps, c.pool, seq, model.param_count()
    );

    // ── Batch-scaling sweep: per-step wall time (async, no per-step sync) ──
    println!("\n── per-step train time vs batch (async) ──");
    println!("   B   | ms/step | ms/step/window | windows/s");
    for &b in &[24usize] {
        let mut gpu = ThermoGpu::new(&ctx, &model, seq, b);
        let t = toks(seq, b, c.vocab);
        let tg = toks(seq, b, c.vocab);
        for _ in 0..5 {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss(); // drain warmup
        let n = 12u32;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss(); // drain → barrier
        let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);
        println!(
            "  {b:>4} | {ms:>7.2} | {:>14.3} | {:>8.0}",
            ms / b as f64,
            b as f64 / (ms / 1e3)
        );
    }

    // ── Correctness: im2col forward must match per-element conv3g ──
    // Same model + input → same forward loss if the conv path is numerically
    // equivalent (f32 GEMM rounding only).
    {
        let mut g_off = ThermoGpu::new(&ctx, &model, seq, 4);
        let mut g_on = ThermoGpu::new(&ctx, &model, seq, 4);
        g_on.conv_im2col = true;
        let tc = toks(seq, 4, c.vocab);
        let l_off = g_off.train_step(&tc, &tc, &adam, true).unwrap();
        let l_on = g_on.train_step(&tc, &tc, &adam, true).unwrap();
        println!(
            "\ncorrectness: loss conv3g {l_off:.6}  im2col {l_on:.6}  |Δ| {:.2e}",
            (l_off - l_on).abs()
        );
    }

    // ── In-process A/B: forward conv per-element vs im2col + tiled GEMM ──
    // Both variants run in ONE process, alternating each round, so they share
    // the same (drifting) GPU boost clock — the per-round delta cancels clock
    // noise. min-over-rounds = best sustained clock. Measures the full train
    // step (fwd+bwd+optim) with only the forward conv path toggled.
    println!("\n── fwd conv: per-element conv3g vs im2col+GEMM @ B=24, in-process A/B ──");
    let b = 24;
    let mut gpu = ThermoGpu::new(&ctx, &model, seq, b);
    let t = toks(seq, b, c.vocab);
    let tg = toks(seq, b, c.vocab);
    let n = 40u32;
    for _ in 0..20 {
        gpu.train_step(&t, &tg, &adam, false);
    }
    let _ = gpu.read_mean_loss();

    let mut min_conv3g = f64::MAX;
    let mut min_im2col = f64::MAX;
    for round in 0..3 {
        gpu.conv_im2col = false;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let c3g = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        gpu.conv_im2col = true;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let im2 = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        min_conv3g = min_conv3g.min(c3g);
        min_im2col = min_im2col.min(im2);
        println!("  round {round}: conv3g {c3g:7.2}  im2col {im2:7.2}  Δ {:+.2} ms", c3g - im2);
    }
    let d = min_conv3g - min_im2col;
    println!("  ── min of 3 rounds (B=24, d={}) ── (flip cfg().d to 4 for the conv-dominated regime)", c.d);
    println!("  per-element conv3g : {min_conv3g:7.2} ms/step");
    println!("  im2col + GEMM      : {min_im2col:7.2} ms/step");
    println!("  Δ = {d:+.2} ms/step ({:+.1}%)", d / min_conv3g * 100.0);

    // ── In-process A/B: im2col forward GEMM f32 vs f16 (both im2col) ──
    if !ctx.has_f16 {
        println!("\nno SHADER_F16 — skipping f16 conv A/B");
        return;
    }
    {
        // Correctness: f16 forward loss vs f32, same model + input.
        let mut g32 = ThermoGpu::new(&ctx, &model, seq, 4);
        let mut g16 = ThermoGpu::new(&ctx, &model, seq, 4);
        g32.conv_im2col = true;
        g16.conv_im2col = true;
        g16.conv_f16 = true;
        let tc = toks(seq, 4, c.vocab);
        let l32 = g32.train_step(&tc, &tc, &adam, true).unwrap();
        let l16 = g16.train_step(&tc, &tc, &adam, true).unwrap();
        println!("\ncorrectness: loss f32 {l32:.6}  f16 {l16:.6}  |Δ| {:.2e}", (l32 - l16).abs());
    }
    println!("\n── fwd conv GEMM: f32 vs f16 @ B=24, in-process A/B (im2col both) ──");
    gpu.conv_im2col = true;
    let mut min_f32 = f64::MAX;
    let mut min_f16 = f64::MAX;
    for round in 0..3 {
        gpu.conv_f16 = false;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let f32ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        gpu.conv_f16 = true;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let f16ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        min_f32 = min_f32.min(f32ms);
        min_f16 = min_f16.min(f16ms);
        println!("  round {round}: f32 {f32ms:7.2}  f16 {f16ms:7.2}  Δ {:+.2} ms", f32ms - f16ms);
    }
    let df = min_f32 - min_f16;
    println!("  ── min of 3 rounds (B=24, d={}) ──", c.d);
    println!("  im2col + f32 GEMM  : {min_f32:7.2} ms/step");
    println!("  im2col + f16 GEMM  : {min_f16:7.2} ms/step");
    println!("  Δ = {df:+.2} ms/step ({:+.1}%)", df / min_f32 * 100.0);
}
