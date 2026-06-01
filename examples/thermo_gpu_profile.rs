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
        let n = 60u32;
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

    // ── In-process A/B: fused 3-add vs 2-vadd chain @ B=24 ──
    // Both variants run in ONE process, alternating each round, so they share
    // the same (drifting) GPU boost clock — the per-round delta cancels clock
    // noise. GPU clocks can't be locked without root here; min-over-rounds is
    // the robust estimator (best round = highest sustained clock).
    println!("\n── d_s add: fused(a+b+c) vs 2×vadd chain @ B=24, in-process A/B ──");
    let b = 24;
    let mut gpu = ThermoGpu::new(&ctx, &model, seq, b);
    let t = toks(seq, b, c.vocab);
    let tg = toks(seq, b, c.vocab);
    let n = 80u32;
    for _ in 0..40 {
        gpu.train_step(&t, &tg, &adam, false);
    }
    let _ = gpu.read_mean_loss();

    let mut min_chain = f64::MAX;
    let mut min_fused = f64::MAX;
    for round in 0..8 {
        gpu.fused_add = false;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let chain = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        gpu.fused_add = true;
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.train_step(&t, &tg, &adam, false);
        }
        let _ = gpu.read_mean_loss();
        let fused = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);

        min_chain = min_chain.min(chain);
        min_fused = min_fused.min(fused);
        println!("  round {round}: chain {chain:7.2}  fused {fused:7.2}  Δ {:+.2} ms", chain - fused);
    }
    let d = min_chain - min_fused;
    println!("  ── min of 8 rounds ──");
    println!("  2-vadd chain (before): {min_chain:7.2} ms/step");
    println!("  fused add3   (after) : {min_fused:7.2} ms/step");
    println!("  Δ = {d:+.2} ms/step ({:+.1}%)  [predicted ~7-8 ms from −64 dispatches]", -d / min_chain * 100.0);
}
