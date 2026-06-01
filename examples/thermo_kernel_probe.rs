#![allow(clippy::cast_precision_loss)]

//! Throwaway kernel-cost probe @ grid 32, B=24 (user's config). Two views:
//!  (A) phase split  — zero_grad / forward / backward / adamw, timed directly
//!      via drains (min-of-rounds to cancel GPU-boost-clock drift).
//!  (B) config ablation — full vs no-diffusion vs moments-only vs K=2, async
//!      train_step timing; deltas attribute cost to subsystems (diffusion,
//!      spectral readout, per-substep conv+transport ∝ K).

use std::time::Instant;

use aurora::gpu::{GpuContext, ThermoGpu};
use aurora::optimizer::AdamWConfig;
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::model::ThermoLM;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SEQ: usize = 16;
const B: usize = 24;
const ROUNDS: usize = 4;
const N: u32 = 10;

fn base_cfg() -> ThermoConfig {
    ThermoConfig {
        vocab: 256,
        h: 32,
        w: 32,
        c: 8,
        conv_hidden: 24,
        conserved: vec![CH_RHO, CH_E],
        k_steps: 4,
        pool: 2,
        dt: 0.1,
        t_init: 1.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: true,
        readout_modes: 6,
        use_diffusion: true,
    }
}

fn toks(vocab: usize) -> Vec<u32> {
    (0..SEQ * B).map(|i| (i as u32 * 37 + 11) % vocab as u32).collect()
}

/// Min-over-rounds async ms/step for a given config (full train_step).
fn ablate(ctx: &GpuContext, cfg: &ThermoConfig, label: &str) -> f64 {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let model = ThermoLM::new(cfg.clone(), &mut rng);
    let adam = AdamWConfig::default();
    let mut gpu = ThermoGpu::new(ctx, &model, SEQ, B);
    let t = toks(cfg.vocab);
    for _ in 0..10 {
        gpu.train_step(&t, &t, &adam, false);
    }
    gpu.drain();
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..N {
            gpu.train_step(&t, &t, &adam, false);
        }
        gpu.drain();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));
    }
    println!("  {label:<22} {best:8.2} ms/step  ({:.0} win/s)", B as f64 / (best / 1e3));
    best
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
    let cfg = base_cfg();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let model = ThermoLM::new(cfg.clone(), &mut rng);
    let adam = AdamWConfig::default();
    println!(
        "cfg: grid 32×32×8 K4 pool2 readout both modes6 seq{SEQ} B{B} · params {}",
        model.param_count()
    );

    // ── (A) phase split, timed directly via drains ──
    let mut gpu = ThermoGpu::new(&ctx, &model, SEQ, B);
    let t = toks(cfg.vocab);
    for _ in 0..10 {
        gpu.train_step(&t, &t, &adam, false);
    }
    gpu.drain();

    let (mut zero, mut fwd, mut bwd, mut adamw, mut full) =
        (f64::MAX, f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..ROUNDS {
        gpu.drain();
        let t0 = Instant::now();
        for _ in 0..N { gpu.zero_grad(); }
        gpu.drain();
        zero = zero.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));

        gpu.drain();
        let t0 = Instant::now();
        for _ in 0..N { gpu.forward(&t); }
        gpu.drain();
        fwd = fwd.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));

        gpu.forward(&t); // caches for backward
        gpu.drain();
        let t0 = Instant::now();
        for _ in 0..N { gpu.backward_dispatch(&t, &t); }
        gpu.drain();
        bwd = bwd.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));

        gpu.drain();
        let t0 = Instant::now();
        for _ in 0..N { gpu.step_adamw(&adam); }
        gpu.drain();
        adamw = adamw.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));

        gpu.drain();
        let t0 = Instant::now();
        for _ in 0..N { gpu.train_step(&t, &t, &adam, false); }
        gpu.drain();
        full = full.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));
    }
    let sum = zero + fwd + bwd + adamw;
    println!("\n── (A) phase split (min of {ROUNDS}×{N}) ──");
    println!("  zero_grad : {zero:7.2} ms  ({:4.1}%)", 100.0 * zero / full);
    println!("  forward   : {fwd:7.2} ms  ({:4.1}%)", 100.0 * fwd / full);
    println!("  backward  : {bwd:7.2} ms  ({:4.1}%)", 100.0 * bwd / full);
    println!("  adamw     : {adamw:7.2} ms  ({:4.1}%)", 100.0 * adamw / full);
    println!("  ─ sum     : {sum:7.2} ms   vs full {full:7.2} ms");

    // ── (B) config ablation (min of {ROUNDS}×{N} async) ──
    println!("\n── (B) ablation (Δ vs full = subsystem cost) ──");
    let f_full = ablate(&ctx, &cfg, "full");
    let mut c = cfg.clone();
    c.use_diffusion = false;
    let f_nodiff = ablate(&ctx, &c, "no-diffusion");
    let mut c = cfg.clone();
    c.readout_spectral = false;
    let f_mom = ablate(&ctx, &c, "moments-only");
    let mut c = cfg.clone();
    c.k_steps = 2;
    let f_k2 = ablate(&ctx, &c, "K=2 (half substeps)");
    println!("\n  diffusion path   ≈ {:7.2} ms ({:4.1}%)", f_full - f_nodiff, 100.0 * (f_full - f_nodiff) / f_full);
    println!("  spectral readout ≈ {:7.2} ms ({:4.1}%)", f_full - f_mom, 100.0 * (f_full - f_mom) / f_full);
    println!("  2 substeps' work ≈ {:7.2} ms ({:4.1}%)  → per-substep ≈ {:.2} ms", f_full - f_k2, 100.0 * (f_full - f_k2) / f_full, (f_full - f_k2) / 2.0);
}
