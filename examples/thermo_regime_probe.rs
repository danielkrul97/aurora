#![allow(clippy::cast_precision_loss)]

//! Throwaway regime probe: per-step wall time vs (grid, batch) at the user's
//! actual thermo config (readout both, modes 6, pool 2, K 4, seq 16).
//! Flat ms/step in B ⇒ dispatch/overhead-bound; ~linear in B ⇒ compute-bound.
//! Compare grid 16 vs 32 at B=24 to size the fixed-overhead D vs compute C.

use std::time::Instant;

use aurora::gpu::{GpuContext, ThermoGpu};
use aurora::optimizer::AdamWConfig;
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::model::ThermoLM;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cfg(grid: usize) -> ThermoConfig {
    ThermoConfig {
        vocab: 256,
        d: 1,
        h: grid,
        w: grid,
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
    let seq = 16;
    let adam = AdamWConfig::default();

    for &grid in &[16usize, 32] {
        let c = cfg(grid);
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let model = ThermoLM::new(c.clone(), &mut rng);
        println!(
            "\n── grid {0}×{0}×{1} convH {2} K {3} pool {4} readout both modes {5} · params {6} ──",
            c.h, c.c, c.conv_hidden, c.k_steps, c.pool, c.readout_modes, model.param_count()
        );
        println!("   B   | ms/step | ms/step/window");
        for &b in &[1usize, 4, 12, 24] {
            let mut gpu = ThermoGpu::new(&ctx, &model, seq, b);
            let t = toks(seq, b, c.vocab);
            let tg = toks(seq, b, c.vocab);
            for _ in 0..5 {
                gpu.train_step(&t, &tg, &adam, false);
            }
            let _ = gpu.read_mean_loss(); // drain warmup → barrier
            let n = 40u32;
            let t0 = Instant::now();
            for _ in 0..n {
                gpu.train_step(&t, &tg, &adam, false);
            }
            let _ = gpu.read_mean_loss(); // drain → barrier
            let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);
            println!("  {b:>4} | {ms:>7.2} | {:>14.3}", ms / b as f64);
        }
    }
}
