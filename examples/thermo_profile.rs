#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

//! Throwaway profiling harness for the thermo-LM CPU hot path.
//!
//! perf_event_paranoid is locked down on this box, so instead of sampling we
//! instrument: a counting global allocator + per-op timers driven through the
//! same public API the trainer uses. Mirrors the default CLI config of
//! `examples/thermo_train.rs` (grid 16, c 8, conv_hidden 24, K 4, pool 1,
//! seq 16, batch 24, readout moments, diffusion on).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use aurora::thermo::field::{init_field, ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::{
    anneal_at, conv_backward, conv_forward, diffusion_backward, diffusion_forward,
    flux_div_backward, flux_div_forward, inject_backward, inject_forward, readout_backward,
    readout_forward, ConvGrads, DiffusionGrads, InjectGrads, ReadoutGrads, StepGrads, ThermoGrads,
    ThermoLM,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Counting;
static N_ALLOC: AtomicU64 = AtomicU64::new(0);
static B_ALLOC: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        N_ALLOC.fetch_add(1, Relaxed);
        B_ALLOC.fetch_add(l.size() as u64, Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
    }
}

#[global_allocator]
static A: Counting = Counting;

fn snap() -> (u64, u64) {
    (N_ALLOC.load(Relaxed), B_ALLOC.load(Relaxed))
}

/// Time `f` over `iters` calls and report ns/call + allocs/call + bytes/call.
fn bench(name: &str, iters: u64, mut f: impl FnMut()) {
    for _ in 0..3 {
        f();
    } // warm
    let (n0, b0) = snap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let el = t0.elapsed();
    let (n1, b1) = snap();
    let per = el.as_nanos() as f64 / iters as f64;
    let na = (n1 - n0) as f64 / iters as f64;
    let ba = (b1 - b0) as f64 / iters as f64;
    println!(
        "  {name:<22} {per:>10.0} ns   {na:>6.1} allocs   {ba:>9.0} B",
        name = name,
        per = per,
        na = na,
        ba = ba
    );
}

fn main() {
    let cfg = ThermoConfig {
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
    };
    let (seq, batch) = (16usize, 24usize);
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let model = ThermoLM::new(cfg.clone(), &mut rng);

    println!("══════════════════════════════════════════════════════════════════");
    println!(" thermo profile · grid {0}×{0}×{1} convH {2} K {3} pool {4} seq {5} batch {6}",
        cfg.h, cfg.c, cfg.conv_hidden, cfg.k_steps, cfg.pool, seq, batch);
    println!(" params {} · readout_dim {} · cells {} · c_out {}",
        model.param_count(), cfg.readout_dim(), cfg.cells(), cfg.c_out());
    println!("══════════════════════════════════════════════════════════════════");

    let tokens: Vec<u32> = (0..seq as u32).map(|i| (i * 37 + 11) % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    // ── A. End-to-end forward + backward over one window ──────────────────
    println!("\n── end-to-end per window (T={seq}, single-thread) ──");
    let mut grads = ThermoGrads::zeros(&cfg);
    let mut fwd_cache = model.forward(&tokens);
    bench("forward", 200, || {
        fwd_cache = model.forward(&tokens);
    });
    let cache = model.forward(&tokens);
    bench("backward", 200, || {
        grads.zero();
        let _ = model.backward(&cache, &targets, &mut grads);
    });
    let _ = &fwd_cache;

    let (fn0, fb0) = snap();
    let c = model.forward(&tokens);
    let (fn1, fb1) = snap();
    grads.zero();
    let (bn0, bb0) = snap();
    let _ = model.backward(&c, &targets, &mut grads);
    let (bn1, bb1) = snap();
    println!(
        "  forward  allocs/window = {}, bytes/window = {}",
        fn1 - fn0,
        fb1 - fb0
    );
    println!(
        "  backward allocs/window = {}, bytes/window = {}",
        bn1 - bn0,
        bb1 - bb0
    );
    let per_window = (fn1 - fn0) + (bn1 - bn0);
    println!(
        "  → per train step (×batch {batch}) ≈ {} allocs, ×1500 steps ≈ {:.1}M allocs",
        per_window * batch as u64,
        (per_window * batch as u64 * 1500) as f64 / 1e6
    );

    // ── B. Leaf-op micro-bench on a representative evolved field ──────────
    println!("\n── leaf ops (per call, single-thread) ──");
    // Evolve a few steps so the field is realistic (not the uniform init).
    let mut field = init_field(&cfg);
    for _ in 0..seq.min(4) {
        let (f, _) = inject_forward(&field, 7, 1.0, &model.inject, &cfg);
        field = f;
        for kk in 0..cfg.k_steps {
            let (f, _) = aurora::thermo::step_forward(&field, &model.step, anneal_at(kk, &cfg), &cfg);
            field = f;
        }
    }

    let (raw, conv_cache) = conv_forward(&model.step.conv, &field, &cfg);
    let (diff_delta, diff_cache) = diffusion_forward(&field, &model.step.diffusion, &cfg);
    let nc = cfg.n_conserved();
    let cells = cfg.cells();
    let fx = vec![0.05_f32; cells * nc];
    let fy = vec![0.03_f32; cells * nc];
    let div = flux_div_forward(&fx, &fy, &cfg);
    let (_logits, ro_cache) = readout_forward(&field, &model.readout, &cfg);

    let mut cg = ConvGrads::zeros(&cfg);
    let mut dg = DiffusionGrads::zeros(&cfg);
    let mut ig = InjectGrads::zeros(&cfg);
    let mut rg = ReadoutGrads::zeros(&cfg);
    let mut sg = StepGrads::zeros(&cfg);

    let d_raw = vec![0.01_f32; raw.len()];
    let d_field = vec![0.01_f32; cfg.field_len()];
    let d_delta = vec![0.01_f32; diff_delta.len()];
    let d_div = vec![0.01_f32; div.len()];
    let d_logits = vec![0.01_f32; cfg.vocab];
    let inj_cache = inject_forward(&field, 7, 1.0, &model.inject, &cfg).1;

    bench("inject_forward", 5000, || {
        let _ = inject_forward(&field, 7, 1.0, &model.inject, &cfg);
    });
    bench("inject_backward", 5000, || {
        ig.zero();
        let _ = inject_backward(&inj_cache, &d_field, &model.inject, &mut ig, &cfg);
    });
    bench("conv_forward", 2000, || {
        let _ = conv_forward(&model.step.conv, &field, &cfg);
    });
    bench("conv_backward", 2000, || {
        cg.zero();
        let _ = conv_backward(&model.step.conv, &conv_cache, &d_raw, &mut cg, &cfg);
    });
    bench("diffusion_forward", 5000, || {
        let _ = diffusion_forward(&field, &model.step.diffusion, &cfg);
    });
    bench("diffusion_backward", 5000, || {
        dg.zero();
        let _ = diffusion_backward(&model.step.diffusion, &diff_cache, &d_delta, &mut dg, &cfg);
    });
    bench("flux_div_forward", 5000, || {
        let _ = flux_div_forward(&fx, &fy, &cfg);
    });
    bench("flux_div_backward", 5000, || {
        let _ = flux_div_backward(&d_div, &cfg);
    });
    let anneal = anneal_at(1, &cfg);
    bench("step_forward", 2000, || {
        let _ = aurora::thermo::step_forward(&field, &model.step, anneal, &cfg);
    });
    let step_cache = aurora::thermo::step_forward(&field, &model.step, anneal, &cfg).1;
    bench("step_backward", 2000, || {
        sg.zero();
        let _ = aurora::thermo::step_backward(&model.step, &step_cache, &d_field, &mut sg, &cfg);
    });
    bench("readout_forward", 5000, || {
        let _ = readout_forward(&field, &model.readout, &cfg);
    });
    bench("readout_backward", 5000, || {
        rg.zero();
        let _ = readout_backward(&model.readout, &ro_cache, &d_logits, &mut rg, &cfg);
    });

    // gelu/tanh share of conv: c_h map = h*w*conv_hidden elements.
    let hp = vec![0.3_f32; cfg.cells() * cfg.conv_hidden];
    bench("gelu_map(h*w*c_h)", 5000, || {
        let _: Vec<f32> = hp.iter().map(|&v| aurora::linalg::gelu(v)).collect();
    });
    bench("gelu_bwd_map", 5000, || {
        let _: Vec<f32> = hp.iter().map(|&v| aurora::linalg::gelu_backward(v)).collect();
    });

    // ── C. Rollout cost attribution (per call × call count per window) ────
    println!("\n── rollout call counts per window ──");
    println!("  inject  f/b : {seq:>3} each");
    println!("  step    f/b : {:>3} each  (T·K)", seq * cfg.k_steps);
    println!("  readout f/b : {seq:>3} each");
    println!("\n(step = conv + flux + diffusion + assemble; compare step vs its parts above)");
}
