#![allow(clippy::cast_precision_loss)]

//! Isolated eval-cost A/B: CPU `evaluate` (rayon over windows, no cache) vs
//! on-device `ThermoGpu::eval_nats` (batched forward + softmax-CE), at the
//! standard T0 config. Settles whether H1 (GPU-resident eval) actually beats
//! the rayon CPU eval, independent of the training loop.

use std::time::Instant;

use aurora::gpu::{GpuContext, ThermoGpu};
use aurora::optimizer::AdamWConfig;
use aurora::text_io::bits_per_byte;
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::model::ThermoLM;
use aurora::transformer::loss::softmax;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

fn cfg() -> ThermoConfig {
    ThermoConfig {
        vocab: 256, h: 16, w: 16, c: 8, conv_hidden: 24,
        conserved: vec![CH_RHO, CH_E], k_steps: 4, pool: 1, dt: 0.1,
        t_init: 1.0, anneal_tmax: 1.0, anneal_alpha: 1.0,
        readout_moments: true, readout_spectral: false, readout_modes: 4,
        use_diffusion: true,
    }
}

fn cpu_eval(model: &ThermoLM, val: &[u32], seq: usize, comp: f32, max_windows: usize) -> (f32, f64) {
    let v = model.config.vocab;
    let all: Vec<usize> = (0..).map(|i| i * seq).take_while(|&p| p + seq + 1 < val.len()).collect();
    let windows: Vec<usize> = if max_windows == 0 || max_windows >= all.len() {
        all
    } else {
        (0..max_windows).map(|i| all[i * all.len() / max_windows]).collect()
    };
    let t0 = Instant::now();
    let (nats, toks): (f32, usize) = windows
        .par_iter()
        .map(|&p| {
            let logits = model.forward(&val[p..p + seq]).logits;
            let mut n = 0.0_f32;
            for ti in 0..seq {
                let probs = softmax(&logits[ti * v..(ti + 1) * v]);
                n += -probs[val[p + ti + 1] as usize].max(1e-12).ln();
            }
            (n, seq)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let bpb = bits_per_byte(nats, (toks as f32 * comp) as usize);
    (bpb, t0.elapsed().as_secs_f64() * 1e3)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("no GPU: {e}"); return; }
    };
    println!("GPU: {}", ctx.adapter_info().name);
    let c = cfg();
    let seq = 16;
    let b = 24;
    let comp = 1.0_f32;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let model = ThermoLM::new(c.clone(), &mut rng);
    // Synthetic val stream (eval cost is data-independent).
    let val: Vec<u32> = (0..20_000).map(|i| (i as u32 * 53 + 7) % c.vocab as u32).collect();

    let mut gpu = ThermoGpu::new(&ctx, &model, seq, b);
    let adam = AdamWConfig::default();
    gpu.train_step(&vec![0u32; seq * b], &vec![0u32; seq * b], &adam, false);
    gpu.drain();

    println!("\n  windows |  CPU eval (rayon)  |  GPU eval_nats  | speedup | bpb match");
    for &mw in &[256usize, 1024, 0] {
        let (bpb_cpu, ms_cpu) = cpu_eval(&model, &val, seq, comp, mw);
        // warm + timed GPU
        let _ = gpu.eval_nats(&val, mw);
        gpu.drain();
        let t0 = Instant::now();
        let (nats_gpu, ntok) = gpu.eval_nats(&val, mw);
        let ms_gpu = t0.elapsed().as_secs_f64() * 1e3;
        let bpb_gpu = bits_per_byte(nats_gpu as f32, (ntok as f32 * comp) as usize);
        let label = if mw == 0 { "full".to_string() } else { mw.to_string() };
        println!(
            "  {label:>7} | {ms_cpu:>10.1} ms     | {ms_gpu:>9.1} ms   | {:>6.2}x | {bpb_cpu:.4}/{bpb_gpu:.4}",
            ms_cpu / ms_gpu
        );
    }
}
