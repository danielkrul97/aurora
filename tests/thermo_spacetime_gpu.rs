//! CPU↔GPU parity for the spacetime LM (`docs/13` §23). Skips cleanly when no
//! GPU is present. Forward logits + backward param-grads + loss must match the
//! CPU `StLM` within f32 drift tolerance.

#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

use aurora::gpu::{GpuContext, StLmGpu};
use aurora::thermo::spacetime_model::{StConfig, StGrads, StLM};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skip (no GPU): {e}");
            None
        }
    }
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut w = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs().max(y.abs()).max(1e-6);
        w = w.max((x - y).abs() / d);
    }
    w
}

#[test]
fn st_forward_logits_match_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x57F0);
    for &(c, ch, ks, taps, dilb) in &[(8usize, 12usize, 3usize, 2usize, 2usize), (16, 24, 4, 2, 2)] {
        let cfg = StConfig { vocab: 16, c, conv_hidden: ch, k_steps: ks, taps, dilation_base: dilb, dt: 0.8 };
        let l = 16usize;
        let model = StLM::new(cfg.clone(), &mut rng);
        let tokens: Vec<u32> = (0..l).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
        let logits_cpu = model.forward(&tokens).logits;
        let gpu = StLmGpu::new(&ctx, &model, l, 1);
        gpu.forward(&tokens);
        let logits_gpu = gpu.download_logits();
        let err = max_rel_err(&logits_cpu, &logits_gpu);
        assert!(err < 5e-3, "fwd logits c{c} ch{ch} K{ks}: rel err {err}");
    }
}

#[test]
fn st_backward_param_grads_match_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x57F1);
    let cfg = StConfig { vocab: 16, c: 12, conv_hidden: 20, k_steps: 4, taps: 2, dilation_base: 2, dt: 0.7 };
    let l = 24usize;
    let model = StLM::new(cfg.clone(), &mut rng);
    let tokens: Vec<u32> = (0..l).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..l).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    // CPU
    let cache = model.forward(&tokens);
    let mut g_cpu = StGrads::zeros(&model);
    let loss_cpu = model.backward(&cache, &tokens, &targets, &mut g_cpu);

    // GPU (batch 1)
    let mut gpu = StLmGpu::new(&ctx, &model, l, 1);
    gpu.zero_grad();
    gpu.forward(&tokens);
    gpu.backward_dispatch(&tokens, &targets);
    let loss_gpu = gpu.read_mean_loss();
    let g_gpu = gpu.download_grads();

    assert!((loss_cpu - loss_gpu).abs() < 1e-2, "loss {loss_cpu} vs {loss_gpu}");
    let chk = |name: &str, cpu: &[f32], gpu: &[f32]| {
        let e = max_rel_err(cpu, gpu);
        assert!(e < 2e-2, "{name} rel err {e}");
    };
    chk("d_w1", &g_cpu.conv.d_w1, &g_gpu.conv.d_w1);
    chk("d_b1", &g_cpu.conv.d_b1, &g_gpu.conv.d_b1);
    chk("d_w2", &g_cpu.conv.d_w2, &g_gpu.conv.d_w2);
    chk("d_b2", &g_cpu.conv.d_b2, &g_gpu.conv.d_b2);
    chk("d_embed", &g_cpu.d_embed, &g_gpu.d_embed);
    chk("d_ro_w", &g_cpu.d_ro_w, &g_gpu.d_ro_w);
    chk("d_ro_bias", &g_cpu.d_ro_bias, &g_gpu.d_ro_bias);
}
