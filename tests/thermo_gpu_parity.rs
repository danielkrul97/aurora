//! GPU↔CPU parity tests for thermodynamic-LM kernels (T2).
//!
//! Each test runs the CPU primitive and its GPU twin on a tiny problem
//! and asserts agreement within tolerance. Skips silently with no GPU.

#![cfg(feature = "gpu")]
#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use aurora::gpu::buffers::GpuBuffer;
use aurora::gpu::{
    GeluBackwardKernel, GemmKernel, GemmTransposeBKernel, GpuContext, OuterProductAccumKernel,
    RowReduceAddKernel, ThermoAvgPoolBackwardKernel, ThermoAvgPoolKernel, VectorAddKernel,
    ThermoConv1x1Kernel, ThermoConv3x3Dw1Kernel, ThermoConv3x3DsKernel, ThermoConv3x3GeluKernel,
    ThermoDiffusionBwdDsKernel, ThermoDiffusionBwdParamKernel, ThermoDiffusionBwdPreKernel,
    ThermoDiffusionForwardKernel, ThermoFluxDivBackwardKernel, ThermoFluxDivKernel,
    ThermoDisassembleKernel, ThermoFluxScatterKernel, ThermoGpu, ThermoGpuForward,
    ThermoInjectBackwardKernel, ThermoInjectKernel, ThermoScaleKernel,
};
use aurora::thermo::conv::{conv_backward, conv_forward, ConvGrads, ConvWeights};
use aurora::thermo::diffusion::{diffusion_backward, diffusion_forward, DiffusionGrads, DiffusionWeights};
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::flux::{flux_div_backward, flux_div_forward};
use aurora::thermo::inject::{inject_backward, inject_forward, InjectGrads, InjectWeights};
use aurora::thermo::model::{ThermoGrads, ThermoLM};
use aurora::thermo::readout::{readout_backward, readout_forward, ReadoutGrads, ReadoutWeights};
use aurora::thermo::step::{step_backward, step_forward, StepGrads, StepWeights};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => {
            eprintln!("GPU adapter: {}", c.adapter_info().name);
            Some(c)
        }
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            None
        }
    }
}

fn random_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1e-6);
        worst = worst.max((x - y).abs() / denom);
    }
    worst
}

fn test_cfg(h: usize, w: usize, c: usize, conv_hidden: usize) -> ThermoConfig {
    let mut cfg = ThermoConfig::toy(16);
    cfg.h = h;
    cfg.w = w;
    cfg.c = c;
    cfg.conv_hidden = conv_hidden;
    cfg
}

#[test]
fn conv_forward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let k3 = ThermoConv3x3GeluKernel::new(&ctx);
    let k1 = ThermoConv1x1Kernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x60FF);

    for &(h, w, c, ch) in &[(4, 4, 6, 6), (8, 8, 8, 16), (16, 16, 8, 24)] {
        let cfg = test_cfg(h, w, c, ch);
        let cells = cfg.cells();
        let cout = cfg.c_out();
        let mut weights = ConvWeights::new(&cfg, &mut rng);
        // Randomize biases too (new() zeros them) to exercise the bias path.
        weights.b1 = random_vec(ch, &mut rng);
        weights.b2 = random_vec(cout, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);

        let (raw_cpu, _) = conv_forward(&weights, &s, &cfg);

        let s_gpu = GpuBuffer::from_slice(&ctx, "s", &s);
        let w1_gpu = GpuBuffer::from_slice(&ctx, "w1", &weights.w1);
        let b1_gpu = GpuBuffer::from_slice(&ctx, "b1", &weights.b1);
        let w2_gpu = GpuBuffer::from_slice(&ctx, "w2", &weights.w2);
        let b2_gpu = GpuBuffer::from_slice(&ctx, "b2", &weights.b2);
        let hp_gpu = GpuBuffer::zeros(&ctx, "hp", cells * ch);
        let raw_gpu = GpuBuffer::zeros(&ctx, "raw", cells * cout);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("conv fwd") });
        k3.dispatch_into(&ctx, &mut enc, &s_gpu, &w1_gpu, &b1_gpu, &hp_gpu,
            h as u32, w as u32, c as u32, ch as u32);
        k1.dispatch_into(&ctx, &mut enc, &hp_gpu, &w2_gpu, &b2_gpu, &raw_gpu,
            cells as u32, ch as u32, cout as u32);
        ctx.queue.submit(Some(enc.finish()));
        let raw_gpu_data = raw_gpu.download(&ctx);

        let err = max_rel_err(&raw_cpu, &raw_gpu_data);
        assert!(err < 5e-3, "conv fwd {h}x{w}x{c} ch{ch}: max_rel_err {err}");
    }
}

#[test]
fn conv_backward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let gemm = GemmKernel::new(&ctx);
    let outer = OuterProductAccumKernel::new(&ctx);
    let row = RowReduceAddKernel::new(&ctx);
    let gelu_bwd = GeluBackwardKernel::new(&ctx);
    let k_ds = ThermoConv3x3DsKernel::new(&ctx);
    let k_dw1 = ThermoConv3x3Dw1Kernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xBACC);

    for &(h, w, c, ch) in &[(4, 4, 6, 6), (8, 8, 8, 16), (16, 16, 8, 24)] {
        let cfg = test_cfg(h, w, c, ch);
        let cells = cfg.cells();
        let cout = cfg.c_out();
        let mut weights = ConvWeights::new(&cfg, &mut rng);
        weights.b1 = random_vec(ch, &mut rng);
        weights.b2 = random_vec(cout, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let d_raw = random_vec(cells * cout, &mut rng);

        // CPU reference.
        let (_raw, cache) = conv_forward(&weights, &s, &cfg);
        let mut g_cpu = ConvGrads::zeros(&cfg);
        let d_s_cpu = conv_backward(&weights, &cache, &d_raw, &mut g_cpu, &cfg);

        // GPU: upload forward cache + weights + upstream.
        let s_g = GpuBuffer::from_slice(&ctx, "s", &s);
        let hpre_g = GpuBuffer::from_slice(&ctx, "hpre", &cache.h_pre);
        let hpost_g = GpuBuffer::from_slice(&ctx, "hpost", &cache.h_post);
        let w1_g = GpuBuffer::from_slice(&ctx, "w1", &weights.w1);
        let w2_g = GpuBuffer::from_slice(&ctx, "w2", &weights.w2);
        let draw_g = GpuBuffer::from_slice(&ctx, "draw", &d_raw);
        // outputs (zeroed; the accumulating reuse-kernels need a zero start).
        let dhpost_g = GpuBuffer::zeros(&ctx, "dhpost", cells * ch);
        let dhpre_g = GpuBuffer::zeros(&ctx, "dhpre", cells * ch);
        let ds_g = GpuBuffer::zeros(&ctx, "ds", cells * c);
        let dw1_g = GpuBuffer::zeros(&ctx, "dw1", ch * c * 9);
        let db1_g = GpuBuffer::zeros(&ctx, "db1", ch);
        let dw2_g = GpuBuffer::zeros(&ctx, "dw2", cout * ch);
        let db2_g = GpuBuffer::zeros(&ctx, "db2", cout);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("conv bwd") });
        // conv1x1 backward.
        gemm.dispatch_into(&ctx, &mut enc, &draw_g, &w2_g, &dhpost_g, cells as u32, ch as u32, cout as u32);
        // OuterProductAccum: dest[out,in] = Σ_t d_y[t,out]·x[t,in], so
        // x carries the input dim (h_post, ch) and d_y the output dim (d_raw, cout).
        outer.dispatch_into(&ctx, &mut enc, &hpost_g, &draw_g, &dw2_g, cout as u32, ch as u32, cells as u32);
        row.dispatch_into(&ctx, &mut enc, &draw_g, &db2_g, cells as u32, cout as u32);
        // GELU backward.
        gelu_bwd.dispatch_into(&ctx, &mut enc, &hpre_g, &dhpost_g, &dhpre_g, (cells * ch) as u32);
        // conv3x3 backward.
        k_ds.dispatch_into(&ctx, &mut enc, &dhpre_g, &w1_g, &ds_g, h as u32, w as u32, c as u32, ch as u32);
        k_dw1.dispatch_into(&ctx, &mut enc, &dhpre_g, &s_g, &dw1_g, h as u32, w as u32, c as u32, ch as u32);
        row.dispatch_into(&ctx, &mut enc, &dhpre_g, &db1_g, cells as u32, ch as u32);
        ctx.queue.submit(Some(enc.finish()));

        let chk = |name: &str, cpu: &[f32], gpu: Vec<f32>| {
            let err = max_rel_err(cpu, &gpu);
            assert!(err < 5e-3, "conv bwd {h}x{w}x{c} ch{ch} {name}: max_rel_err {err}");
        };
        chk("d_s", &d_s_cpu, ds_g.download(&ctx));
        chk("d_w1", &g_cpu.d_w1, dw1_g.download(&ctx));
        chk("d_b1", &g_cpu.d_b1, db1_g.download(&ctx));
        chk("d_w2", &g_cpu.d_w2, dw2_g.download(&ctx));
        chk("d_b2", &g_cpu.d_b2, db2_g.download(&ctx));
    }
}

#[test]
fn flux_div_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kf = ThermoFluxDivKernel::new(&ctx);
    let kb = ThermoFluxDivBackwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xF111);

    for &(h, w) in &[(4, 4), (8, 8), (16, 16)] {
        let cfg = test_cfg(h, w, 8, 16);
        let nc = cfg.n_conserved();
        let n = cfg.cells() * nc;
        let fx = random_vec(n, &mut rng);
        let fy = random_vec(n, &mut rng);
        let d_div = random_vec(n, &mut rng);

        // forward
        let div_cpu = flux_div_forward(&fx, &fy, &cfg);
        let fx_g = GpuBuffer::from_slice(&ctx, "fx", &fx);
        let fy_g = GpuBuffer::from_slice(&ctx, "fy", &fy);
        let div_g = GpuBuffer::zeros(&ctx, "div", n);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("flux fwd") });
        kf.dispatch_into(&ctx, &mut enc, &fx_g, &fy_g, &div_g, h as u32, w as u32, nc as u32);
        ctx.queue.submit(Some(enc.finish()));
        assert!(max_rel_err(&div_cpu, &div_g.download(&ctx)) < 5e-3, "flux fwd {h}x{w}");

        // backward
        let (dfx_cpu, dfy_cpu) = flux_div_backward(&d_div, &cfg);
        let dd_g = GpuBuffer::from_slice(&ctx, "dd", &d_div);
        let dfx_g = GpuBuffer::zeros(&ctx, "dfx", n);
        let dfy_g = GpuBuffer::zeros(&ctx, "dfy", n);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("flux bwd") });
        kb.dispatch_into(&ctx, &mut enc, &dd_g, &dfx_g, &dfy_g, h as u32, w as u32, nc as u32);
        ctx.queue.submit(Some(enc.finish()));
        assert!(max_rel_err(&dfx_cpu, &dfx_g.download(&ctx)) < 5e-3, "flux bwd d_fx {h}x{w}");
        assert!(max_rel_err(&dfy_cpu, &dfy_g.download(&ctx)) < 5e-3, "flux bwd d_fy {h}x{w}");
    }
}

#[test]
fn diffusion_forward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kd = ThermoDiffusionForwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xD1FF);

    for &(h, w, c) in &[(4, 4, 6), (8, 8, 8), (16, 16, 8)] {
        let cfg = test_cfg(h, w, c, 16);
        let mut weights = DiffusionWeights::new(&cfg);
        weights.k0 = random_vec(c, &mut rng).iter().map(|v| 0.2 + 0.1 * v).collect();
        weights.a = random_vec(c, &mut rng);
        weights.b = random_vec(c, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);

        let (delta_cpu, cache) = diffusion_forward(&s, &weights, &cfg);

        let s_g = GpuBuffer::from_slice(&ctx, "s", &s);
        let k0_g = GpuBuffer::from_slice(&ctx, "k0", &weights.k0);
        let a_g = GpuBuffer::from_slice(&ctx, "a", &weights.a);
        let b_g = GpuBuffer::from_slice(&ctx, "b", &weights.b);
        let delta_g = GpuBuffer::zeros(&ctx, "delta", cfg.field_len());
        let lap_g = GpuBuffer::zeros(&ctx, "lap", cfg.field_len());
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diff fwd") });
        kd.dispatch_into(&ctx, &mut enc, &s_g, &k0_g, &a_g, &b_g, &delta_g, &lap_g, h as u32, w as u32, c as u32);
        ctx.queue.submit(Some(enc.finish()));

        assert!(max_rel_err(&delta_cpu, &delta_g.download(&ctx)) < 5e-3, "diffusion delta {h}x{w}x{c}");
        assert!(max_rel_err(&cache.lap, &lap_g.download(&ctx)) < 5e-3, "diffusion lap {h}x{w}x{c}");
    }
}

#[test]
fn diffusion_backward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pre = ThermoDiffusionBwdPreKernel::new(&ctx);
    let param = ThermoDiffusionBwdParamKernel::new(&ctx);
    let ds = ThermoDiffusionBwdDsKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xD1BB);

    for &(h, w, c) in &[(4, 4, 6), (8, 8, 8), (16, 16, 8)] {
        let cfg = test_cfg(h, w, c, 16);
        let cells = cfg.cells();
        let mut weights = DiffusionWeights::new(&cfg);
        weights.k0 = random_vec(c, &mut rng).iter().map(|v| 0.2 + 0.1 * v).collect();
        weights.a = random_vec(c, &mut rng);
        weights.b = random_vec(c, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let d_delta = random_vec(cells * c, &mut rng);

        // CPU reference.
        let (_delta, cache) = diffusion_forward(&s, &weights, &cfg);
        let mut g_cpu = DiffusionGrads::zeros(&cfg);
        let d_s_cpu = diffusion_backward(&weights, &cache, &d_delta, &mut g_cpu, &cfg);

        // GPU: packed gate [k0|a|b].
        let mut gate = Vec::with_capacity(3 * c);
        gate.extend_from_slice(&weights.k0);
        gate.extend_from_slice(&weights.a);
        gate.extend_from_slice(&weights.b);

        let dd_g = GpuBuffer::from_slice(&ctx, "dd", &d_delta);
        let lap_g = GpuBuffer::from_slice(&ctx, "lap", &cache.lap);
        let s_g = GpuBuffer::from_slice(&ctx, "s", &s);
        let gate_g = GpuBuffer::from_slice(&ctx, "gate", &gate);
        let gbuf_g = GpuBuffer::zeros(&ctx, "gbuf", cells * c);
        let ct_g = GpuBuffer::zeros(&ctx, "cT", cells * c);
        let ds_g = GpuBuffer::zeros(&ctx, "ds", cells * c);
        let dk0_g = GpuBuffer::zeros(&ctx, "dk0", c);
        let da_g = GpuBuffer::zeros(&ctx, "da", c);
        let db_g = GpuBuffer::zeros(&ctx, "db", c);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diff bwd") });
        pre.dispatch_into(&ctx, &mut enc, &dd_g, &lap_g, &s_g, &gate_g, &gbuf_g, &ct_g, h as u32, w as u32, c as u32);
        param.dispatch_into(&ctx, &mut enc, &dd_g, &lap_g, &s_g, &gate_g, &dk0_g, &da_g, &db_g, h as u32, w as u32, c as u32);
        ds.dispatch_into(&ctx, &mut enc, &gbuf_g, &ct_g, &ds_g, h as u32, w as u32, c as u32);
        ctx.queue.submit(Some(enc.finish()));

        let chk = |name: &str, cpu: &[f32], gpu: Vec<f32>| {
            let err = max_rel_err(cpu, &gpu);
            assert!(err < 5e-3, "diff bwd {h}x{w}x{c} {name}: max_rel_err {err}");
        };
        chk("d_s", &d_s_cpu, ds_g.download(&ctx));
        chk("d_k0", &g_cpu.d_k0, dk0_g.download(&ctx));
        chk("d_a", &g_cpu.d_a, da_g.download(&ctx));
        chk("d_b", &g_cpu.d_b, db_g.download(&ctx));
    }
}

#[test]
fn inject_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kf = ThermoInjectKernel::new(&ctx);
    let kb = ThermoInjectBackwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x123E);

    for &(h, w, c) in &[(4, 4, 6), (8, 8, 8), (16, 16, 8)] {
        let cfg = test_cfg(h, w, c, 16);
        let cells = cfg.cells();
        let weights = InjectWeights::new(&cfg, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let d_out = random_vec(cfg.field_len(), &mut rng);
        let token = 3u32;
        let anneal = 0.7_f32;

        // CPU.
        let (out_cpu, cache) = inject_forward(&s, token, anneal, &weights, &cfg);
        let mut g_cpu = InjectGrads::zeros(&cfg);
        let _ = inject_backward(&cache, &d_out, &weights, &mut g_cpu, &cfg);

        // GPU forward.
        let s_g = GpuBuffer::from_slice(&ctx, "s", &s);
        let prof_g = GpuBuffer::from_slice(&ctx, "prof", &weights.profile);
        let inj_g = GpuBuffer::from_slice(&ctx, "inj", &weights.inj);
        let out_g = GpuBuffer::zeros(&ctx, "out", cfg.field_len());
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inj fwd") });
        kf.dispatch_into(&ctx, &mut enc, &s_g, &prof_g, &inj_g, &out_g, cells as u32, c as u32, token, anneal);
        ctx.queue.submit(Some(enc.finish()));
        assert!(max_rel_err(&out_cpu, &out_g.download(&ctx)) < 5e-3, "inject fwd {h}x{w}x{c}");

        // GPU backward (d_inj).
        let dout_g = GpuBuffer::from_slice(&ctx, "dout", &d_out);
        let dinj_g = GpuBuffer::zeros(&ctx, "dinj", cfg.vocab * c);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inj bwd") });
        kb.dispatch_into(&ctx, &mut enc, &prof_g, &dout_g, &dinj_g, cells as u32, c as u32, token, anneal);
        ctx.queue.submit(Some(enc.finish()));
        assert!(max_rel_err(&g_cpu.d_inj, &dinj_g.download(&ctx)) < 5e-3, "inject bwd d_inj {h}x{w}x{c}");
    }
}

#[test]
fn readout_moments_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = ThermoAvgPoolKernel::new(&ctx);
    let pool_bwd = ThermoAvgPoolBackwardKernel::new(&ctx);
    let gtb = GemmTransposeBKernel::new(&ctx);
    let gemm = GemmKernel::new(&ctx);
    let outer = OuterProductAccumKernel::new(&ctx);
    let row = RowReduceAddKernel::new(&ctx);
    let vadd = VectorAddKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x4EAD);

    for &(h, w, c) in &[(4, 4, 6), (8, 8, 8), (16, 16, 8)] {
        let mut cfg = test_cfg(h, w, c, 16);
        cfg.readout_moments = true;
        cfg.readout_spectral = false; // moments-only path
        let p = cfg.pool;
        let pdim = cfg.pooled_dim();
        let vocab = cfg.vocab;
        let cells = cfg.cells();
        let weights = ReadoutWeights::new(&cfg, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let d_logits = random_vec(vocab, &mut rng);

        // CPU.
        let (logits_cpu, cache) = readout_forward(&s, &weights, &cfg);
        let mut g_cpu = ReadoutGrads::zeros(&cfg);
        let d_s_cpu = readout_backward(&weights, &cache, &d_logits, &mut g_cpu, &cfg);

        // GPU buffers.
        let s_g = GpuBuffer::from_slice(&ctx, "s", &s);
        let projw_g = GpuBuffer::from_slice(&ctx, "projw", &weights.proj.w);
        let bias_g = GpuBuffer::from_slice(&ctx, "bias", &weights.bias);
        let pooled_g = GpuBuffer::zeros(&ctx, "pooled", pdim);
        let lraw_g = GpuBuffer::zeros(&ctx, "lraw", vocab);
        let logits_g = GpuBuffer::zeros(&ctx, "logits", vocab);

        // forward: avgpool → gemm(pooled @ projᵀ) → +bias.
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ro fwd") });
        pool.dispatch_into(&ctx, &mut enc, &s_g, &pooled_g, h as u32, w as u32, c as u32, p as u32);
        gtb.dispatch_into(&ctx, &mut enc, &pooled_g, &projw_g, &lraw_g, 1, vocab as u32, pdim as u32);
        vadd.dispatch_into(&ctx, &mut enc, &lraw_g, &bias_g, &logits_g, vocab as u32);
        ctx.queue.submit(Some(enc.finish()));
        assert!(max_rel_err(&logits_cpu, &logits_g.download(&ctx)) < 5e-3, "readout fwd {h}x{w}x{c}");

        // backward.
        let dl_g = GpuBuffer::from_slice(&ctx, "dl", &d_logits);
        let dbias_g = GpuBuffer::zeros(&ctx, "dbias", vocab);
        let dproj_g = GpuBuffer::zeros(&ctx, "dproj", vocab * pdim);
        let dpooled_g = GpuBuffer::zeros(&ctx, "dpooled", pdim);
        let ds_g = GpuBuffer::zeros(&ctx, "ds", cells * c);
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ro bwd") });
        row.dispatch_into(&ctx, &mut enc, &dl_g, &dbias_g, 1, vocab as u32);
        outer.dispatch_into(&ctx, &mut enc, &pooled_g, &dl_g, &dproj_g, vocab as u32, pdim as u32, 1);
        gemm.dispatch_into(&ctx, &mut enc, &dl_g, &projw_g, &dpooled_g, 1, pdim as u32, vocab as u32);
        pool_bwd.dispatch_into(&ctx, &mut enc, &dpooled_g, &ds_g, h as u32, w as u32, c as u32, p as u32);
        ctx.queue.submit(Some(enc.finish()));

        let chk = |name: &str, cpu: &[f32], gpu: Vec<f32>| {
            let err = max_rel_err(cpu, &gpu);
            assert!(err < 5e-3, "readout bwd {h}x{w}x{c} {name}: max_rel_err {err}");
        };
        chk("d_bias", &g_cpu.d_bias, dbias_g.download(&ctx));
        chk("d_proj", &g_cpu.d_proj.d_w, dproj_g.download(&ctx));
        chk("d_s", &d_s_cpu, ds_g.download(&ctx));
    }
}

#[test]
fn rollout_forward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x5011);
    let cfg = ThermoConfig {
        vocab: 16,
        h: 8,
        w: 8,
        c: 6,
        conv_hidden: 8,
        conserved: vec![CH_RHO, CH_E],
        k_steps: 2,
        pool: 2,
        dt: 0.1,
        t_init: 1.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: false, // GPU forward = moments-only
        readout_modes: 4,
        use_diffusion: true,
    };
    let model = ThermoLM::new(cfg.clone(), &mut rng);
    let tokens: Vec<u32> = (0..5).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    let logits_cpu = model.forward(&tokens).logits;
    let gpu = ThermoGpuForward::new(&ctx, &model);
    let logits_gpu = gpu.forward(&tokens);

    // Full T×K rollout accumulates f32 drift; the project bar is < 5e-2.
    let err = max_rel_err(&logits_cpu, &logits_gpu);
    assert!(err < 2e-2, "rollout forward parity: max_rel_err {err}");
}

#[test]
fn step_backward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    // kernels
    let scale = ThermoScaleKernel::new(&ctx);
    let pre = aurora::gpu::ThermoDiffusionBwdPreKernel::new(&ctx);
    let param = aurora::gpu::ThermoDiffusionBwdParamKernel::new(&ctx);
    let dds = aurora::gpu::ThermoDiffusionBwdDsKernel::new(&ctx);
    let disasm = ThermoDisassembleKernel::new(&ctx);
    let fluxb = ThermoFluxDivBackwardKernel::new(&ctx);
    let scatter = ThermoFluxScatterKernel::new(&ctx);
    let gemm = GemmKernel::new(&ctx);
    let outer = OuterProductAccumKernel::new(&ctx);
    let row = RowReduceAddKernel::new(&ctx);
    let gelu_bwd = GeluBackwardKernel::new(&ctx);
    let k_ds = ThermoConv3x3DsKernel::new(&ctx);
    let k_dw1 = ThermoConv3x3Dw1Kernel::new(&ctx);
    let vadd = VectorAddKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x57EB);

    for &(h, w, c, ch) in &[(4, 4, 6, 6), (8, 8, 8, 16)] {
        let cfg = test_cfg(h, w, c, ch); // conserved [rho,e], use_diffusion true
        let cells = cfg.cells();
        let cout = cfg.c_out();
        let nc = cfg.n_conserved();
        let nfree = c - nc;
        let anneal = 0.6_f32;
        let dt = cfg.dt;

        let weights = StepWeights::new(&cfg, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let d_out = random_vec(cfg.field_len(), &mut rng);

        // CPU.
        let (_out, cache) = step_forward(&s, &weights, anneal, &cfg);
        let mut g_cpu = StepGrads::zeros(&cfg);
        let d_s_cpu = step_backward(&weights, &cache, &d_out, &mut g_cpu, &cfg);

        // gate packing for diffusion backward.
        let mut gate = Vec::with_capacity(3 * c);
        gate.extend_from_slice(&weights.diffusion.k0);
        gate.extend_from_slice(&weights.diffusion.a);
        gate.extend_from_slice(&weights.diffusion.b);

        // channel maps
        let free: Vec<usize> = (0..c).filter(|x| !cfg.conserved.contains(x)).collect();
        let mut kind = vec![0u32; c];
        let mut slot = vec![0u32; c];
        for (m, &cc) in cfg.conserved.iter().enumerate() { kind[cc] = 1; slot[cc] = m as u32; }
        for (k, &cc) in free.iter().enumerate() { slot[cc] = k as u32; }

        // uploads
        let s_g = GpuBuffer::from_slice(&ctx, "s", &cache.conv.s);
        let hpre_g = GpuBuffer::from_slice(&ctx, "hpre", &cache.conv.h_pre);
        let hpost_g = GpuBuffer::from_slice(&ctx, "hpost", &cache.conv.h_post);
        let lap_g = GpuBuffer::from_slice(&ctx, "lap", &cache.diff.lap);
        let w1_g = GpuBuffer::from_slice(&ctx, "w1", &weights.conv.w1);
        let w2_g = GpuBuffer::from_slice(&ctx, "w2", &weights.conv.w2);
        let gate_g = GpuBuffer::from_slice(&ctx, "gate", &gate);
        let dout_g = GpuBuffer::from_slice(&ctx, "dout", &d_out);
        let kind_g = GpuBuffer::from_u32_slice(&ctx, "kind", &kind);
        let slot_g = GpuBuffer::from_u32_slice(&ctx, "slot", &slot);
        // scratch / outputs (zeroed)
        let ddelta_g = GpuBuffer::zeros(&ctx, "ddelta", cells * c);
        let gbuf_g = GpuBuffer::zeros(&ctx, "gbuf", cells * c);
        let ct_g = GpuBuffer::zeros(&ctx, "ct", cells * c);
        let dsdiff_g = GpuBuffer::zeros(&ctx, "dsdiff", cells * c);
        let ddiv_g = GpuBuffer::zeros(&ctx, "ddiv", cells * nc);
        let draw_g = GpuBuffer::zeros(&ctx, "draw", cells * cout);
        let dfx_g = GpuBuffer::zeros(&ctx, "dfx", cells * nc);
        let dfy_g = GpuBuffer::zeros(&ctx, "dfy", cells * nc);
        let dhpost_g = GpuBuffer::zeros(&ctx, "dhpost", cells * ch);
        let dhpre_g = GpuBuffer::zeros(&ctx, "dhpre", cells * ch);
        let dsconv_g = GpuBuffer::zeros(&ctx, "dsconv", cells * c);
        let tmp_g = GpuBuffer::zeros(&ctx, "tmp", cells * c);
        let ds_g = GpuBuffer::zeros(&ctx, "ds", cells * c);
        let dw1_g = GpuBuffer::zeros(&ctx, "dw1", ch * c * 9);
        let db1_g = GpuBuffer::zeros(&ctx, "db1", ch);
        let dw2_g = GpuBuffer::zeros(&ctx, "dw2", cout * ch);
        let db2_g = GpuBuffer::zeros(&ctx, "db2", cout);
        let dk0_g = GpuBuffer::zeros(&ctx, "dk0", c);
        let da_g = GpuBuffer::zeros(&ctx, "da", c);
        let db_g = GpuBuffer::zeros(&ctx, "dbb", c);

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("step bwd") });
        let (hu, wu, cu, ncu, coutu, chu) = (h as u32, w as u32, c as u32, nc as u32, cout as u32, ch as u32);

        // diffusion path: d_diff_delta = dt*anneal*d_out → PRE → PARAM → DS
        scale.dispatch_into(&ctx, &mut enc, &dout_g, &ddelta_g, (cells * c) as u32, dt * anneal);
        pre.dispatch_into(&ctx, &mut enc, &ddelta_g, &lap_g, &s_g, &gate_g, &gbuf_g, &ct_g, hu, wu, cu);
        param.dispatch_into(&ctx, &mut enc, &ddelta_g, &lap_g, &s_g, &gate_g, &dk0_g, &da_g, &db_g, hu, wu, cu);
        dds.dispatch_into(&ctx, &mut enc, &gbuf_g, &ct_g, &dsdiff_g, hu, wu, cu);

        // transport path
        disasm.dispatch_into(&ctx, &mut enc, &dout_g, &kind_g, &slot_g, &ddiv_g, &draw_g, hu, wu, cu, ncu, coutu, dt);
        fluxb.dispatch_into(&ctx, &mut enc, &ddiv_g, &dfx_g, &dfy_g, hu, wu, ncu);
        scatter.dispatch_into(&ctx, &mut enc, &dfx_g, &dfy_g, &draw_g, hu, wu, ncu, nfree as u32, coutu);
        // conv backward (d_raw, cache) → dsconv + conv grads
        gemm.dispatch_into(&ctx, &mut enc, &draw_g, &w2_g, &dhpost_g, cells as u32, chu, coutu);
        outer.dispatch_into(&ctx, &mut enc, &hpost_g, &draw_g, &dw2_g, coutu, chu, cells as u32);
        row.dispatch_into(&ctx, &mut enc, &draw_g, &db2_g, cells as u32, coutu);
        gelu_bwd.dispatch_into(&ctx, &mut enc, &hpre_g, &dhpost_g, &dhpre_g, (cells * ch) as u32);
        k_ds.dispatch_into(&ctx, &mut enc, &dhpre_g, &w1_g, &dsconv_g, hu, wu, cu, chu);
        k_dw1.dispatch_into(&ctx, &mut enc, &dhpre_g, &s_g, &dw1_g, hu, wu, cu, chu);
        row.dispatch_into(&ctx, &mut enc, &dhpre_g, &db1_g, cells as u32, chu);

        // d_s = d_out + d_s_diff + d_s_conv
        vadd.dispatch_into(&ctx, &mut enc, &dout_g, &dsdiff_g, &tmp_g, (cells * c) as u32);
        vadd.dispatch_into(&ctx, &mut enc, &tmp_g, &dsconv_g, &ds_g, (cells * c) as u32);
        ctx.queue.submit(Some(enc.finish()));

        let chk = |name: &str, cpu: &[f32], gpu: Vec<f32>| {
            let err = max_rel_err(cpu, &gpu);
            assert!(err < 5e-3, "step bwd {h}x{w}x{c} {name}: max_rel_err {err}");
        };
        chk("d_s", &d_s_cpu, ds_g.download(&ctx));
        chk("d_w1", &g_cpu.conv.d_w1, dw1_g.download(&ctx));
        chk("d_b1", &g_cpu.conv.d_b1, db1_g.download(&ctx));
        chk("d_w2", &g_cpu.conv.d_w2, dw2_g.download(&ctx));
        chk("d_b2", &g_cpu.conv.d_b2, db2_g.download(&ctx));
        chk("d_k0", &g_cpu.diffusion.d_k0, dk0_g.download(&ctx));
        chk("d_a", &g_cpu.diffusion.d_a, da_g.download(&ctx));
        chk("d_b", &g_cpu.diffusion.d_b, db_g.download(&ctx));
    }
}

#[test]
fn rollout_backward_param_grads_match_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x6ACC);
    let cfg = ThermoConfig {
        vocab: 16,
        h: 8,
        w: 8,
        c: 6,
        conv_hidden: 8,
        conserved: vec![CH_RHO, CH_E],
        k_steps: 2,
        pool: 2,
        dt: 0.1,
        t_init: 1.0,
        anneal_tmax: 1.0,
        anneal_alpha: 1.0,
        readout_moments: true,
        readout_spectral: false,
        readout_modes: 4,
        use_diffusion: true,
    };
    let model = ThermoLM::new(cfg.clone(), &mut rng);
    let seq_len = 4;
    let tokens: Vec<u32> = (0..seq_len).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..seq_len).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    // CPU forward+backward.
    let cache = model.forward(&tokens);
    let mut g_cpu = ThermoGrads::zeros(&cfg);
    let loss_cpu = model.backward(&cache, &targets, &mut g_cpu);

    // GPU forward+backward.
    let gpu = ThermoGpu::new(&ctx, &model, seq_len);
    gpu.zero_grad();
    gpu.forward(&tokens);
    let loss_gpu = gpu.backward(&tokens, &targets);
    let g_gpu = gpu.download_grads();

    assert!((loss_cpu - loss_gpu).abs() < 1e-2, "loss {loss_cpu} vs {loss_gpu}");
    // Full T×K BPTT accumulates f32 drift; project bar is < 5e-2.
    let chk = |name: &str, cpu: &[f32], gpu: &[f32]| {
        let e = max_rel_err(cpu, gpu);
        assert!(e < 2e-2, "param-grad {name}: max_rel_err {e}");
    };
    chk("d_w1", &g_cpu.step.conv.d_w1, &g_gpu.step.conv.d_w1);
    chk("d_b1", &g_cpu.step.conv.d_b1, &g_gpu.step.conv.d_b1);
    chk("d_w2", &g_cpu.step.conv.d_w2, &g_gpu.step.conv.d_w2);
    chk("d_b2", &g_cpu.step.conv.d_b2, &g_gpu.step.conv.d_b2);
    chk("d_k0", &g_cpu.step.diffusion.d_k0, &g_gpu.step.diffusion.d_k0);
    chk("d_a", &g_cpu.step.diffusion.d_a, &g_gpu.step.diffusion.d_a);
    chk("d_b", &g_cpu.step.diffusion.d_b, &g_gpu.step.diffusion.d_b);
    chk("d_inj", &g_cpu.inject.d_inj, &g_gpu.inject.d_inj);
    chk("d_proj", &g_cpu.readout.d_proj.d_w, &g_gpu.readout.d_proj.d_w);
    chk("d_bias", &g_cpu.readout.d_bias, &g_gpu.readout.d_bias);
}
