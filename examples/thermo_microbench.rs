#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

//! Throwaway per-kernel micro-bench @ grid 32, B=24. Times each thermo kernel
//! in isolation (GPU-only: record N dispatches OUTSIDE the timer, then
//! submit+drain INSIDE), min-of-rounds to cancel GPU-boost-clock drift.
//! Per-substep kernels run K·T=64×/step, so ×64 = their per-step contribution.

use std::time::Instant;

use aurora::gpu::buffers::GpuBuffer;
use aurora::gpu::{
    GpuContext, OuterProductAccumKernel, RowReduceAddKernel, StepTransportBwdKernel,
    StepTransportFwdKernel, ThermoConv1x1Kernel, ThermoConv3x3DsKernel, ThermoConv3x3Dw1Kernel,
    ThermoConv3x3GeluCacheKernel, ThermoConvBwdHiddenKernel, ThermoDiffusionBwdDsKernel,
    ThermoDiffusionBwdParamKernel, ThermoDiffusionBwdPreKernel, ThermoDiffusionForwardKernel,
    ThermoScaleKernel, VectorAdd3Kernel,
};
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};

const ROUNDS: usize = 6;
const N: u32 = 60;
const OCC: f64 = 64.0; // per-substep kernels run K·T = 4·16 times per step

fn bench(ctx: &GpuContext, mut record: impl FnMut(&mut wgpu::CommandEncoder)) -> f64 {
    {
        let mut e = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..N { record(&mut e); }
        ctx.queue.submit(Some(e.finish()));
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
    }
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let mut e = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..N { record(&mut e); }
        let cb = e.finish();
        let t0 = Instant::now();
        ctx.queue.submit(Some(cb));
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3 / f64::from(N));
    }
    best
}

use aurora::gpu::{OuterProductSplitKKernel, ThermoConv3x3Dw1SplitKKernel, ThermoIm2colDw1Kernel};

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    println!("GPU: {}", ctx.adapter_info().name);

    let cfg = ThermoConfig {
        vocab: 256, d: 1, h: 32, w: 32, c: 8, conv_hidden: 24, conserved: vec![CH_RHO, CH_E],
        k_steps: 4, pool: 2, dt: 0.1, t_init: 1.0, anneal_tmax: 1.0, anneal_alpha: 1.0,
        readout_moments: true, readout_spectral: true, readout_modes: 6, use_diffusion: true,
    };
    let (d, h, w, c, ch, b) = (4u32, 16u32, 16u32, 8u32, 24u32, 24u32);
    let cells = (d * h * w) as usize;
    let bcells = b as usize * cells;
    let coutu = cfg.c_out();
    let cout = coutu as u32;
    let nc = cfg.n_conserved() as u32;
    let bcc = bcells * c as usize;
    let bcch = bcells * ch as usize;
    let bcco = bcells * coutu;
    println!("grid {d}×{h}×{w} B{b}: cout={cout} nc={nc}  (per-substep ×{OCC:.0}/step)\n");

    let z = |n: usize| GpuBuffer::zeros(&ctx, "b", n);
    let (s, d_out, ddelta, lap, gbuf, ct, ds_diff, dsconv, d_s_out, diff_delta) =
        (z(bcc), z(bcc), z(bcc), z(bcc), z(bcc), z(bcc), z(bcc), z(bcc), z(bcc), z(bcc));
    let (hpre, hpost, dhpre) = (z(bcch), z(bcch), z(bcch));
    let (raw, draw) = (z(bcco), z(bcco));
    let (w1, w1grad) = (z(ch as usize * c as usize * 27), z(ch as usize * c as usize * 27));
    let (w2, w2grad) = (z(coutu * ch as usize), z(coutu * ch as usize));
    let (b1, b2, b1grad, b2grad) = (z(ch as usize), z(coutu), z(ch as usize), z(coutu));
    let (k0, pa, pb, gate, k0g, pag, pbg) =
        (z(c as usize), z(c as usize), z(c as usize), z(3 * c as usize), z(c as usize), z(c as usize), z(c as usize));
    let cons_ch = GpuBuffer::from_u32_slice(&ctx, "cons", &[0u32, 1]);
    let free_ch = GpuBuffer::from_u32_slice(&ctx, "free", &[2u32, 3, 4, 5, 6, 7]);
    let kind = GpuBuffer::from_u32_slice(&ctx, "kind", &[1u32, 1, 0, 0, 0, 0, 0, 0]);
    let slot = GpuBuffer::from_u32_slice(&ctx, "slot", &[0u32, 1, 0, 0, 0, 0, 0, 0]);

    let conv3g = ThermoConv3x3GeluCacheKernel::new(&ctx);
    let conv1 = ThermoConv1x1Kernel::new(&ctx);
    let diff_fwd = ThermoDiffusionForwardKernel::new(&ctx);
    let transport_fwd = StepTransportFwdKernel::new(&ctx);
    let transport_bwd = StepTransportBwdKernel::new(&ctx);
    let outer = OuterProductAccumKernel::new(&ctx);
    let splitk = OuterProductSplitKKernel::new(&ctx);
    let w2parts = z(OuterProductSplitKKernel::GROUPS as usize * coutu * ch as usize);
    let row = RowReduceAddKernel::new(&ctx);
    let conv_bwd_hidden = ThermoConvBwdHiddenKernel::new(&ctx);
    let conv_ds = ThermoConv3x3DsKernel::new(&ctx);
    let conv_dw1 = ThermoConv3x3Dw1Kernel::new(&ctx);
    let dw1_sk = ThermoConv3x3Dw1SplitKKernel::new(&ctx);
    let im2col = ThermoIm2colDw1Kernel::new(&ctx);
    let patch = z(bcells * c as usize * 27);
    let w1parts_op = z(OuterProductSplitKKernel::GROUPS as usize * ch as usize * c as usize * 27);
    let w1parts = z(b as usize * ch as usize * c as usize * 27);
    let vadd3 = VectorAdd3Kernel::new(&ctx);
    let scale = ThermoScaleKernel::new(&ctx);
    let diff_pre = ThermoDiffusionBwdPreKernel::new(&ctx);
    let diff_param = ThermoDiffusionBwdParamKernel::new(&ctx);
    let diff_ds = ThermoDiffusionBwdDsKernel::new(&ctx);

    let n_bcc = bcc as u32;
    let mut rows: Vec<(&str, &str, f64)> = Vec::new();
    let mut add = |phase: &'static str, name: &'static str, ms: f64| rows.push((phase, name, ms));

    // ── forward kernels ──
    add("fwd", "conv3g", bench(&ctx, |e| conv3g.dispatch_b(&ctx, e, &s, &w1, &b1, &hpre, &hpost, h, w, c, ch, b, d)));
    add("fwd", "conv1", bench(&ctx, |e| conv1.dispatch_into(&ctx, e, &hpost, &w2, &b2, &raw, bcells as u32, ch, cout)));
    add("fwd", "diff_fwd", bench(&ctx, |e| diff_fwd.dispatch_b(&ctx, e, &s, &k0, &pa, &pb, &diff_delta, &lap, h, w, c, b, d)));
    add("fwd", "transport_fwd", bench(&ctx, |e| transport_fwd.dispatch_b(&ctx, e, &s, &raw, &diff_delta, &kind, &slot, &d_s_out, h, w, c, nc, cout, b, 0.1, 1.0, d)));

    // ── backward: conv + transport ──
    add("bwd", "transport_bwd", bench(&ctx, |e| transport_bwd.dispatch_b(&ctx, e, &d_out, &cons_ch, &free_ch, &draw, h, w, c, nc, cout, b, 0.1, d)));
    add("bwd", "outer(w2grad)", bench(&ctx, |e| outer.dispatch_into(&ctx, e, &hpost, &draw, &w2grad, cout, ch, bcells as u32)));
    add("bwd", "d_w2 splitK+row", bench(&ctx, |e| {
        splitk.dispatch_into(&ctx, e, &hpost, &draw, &w2parts, cout, ch, bcells as u32);
        row.dispatch_into(&ctx, e, &w2parts, &w2grad, OuterProductSplitKKernel::groups_for(bcells as u32), cout * ch);
    }));
    add("bwd", "row(b2grad)", bench(&ctx, |e| row.dispatch_into(&ctx, e, &draw, &b2grad, bcells as u32, cout)));
    add("bwd", "conv_bwd_hidden", bench(&ctx, |e| conv_bwd_hidden.dispatch_into(&ctx, e, &draw, &w2, &hpre, &dhpre, bcells as u32, ch, cout)));
    add("bwd", "conv_ds", bench(&ctx, |e| conv_ds.dispatch_b(&ctx, e, &dhpre, &w1, &dsconv, h, w, c, ch, b, d)));
    add("bwd", "conv_dw1", bench(&ctx, |e| conv_dw1.dispatch_b(&ctx, e, &dhpre, &s, &w1grad, h, w, c, ch, b, d)));
    add("bwd", "dw1 splitK+row", bench(&ctx, |e| {
        dw1_sk.dispatch_b(&ctx, e, &dhpre, &s, &w1parts, h, w, c, ch, b, d);
        row.dispatch_into(&ctx, e, &w1parts, &w1grad, b, ch * c * 27);
    }));
    add("bwd", "dw1 im2col+gemm+row", bench(&ctx, |e| {
        im2col.dispatch_b(&ctx, e, &s, &patch, h, w, c, b, d);
        splitk.dispatch_into(&ctx, e, &patch, &dhpre, &w1parts_op, ch, c * 27, bcells as u32);
        row.dispatch_into(&ctx, e, &w1parts_op, &w1grad, OuterProductSplitKKernel::groups_for(bcells as u32), ch * c * 27);
    }));
    add("bwd", "row(b1grad)", bench(&ctx, |e| row.dispatch_into(&ctx, e, &dhpre, &b1grad, bcells as u32, ch)));
    add("bwd", "vadd3", bench(&ctx, |e| vadd3.dispatch_into(&ctx, e, &d_out, &ds_diff, &dsconv, &d_s_out, n_bcc)));

    // ── backward: diffusion ──
    add("bwd", "scale", bench(&ctx, |e| scale.dispatch_into(&ctx, e, &d_out, &ddelta, n_bcc, 0.1)));
    add("bwd", "diff_pre", bench(&ctx, |e| diff_pre.dispatch_b(&ctx, e, &ddelta, &lap, &s, &gate, &gbuf, &ct, h, w, c, b, d)));
    add("bwd", "diff_param", bench(&ctx, |e| diff_param.dispatch_b(&ctx, e, &ddelta, &lap, &s, &gate, &k0g, &pag, &pbg, h, w, c, b, d)));
    add("bwd", "diff_ds", bench(&ctx, |e| diff_ds.dispatch_b(&ctx, e, &gbuf, &ct, &ds_diff, h, w, c, b, d)));

    rows.sort_by(|a, b| (b.2 * OCC).partial_cmp(&(a.2 * OCC)).unwrap());
    println!("  kernel              phase   µs/disp   ×64 ms/step");
    println!("  ────────────────────────────────────────────────");
    let mut total = 0.0;
    for (phase, name, ms) in &rows {
        let contrib = ms * OCC;
        total += contrib;
        println!("  {name:<18} {phase:>5}  {:>8.1}   {contrib:>7.1}", ms * 1e3);
    }
    println!("  ────────────────────────────────────────────────");
    println!("  Σ (×64 substep kernels) = {total:.0} ms/step   [full step ≈ 870 ms; rest = readout+inject+opt]");
}
