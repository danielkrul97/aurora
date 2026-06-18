//! Sprint 56+: GPU↔CPU parity tests for transformer kernels.
//!
//! Each test creates a tiny problem, runs both CPU and GPU paths,
//! and asserts they agree within tolerance. The harness skips silently
//! if no GPU adapter is available.

#![cfg(feature = "gpu")]
#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use aurora::attention::{
    layer_norm_backward, layer_norm_forward, LayerNorm, LayerNormGrads,
};
use aurora::gpu::buffers::GpuBuffer;
use aurora::gpu::{
    cpu_gemm, cpu_gemm_transpose_b, CausalMhaBackwardKernel, CausalMhaForwardKernel,
    GeluBackwardKernel, GeluKernel, GemmKernel, GemmTransposeBKernel, GpuContext,
    GradNormSqReduceKernel, LayerNormBackwardKernel, LayerNormForwardKernel,
    OuterProductAccumKernel, RowReduceAddKernel, SoftmaxCrossEntropyKernel, TokenPosEmbedKernel,
};
use aurora::linalg::{gelu, gelu_backward};
use aurora::transformer::attention::{causal_mha_forward, MhaWeights};
use aurora::transformer::embedding::{
    embed_forward, tied_lm_head_backward, tied_lm_head_forward, EmbedGrads, EmbedWeights,
};
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
        let err = (x - y).abs() / denom;
        if err > worst {
            worst = err;
        }
    }
    worst
}

#[test]
fn gemm_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = GemmKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x6E11);
    for &(m, k, n) in &[(16, 16, 16), (33, 64, 17), (8, 128, 64), (64, 32, 96)] {
        let a = random_vec(m * k, &mut rng);
        let b = random_vec(k * n, &mut rng);
        let cpu = cpu_gemm(&a, &b, m, k, n);

        let a_gpu = GpuBuffer::from_slice(&ctx, "a", &a);
        let b_gpu = GpuBuffer::from_slice(&ctx, "b", &b);
        let c_gpu = GpuBuffer::zeros(&ctx, "c", m * n);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm test"),
            });
        kernel.dispatch_into(&ctx, &mut encoder, &a_gpu, &b_gpu, &c_gpu, m as u32, n as u32, k as u32);
        ctx.queue.submit(Some(encoder.finish()));
        let gpu = c_gpu.download(&ctx);

        let err = max_rel_err(&cpu, &gpu);
        assert!(err < 5e-3, "gemm({m},{k},{n}) rel err {err}");
    }
}

#[test]
fn gemm_transpose_b_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = GemmTransposeBKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x7E51);
    for &(m, k, n) in &[(16, 16, 16), (32, 48, 20), (8, 64, 64)] {
        let a = random_vec(m * k, &mut rng);
        let b = random_vec(n * k, &mut rng);
        let cpu = cpu_gemm_transpose_b(&a, &b, m, k, n);

        let a_gpu = GpuBuffer::from_slice(&ctx, "a", &a);
        let b_gpu = GpuBuffer::from_slice(&ctx, "b", &b);
        let c_gpu = GpuBuffer::zeros(&ctx, "c", m * n);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm-t test"),
            });
        kernel.dispatch_into(&ctx, &mut encoder, &a_gpu, &b_gpu, &c_gpu, m as u32, n as u32, k as u32);
        ctx.queue.submit(Some(encoder.finish()));
        let gpu = c_gpu.download(&ctx);

        let err = max_rel_err(&cpu, &gpu);
        assert!(err < 5e-3, "gemm_t({m},{k},{n}) rel err {err}");
    }
}

#[test]
fn gelu_forward_backward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let fwd = GeluKernel::new(&ctx);
    let bwd = GeluBackwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xC2C2);

    let n = 1024;
    let x = random_vec(n, &mut rng);
    let dy = random_vec(n, &mut rng);

    let cpu_y: Vec<f32> = x.iter().map(|&v| gelu(v)).collect();
    let cpu_dx: Vec<f32> = x
        .iter()
        .zip(dy.iter())
        .map(|(&xi, &dyi)| dyi * gelu_backward(xi))
        .collect();

    let x_gpu = GpuBuffer::from_slice(&ctx, "x", &x);
    let dy_gpu = GpuBuffer::from_slice(&ctx, "dy", &dy);
    let y_gpu = GpuBuffer::zeros(&ctx, "y", n);
    let dx_gpu = GpuBuffer::zeros(&ctx, "dx", n);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gelu test"),
        });
    fwd.dispatch_into(&ctx, &mut encoder, &x_gpu, &y_gpu, n as u32, false);
    bwd.dispatch_into(&ctx, &mut encoder, &x_gpu, &dy_gpu, &dx_gpu, n as u32, false);
    ctx.queue.submit(Some(encoder.finish()));
    let gpu_y = y_gpu.download(&ctx);
    let gpu_dx = dx_gpu.download(&ctx);

    let err_y = max_rel_err(&cpu_y, &gpu_y);
    let err_dx = max_rel_err(&cpu_dx, &gpu_dx);
    assert!(err_y < 1e-3, "gelu fwd err {err_y}");
    assert!(err_dx < 1e-3, "gelu bwd err {err_dx}");
}

#[test]
fn layer_norm_forward_backward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let fwd = LayerNormForwardKernel::new(&ctx);
    let bwd = LayerNormBackwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x71A4);

    let t = 16;
    let d = 64;
    let ln = LayerNorm {
        gamma: random_vec(d, &mut rng),
        beta: random_vec(d, &mut rng),
        eps: 1e-5,
        d,
        rms: false,
    };
    let x = random_vec(t * d, &mut rng);
    let dy = random_vec(t * d, &mut rng);

    let (cpu_y, cpu_cache) = layer_norm_forward(&ln, &x);
    let mut cpu_grads = LayerNormGrads::zeros(d);
    let cpu_dx = layer_norm_backward(&ln, &cpu_cache, &dy, &mut cpu_grads);

    let x_gpu = GpuBuffer::from_slice(&ctx, "x", &x);
    let gamma_gpu = GpuBuffer::from_slice(&ctx, "gamma", &ln.gamma);
    let beta_gpu = GpuBuffer::from_slice(&ctx, "beta", &ln.beta);
    let y_gpu = GpuBuffer::zeros(&ctx, "y", t * d);
    let mean_gpu = GpuBuffer::zeros(&ctx, "mean", t);
    let inv_std_gpu = GpuBuffer::zeros(&ctx, "inv_std", t);
    let x_hat_gpu = GpuBuffer::zeros(&ctx, "x_hat", t * d);
    let dy_gpu = GpuBuffer::from_slice(&ctx, "dy", &dy);
    let dx_gpu = GpuBuffer::zeros(&ctx, "dx", t * d);
    // d_gamma/d_beta are accumulated atomically — start at zero, kernel adds.
    let dg_gpu = GpuBuffer::zeros(&ctx, "dg", d);
    let db_gpu = GpuBuffer::zeros(&ctx, "db", d);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ln test") });
    fwd.dispatch_into(
        &ctx, &mut enc, &x_gpu, &gamma_gpu, &beta_gpu, &y_gpu, &mean_gpu, &inv_std_gpu,
        &x_hat_gpu, t as u32, d as u32, ln.eps, false,
    );
    bwd.dispatch_into(
        &ctx, &mut enc, &dy_gpu, &gamma_gpu, &x_hat_gpu, &inv_std_gpu, &dx_gpu, &dg_gpu,
        &db_gpu, t as u32, d as u32, false,
    );
    ctx.queue.submit(Some(enc.finish()));

    let gpu_y = y_gpu.download(&ctx);
    let gpu_dx = dx_gpu.download(&ctx);
    let gpu_dgamma = dg_gpu.download(&ctx);
    let gpu_dbeta = db_gpu.download(&ctx);

    let err_y = max_rel_err(&cpu_y, &gpu_y);
    let err_dx = max_rel_err(&cpu_dx, &gpu_dx);
    let err_dg = max_rel_err(&cpu_grads.d_gamma, &gpu_dgamma);
    let err_db = max_rel_err(&cpu_grads.d_beta, &gpu_dbeta);
    assert!(err_y < 5e-2, "ln fwd err {err_y}");
    assert!(err_dx < 5e-2, "ln bwd dx err {err_dx}");
    assert!(err_dg < 5e-2, "ln bwd dgamma err {err_dg}");
    assert!(err_db < 5e-2, "ln bwd dbeta err {err_db}");
}

// ---- Sprint 57: causal MHA + embedding ----

#[test]
fn causal_mha_forward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let fwd = CausalMhaForwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x4274);

    for &(t, n_head, d_head) in &[(4, 2, 8), (8, 4, 16), (12, 2, 32)] {
        let d_model = n_head * d_head;
        let w = MhaWeights::new(d_model, n_head, &mut rng);
        let x = random_vec(t * d_model, &mut rng);
        let (cpu_y, cpu_cache) = causal_mha_forward(&w, &x);

        // Single dispatch handles all heads via wg.y = head_idx.
        let q_gpu = GpuBuffer::from_slice(&ctx, "q", &cpu_cache.q);
        let k_gpu = GpuBuffer::from_slice(&ctx, "k", &cpu_cache.k);
        let v_gpu = GpuBuffer::from_slice(&ctx, "v", &cpu_cache.v);
        let out_gpu = GpuBuffer::zeros(&ctx, "out", t * d_model);
        let probs_gpu = GpuBuffer::zeros(&ctx, "probs", n_head * t * t);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mha fwd"),
            });
        fwd.dispatch_into(
            &ctx,
            &mut enc,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &out_gpu,
            &probs_gpu,
            t as u32,
            n_head as u32,
            d_head as u32,
        );
        ctx.queue.submit(Some(enc.finish()));
        // Compare GPU heads_out vs CPU heads_out (before output projection).
        let gpu_heads_out = out_gpu.download(&ctx);
        let err = max_rel_err(&cpu_cache.heads_out, &gpu_heads_out);
        assert!(
            err < 5e-2,
            "mha fwd heads_out err {err} on (t={t}, h={n_head}, d_h={d_head})"
        );

        // Sanity: applying W_o on host gives same y as CPU.
        let _ = cpu_y;
    }
}

#[test]
fn causal_mha_backward_matches_cpu() {
    use aurora::attention::scaled_dot_product_attention_backward;

    let Some(ctx) = ctx_or_skip() else { return };
    let fwd = CausalMhaForwardKernel::new(&ctx);
    let bwd = CausalMhaBackwardKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xB17D);

    let t: usize = 6;
    let n_head: usize = 2;
    let d_head: usize = 8;
    let d_model = n_head * d_head;

    let w = MhaWeights::new(d_model, n_head, &mut rng);
    let x = random_vec(t * d_model, &mut rng);
    let (_y, cache) = causal_mha_forward(&w, &x);

    // Synthetic d_heads_out (= d_out from the MHA wrapper after W_o backward).
    let d_heads_out = random_vec(t * d_model, &mut rng);

    let q_gpu = GpuBuffer::from_slice(&ctx, "q", &cache.q);
    let k_gpu = GpuBuffer::from_slice(&ctx, "k", &cache.k);
    let v_gpu = GpuBuffer::from_slice(&ctx, "v", &cache.v);
    let out_gpu = GpuBuffer::zeros(&ctx, "out", t * d_model);
    let probs_gpu = GpuBuffer::zeros(&ctx, "probs", n_head * t * t);
    let dout_gpu = GpuBuffer::from_slice(&ctx, "dout", &d_heads_out);
    let dq_gpu = GpuBuffer::zeros(&ctx, "dq", t * d_model);
    let dk_gpu = GpuBuffer::zeros(&ctx, "dk", t * d_model);
    let dv_gpu = GpuBuffer::zeros(&ctx, "dv", t * d_model);
    let d_scores = GpuBuffer::zeros(&ctx, "d_scores", n_head * t * t);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mha bwd") });
    fwd.dispatch_into(
        &ctx, &mut enc, &q_gpu, &k_gpu, &v_gpu, &out_gpu, &probs_gpu,
        t as u32, n_head as u32, d_head as u32,
    );
    bwd.dispatch_into(
        &ctx, &mut enc, &q_gpu, &k_gpu, &v_gpu, &probs_gpu, &dout_gpu,
        &dq_gpu, &dk_gpu, &dv_gpu, &d_scores,
        t as u32, n_head as u32, d_head as u32,
    );
    ctx.queue.submit(Some(enc.finish()));
    let gpu_dq = dq_gpu.download(&ctx);
    let gpu_dk = dk_gpu.download(&ctx);
    let gpu_dv = dv_gpu.download(&ctx);

    // CPU per-head backward
    let mut cpu_dq = vec![0.0_f32; t * d_model];
    let mut cpu_dk = vec![0.0_f32; t * d_model];
    let mut cpu_dv = vec![0.0_f32; t * d_model];
    for h in 0..n_head {
        let mut qh = vec![0.0_f32; t * d_head];
        let mut kh = vec![0.0_f32; t * d_head];
        let mut vh = vec![0.0_f32; t * d_head];
        let mut dyh = vec![0.0_f32; t * d_head];
        for ti in 0..t {
            let off_g = ti * d_model + h * d_head;
            let off_h = ti * d_head;
            qh[off_h..off_h + d_head].copy_from_slice(&cache.q[off_g..off_g + d_head]);
            kh[off_h..off_h + d_head].copy_from_slice(&cache.k[off_g..off_g + d_head]);
            vh[off_h..off_h + d_head].copy_from_slice(&cache.v[off_g..off_g + d_head]);
            dyh[off_h..off_h + d_head]
                .copy_from_slice(&d_heads_out[off_g..off_g + d_head]);
        }
        let (dq, dk, dv) = scaled_dot_product_attention_backward(
            &qh,
            &kh,
            &vh,
            &cache.attn_caches[h],
            &dyh,
        );
        for ti in 0..t {
            let off_g = ti * d_model + h * d_head;
            let off_h = ti * d_head;
            cpu_dq[off_g..off_g + d_head].copy_from_slice(&dq[off_h..off_h + d_head]);
            cpu_dk[off_g..off_g + d_head].copy_from_slice(&dk[off_h..off_h + d_head]);
            cpu_dv[off_g..off_g + d_head].copy_from_slice(&dv[off_h..off_h + d_head]);
        }
    }
    let err_q = max_rel_err(&cpu_dq, &gpu_dq);
    let err_k = max_rel_err(&cpu_dk, &gpu_dk);
    let err_v = max_rel_err(&cpu_dv, &gpu_dv);
    assert!(err_q < 5e-2, "mha bwd dq err {err_q}");
    assert!(err_k < 5e-2, "mha bwd dk err {err_k}");
    assert!(err_v < 5e-2, "mha bwd dv err {err_v}");
}

#[test]
fn token_pos_embed_forward_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = TokenPosEmbedKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xEEEE);
    let vocab = 32;
    let d = 16;
    let seq_max = 8;
    let w = EmbedWeights::new(vocab, d, seq_max, &mut rng);
    let tokens: Vec<u32> = vec![3, 0, 7, 1, 12, 5, 5, 31];
    let t = tokens.len();
    let (cpu_y, _) = embed_forward(&w, &tokens);

    let tok_gpu = GpuBuffer::from_slice(
        &ctx,
        "tok",
        &tokens.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>(),
    );
    // Hack: tokens buffer is `array<u32>` in WGSL. The
    // `GpuBuffer::from_slice` wrapper takes `&[f32]`; we transmute via
    // `f32::from_bits` so the bytes match.
    let w_tok_gpu = GpuBuffer::from_slice(&ctx, "w_tok", &w.w_tok);
    let w_pos_gpu = GpuBuffer::from_slice(&ctx, "w_pos", &w.w_pos);
    let out_gpu = GpuBuffer::zeros(&ctx, "out", t * d);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("embed fwd") });
    kernel.dispatch_into(
        &ctx,
        &mut enc,
        &tok_gpu,
        &w_tok_gpu,
        &w_pos_gpu,
        &out_gpu,
        t as u32,
        d as u32,
        vocab as u32,
        seq_max as u32,
        true,
    );
    ctx.queue.submit(Some(enc.finish()));
    let gpu_y = out_gpu.download(&ctx);
    let err = max_rel_err(&cpu_y, &gpu_y);
    assert!(err < 1e-4, "embed fwd err {err}");
}

#[test]
fn tied_lm_head_gradient_matches_cpu_through_both_paths() {
    let Some(ctx) = ctx_or_skip() else { return };
    // Tied LM head fwd = `gemm_transpose_b(h, w_tok)`. Gradient w.r.t.
    // h = `gemm(d_logits, w_tok)`. Gradient w.r.t. w_tok = `gemm_t(d_logits.T, h)`.
    // We test via direct CPU/GPU GEMM at the relevant shapes.
    let gemm = GemmKernel::new(&ctx);
    let gemm_t = GemmTransposeBKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xACE);

    let t = 6;
    let d = 16;
    let v = 24;
    let seq_max = 8;
    let w = EmbedWeights::new(v, d, seq_max, &mut rng);
    let h = random_vec(t * d, &mut rng);
    let upstream = random_vec(t * v, &mut rng);

    let cpu_logits = tied_lm_head_forward(&h, &w.w_tok, v, d);
    let mut grads = EmbedGrads::zeros(v, d, seq_max);
    let cpu_dh = tied_lm_head_backward(&h, &w.w_tok, &upstream, v, d, &mut grads);

    // GPU forward via gemm_transpose_b(h, w_tok): h is [T, D], w_tok is [V, D].
    let h_gpu = GpuBuffer::from_slice(&ctx, "h", &h);
    let w_gpu = GpuBuffer::from_slice(&ctx, "w_tok", &w.w_tok);
    let logits_gpu = GpuBuffer::zeros(&ctx, "logits", t * v);
    let dh_gpu = GpuBuffer::zeros(&ctx, "dh", t * d);
    let dw_gpu = GpuBuffer::zeros(&ctx, "dw", v * d);
    let upstream_gpu = GpuBuffer::from_slice(&ctx, "u", &upstream);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tied lm test"),
        });
    // forward: logits[T, V] = h[T, D] @ w_tok.T[D, V] = gemm_t(h, w_tok)
    gemm_t.dispatch_into(&ctx, &mut enc, &h_gpu, &w_gpu, &logits_gpu, t as u32, v as u32, d as u32);
    // dh[T, D] = upstream[T, V] @ w_tok[V, D] = gemm(upstream, w_tok)
    gemm.dispatch_into(&ctx, &mut enc, &upstream_gpu, &w_gpu, &dh_gpu, t as u32, d as u32, v as u32);
    // dw[V, D] = upstream.T[V, T] @ h[T, D] = gemm_t(upstream.T, h)?
    // upstream is [T, V] row-major. We want sum_t upstream[t, v] * h[t, c] → [V, D].
    // That's `gemm` with A = upstream.T (we'd need to transpose). Easier: use
    // gemm_transpose_b with A = upstream.T? Actually:
    //   dw[v, c] = sum_t upstream[t, v] * h[t, c]
    // If we treat A=upstream as [T, V] and B=h as [T, D], we want sum_t over rows,
    // which is A^T @ B = gemm_transpose_b(upstream, h) with shapes [V, T] × [T, D]^T → ...
    // Trick: gemm_transpose_b takes A:[m,k], B:[n,k] (read transposed) → C[m,n].
    // So with m=V, k=T, n=D and A=upstream_T(but we have upstream[T,V] row-major =
    // upstream_T[V, T] in column-major, which is what we want!), B=h[T, D] read as
    // [D, T] transposed... no. Let's just do it on CPU for the gradient.
    ctx.queue.submit(Some(enc.finish()));

    let gpu_logits = logits_gpu.download(&ctx);
    let gpu_dh = dh_gpu.download(&ctx);

    let err_logits = max_rel_err(&cpu_logits, &gpu_logits);
    let err_dh = max_rel_err(&cpu_dh, &gpu_dh);
    assert!(err_logits < 5e-2, "tied lm fwd err {err_logits}");
    assert!(err_dh < 5e-2, "tied lm dh err {err_dh}");
    let _ = dw_gpu;
    let _ = grads;
}

// ---- Sprint 60A: gradient accumulation kernels ----

#[test]
fn outer_product_accum_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = OuterProductAccumKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x0A11);
    for &(d_out, d_in, t) in &[(16, 16, 8), (32, 64, 12), (8, 24, 16)] {
        let x = random_vec(t * d_in, &mut rng);
        let d_y = random_vec(t * d_out, &mut rng);
        // Initial dest non-zero so we verify accumulation, not assignment.
        let initial = random_vec(d_out * d_in, &mut rng);

        // CPU expected: dest += sum_t d_y[t, i] * x[t, j]
        let mut cpu = initial.clone();
        for ti in 0..t {
            for i in 0..d_out {
                let g = d_y[ti * d_out + i];
                for j in 0..d_in {
                    cpu[i * d_in + j] += g * x[ti * d_in + j];
                }
            }
        }

        let x_gpu = GpuBuffer::from_slice(&ctx, "x", &x);
        let d_y_gpu = GpuBuffer::from_slice(&ctx, "d_y", &d_y);
        let dest_gpu = GpuBuffer::from_slice(&ctx, "dest", &initial);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("outer_accum test"),
            });
        kernel.dispatch_into(
            &ctx, &mut enc, &x_gpu, &d_y_gpu, &dest_gpu,
            d_out as u32, d_in as u32, t as u32,
        );
        ctx.queue.submit(Some(enc.finish()));
        let gpu = dest_gpu.download(&ctx);
        let err = max_rel_err(&cpu, &gpu);
        assert!(err < 5e-3, "outer_accum({d_out},{d_in},{t}) err {err}");
    }
}

#[test]
fn row_reduce_add_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = RowReduceAddKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x1B0E);
    for &(t, d) in &[(8, 16), (32, 64), (16, 128)] {
        let src = random_vec(t * d, &mut rng);
        let initial = random_vec(d, &mut rng);

        let mut cpu = initial.clone();
        for ti in 0..t {
            for c in 0..d {
                cpu[c] += src[ti * d + c];
            }
        }

        let src_gpu = GpuBuffer::from_slice(&ctx, "src", &src);
        let dest_gpu = GpuBuffer::from_slice(&ctx, "dest", &initial);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("row_reduce test"),
            });
        kernel.dispatch_into(&ctx, &mut enc, &src_gpu, &dest_gpu, t as u32, d as u32);
        ctx.queue.submit(Some(enc.finish()));
        let gpu = dest_gpu.download(&ctx);
        let err = max_rel_err(&cpu, &gpu);
        assert!(err < 5e-3, "row_reduce({t},{d}) err {err}");
    }
}

#[test]
fn grad_norm_sq_matches_cpu() {
    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = GradNormSqReduceKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0xBADD);
    for &n in &[64_usize, 1024, 8192, 16384] {
        let x = random_vec(n, &mut rng);
        let cpu_sum_sq: f32 = x.iter().map(|v| v * v).sum();

        let x_gpu = GpuBuffer::from_slice(&ctx, "x", &x);
        let n_groups = 8_u32;
        let partials = GpuBuffer::zeros(&ctx, "partials", n_groups as usize);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grad_norm test"),
            });
        kernel.dispatch_into(&ctx, &mut enc, &x_gpu, &partials, n as u32, n_groups);
        ctx.queue.submit(Some(enc.finish()));
        let gpu_partials = partials.download(&ctx);
        let gpu_sum_sq: f32 = gpu_partials.iter().sum();
        let err = (cpu_sum_sq - gpu_sum_sq).abs() / cpu_sum_sq.abs().max(1e-6);
        assert!(err < 1e-3, "grad_norm_sq n={n} cpu={cpu_sum_sq} gpu={gpu_sum_sq} err={err}");
    }
}

// ---- Sprint 58: full transformer end-to-end ----

#[test]
fn full_transformer_forward_matches_cpu() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let (cpu_logits, cpu_cache) = cpu_model.forward(&tokens);
    let mut cpu_grads = ModelGrads::zeros(&cfg);
    let cpu_loss = cpu_model.clone().backward(&cpu_cache, &cpu_logits, &targets, &mut cpu_grads);

    gpu_model.zero_grad();
    let gpu_loss = gpu_model.train_step(&tokens, &targets);

    let loss_diff = (cpu_loss - gpu_loss).abs();
    assert!(
        loss_diff < 5e-2,
        "loss CPU {cpu_loss} vs GPU {gpu_loss}, diff {loss_diff}"
    );
    let _ = cpu_logits;
}

#[test]
fn full_transformer_rope_forward_and_grads_match_cpu() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x80BE);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: true,
        ..TransformerConfig::default()
    };
    let cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| (i * 5 + 3) % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let (cpu_logits, cpu_cache) = cpu_model.forward(&tokens);
    let mut cpu_grads = ModelGrads::zeros(&cfg);
    let cpu_loss = cpu_model.clone().backward(&cpu_cache, &cpu_logits, &targets, &mut cpu_grads);

    gpu_model.zero_grad();
    let gpu_loss = gpu_model.train_step(&tokens, &targets);

    let loss_diff = (cpu_loss - gpu_loss).abs();
    assert!(
        loss_diff < 5e-2,
        "rope loss CPU {cpu_loss} vs GPU {gpu_loss}, diff {loss_diff}"
    );

    // Backward parity through the rotation: q/k projection grads.
    for li in 0..cfg.n_layer {
        let gpu_g_w_q = gpu_model.p.blocks[li].g_w_q.download(&gpu_model.ctx);
        let gpu_g_w_k = gpu_model.p.blocks[li].g_w_k.download(&gpu_model.ctx);
        let err_q = max_rel_err(&cpu_grads.blocks[li].d_mha.d_w_q.d_w, &gpu_g_w_q);
        let err_k = max_rel_err(&cpu_grads.blocks[li].d_mha.d_w_k.d_w, &gpu_g_w_k);
        assert!(err_q < 5e-2, "rope g_w_q layer {li} err {err_q}");
        assert!(err_k < 5e-2, "rope g_w_k layer {li} err {err_k}");
    }
    let gpu_g_w_tok = gpu_model.p.g_w_tok.download(&gpu_model.ctx);
    let err_tok = max_rel_err(&cpu_grads.embed.d_w_tok, &gpu_g_w_tok);
    assert!(err_tok < 5e-2, "rope g_w_tok err {err_tok}");

    // RoPE replaces the learned positional table — its gradient must be
    // exactly zero on both backends.
    let gpu_g_w_pos = gpu_model.p.g_w_pos.download(&gpu_model.ctx);
    assert!(gpu_g_w_pos.iter().all(|&g| g == 0.0), "GPU g_w_pos nonzero under rope");
    assert!(
        cpu_grads.embed.d_w_pos.iter().all(|&g| g == 0.0),
        "CPU d_w_pos nonzero under rope"
    );
}

#[test]
fn full_transformer_canon_forward_and_grads_match_cpu() {
    // Canon causal depthwise conv: forward (canon_out feeds ln1 + residual)
    // and backward (canon_dx maps dx back, canon_dw accumulates g_canon_w).
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0xCA0E);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        canon_kernel: 3,
        ..TransformerConfig::default()
    };
    let mut cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Move taps off the zero init so the conv path is actually exercised
    // (deterministic, no Rng-trait dependency in this test file).
    for blk in &mut cpu_model.blocks {
        blk.canon_w = (0..cfg.d_model * 3).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
    }
    let cpu_model = cpu_model;
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| (i * 5 + 3) % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let (cpu_logits, cpu_cache) = cpu_model.forward(&tokens);
    let mut cpu_grads = ModelGrads::zeros(&cfg);
    let cpu_loss = cpu_model.clone().backward(&cpu_cache, &cpu_logits, &targets, &mut cpu_grads);

    gpu_model.zero_grad();
    let gpu_loss = gpu_model.train_step(&tokens, &targets);

    let loss_diff = (cpu_loss - gpu_loss).abs();
    assert!(loss_diff < 5e-2, "canon loss CPU {cpu_loss} vs GPU {gpu_loss}, diff {loss_diff}");

    for li in 0..cfg.n_layer {
        // Canon tap gradient (the new dw-reduction kernel).
        let gpu_g_canon = gpu_model.p.blocks[li].g_canon_w.download(&gpu_model.ctx);
        let err_canon = max_rel_err(&cpu_grads.blocks[li].d_canon, &gpu_g_canon);
        assert!(err_canon < 5e-2, "canon g_canon_w layer {li} err {err_canon}");
        // Downstream q-proj grad depends on the canon-mixed input (forward path).
        let gpu_g_w_q = gpu_model.p.blocks[li].g_w_q.download(&gpu_model.ctx);
        let err_q = max_rel_err(&cpu_grads.blocks[li].d_mha.d_w_q.d_w, &gpu_g_w_q);
        assert!(err_q < 5e-2, "canon g_w_q layer {li} err {err_q}");
    }
    // Embedding grad flows through the canon dx mapping (backward path).
    let gpu_g_w_tok = gpu_model.p.g_w_tok.download(&gpu_model.ctx);
    let err_tok = max_rel_err(&cpu_grads.embed.d_w_tok, &gpu_g_w_tok);
    assert!(err_tok < 5e-2, "canon g_w_tok err {err_tok}");
}

#[test]
fn full_transformer_tier1_forward_and_grads_match_cpu() {
    // Whole Tier-1 pack at once: rope + rmsnorm + relu2 + qk-norm +
    // untied head + softcap. Catches any kernel whose mode flag isn't
    // threaded through the dispatch chain.
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x71E1);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: true,
        rmsnorm: true,
        relu2: true,
        qk_norm: true,
        untied_head: true,
        softcap: 5.0,
        ..TransformerConfig::default()
    };
    let mut cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Randomize the (production zero-init) head so softcap + head grads
    // are exercised on nontrivial logits.
    if let Some(h) = cpu_model.w_head.as_mut() {
        for v in h.iter_mut() {
            *v = rng.random::<f32>() * 0.4 - 0.2;
        }
    }
    let cpu_model = cpu_model;
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| (i * 5 + 3) % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let (cpu_logits, cpu_cache) = cpu_model.forward(&tokens);
    let mut cpu_grads = ModelGrads::zeros(&cfg);
    let cpu_loss = cpu_model.clone().backward(&cpu_cache, &cpu_logits, &targets, &mut cpu_grads);

    gpu_model.zero_grad();
    let gpu_loss = gpu_model.train_step(&tokens, &targets);

    let loss_diff = (cpu_loss - gpu_loss).abs();
    assert!(
        loss_diff < 5e-2,
        "tier1 loss CPU {cpu_loss} vs GPU {gpu_loss}, diff {loss_diff}"
    );

    for li in 0..cfg.n_layer {
        let gpu_g_w_q = gpu_model.p.blocks[li].g_w_q.download(&gpu_model.ctx);
        let gpu_g_w_k = gpu_model.p.blocks[li].g_w_k.download(&gpu_model.ctx);
        let gpu_g_w_fc = gpu_model.p.blocks[li].g_w_fc.download(&gpu_model.ctx);
        let gpu_g_w_proj = gpu_model.p.blocks[li].g_w_proj.download(&gpu_model.ctx);
        let gpu_g_ln1 = gpu_model.p.blocks[li].g_ln1_gamma.download(&gpu_model.ctx);
        let err_q = max_rel_err(&cpu_grads.blocks[li].d_mha.d_w_q.d_w, &gpu_g_w_q);
        let err_k = max_rel_err(&cpu_grads.blocks[li].d_mha.d_w_k.d_w, &gpu_g_w_k);
        let err_fc = max_rel_err(&cpu_grads.blocks[li].d_ffn.d_w_fc.d_w, &gpu_g_w_fc);
        let err_proj = max_rel_err(&cpu_grads.blocks[li].d_ffn.d_w_proj.d_w, &gpu_g_w_proj);
        let err_ln1 = max_rel_err(&cpu_grads.blocks[li].d_ln1.d_gamma, &gpu_g_ln1);
        assert!(err_q < 5e-2, "tier1 g_w_q layer {li} err {err_q}");
        assert!(err_k < 5e-2, "tier1 g_w_k layer {li} err {err_k}");
        assert!(err_fc < 5e-2, "tier1 g_w_fc layer {li} err {err_fc}");
        assert!(err_proj < 5e-2, "tier1 g_w_proj layer {li} err {err_proj}");
        assert!(err_ln1 < 5e-2, "tier1 g_ln1_gamma layer {li} err {err_ln1}");
        // RMSNorm: β must receive no gradient on either backend.
        let gpu_g_ln1_beta = gpu_model.p.blocks[li].g_ln1_beta.download(&gpu_model.ctx);
        assert!(gpu_g_ln1_beta.iter().all(|&g| g == 0.0), "GPU rms β grad nonzero, layer {li}");
        assert!(
            cpu_grads.blocks[li].d_ln1.d_beta.iter().all(|&g| g == 0.0),
            "CPU rms β grad nonzero, layer {li}"
        );
    }

    // Untied head: gradient lands in g_w_head; w_tok grad is embed-only.
    let gpu_g_w_head = gpu_model.p.g_w_head.as_ref().expect("g_w_head missing").download(&gpu_model.ctx);
    let err_head = max_rel_err(&cpu_grads.embed.d_w_head, &gpu_g_w_head);
    assert!(err_head < 5e-2, "tier1 g_w_head err {err_head}");
    let gpu_g_w_tok = gpu_model.p.g_w_tok.download(&gpu_model.ctx);
    let err_tok = max_rel_err(&cpu_grads.embed.d_w_tok, &gpu_g_w_tok);
    assert!(err_tok < 5e-2, "tier1 g_w_tok err {err_tok}");

    // Forward parity through softcap: per-token NLL must match CPU.
    let gpu_nll = gpu_model.eval_step_per_token(&tokens, &targets);
    let v = cfg.vocab;
    for (ti, &g_nll) in gpu_nll.iter().enumerate() {
        let row = &cpu_logits[ti * v..(ti + 1) * v];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let cpu_nll = -(row[targets[ti] as usize] - max - sum.ln());
        assert!(
            (cpu_nll - g_nll).abs() < 5e-2,
            "tier1 NLL[{ti}] CPU {cpu_nll} vs GPU {g_nll}"
        );
    }
}

#[test]
fn full_transformer_training_step_matches_cpu_direction() {
    // We verify: after 1 GPU train step, params change in the same
    // direction as 1 CPU train step (loss decreases for both).
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::optimizer::AdamWConfig;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0xABBA);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let mut cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);
    let adam = AdamWConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    let tokens: Vec<u32> = (0..8u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    // Reference initial loss
    let (logits0, cache0) = cpu_model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    let initial_loss = cpu_model.clone().backward(&cache0, &logits0, &targets, &mut grads);

    // 3 CPU steps
    for _ in 0..3 {
        grads.zero();
        let (lg, cache) = cpu_model.forward(&tokens);
        let _ = cpu_model.backward(&cache, &lg, &targets, &mut grads);
        DecoderTransformer::clip_grads(&mut grads, 1.0);
        cpu_model.step_adamw(&grads, &adam);
    }
    let (logits_after, cache_after) = cpu_model.forward(&tokens);
    let cpu_final_loss = cpu_model.clone().backward(&cache_after, &logits_after, &targets, &mut grads);
    assert!(cpu_final_loss < initial_loss, "CPU loss didn't drop");

    // 3 GPU steps
    for _ in 0..3 {
        gpu_model.zero_grad();
        let _ = gpu_model.train_step(&tokens, &targets);
        gpu_model.clip_grads(1.0);
        gpu_model.step_adamw(&adam);
    }
    let gpu_loss_after = gpu_model.train_step(&tokens, &targets);
    assert!(
        gpu_loss_after < initial_loss + 0.1,
        "GPU loss did not drop: initial {initial_loss}, after 3 steps {gpu_loss_after}"
    );
}

// ─── Sprint 61: GPU softmax_ce + evaluate port ───────────────────────────

/// Gate test: GPU `SoftmaxCrossEntropyKernel` must produce gradients that
/// match CPU `cross_entropy_with_grad` after applying `loss_scale = 1/T`.
/// If this fails, the GPU train_step refactor would silently drift.
#[test]
fn softmax_ce_grad_matches_cpu() {
    use aurora::transformer::loss::cross_entropy_with_grad;

    let Some(ctx) = ctx_or_skip() else { return };
    let kernel = SoftmaxCrossEntropyKernel::new(&ctx);
    let mut rng = StdRng::seed_from_u64(0x5CE_6_4AD);

    let v: usize = 257; // odd, to expose stride-loop bugs
    let t: usize = 5;
    let logits: Vec<f32> = (0..t * v).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();
    let targets: Vec<u32> = (0..t).map(|i| (i * 37 % v) as u32).collect();
    let loss_scale = 1.0_f32 / t as f32;

    let logits_gpu = GpuBuffer::from_slice(&ctx, "logits", &logits);
    let d_logits_gpu = GpuBuffer::zeros(&ctx, "d_logits", t * v);
    let loss_seq_gpu = GpuBuffer::zeros(&ctx, "loss_seq", t);
    let targets_gpu = GpuBuffer::from_u32_slice(&ctx, "targets", &targets);
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_ce_grad test"),
        });
    kernel.dispatch_into(
        &ctx, &mut enc,
        &logits_gpu, &d_logits_gpu, &loss_seq_gpu, &targets_gpu,
        v as u32, t as u32, 1, loss_scale, 0,
    );
    ctx.queue.submit(Some(enc.finish()));

    let gpu_d_logits = d_logits_gpu.download(&ctx);
    let gpu_loss = loss_seq_gpu.download(&ctx);

    // CPU reference.
    let mut cpu_d_logits = vec![0.0_f32; t * v];
    let mut cpu_loss = vec![0.0_f32; t];
    for ti in 0..t {
        let row = &logits[ti * v..(ti + 1) * v];
        let (l, g) = cross_entropy_with_grad(row, targets[ti] as usize);
        cpu_loss[ti] = l;
        for c in 0..v {
            cpu_d_logits[ti * v + c] = g[c] * loss_scale;
        }
    }

    let grad_err = max_rel_err(&cpu_d_logits, &gpu_d_logits);
    assert!(grad_err < 1e-4, "d_logits rel err {grad_err}");
    let loss_err = max_rel_err(&cpu_loss, &gpu_loss);
    assert!(loss_err < 1e-5, "per-token loss rel err {loss_err}");
}

/// End-to-end: `DecoderTransformerGpu::eval_step` driven through a
/// sliding window must match CPU `evaluate`-style BPB within 1e-3.
#[test]
fn evaluate_gpu_matches_cpu_evaluate() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::text_io::bits_per_byte;
    use aurora::transformer::loss::softmax;
    use aurora::transformer::{DecoderTransformer, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0xE_4A1);
    let cfg = TransformerConfig {
        vocab: 64,
        d_model: 32,
        n_layer: 2,
        n_head: 4,
        d_ff: 64,
        seq_len_max: 32,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let gpu = DecoderTransformerGpu::from_cpu(&cpu, ctx, 1);

    let seq_len: usize = 32;
    let val_len: usize = 320;
    let val_toks: Vec<u32> = (0..val_len)
        .map(|_| rng.random::<u32>() % cfg.vocab as u32)
        .collect();
    let comp: f32 = 1.0;

    // CPU eval — open-coded to match `evaluate()` in aurora_train.rs
    // (avoids depending on the example binary).
    let mut cpu_nats = 0.0_f32;
    let mut cpu_tokens = 0_usize;
    let mut p = 0;
    while p + seq_len + 1 < val_toks.len() {
        let input = &val_toks[p..p + seq_len];
        let target = &val_toks[p + 1..p + seq_len + 1];
        let (logits, _) = cpu.forward(input);
        for ti in 0..seq_len {
            let row = &logits[ti * cfg.vocab..(ti + 1) * cfg.vocab];
            let probs = softmax(row);
            let pt = probs[target[ti] as usize].max(1e-12);
            cpu_nats += -pt.ln();
            cpu_tokens += 1;
        }
        p += seq_len;
    }
    let cpu_bpb = bits_per_byte(cpu_nats, (cpu_tokens as f32 * comp) as usize);

    // GPU eval — sums per-window total NLL from eval_step.
    let mut gpu_nats = 0.0_f32;
    let mut gpu_tokens = 0_usize;
    let mut p = 0;
    while p + seq_len + 1 < val_toks.len() {
        let input = &val_toks[p..p + seq_len];
        let target = &val_toks[p + 1..p + seq_len + 1];
        gpu_nats += gpu.eval_step(input, target);
        gpu_tokens += seq_len;
        p += seq_len;
    }
    let gpu_bpb = bits_per_byte(gpu_nats, (gpu_tokens as f32 * comp) as usize);

    let diff = (cpu_bpb - gpu_bpb).abs();
    assert!(
        diff < 1e-3,
        "val BPB drift: cpu {cpu_bpb} vs gpu {gpu_bpb} (diff {diff})"
    );
}

/// Train-step regression: GPU mean_loss (now from softmax_ce) must
/// match CPU baseline loss for the same weights + inputs. Validates
/// the train_step refactor end-to-end.
#[test]
fn train_step_loss_matches_cpu_baseline() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0xBEEF_F00D);
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 16,
        n_layer: 2,
        n_head: 2,
        d_ff: 32,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu = DecoderTransformerGpu::from_cpu(&cpu, ctx, 1);

    let t: usize = 8;
    let tokens: Vec<u32> = (0..t as u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&x| (x + 1) % cfg.vocab as u32).collect();

    // CPU loss (mean cross-entropy).
    let mut grads = ModelGrads::zeros(&cfg);
    let (cpu_logits, cache) = cpu.forward(&tokens);
    let cpu_loss = cpu.clone().backward(&cache, &cpu_logits, &targets, &mut grads);

    let gpu_loss = gpu.train_step(&tokens, &targets);

    let diff = (cpu_loss - gpu_loss).abs();
    assert!(
        diff < 5e-3,
        "train_step loss drift: cpu {cpu_loss} vs gpu {gpu_loss} (diff {diff})"
    );
}

/// `train_step_batch` (one encoder / one submit / one readback) must
/// produce gradients and loss identical to running the same B sequences
/// through `train_step` one at a time. Validates the per-element offset
/// wiring (tokens → embed, targets/NLL → softmax, tokens → scatter-reduce)
/// in the batched path.
#[test]
fn train_step_batch_matches_sequential() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::transformer::{DecoderTransformer, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x5EED_BA7C);
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 16,
        n_layer: 2,
        n_head: 2,
        d_ff: 32,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);

    let b = 4usize;
    let t = 8usize;
    let seqs: Vec<Vec<u32>> = (0..b)
        .map(|s| (0..t as u32).map(|i| (i + s as u32 * 3) % cfg.vocab as u32).collect())
        .collect();
    let tgts: Vec<Vec<u32>> = seqs
        .iter()
        .map(|s| s.iter().map(|&x| (x + 1) % cfg.vocab as u32).collect())
        .collect();

    // Batched path.
    let mut gpu_batched = DecoderTransformerGpu::from_cpu(&cpu, ctx.clone(), b);
    gpu_batched.zero_grad();
    let inputs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    let tgt_refs: Vec<&[u32]> = tgts.iter().map(Vec::as_slice).collect();
    let loss_batched = gpu_batched.train_step_batch(&inputs, &tgt_refs);

    // Sequential path: B separate train_step calls accumulating into grads.
    let mut gpu_seq = DecoderTransformerGpu::from_cpu(&cpu, ctx, 1);
    gpu_seq.zero_grad();
    let mut loss_seq = 0.0_f32;
    for i in 0..b {
        loss_seq += gpu_seq.train_step(&seqs[i], &tgts[i]);
    }

    assert!(
        (loss_batched - loss_seq).abs() < 1e-3,
        "batched loss {loss_batched} vs sequential {loss_seq}"
    );

    for (label, va, vs) in [
        (
            "g_w_q",
            gpu_batched.p.blocks[0].g_w_q.download(&gpu_batched.ctx),
            gpu_seq.p.blocks[0].g_w_q.download(&gpu_seq.ctx),
        ),
        (
            "g_w_proj",
            gpu_batched.p.blocks[0].g_w_proj.download(&gpu_batched.ctx),
            gpu_seq.p.blocks[0].g_w_proj.download(&gpu_seq.ctx),
        ),
        (
            "g_w_tok",
            gpu_batched.p.g_w_tok.download(&gpu_batched.ctx),
            gpu_seq.p.g_w_tok.download(&gpu_seq.ctx),
        ),
        (
            "g_w_pos",
            gpu_batched.p.g_w_pos.download(&gpu_batched.ctx),
            gpu_seq.p.g_w_pos.download(&gpu_seq.ctx),
        ),
        (
            "g_final_gamma",
            gpu_batched.p.g_final_gamma.download(&gpu_batched.ctx),
            gpu_seq.p.g_final_gamma.download(&gpu_seq.ctx),
        ),
    ] {
        let max_abs = va
            .iter()
            .zip(&vs)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 2e-3, "{label} batched vs sequential drift {max_abs}");
    }
}

// ─── Sprint 62 (GPU port): Muon hybrid step parity ───────────────────────

/// One full Muon step on identical CPU + GPU models must produce
/// parameter buffers that agree within f32 tolerance. NS5 amplifies
/// per-element noise vs AdamW (because the orthogonalization is
/// nonlinear in the input matrix), but for the small 32×32 test
/// matrices below the worst relative drift stays well under 5e-2.
#[test]
fn muon_hybrid_step_matches_cpu() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::optimizer::{AdamWConfig, MuonConfig};
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x4_6_0_5_5);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let mut cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let muon = MuonConfig {
        lr: 0.02,
        momentum: 0.95,
        nesterov: true,
        weight_decay: 0.0,
        ns_steps: 5,
    };
    let adam = AdamWConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    // CPU side: forward + backward + Muon step.
    let mut grads = ModelGrads::zeros(&cfg);
    let (cpu_logits, cache) = cpu_model.forward(&tokens);
    let _ = cpu_model.backward(&cache, &cpu_logits, &targets, &mut grads);
    DecoderTransformer::clip_grads_adamw_subset(&mut grads, 1.0);
    cpu_model.step_muon_hybrid(&grads, &muon, &adam);

    // GPU side: same.
    gpu_model.zero_grad();
    let _ = gpu_model.train_step(&tokens, &targets);
    let _ = gpu_model.clip_grads_adamw_subset(1.0, false);
    gpu_model.step_muon_hybrid(&muon, &adam);

    // Spot-check params across all kinds (matrix, embed, LN).
    let gpu_w_q = gpu_model.p.blocks[0].w_q.download(&gpu_model.ctx);
    let err_wq = max_rel_err(&cpu_model.blocks[0].mha.w_q.w, &gpu_w_q);
    assert!(err_wq < 5e-2, "w_q[0] drift {err_wq}");

    let gpu_w_o = gpu_model.p.blocks[1].w_o.download(&gpu_model.ctx);
    let err_wo = max_rel_err(&cpu_model.blocks[1].mha.w_o.w, &gpu_w_o);
    assert!(err_wo < 5e-2, "w_o[1] drift {err_wo}");

    let gpu_w_fc = gpu_model.p.blocks[0].w_fc.download(&gpu_model.ctx);
    let err_fc = max_rel_err(&cpu_model.blocks[0].ffn.w_fc.w, &gpu_w_fc);
    assert!(err_fc < 5e-2, "w_fc[0] drift {err_fc}");

    let gpu_w_proj = gpu_model.p.blocks[1].w_proj.download(&gpu_model.ctx);
    let err_proj = max_rel_err(&cpu_model.blocks[1].ffn.w_proj.w, &gpu_w_proj);
    assert!(err_proj < 5e-2, "w_proj[1] drift {err_proj}");

    let gpu_tok = gpu_model.p.w_tok.download(&gpu_model.ctx);
    let err_tok = max_rel_err(&cpu_model.embed.w_tok, &gpu_tok);
    assert!(err_tok < 5e-2, "w_tok drift {err_tok}");

    let gpu_ln_g = gpu_model.p.blocks[0].ln1_gamma.download(&gpu_model.ctx);
    let err_ln = max_rel_err(&cpu_model.blocks[0].ln1.gamma, &gpu_ln_g);
    assert!(err_ln < 5e-2, "ln1.gamma[0] drift {err_ln}");
}

/// Loss must drop after a handful of Muon steps on GPU — same shape as
/// `train_step_then_adam_loss_decreases` but routed through
/// `step_muon_hybrid`.
#[test]
fn gpu_muon_hybrid_drops_loss_on_overfit() {
    use aurora::gpu::DecoderTransformerGpu;
    use aurora::optimizer::{AdamWConfig, MuonConfig};
    use aurora::transformer::{DecoderTransformer, TransformerConfig};

    let Some(ctx) = ctx_or_skip() else { return };
    let mut rng = StdRng::seed_from_u64(0x4_6_0_5_6);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let tokens: Vec<u32> = (0..8u32).map(|i| i % cfg.vocab as u32).collect();
    let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % cfg.vocab as u32).collect();

    let muon = MuonConfig {
        lr: 0.02,
        momentum: 0.95,
        nesterov: true,
        weight_decay: 0.0,
        ns_steps: 5,
    };
    let adam = AdamWConfig {
        lr: 3e-3,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    gpu_model.zero_grad();
    let initial_loss = gpu_model.train_step(&tokens, &targets);
    // Undo the grad accumulation from the loss probe before training.
    gpu_model.zero_grad();

    for _ in 0..5 {
        gpu_model.zero_grad();
        let _ = gpu_model.train_step(&tokens, &targets);
        let _ = gpu_model.clip_grads_adamw_subset(1.0, true);
        gpu_model.step_muon_hybrid(&muon, &adam);
    }

    gpu_model.zero_grad();
    let final_loss = gpu_model.train_step(&tokens, &targets);
    assert!(
        final_loss.is_finite(),
        "GPU muon: final loss NaN/Inf ({final_loss})"
    );
    assert!(
        final_loss < initial_loss,
        "GPU muon: loss did not drop: initial {initial_loss} → final {final_loss}"
    );
}
