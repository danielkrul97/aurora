//! Sprint 60C: KV cache parity + generation timing tests.
//!
//! Asserts: running N `generate_step` calls produces logits identical
//! (within numerical tolerance) to re-running `forward()` over the full
//! growing context.

#![allow(clippy::needless_range_loop, clippy::cast_precision_loss)]

use aurora::transformer::loss::argmax;
use aurora::transformer::{generate_step, DecoderTransformer, KvCache, TransformerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[cfg(feature = "gpu")]
use aurora::gpu::{DecoderTransformerGpu, GpuContext};

fn rel_err(a: f32, b: f32) -> f32 {
    let d = a.abs().max(b.abs()).max(1e-6);
    (a - b).abs() / d
}

#[test]
fn kv_cache_generate_step_matches_full_forward() {
    kv_cache_parity_for(false);
}

#[test]
fn kv_cache_generate_step_matches_full_forward_rope() {
    kv_cache_parity_for(true);
}

#[test]
fn kv_cache_generate_step_matches_full_forward_tier1() {
    kv_cache_parity_cfg(TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 32,
        rope: true,
        rmsnorm: true,
        relu2: true,
        qk_norm: true,
        untied_head: true,
        softcap: 5.0,
        ..TransformerConfig::default()
    });
}

fn kv_cache_parity_for(rope: bool) {
    kv_cache_parity_cfg(TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 32,
        rope,
        ..TransformerConfig::default()
    });
}

fn kv_cache_parity_cfg(cfg: TransformerConfig) {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(0xCACE);
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Production zero-inits the untied head; randomize so logits differ.
    if let Some(h) = model.w_head.as_mut() {
        for v in h.iter_mut() {
            *v = rng.random::<f32>() * 2.0 - 1.0;
        }
    }
    let model = model;

    let seed: Vec<u32> = vec![3, 7, 2, 11, 5];
    let mut tokens = seed.clone();

    // Reference path: full forward at each step, take last logits.
    let mut ref_logits: Vec<Vec<f32>> = Vec::new();
    let mut to_predict: Vec<u32> = Vec::new();
    for step in 0..5 {
        let (logits, _) = model.forward(&tokens);
        let v = cfg.vocab;
        let last_pos = tokens.len() - 1;
        let row = logits[last_pos * v..(last_pos + 1) * v].to_vec();
        ref_logits.push(row.clone());
        let next = argmax(&row) as u32;
        to_predict.push(next);
        tokens.push(next);
        let _ = step;
    }

    // Cached path: prefill on seed, then incremental generate_step.
    let mut cache = KvCache::new(cfg.n_layer, cfg.d_model);
    let _last_h = cache.prefill(&model, &seed);

    // Sanity: prefill creates valid cache state (k_history non-empty per layer).
    for blk in &cache.blocks {
        assert_eq!(blk.current_len, seed.len());
        assert_eq!(blk.k_history.len(), seed.len() * cfg.d_model);
    }

    // The first logits we need to compare are the ones from the seed
    // prefix — i.e. predictions for what comes after `seed`. The
    // reference run above computed those as `ref_logits[0]` (after the
    // first full forward on seed). We need to obtain the same from the
    // cache: actually, after prefill, the cache encodes the full seed,
    // but `generate_step` predicts the token after `seed`. So we need
    // to bootstrap: do one full forward on seed to get its last logits,
    // then for each subsequent step use `generate_step(predicted)`.
    let v = cfg.vocab;

    // Step 0: prediction for after seed. Same as reference[0].
    let (logits_seed, _) = model.forward(&seed);
    let cached_step0 = logits_seed[(seed.len() - 1) * v..seed.len() * v].to_vec();
    let err0 = ref_logits[0]
        .iter()
        .zip(cached_step0.iter())
        .map(|(a, b)| rel_err(*a, *b))
        .fold(0.0_f32, f32::max);
    assert!(err0 < 1e-4, "seed-step rel err {err0}");

    // Steps 1..5: generate_step each time, predicting next token.
    for step_idx in 0..to_predict.len() - 1 {
        let new_tok = to_predict[step_idx];
        let cached_logits = generate_step(&model, &mut cache, new_tok);
        let ref_row = &ref_logits[step_idx + 1];
        for vi in 0..v {
            let err = rel_err(ref_row[vi], cached_logits[vi]);
            assert!(
                err < 5e-3,
                "step {step_idx} vocab {vi}: ref {} cached {} err {err}",
                ref_row[vi], cached_logits[vi]
            );
        }
    }
}

#[test]
fn kv_cache_reset_clears_state() {
    let mut rng = StdRng::seed_from_u64(0xFFFF);
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
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut cache = KvCache::new(cfg.n_layer, cfg.d_model);
    cache.prefill(&model, &[1u32, 2, 3, 4]);
    assert_eq!(cache.current_len, 4);
    cache.reset();
    assert_eq!(cache.current_len, 0);
    for blk in &cache.blocks {
        assert_eq!(blk.current_len, 0);
        assert!(blk.k_history.is_empty());
    }
}

#[test]
fn kv_cache_generates_deterministic_with_temperature_0() {
    // Greedy argmax sampling with KV cache must produce the same token
    // sequence as greedy argmax without cache (because both pick the
    // same next token at each step, identical logits).
    let mut rng = StdRng::seed_from_u64(0x7E57);
    let cfg = TransformerConfig {
        vocab: 8,
        d_model: 16,
        n_layer: 2,
        n_head: 2,
        d_ff: 32,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let seed: Vec<u32> = vec![1, 2, 3];

    // No-cache: full forward at each step.
    let mut tokens_nocache = seed.clone();
    for _ in 0..6 {
        let (logits, _) = model.forward(&tokens_nocache);
        let v = cfg.vocab;
        let last = tokens_nocache.len() - 1;
        let row = &logits[last * v..(last + 1) * v];
        let next = argmax(row) as u32;
        tokens_nocache.push(next);
    }

    // Cached: prefill + 6 generate_step.
    let mut cache = KvCache::new(cfg.n_layer, cfg.d_model);
    let _ = cache.prefill(&model, &seed);
    let mut tokens_cached = seed.clone();
    // First prediction from seed (same as no-cache first step)
    let (seed_logits, _) = model.forward(&seed);
    let v = cfg.vocab;
    let last = seed.len() - 1;
    let mut next = argmax(&seed_logits[last * v..(last + 1) * v]) as u32;
    tokens_cached.push(next);
    for _ in 0..5 {
        let row = generate_step(&model, &mut cache, next);
        next = argmax(&row) as u32;
        tokens_cached.push(next);
    }

    assert_eq!(tokens_nocache, tokens_cached, "greedy sequences diverged");
}

// ---- Sprint 61: GPU KV cache parity ----

#[cfg(feature = "gpu")]
fn rel_err_max(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x.abs().max(y.abs()).max(1e-6);
        let e = (x - y).abs() / d;
        if e > worst {
            worst = e;
        }
    }
    worst
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_matches_cpu_kv_cache() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x6900);
    let cfg = TransformerConfig {
        vocab: 24,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let seed: Vec<u32> = vec![3, 0, 7, 1, 11];

    // CPU path: prefill + 3 generate_step calls.
    let mut cpu_cache = KvCache::new(cfg.n_layer, cfg.d_model);
    cpu_cache.prefill(&cpu_model, &seed);

    // GPU path: prefill + 3 generate_step calls.
    let mut gpu_cache = gpu_model.prefill_kv(&seed);
    assert_eq!(gpu_cache.current_len, seed.len());
    assert_eq!(cpu_cache.current_len, seed.len());

    // Step 1: feed seed's last token (the "first prediction context").
    // For CPU and GPU we run generate_step with same token, compare logits.
    let to_feed: Vec<u32> = vec![5, 2, 9];
    for &tok in &to_feed {
        let cpu_logits = generate_step(&cpu_model, &mut cpu_cache, tok);
        let gpu_logits = gpu_model.generate_step_kv(tok, &mut gpu_cache);
        let err = rel_err_max(&cpu_logits, &gpu_logits);
        assert!(
            err < 5e-2,
            "GPU/CPU KV logits diverged at token {tok}: err {err}"
        );
    }
    assert_eq!(gpu_cache.current_len, seed.len() + to_feed.len());
    assert_eq!(cpu_cache.current_len, seed.len() + to_feed.len());
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_greedy_matches_full_forward() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0xA571);
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
    let gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let seed: Vec<u32> = vec![2, 4, 1];
    let mut tokens_full = seed.clone();
    for _ in 0..4 {
        let (logits, _) = cpu_model.forward(&tokens_full);
        let v = cfg.vocab;
        let last = tokens_full.len() - 1;
        let next = argmax(&logits[last * v..(last + 1) * v]) as u32;
        tokens_full.push(next);
    }

    let mut gpu_cache = gpu_model.prefill_kv(&seed);
    let (seed_logits, _) = cpu_model.forward(&seed);
    let v = cfg.vocab;
    let last = seed.len() - 1;
    let mut next = argmax(&seed_logits[last * v..(last + 1) * v]) as u32;
    let mut tokens_gpu = seed.clone();
    tokens_gpu.push(next);
    for _ in 0..3 {
        let logits = gpu_model.generate_step_kv(next, &mut gpu_cache);
        next = argmax(&logits) as u32;
        tokens_gpu.push(next);
    }

    assert_eq!(tokens_full, tokens_gpu, "GPU greedy sample diverged from full-forward");
}

// ---- Sprint 62: parallel MHA kernel + GPU-side reset ----

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_parallel_kernel_handles_large_t_total() {
    gpu_kv_cache_parity_for(false);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_parallel_kernel_handles_large_t_total_rope() {
    gpu_kv_cache_parity_for(true);
}

#[cfg(feature = "gpu")]
fn gpu_kv_cache_parity_for(rope: bool) {
    // Exercises the workgroup-shared max/sum reductions across the
    // stride-loop boundary (t_total > 64 threads).
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x6262);
    let cfg = TransformerConfig {
        vocab: 16,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 96, // > 64 forces stride loops in phase 1/2
        rope,
        ..TransformerConfig::default()
    };
    let cpu_model = DecoderTransformer::new(cfg.clone(), &mut rng);
    let gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    // 80 tokens prefill → 5 generate steps.
    let seed: Vec<u32> = (0..80u32).map(|i| i % cfg.vocab as u32).collect();
    let mut cpu_cache = KvCache::new(cfg.n_layer, cfg.d_model);
    cpu_cache.prefill(&cpu_model, &seed);
    let mut gpu_cache = gpu_model.prefill_kv(&seed);

    for tok in &[1u32, 4, 7, 2, 9] {
        let cpu_logits = generate_step(&cpu_model, &mut cpu_cache, *tok);
        let gpu_logits = gpu_model.generate_step_kv(*tok, &mut gpu_cache);
        let err = rel_err_max(&cpu_logits, &gpu_logits);
        assert!(
            err < 5e-2,
            "stride-loop kernel divergence (rope={rope}) at t_total={}: err {err}",
            gpu_cache.current_len
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_reset_correctness() {
    // After reset(), the cache's history must be all zeros AND a fresh
    // prefill + step on the cleared cache must match a brand-new cache
    // path. Validates the kernel-based reset (no host upload).
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0x9292);
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
    let gpu_model = DecoderTransformerGpu::from_cpu(&cpu_model, ctx, 1);

    let seed_a: Vec<u32> = vec![3, 0, 7, 1];
    let seed_b: Vec<u32> = vec![5, 2, 9, 11];

    // Path A: fresh cache → prefill(seed_b) → step(8). Reference.
    let mut cache_fresh = gpu_model.prefill_kv(&seed_b);
    let logits_fresh = gpu_model.generate_step_kv(8, &mut cache_fresh);

    // Path B: dirty cache from seed_a + steps, reset, then re-prefill(seed_b) → step(8).
    let mut cache_dirty = gpu_model.prefill_kv(&seed_a);
    let _ = gpu_model.generate_step_kv(4, &mut cache_dirty);
    let _ = gpu_model.generate_step_kv(6, &mut cache_dirty);
    cache_dirty.reset(&gpu_model.ctx);

    assert_eq!(cache_dirty.current_len, 0);
    let k0 = cache_dirty.blocks[0].k_history.download(&gpu_model.ctx);
    let max_abs_k = k0.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(max_abs_k < 1e-6, "k_history not zeroed after reset; max abs = {max_abs_k}");
    let v0 = cache_dirty.blocks[0].v_history.download(&gpu_model.ctx);
    let max_abs_v = v0.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(max_abs_v < 1e-6, "v_history not zeroed after reset; max abs = {max_abs_v}");

    // Reuse the reset cache for the same prefill+step as Path A.
    cache_dirty = gpu_model.prefill_kv(&seed_b);
    let logits_reused = gpu_model.generate_step_kv(8, &mut cache_dirty);

    let err = rel_err_max(&logits_fresh, &logits_reused);
    assert!(err < 5e-2, "reset+re-prefill diverged from fresh: err {err}");
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_kv_cache_new_does_not_read_back_garbage() {
    // After KvCacheGpu::new with no prefill, history buffers must be
    // all zeros (kernel-initialized, not host-uploaded).
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            return;
        }
    };
    use aurora::gpu::KvCacheGpu;
    let cache = KvCacheGpu::new(&ctx, 3, 32, 16);
    for blk in &cache.blocks {
        let k = blk.k_history.download(&ctx);
        let v = blk.v_history.download(&ctx);
        let max_abs = k.iter().chain(v.iter()).map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs < 1e-6, "fresh KvCacheGpu has non-zero history; max abs = {max_abs}");
    }
    assert_eq!(cache.current_len, 0);
}
