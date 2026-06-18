//! PHAGE governor — GPU device snapshot/restore (the primitive that lets
//! reflex rollback and paired-rollout trials run on-device under --gpu).
//! Skips cleanly when no GPU adapter is present.
#![cfg(feature = "gpu")]

use aurora::gpu::{DecoderTransformerGpu, GpuContext};
use aurora::optimizer::{AdamWConfig, MuonConfig};
use aurora::transformer::{DecoderTransformer, TransformerConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("GPU not available, skipping: {e}");
            None
        }
    }
}

fn tiny_cfg() -> TransformerConfig {
    TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        ..TransformerConfig::tiny(32)
    }
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// One real training step on the shared (input, target) window.
fn step(gpu: &mut DecoderTransformerGpu, input: &[u32], target: &[u32]) {
    let adam = AdamWConfig { lr: 0.2, ..AdamWConfig::default() };
    let muon = MuonConfig { lr: 0.2, ..MuonConfig::default() };
    gpu.zero_grad();
    gpu.train_step_batch(&[input], &[target]);
    gpu.clip_grads_adamw_subset(1e9, false);
    gpu.step_muon_hybrid(&muon, &adam);
}

#[test]
fn gpu_snapshot_restore_is_bit_exact() {
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let cfg = tiny_cfg();
    let mut rng = StdRng::seed_from_u64(1);
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu = DecoderTransformerGpu::from_cpu(&cpu, ctx, 4);

    let toks: Vec<u32> = (0..200u32).map(|i| i % 32).collect();
    let input = toks[0..16].to_vec();
    let target = toks[1..17].to_vec();

    // Warm up so the zero-init attention/FFN output projections become
    // non-zero — the very first step has exactly-zero matrix grads (Muon
    // can't move w_q until w_o != 0), which would make the check below
    // vacuous for the Muon path.
    for _ in 0..3 {
        step(&mut gpu, &input, &target);
    }

    let before = gpu.to_cpu();
    let snap = gpu.snapshot_state();

    // One more step now moves both the Muon (w_q) and AdamW (w_tok) params.
    step(&mut gpu, &input, &target);
    let after = gpu.to_cpu();
    let dq = maxabs(&before.blocks[0].mha.w_q.w, &after.blocks[0].mha.w_q.w);
    let dtok = maxabs(&before.embed.w_tok, &after.embed.w_tok);
    assert!(dq > 0.0, "warmed-up Muon step did not move w_q (dq={dq})");
    assert!(dtok > 0.0, "AdamW step did not move w_tok (dtok={dtok})");

    // Restore must bring every state buffer back bit-exact.
    gpu.restore_state(&snap);
    let restored = gpu.to_cpu();
    assert_eq!(before.embed.w_tok, restored.embed.w_tok, "w_tok not restored");
    assert_eq!(before.embed.w_pos, restored.embed.w_pos, "w_pos not restored");
    assert_eq!(
        before.blocks[0].mha.w_q.w, restored.blocks[0].mha.w_q.w,
        "block0 w_q (Muon) not restored"
    );
    assert_eq!(
        before.blocks[1].ffn.w_proj.w, restored.blocks[1].ffn.w_proj.w,
        "block1 w_proj (Muon) not restored"
    );
    assert_eq!(
        before.final_ln.gamma, restored.final_ln.gamma,
        "final_ln gamma not restored"
    );
}

#[test]
fn gpu_refresh_recaptures_current_weights() {
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let cfg = tiny_cfg();
    let mut rng = StdRng::seed_from_u64(2);
    let cpu = DecoderTransformer::new(cfg.clone(), &mut rng);
    let mut gpu = DecoderTransformerGpu::from_cpu(&cpu, ctx, 4);

    let toks: Vec<u32> = (0..200u32).map(|i| (i * 7) % 32).collect();
    let input = toks[0..16].to_vec();
    let target = toks[1..17].to_vec();

    for _ in 0..3 {
        step(&mut gpu, &input, &target);
    }

    let mut snap = gpu.snapshot_state();
    // Train, then refresh the snapshot to the *new* weights.
    step(&mut gpu, &input, &target);
    let trained = gpu.to_cpu();
    gpu.refresh_state(&mut snap);

    // Move away again, then restore to the refreshed point.
    step(&mut gpu, &input, &target);
    let moved = gpu.to_cpu();
    gpu.restore_state(&snap);
    let restored = gpu.to_cpu();

    assert!(
        maxabs(&trained.blocks[0].mha.w_q.w, &moved.blocks[0].mha.w_q.w) > 0.0,
        "second step did not move w_q — test would be vacuous"
    );
    assert_eq!(
        trained.blocks[0].mha.w_q.w, restored.blocks[0].mha.w_q.w,
        "refresh did not recapture the trained weights"
    );
}
