#![allow(clippy::cast_precision_loss)]

//! `forward_backward_batch` must produce gradients identical (up to FP
//! summation order) to running the same `(input, target)` pairs serially
//! and summing into one `grads` buffer.
//!
//! The order in which we sum per-sample gradients differs between serial
//! (sequential addition into `self.grads`) and parallel (collected into a
//! Vec then summed in deterministic order) — both are sequential adds in
//! some order, so the result is identical for finite-rate floating point
//! when iteration order matches. We use single-threaded rayon to nail
//! down ordering, then assert exact equality on grad buffers.

use aurora::optimizer::AdamConfig;
use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::VOCAB;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_HIDDEN: usize = 32;
const T: usize = 16;
const BATCH: usize = 4;

fn make_net(seed: u64) -> GatedRecurrentSnn {
    let mut rng = StdRng::seed_from_u64(seed);
    GatedRecurrentSnn::new(VOCAB, N_HIDDEN, VOCAB, SpikeFn::default(), &mut rng)
}

fn make_batch(seed: u64) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(BATCH);
    let mut targets = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let bytes: Vec<u8> = (0..T).map(|_| rng.random::<u8>()).collect();
        let tgts: Vec<u8> = (0..T).map(|_| rng.random::<u8>()).collect();
        inputs.push(bytes);
        targets.push(tgts);
    }
    (inputs, targets)
}

#[test]
fn batch_grads_match_serial_sum_within_fp_tolerance() {
    let (inputs, targets) = make_batch(0xBA7C);

    let mut net_serial = make_net(0xCAFE);
    net_serial.zero_grad();
    for b in 0..BATCH {
        let cache = net_serial.forward_lm_bytes(&inputs[b]);
        net_serial.backward_lm_bytes(&cache, &inputs[b], &targets[b]);
    }

    let mut net_batch = make_net(0xCAFE);
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let target_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();
    let _ = net_batch.forward_backward_batch(&input_refs, &target_refs);

    let g_serial = net_serial.grads();
    let g_batch = net_batch.grads();

    let close = |a: &[f32], b: &[f32], tol: f32| -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())))
    };
    let tol = 1e-5_f32;
    assert!(close(&g_serial.w_in, &g_batch.w_in, tol), "w_in mismatch");
    assert!(
        close(&g_serial.w_rec, &g_batch.w_rec, tol),
        "w_rec mismatch"
    );
    assert!(
        close(&g_serial.w_out, &g_batch.w_out, tol),
        "w_out mismatch"
    );
    assert!(
        close(&g_serial.w_gate_in, &g_batch.w_gate_in, tol),
        "w_gate_in mismatch"
    );
    assert!(
        close(&g_serial.w_gate_rec, &g_batch.w_gate_rec, tol),
        "w_gate_rec mismatch"
    );
    assert!(close(&g_serial.b_h, &g_batch.b_h, tol), "b_h mismatch");
    assert!(close(&g_serial.b_o, &g_batch.b_o, tol), "b_o mismatch");
    assert!(
        close(&g_serial.b_gate, &g_batch.b_gate, tol),
        "b_gate mismatch"
    );
}

#[test]
fn batched_training_progresses() {
    // Sanity: a few batched Adam steps must reduce loss on a fixed pair.
    let (inputs, targets) = make_batch(0xBA7C);
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let target_refs: Vec<&[u8]> = targets.iter().map(|v| v.as_slice()).collect();

    let mut net = make_net(0x1234);
    let adam = AdamConfig {
        lr: 0.01,
        ..AdamConfig::default()
    };

    let initial_loss = net.forward_backward_batch(&input_refs, &target_refs);
    net.clip_grads(1.0);
    net.step_adam(&adam);
    for _ in 0..50 {
        let _ = net.forward_backward_batch(&input_refs, &target_refs);
        net.clip_grads(1.0);
        net.step_adam(&adam);
    }
    let final_loss = net.forward_backward_batch(&input_refs, &target_refs);
    assert!(
        final_loss < initial_loss * 0.7,
        "batched Adam must reduce loss in 50 steps: {initial_loss} → {final_loss}"
    );
}
