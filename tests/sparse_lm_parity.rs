#![allow(clippy::cast_possible_truncation)]

//! Sparse `forward_lm_bytes` / `backward_lm_bytes` must produce
//! bit-for-bit identical results to the dense `forward_lm` /
//! `backward_lm` when inputs are one-hot and spikes are binary —
//! IEEE 754 guarantees `x + 0.0 == x`, so skipping zero-valued summands
//! in the same iteration order yields identical partial sums.

use aurora::snn_recurrent::RecurrentSurrogateSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::{one_hot, VOCAB};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_HIDDEN: usize = 32;
const T: usize = 24;

fn make_net() -> RecurrentSurrogateSnn {
    let mut rng = StdRng::seed_from_u64(0xCAFE_F00D);
    RecurrentSurrogateSnn::new(
        VOCAB,
        N_HIDDEN,
        VOCAB,
        SpikeFn::default(),
        5.0,
        1.0,
        &mut rng,
    )
}

#[test]
fn sparse_forward_matches_dense_bitwise() {
    let mut rng = StdRng::seed_from_u64(0xABCD);
    let bytes: Vec<u8> = (0..T).map(|_| rng.random::<u8>()).collect();
    let dense_inputs: Vec<Vec<f32>> = bytes.iter().map(|&b| one_hot(b)).collect();

    let net = make_net();
    let cache_dense = net.forward_lm(&dense_inputs);
    let cache_sparse = net.forward_lm_bytes(&bytes);

    for t in 0..T {
        assert_eq!(
            cache_dense.h_v_pre[t], cache_sparse.h_v_pre[t],
            "h_v_pre mismatch at t={t}"
        );
        assert_eq!(
            cache_dense.h_spike[t], cache_sparse.h_spike[t],
            "h_spike mismatch at t={t}"
        );
        assert_eq!(
            cache_dense.h_v_after[t], cache_sparse.h_v_after[t],
            "h_v_after mismatch at t={t}"
        );
        assert_eq!(
            cache_dense.logits[t], cache_sparse.logits[t],
            "logits mismatch at t={t}"
        );
    }
}

#[test]
fn sparse_backward_matches_dense_bitwise() {
    let mut rng = StdRng::seed_from_u64(0xABCD);
    let bytes: Vec<u8> = (0..T).map(|_| rng.random::<u8>()).collect();
    let targets: Vec<u8> = (0..T).map(|_| rng.random::<u8>()).collect();
    let dense_inputs: Vec<Vec<f32>> = bytes.iter().map(|&b| one_hot(b)).collect();

    let mut net_dense = make_net();
    let mut net_sparse = make_net();

    let cache_d = net_dense.forward_lm(&dense_inputs);
    let cache_s = net_sparse.forward_lm_bytes(&bytes);
    net_dense.zero_grad();
    net_sparse.zero_grad();
    let loss_d = net_dense.backward_lm(&cache_d, &dense_inputs, &targets);
    let loss_s = net_sparse.backward_lm_bytes(&cache_s, &bytes, &targets);
    assert_eq!(loss_d, loss_s, "loss mismatch");

    assert_eq!(
        net_dense.grad_w_in(),
        net_sparse.grad_w_in(),
        "grad_w_in mismatch"
    );
    assert_eq!(
        net_dense.grad_w_rec(),
        net_sparse.grad_w_rec(),
        "grad_w_rec mismatch"
    );
    assert_eq!(
        net_dense.grad_w_out(),
        net_sparse.grad_w_out(),
        "grad_w_out mismatch"
    );
    assert_eq!(
        net_dense.grad_b_h(),
        net_sparse.grad_b_h(),
        "grad_b_h mismatch"
    );
    assert_eq!(
        net_dense.grad_b_o(),
        net_sparse.grad_b_o(),
        "grad_b_o mismatch"
    );
}
