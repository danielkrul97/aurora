//! Stateful TBPTT carries hidden state across chunk boundaries with
//! gradients detached. Two checks:
//!
//! 1. `forward_lm_bytes_stateful` with zero state == `forward_lm_bytes`
//!    (bit-identical, by construction).
//! 2. State written back at end of forward equals what subsequent
//!    `cache.initial_h_*` reflects on the next call.

use aurora::snn_gated::GatedRecurrentSnn;
use aurora::surrogate::SpikeFn;
use aurora::text_io::VOCAB;
use rand::rngs::StdRng;
use rand::SeedableRng;

const N_HIDDEN: usize = 64;

#[test]
fn stateful_with_zero_state_matches_stateless() {
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let net = GatedRecurrentSnn::new(VOCAB, N_HIDDEN, VOCAB, SpikeFn::default(), &mut rng);

    let bytes: Vec<u8> = (0..32).map(|i| (i * 7 % 251) as u8).collect();

    let cache_stateless = net.forward_lm_bytes(&bytes);
    let mut state = net.fresh_state();
    let cache_stateful = net.forward_lm_bytes_stateful(&bytes, &mut state);

    for t in 0..bytes.len() {
        assert_eq!(cache_stateless.h_v_pre[t], cache_stateful.h_v_pre[t]);
        assert_eq!(cache_stateless.h_spike[t], cache_stateful.h_spike[t]);
        assert_eq!(cache_stateless.logits[t], cache_stateful.logits[t]);
    }
}

#[test]
fn stateful_state_persists_across_chunks() {
    // Smoke test that stateful API actually carries non-trivial state
    // forward. After feeding an active chunk, state.h_v_after must move
    // away from zero; feeding a second chunk reads that state and the
    // cache.initial_h_* fields must reflect it.

    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let net = GatedRecurrentSnn::new(VOCAB, N_HIDDEN, VOCAB, SpikeFn::default(), &mut rng);

    let chunk_a: Vec<u8> = (0..16).map(|i| (i * 13 + 7) as u8).collect();
    let chunk_b: Vec<u8> = (0..16).map(|i| (i * 17 + 3) as u8).collect();

    let mut state = net.fresh_state();
    let _ = net.forward_lm_bytes_stateful(&chunk_a, &mut state);
    let after_a_h_v_after = state.h_v_after.clone();
    let after_a_h_spike = state.h_spike.clone();

    let nonzero_h_v: usize = after_a_h_v_after.iter().filter(|&&v| v != 0.0).count();
    let nonzero_spike: usize = after_a_h_spike.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero_h_v > 0,
        "after a non-empty chunk, h_v_after must have non-zero entries"
    );
    let _ = nonzero_spike; // spikes can be all zero by chance; h_v_after is the stronger check

    let cache_b = net.forward_lm_bytes_stateful(&chunk_b, &mut state);
    assert_eq!(
        cache_b.initial_h_v_after, after_a_h_v_after,
        "cache.initial_h_v_after must equal incoming state for backward use"
    );
    assert_eq!(
        cache_b.initial_h_spike, after_a_h_spike,
        "cache.initial_h_spike must equal incoming state for backward use"
    );
}
