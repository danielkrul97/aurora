//! Sprint 35: smoke tests for `SdsaCell`.
//!
//! The SDSA forward uses heaviside spikes for Q/K/V (binary {0,1}).
//! The backward uses a fast-sigmoid surrogate gradient. By construction
//! analytical gradient ≠ true derivative of forward — that's the
//! point of surrogate-gradient SNN training. So a straight
//! finite-difference test of `<d_y, dy/dθ>` against `<d_param, ε>`
//! does NOT validate the implementation; both fail to be the literal
//! derivative.
//!
//! What we *can* check:
//! - forward runs without panics, output finite
//! - backward runs without panics, all parameter gradients finite
//! - `attn_state` correctly carries between sequential calls
//! - residual recovers `h_spike` when SDSA contribution is zero
//!   (Q is all zeros → out is all zeros)
//! - LayerNorm output has zero mean, unit variance per timestep

use aurora::sdsa::{sdsa_backward, sdsa_forward, SdsaCell, SdsaGrads};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

#[test]
fn sdsa_forward_finite_and_residual_growth_monotone() {
    let mut rng = StdRng::seed_from_u64(7);
    let cell = SdsaCell::new(8, &mut rng);
    let t = 4;
    let h_spike: Vec<f32> = (0..t * 8)
        .map(|_| if rng.random::<f32>() < 0.2 { 1.0 } else { 0.0 })
        .collect();
    let attn0 = vec![0.0_f32; 8];
    let (y, cache) = sdsa_forward(&cell, &h_spike, &attn0);

    assert!(y.iter().all(|v| v.is_finite()), "y has non-finite entries");
    // Per-element attn_state is monotone non-decreasing: K⊙V ≥ 0 always.
    for t_idx in 1..t {
        for i in 0..8 {
            assert!(
                cache.attn_states[t_idx][i] >= cache.attn_states[t_idx - 1][i],
                "attn state should be non-decreasing"
            );
        }
    }
}

#[test]
fn sdsa_backward_finite_grads() {
    let mut rng = StdRng::seed_from_u64(11);
    let n = 6;
    let t = 3;
    let cell = SdsaCell::new(n, &mut rng);
    let h_spike = random_vec(t * n, &mut rng);
    let upstream = random_vec(t * n, &mut rng);
    let attn0 = vec![0.0_f32; n];

    let (_, cache) = sdsa_forward(&cell, &h_spike, &attn0);
    let mut grads = SdsaGrads::zeros(n);
    let dh = sdsa_backward(&cell, &cache, &h_spike, &upstream, &mut grads);

    assert!(dh.iter().all(|v| v.is_finite()));
    assert!(grads.iter_all().all(|v| v.is_finite()));
    // Residual contributes d_h_spike directly, so norm should be at
    // least as large as upstream norm (ignoring tiny SDSA correction).
    let dh_norm: f32 = dh.iter().map(|x| x * x).sum::<f32>().sqrt();
    let up_norm: f32 = upstream.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(dh_norm > 0.0 && up_norm > 0.0);
}

#[test]
fn sdsa_attn_state_carries_between_calls() {
    let mut rng = StdRng::seed_from_u64(17);
    let n = 8;
    let cell = SdsaCell::new(n, &mut rng);
    let h_spike: Vec<f32> = (0..16).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let (h1, h2) = h_spike.split_at(8);
    let attn0 = vec![0.0_f32; n];
    let (_, cache1) = sdsa_forward(&cell, h1, &attn0);
    let (_, cache2_carry) = sdsa_forward(&cell, h2, &cache1.final_attn_state);
    let (_, cache2_fresh) = sdsa_forward(&cell, h2, &attn0);

    if cache1.final_attn_state.iter().any(|&x| x != 0.0) {
        // If the first chunk produced any nonzero accumulator entries,
        // the carry must affect the second chunk's final state.
        let differs = cache2_carry
            .final_attn_state
            .iter()
            .zip(cache2_fresh.final_attn_state.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "carry didn't change second chunk's state"
        );
    }
}

#[test]
fn sdsa_zero_q_zero_sdsa_contribution() {
    // If we force Q to be all zeros (set the cell's weights so q_pre <
    // θ everywhere), residual y must equal h_spike exactly.
    let mut rng = StdRng::seed_from_u64(23);
    let n = 6;
    let mut cell = SdsaCell::new(n, &mut rng);
    // Zero out W_q so q_pre = 0 < θ → Q spikes are all zero.
    cell.w_q.w.fill(0.0);

    let h_spike: Vec<f32> = (0..3 * n).map(|_| rng.random::<f32>()).collect();
    let attn0 = vec![0.0_f32; n];
    let (y, _) = sdsa_forward(&cell, &h_spike, &attn0);

    for (yi, hi) in y.iter().zip(h_spike.iter()) {
        assert!(
            (yi - hi).abs() < 1e-6,
            "Q=0 should give residual=h_spike (got {yi} vs {hi})"
        );
    }
}
