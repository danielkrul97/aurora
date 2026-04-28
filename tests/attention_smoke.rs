//! Sprint 27: stand-alone tests for spiking-attention building blocks.
//!
//! All backward passes are validated with central-difference numerical
//! gradients on small random inputs.

#![allow(clippy::needless_range_loop)]

use aurora::attention::{
    layer_norm_backward, layer_norm_forward, linear_backward, linear_forward,
    scaled_dot_product_attention, scaled_dot_product_attention_backward,
    spike_gated_attention, LayerNorm, LayerNormGrads, LinearGrads, LinearProj,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPS: f32 = 1e-3;
const TOL: f32 = 5e-2;

fn random_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn rel_err(a: f32, b: f32) -> f32 {
    let denom = a.abs().max(b.abs()).max(1e-6);
    (a - b).abs() / denom
}

// --- LayerNorm ----------------------------------------------------------

#[test]
fn layer_norm_forward_normalizes_each_token() {
    let ln = LayerNorm::new(8);
    let mut rng = StdRng::seed_from_u64(1);
    let x = random_vec(4 * 8, &mut rng);
    let (y, _) = layer_norm_forward(&ln, &x);
    for t in 0..4 {
        let row = &y[t * 8..(t + 1) * 8];
        let mean: f32 = row.iter().sum::<f32>() / 8.0;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-4, "mean {mean} not ≈ 0");
        assert!((var - 1.0).abs() < 1e-3, "var {var} not ≈ 1");
    }
}

#[test]
fn layer_norm_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(7);
    let d = 6;
    let t = 3;
    let ln = LayerNorm {
        gamma: random_vec(d, &mut rng),
        beta: random_vec(d, &mut rng),
        eps: 1e-5,
        d,
    };
    let x = random_vec(t * d, &mut rng);
    let upstream = random_vec(t * d, &mut rng);

    let (_y, cache) = layer_norm_forward(&ln, &x);
    let mut grads = LayerNormGrads::default();
    let dx = layer_norm_backward(&ln, &cache, &upstream, &mut grads);

    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = layer_norm_forward(&ln, &xp);
        let (ym, _) = layer_norm_forward(&ln, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "dx[{i}]: analytical {} vs numerical {} (err {err})", dx[i], g);
    }

    // dγ check
    for j in 0..d {
        let mut lp = ln.clone();
        let mut lm = ln.clone();
        lp.gamma[j] += EPS;
        lm.gamma[j] -= EPS;
        let (yp, _) = layer_norm_forward(&lp, &x);
        let (ym, _) = layer_norm_forward(&lm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_gamma[j], g);
        assert!(err < TOL, "dγ[{j}]: analytical {} vs numerical {} (err {err})", grads.d_gamma[j], g);
    }

    // dβ check
    for j in 0..d {
        let mut lp = ln.clone();
        let mut lm = ln.clone();
        lp.beta[j] += EPS;
        lm.beta[j] -= EPS;
        let (yp, _) = layer_norm_forward(&lp, &x);
        let (ym, _) = layer_norm_forward(&lm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_beta[j], g);
        assert!(err < TOL, "dβ[{j}]: analytical {} vs numerical {} (err {err})", grads.d_beta[j], g);
    }
}

// --- Linear projection --------------------------------------------------

#[test]
fn linear_forward_shape_and_zero_input() {
    let mut rng = StdRng::seed_from_u64(2);
    let proj = LinearProj::new(5, 7, &mut rng);
    let x = vec![0.0_f32; 3 * 5];
    let y = linear_forward(&proj, &x);
    assert_eq!(y.len(), 3 * 7);
    assert!(y.iter().all(|&v| v == 0.0));
}

#[test]
fn linear_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(11);
    let proj = LinearProj::new(4, 6, &mut rng);
    let t = 3;
    let x = random_vec(t * 4, &mut rng);
    let upstream = random_vec(t * 6, &mut rng);

    let _y = linear_forward(&proj, &x);
    let mut grads = LinearGrads::default();
    let dx = linear_backward(&proj, &x, &upstream, &mut grads);

    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let g = (dot(&linear_forward(&proj, &xp), &upstream)
            - dot(&linear_forward(&proj, &xm), &upstream))
            / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "dx[{i}]: {} vs {} (err {err})", dx[i], g);
    }

    // Spot-check a few weight gradients (full grid is overkill).
    let n_weights = proj.w.len();
    for &widx in &[0, 1, 5, n_weights - 1] {
        let mut pp = proj.clone();
        let mut pm = proj.clone();
        pp.w[widx] += EPS;
        pm.w[widx] -= EPS;
        let g = (dot(&linear_forward(&pp, &x), &upstream)
            - dot(&linear_forward(&pm, &x), &upstream))
            / (2.0 * EPS);
        let err = rel_err(grads.d_w[widx], g);
        assert!(err < TOL, "dW[{widx}]: {} vs {} (err {err})", grads.d_w[widx], g);
    }
}

// --- Scaled dot-product attention --------------------------------------

#[test]
fn sdpa_softmax_rows_sum_to_one() {
    let mut rng = StdRng::seed_from_u64(3);
    let t_q = 4;
    let t_k = 5;
    let d_k = 8;
    let d_v = 6;
    let q = random_vec(t_q * d_k, &mut rng);
    let k = random_vec(t_k * d_k, &mut rng);
    let v = random_vec(t_k * d_v, &mut rng);
    let (out, cache) = scaled_dot_product_attention(&q, &k, &v, d_k, d_v, None);
    assert_eq!(out.len(), t_q * d_v);
    for i in 0..t_q {
        let row_sum: f32 = cache.probs[i * t_k..(i + 1) * t_k].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sum {row_sum} ≠ 1");
    }
}

#[test]
fn sdpa_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(13);
    let t_q = 3;
    let t_k = 4;
    let d_k = 5;
    let d_v = 4;
    let q = random_vec(t_q * d_k, &mut rng);
    let k = random_vec(t_k * d_k, &mut rng);
    let v = random_vec(t_k * d_v, &mut rng);
    let upstream = random_vec(t_q * d_v, &mut rng);

    let (_out, cache) = scaled_dot_product_attention(&q, &k, &v, d_k, d_v, None);
    let (dq, dk, dv) =
        scaled_dot_product_attention_backward(&q, &k, &v, &cache, &upstream);

    let f = |q: &[f32], k: &[f32], v: &[f32]| {
        let (o, _) = scaled_dot_product_attention(q, k, v, d_k, d_v, None);
        dot(&o, &upstream)
    };

    for (label, vec, dvec) in [("q", &q, &dq), ("k", &k, &dk), ("v", &v, &dv)] {
        for i in 0..vec.len() {
            let mut vp = vec.clone();
            let mut vm = vec.clone();
            vp[i] += EPS;
            vm[i] -= EPS;
            let (a, b) = match label {
                "q" => (f(&vp, &k, &v), f(&vm, &k, &v)),
                "k" => (f(&q, &vp, &v), f(&q, &vm, &v)),
                _ => (f(&q, &k, &vp), f(&q, &k, &vm)),
            };
            let g = (a - b) / (2.0 * EPS);
            let err = rel_err(dvec[i], g);
            assert!(err < TOL, "d{label}[{i}]: {} vs {} (err {err})", dvec[i], g);
        }
    }
}

// --- Spike-gated attention ---------------------------------------------

#[test]
fn spike_gated_all_ones_equals_plain() {
    let mut rng = StdRng::seed_from_u64(4);
    let t_q = 3;
    let t_k = 4;
    let d_k = 5;
    let d_v = 5;
    let q = random_vec(t_q * d_k, &mut rng);
    let k = random_vec(t_k * d_k, &mut rng);
    let v = random_vec(t_k * d_v, &mut rng);
    let mask = vec![1.0_f32; t_k];

    let (out_plain, _) = scaled_dot_product_attention(&q, &k, &v, d_k, d_v, None);
    let (out_gated, _) = spike_gated_attention(&q, &k, &v, d_k, d_v, &mask);
    for (a, b) in out_plain.iter().zip(out_gated.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn spike_gated_zero_masks_dont_contribute() {
    let mut rng = StdRng::seed_from_u64(5);
    let t_q = 2;
    let t_k = 4;
    let d_k = 4;
    let d_v = 3;
    let q = random_vec(t_q * d_k, &mut rng);
    let k = random_vec(t_k * d_k, &mut rng);
    let v = random_vec(t_k * d_v, &mut rng);
    // Mask out positions 1 and 3 entirely.
    let mask = vec![1.0, 0.0, 1.0, 0.0];

    let (_out, cache) = spike_gated_attention(&q, &k, &v, d_k, d_v, &mask);
    for i in 0..t_q {
        for j in 0..t_k {
            if mask[j] == 0.0 {
                let p = cache.probs[i * t_k + j];
                assert!(p.abs() < 1e-12, "probs[{i},{j}] should be 0, got {p}");
            }
        }
    }

    // Output should be identical to plain SDPA over only the unmasked
    // keys/values — verify by reconstructing the smaller problem.
    let kept: Vec<usize> = (0..t_k).filter(|&j| mask[j] != 0.0).collect();
    let mut k_small = vec![0.0_f32; kept.len() * d_k];
    let mut v_small = vec![0.0_f32; kept.len() * d_v];
    for (new_j, &orig_j) in kept.iter().enumerate() {
        k_small[new_j * d_k..(new_j + 1) * d_k]
            .copy_from_slice(&k[orig_j * d_k..(orig_j + 1) * d_k]);
        v_small[new_j * d_v..(new_j + 1) * d_v]
            .copy_from_slice(&v[orig_j * d_v..(orig_j + 1) * d_v]);
    }
    let (out_full, _) = spike_gated_attention(&q, &k, &v, d_k, d_v, &mask);
    let (out_small, _) =
        scaled_dot_product_attention(&q, &k_small, &v_small, d_k, d_v, None);
    for (a, b) in out_full.iter().zip(out_small.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}
