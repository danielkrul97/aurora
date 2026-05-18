//! Sprint 52: numerical gradient checks for transformer primitives.
//!
//! Central-difference epsilon = 1e-3, tolerance = 5e-2 (relative).
//! Same convention as the old `attention_smoke.rs` from the SNN era.

#![allow(clippy::needless_range_loop)]

use aurora::attention::{LinearGrads, LinearProj};
use aurora::linalg::{gelu, gelu_backward};
use aurora::transformer::attention::{
    causal_mha_backward, causal_mha_forward, MhaGrads, MhaWeights,
};
use aurora::transformer::embedding::{
    embed_backward, embed_forward, tied_lm_head_backward, tied_lm_head_forward, EmbedGrads,
    EmbedWeights,
};
use aurora::transformer::ffn::{ffn_backward, ffn_forward, FfnGrads, FfnWeights};
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

// ---------- 1. Causal MHA ----------

#[test]
fn mha_causal_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xC0DE);
    let d_model = 8;
    let n_head = 2;
    let t = 4;

    let w = MhaWeights::new(d_model, n_head, 2, &mut rng);
    let x = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = causal_mha_forward(&w, &x);
    let mut grads = MhaGrads::zeros(d_model);
    let dx = causal_mha_backward(&w, &cache, &upstream, &mut grads);

    // dx parity
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = causal_mha_forward(&w, &xp);
        let (ym, _) = causal_mha_forward(&w, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }

    // dW_q parity on a few elements (full check is O(D^2) × O(D^2) — overkill).
    for i in [0, 5, 17, 31] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_q.w[i] += EPS;
        wm.w_q.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_q.d_w[i], g);
        assert!(err < TOL, "dW_q[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_q.d_w[i], g);
    }

    // dW_o parity sample
    for i in [0, 7, 23] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_o.w[i] += EPS;
        wm.w_o.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_o.d_w[i], g);
        assert!(err < TOL, "dW_o[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_o.d_w[i], g);
    }
}

#[test]
fn mha_causal_mask_blocks_future() {
    // Verify that swapping out a future token (j > i) does not change
    // output at position i. This is a structural check, not a gradient
    // check.
    let mut rng = StdRng::seed_from_u64(1);
    let d_model = 8;
    let n_head = 2;
    let t = 4;
    let w = MhaWeights::new(d_model, n_head, 2, &mut rng);
    let mut x = random_vec(t * d_model, &mut rng);
    let (y_before, _) = causal_mha_forward(&w, &x);

    // perturb token at position t-1 (future relative to t-2, t-3, etc.)
    for c in 0..d_model {
        x[(t - 1) * d_model + c] += 5.0;
    }
    let (y_after, _) = causal_mha_forward(&w, &x);

    // Positions 0..t-1 must be unchanged.
    for ti in 0..t - 1 {
        for c in 0..d_model {
            let i = ti * d_model + c;
            assert!(
                (y_before[i] - y_after[i]).abs() < 1e-5,
                "causal mask leaked future into y[{ti}, {c}]: before {} vs after {}",
                y_before[i], y_after[i]
            );
        }
    }
}

// ---------- 2. GELU ----------

#[test]
fn gelu_backward_matches_numerical() {
    let xs = [-2.0_f32, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5];
    for &x in &xs {
        let g_analytic = gelu_backward(x);
        let g_numeric = (gelu(x + EPS) - gelu(x - EPS)) / (2.0 * EPS);
        let err = rel_err(g_analytic, g_numeric);
        assert!(
            err < TOL,
            "gelu_backward({x}): analytic {} vs numeric {} (err {err})",
            g_analytic, g_numeric
        );
    }
}

// ---------- 3. FFN ----------

#[test]
fn ffn_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x7E57);
    let d_model = 6;
    let d_ff = 12;
    let t = 3;

    let w = FfnWeights::new(d_model, d_ff, 2, &mut rng);
    let x = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = ffn_forward(&w, &x);
    let mut grads = FfnGrads::zeros(d_model, d_ff);
    let dx = ffn_backward(&w, &cache, &upstream, &mut grads);

    // dx parity
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = ffn_forward(&w, &xp);
        let (ym, _) = ffn_forward(&w, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "ffn dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }

    // d_w_fc + d_w_proj sample
    for i in [0, 5, 17, 35] {
        if i >= grads.d_w_fc.d_w.len() {
            continue;
        }
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_fc.w[i] += EPS;
        wm.w_fc.w[i] -= EPS;
        let (yp, _) = ffn_forward(&wp, &x);
        let (ym, _) = ffn_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_fc.d_w[i], g);
        assert!(err < TOL, "dW_fc[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_fc.d_w[i], g);
    }
    for i in [0, 8, 23] {
        if i >= grads.d_w_proj.d_w.len() {
            continue;
        }
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_proj.w[i] += EPS;
        wm.w_proj.w[i] -= EPS;
        let (yp, _) = ffn_forward(&wp, &x);
        let (ym, _) = ffn_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_proj.d_w[i], g);
        assert!(err < TOL, "dW_proj[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_proj.d_w[i], g);
    }
}

// ---------- 4. Token + positional embedding ----------

#[test]
fn token_pos_embed_backward_scatters_correctly() {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let vocab = 16;
    let d_model = 6;
    let seq_len_max = 8;

    let w = EmbedWeights::new(vocab, d_model, seq_len_max, &mut rng);
    let tokens: Vec<u32> = vec![3, 1, 7, 3, 11];
    let t = tokens.len();
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = embed_forward(&w, &tokens);
    let mut grads = EmbedGrads::zeros(vocab, d_model, seq_len_max);
    embed_backward(&w, &cache, &upstream, &mut grads);

    // Token 3 appears at positions 0 and 3, so its gradient is the
    // sum of `upstream[0..]` and `upstream[3..]`.
    for c in 0..d_model {
        let expected = upstream[0 * d_model + c] + upstream[3 * d_model + c];
        let got = grads.d_w_tok[3 * d_model + c];
        let err = (expected - got).abs();
        assert!(err < 1e-5, "d_w_tok[3, {c}] = {got}, expected {expected}");
    }
    // Token 1 (position 1) gradient = upstream[1*D..]
    for c in 0..d_model {
        let expected = upstream[1 * d_model + c];
        let got = grads.d_w_tok[1 * d_model + c];
        assert!((expected - got).abs() < 1e-5, "d_w_tok[1, {c}] mismatch");
    }
    // Position 0 gradient = upstream[0*D..] (token 3 at pos 0)
    for c in 0..d_model {
        let expected = upstream[0 * d_model + c];
        let got = grads.d_w_pos[0 * d_model + c];
        assert!((expected - got).abs() < 1e-5, "d_w_pos[0, {c}] mismatch");
    }
    // Unused position 7 must stay zero.
    for c in 0..d_model {
        assert!(grads.d_w_pos[7 * d_model + c].abs() < 1e-6);
    }
}

// ---------- 5. Tied LM head ----------

#[test]
fn tied_lm_head_gradient_accumulates_through_both_paths() {
    let mut rng = StdRng::seed_from_u64(0xFADE);
    let vocab = 12;
    let d_model = 6;
    let seq_len_max = 8;
    let t = 3;

    let w = EmbedWeights::new(vocab, d_model, seq_len_max, &mut rng);
    let h = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * vocab, &mut rng);

    let logits = tied_lm_head_forward(&h, &w.w_tok, vocab, d_model);
    assert_eq!(logits.len(), t * vocab);

    let mut grads = EmbedGrads::zeros(vocab, d_model, seq_len_max);
    let dh = tied_lm_head_backward(&h, &w.w_tok, &upstream, vocab, d_model, &mut grads);

    // dh parity
    for i in 0..h.len() {
        let mut hp = h.clone();
        let mut hm = h.clone();
        hp[i] += EPS;
        hm[i] -= EPS;
        let lp = tied_lm_head_forward(&hp, &w.w_tok, vocab, d_model);
        let lm = tied_lm_head_forward(&hm, &w.w_tok, vocab, d_model);
        let g = (dot(&lp, &upstream) - dot(&lm, &upstream)) / (2.0 * EPS);
        let err = rel_err(dh[i], g);
        assert!(err < TOL, "dh[{i}]: analytic {} vs numeric {} (err {err})", dh[i], g);
    }

    // d_w_tok parity: sample a few elements.
    let mut w_perturb = w.clone();
    for i in [0, 7, 23, 41] {
        if i >= w_perturb.w_tok.len() {
            continue;
        }
        let saved = w_perturb.w_tok[i];
        w_perturb.w_tok[i] = saved + EPS;
        let lp = tied_lm_head_forward(&h, &w_perturb.w_tok, vocab, d_model);
        w_perturb.w_tok[i] = saved - EPS;
        let lm = tied_lm_head_forward(&h, &w_perturb.w_tok, vocab, d_model);
        w_perturb.w_tok[i] = saved;
        let g = (dot(&lp, &upstream) - dot(&lm, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_tok[i], g);
        assert!(
            err < TOL,
            "d_w_tok[{i}]: analytic {} vs numeric {} (err {err})",
            grads.d_w_tok[i], g
        );
    }

    // Crucial: confirm both forward paths can write into the same buffer.
    // Run embed_backward then tied_lm_head_backward; gradient on shared
    // tokens accumulates additively rather than overwriting.
    let mut grads2 = EmbedGrads::zeros(vocab, d_model, seq_len_max);
    let tokens: Vec<u32> = vec![5, 3, 5];
    let dy_embed: Vec<f32> = random_vec(t * d_model, &mut rng);
    let (_, cache) = embed_forward(&w, &tokens);
    embed_backward(&w, &cache, &dy_embed, &mut grads2);
    let before_tok5 = grads2.d_w_tok[5 * d_model];
    tied_lm_head_backward(&h, &w.w_tok, &upstream, vocab, d_model, &mut grads2);
    let after_tok5 = grads2.d_w_tok[5 * d_model];
    assert!(
        (after_tok5 - before_tok5).abs() > 1e-6,
        "tied lm head backward did not accumulate into d_w_tok[5]"
    );
}

// Silence dead_code on Linear* re-exports if no test uses them yet.
#[allow(dead_code)]
fn _types_used() {
    let _: Option<LinearProj> = None;
    let _: Option<LinearGrads> = None;
}
