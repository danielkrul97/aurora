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

    let w = MhaWeights::new(d_model, n_head, &mut rng);
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
fn mha_rope_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x80BE);
    let d_model = 8;
    let n_head = 2;
    let t = 4;

    let mut w = MhaWeights::new(d_model, n_head, &mut rng);
    w.rope = true;
    let x = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = causal_mha_forward(&w, &x);
    let mut grads = MhaGrads::zeros(d_model);
    let dx = causal_mha_backward(&w, &cache, &upstream, &mut grads);

    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = causal_mha_forward(&w, &xp);
        let (ym, _) = causal_mha_forward(&w, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "rope dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }

    for i in [0, 5, 17, 31] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_q.w[i] += EPS;
        wm.w_q.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_q.d_w[i], g);
        assert!(err < TOL, "rope dW_q[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_q.d_w[i], g);
    }

    for i in [0, 5, 17, 31] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_k.w[i] += EPS;
        wm.w_k.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_k.d_w[i], g);
        assert!(err < TOL, "rope dW_k[{i}]: analytic {} vs numeric {} (err {err})", grads.d_w_k.d_w[i], g);
    }
}

#[test]
fn rope_relative_shift_invariance() {
    // RoPE's defining property: attention scores depend only on the
    // relative offset, so shifting the absolute window start must not
    // change q·k for a fixed pair of (unrotated) vectors.
    use aurora::attention::rope_rotate;
    let mut rng = StdRng::seed_from_u64(7);
    let n_head = 2;
    let d_head = 4;
    let d = n_head * d_head;
    let q0 = random_vec(d, &mut rng);
    let k0 = random_vec(d, &mut rng);

    let score = |offset: usize| -> f32 {
        // q at position offset+3, k at position offset+1.
        let mut q = q0.clone();
        let mut k = k0.clone();
        rope_rotate(&mut q, n_head, d_head, offset + 3, false);
        rope_rotate(&mut k, n_head, d_head, offset + 1, false);
        dot(&q, &k)
    };
    let s0 = score(0);
    for offset in [1, 10, 100] {
        let s = score(offset);
        assert!(
            (s - s0).abs() < 1e-4,
            "rope score not shift-invariant: offset {offset} gives {s} vs {s0}"
        );
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
    let w = MhaWeights::new(d_model, n_head, &mut rng);
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

    let w = FfnWeights::new(d_model, d_ff, 2, false, &mut rng);
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

// ---------- 6. Tier-1 pack: RMSNorm / ReLU² / QK-norm / softcap+untied ----------

#[test]
fn rmsnorm_backward_matches_numerical() {
    use aurora::attention::{layer_norm_backward, layer_norm_forward, LayerNorm, LayerNormGrads};
    let mut rng = StdRng::seed_from_u64(0x4452);
    let d = 8;
    let t = 3;
    let mut ln = LayerNorm::new(d);
    ln.rms = true;
    ln.gamma = random_vec(d, &mut rng).iter().map(|v| 1.0 + 0.3 * v).collect();

    let x = random_vec(t * d, &mut rng);
    let upstream = random_vec(t * d, &mut rng);

    let (_y, cache) = layer_norm_forward(&ln, &x);
    let mut grads = LayerNormGrads::zeros(d);
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
        assert!(err < TOL, "rms dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }
    for i in 0..d {
        let mut lp = ln.clone();
        let mut lm = ln.clone();
        lp.gamma[i] += EPS;
        lm.gamma[i] -= EPS;
        let (yp, _) = layer_norm_forward(&lp, &x);
        let (ym, _) = layer_norm_forward(&lm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_gamma[i], g);
        assert!(err < TOL, "rms d_gamma[{i}] err {err}");
    }
    // β is unused in RMS mode and must receive no gradient.
    assert!(grads.d_beta.iter().all(|&g| g == 0.0), "rms mode gradded beta");
}

#[test]
fn relu2_ffn_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x4E12);
    let d_model = 6;
    let d_ff = 12;
    let t = 3;

    let mut w = FfnWeights::new(d_model, d_ff, 2, false, &mut rng);
    w.relu2 = true;
    let x = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = ffn_forward(&w, &x);
    let mut grads = FfnGrads::zeros(d_model, d_ff);
    let dx = ffn_backward(&w, &cache, &upstream, &mut grads);

    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = ffn_forward(&w, &xp);
        let (ym, _) = ffn_forward(&w, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "relu2 ffn dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }
    for i in [0, 5, 17, 35] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_fc.w[i] += EPS;
        wm.w_fc.w[i] -= EPS;
        let (yp, _) = ffn_forward(&wp, &x);
        let (ym, _) = ffn_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_fc.d_w[i], g);
        assert!(err < TOL, "relu2 dW_fc[{i}] err {err}");
    }
}

#[test]
fn mha_qk_norm_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x96B0);
    let d_model = 8;
    let n_head = 2;
    let t = 4;

    let mut w = MhaWeights::new(d_model, n_head, &mut rng);
    w.qk_norm = true;
    w.rope = true;
    let x = random_vec(t * d_model, &mut rng);
    let upstream = random_vec(t * d_model, &mut rng);

    let (_y, cache) = causal_mha_forward(&w, &x);
    let mut grads = MhaGrads::zeros(d_model);
    let dx = causal_mha_backward(&w, &cache, &upstream, &mut grads);

    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let (yp, _) = causal_mha_forward(&w, &xp);
        let (ym, _) = causal_mha_forward(&w, &xm);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(dx[i], g);
        assert!(err < TOL, "qk_norm dx[{i}]: analytic {} vs numeric {} (err {err})", dx[i], g);
    }
    for i in [0, 5, 17, 31] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_q.w[i] += EPS;
        wm.w_q.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_q.d_w[i], g);
        assert!(err < TOL, "qk_norm dW_q[{i}] err {err}");
    }
    for i in [0, 5, 17, 31] {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp.w_k.w[i] += EPS;
        wm.w_k.w[i] -= EPS;
        let (yp, _) = causal_mha_forward(&wp, &x);
        let (ym, _) = causal_mha_forward(&wm, &x);
        let g = (dot(&yp, &upstream) - dot(&ym, &upstream)) / (2.0 * EPS);
        let err = rel_err(grads.d_w_k.d_w[i], g);
        assert!(err < TOL, "qk_norm dW_k[{i}] err {err}");
    }
}

#[test]
fn softcap_untied_head_model_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x50F7);
    let cfg = TransformerConfig {
        vocab: 12,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        // Low cap so the tanh is far from its linear regime.
        softcap: 2.0,
        untied_head: true,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // The head is zero-initialized in production; randomize it so the
    // softcap nonlinearity and head gradient are actually exercised.
    model.w_head = Some(random_vec(cfg.vocab * cfg.d_model, &mut rng));

    let tokens: Vec<u32> = vec![3, 7, 1, 9, 4];
    let targets: Vec<u32> = vec![7, 1, 9, 4, 2];

    let ce_loss = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    let loss = model.backward(&cache, &logits, &targets, &mut grads);
    assert!((loss - ce_loss(&model)).abs() < 1e-4, "backward loss != forward CE");

    // d_w_head parity (head-only path).
    for i in [0, 13, 41, 77] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.w_head.as_mut().unwrap()[i] += EPS;
        mm.w_head.as_mut().unwrap()[i] -= EPS;
        let g = (ce_loss(&mp) - ce_loss(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.embed.d_w_head[i], g);
        assert!(err < TOL, "d_w_head[{i}]: analytic {} vs numeric {} (err {err})", grads.embed.d_w_head[i], g);
    }

    // d_w_tok parity — with the head untied this is the embedding path only.
    for i in [0, 25, 60] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.embed.w_tok[i] += EPS;
        mm.embed.w_tok[i] -= EPS;
        let g = (ce_loss(&mp) - ce_loss(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.embed.d_w_tok[i], g);
        assert!(err < TOL, "untied d_w_tok[{i}]: analytic {} vs numeric {} (err {err})", grads.embed.d_w_tok[i], g);
    }
}

#[test]
fn zero_ffn_proj_zeroes_only_proj_init() {
    use aurora::transformer::{DecoderTransformer, TransformerConfig};
    let mut rng = StdRng::seed_from_u64(0x0F);
    let cfg = TransformerConfig {
        vocab: 8,
        d_model: 8,
        n_layer: 2,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        zero_ffn_proj: true,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg, &mut rng);
    for blk in &model.blocks {
        assert!(blk.ffn.w_proj.w.iter().all(|&v| v == 0.0), "w_proj not zeroed");
        assert!(blk.ffn.w_fc.w.iter().any(|&v| v != 0.0), "w_fc unexpectedly zeroed");
    }
}

#[test]
fn z_loss_model_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x2105);
    let cfg = TransformerConfig {
        vocab: 12,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        z_loss: 0.3,
        // Exercise the z-loss × softcap chain factor together.
        softcap: 2.0,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = vec![3, 7, 1, 9, 4];
    let targets: Vec<u32> = vec![7, 1, 9, 4, 2];

    // Objective must match backward(): mean CE + z_loss·mean(logsumexp²),
    // both over the SAME post-softcap logits the model emits.
    let obj = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            let z = max + sum.ln();
            total += -(row[targets[ti] as usize] - z) + m.config.z_loss * z * z;
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    let loss = model.backward(&cache, &logits, &targets, &mut grads);
    assert!(
        (loss - obj(&model)).abs() < 1e-4,
        "backward loss {loss} != CE+z-loss objective {}",
        obj(&model)
    );

    for i in [0, 25, 60] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.embed.w_tok[i] += EPS;
        mm.embed.w_tok[i] -= EPS;
        let g = (obj(&mp) - obj(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.embed.d_w_tok[i], g);
        assert!(err < TOL, "z-loss d_w_tok[{i}]: analytic {} vs numeric {g} (err {err})", grads.embed.d_w_tok[i]);
    }
    for i in [0, 5, 17] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].mha.w_q.w[i] += EPS;
        mm.blocks[0].mha.w_q.w[i] -= EPS;
        let g = (obj(&mp) - obj(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.blocks[0].d_mha.d_w_q.d_w[i], g);
        assert!(err < TOL, "z-loss d_w_q[{i}]: analytic {} vs numeric {g} (err {err})", grads.blocks[0].d_mha.d_w_q.d_w[i]);
    }
}

#[test]
fn head_bias_model_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0xB1A5);
    let cfg = TransformerConfig {
        vocab: 12,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        head_bias: true,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Randomize the bias so its gradient path is actually exercised.
    model.head_bias = Some(random_vec(cfg.vocab, &mut rng));

    let tokens: Vec<u32> = vec![3, 7, 1, 9, 4];
    let targets: Vec<u32> = vec![7, 1, 9, 4, 2];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    let loss = model.backward(&cache, &logits, &targets, &mut grads);
    assert!((loss - ce(&model)).abs() < 1e-4, "backward loss != forward CE");

    for i in [0, 4, 9, 11] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.head_bias.as_mut().unwrap()[i] += EPS;
        mm.head_bias.as_mut().unwrap()[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.embed.d_head_bias[i], g);
        assert!(err < TOL, "d_head_bias[{i}]: analytic {} vs numeric {g} (err {err})", grads.embed.d_head_bias[i]);
    }
    for i in [0, 25, 60] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.embed.w_tok[i] += EPS;
        mm.embed.w_tok[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.embed.d_w_tok[i], g);
        assert!(err < TOL, "head_bias d_w_tok[{i}]: analytic {} vs numeric {g} (err {err})", grads.embed.d_w_tok[i]);
    }
}

#[test]
fn parallel_block_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x9A2A);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 2,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        parallel_block: true,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    // FFN reads x in parallel mode — its grad path differs from sequential.
    for i in [0, 7, 31] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[1].ffn.w_fc.w[i] += EPS;
        mm.blocks[1].ffn.w_fc.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.blocks[1].d_ffn.d_w_fc.d_w[i], g);
        assert!(err < TOL, "parallel d_w_fc[{i}]: analytic {} vs numeric {g} (err {err})", grads.blocks[1].d_ffn.d_w_fc.d_w[i]);
    }
    for i in [0, 3, 7] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].ln2.gamma[i] += EPS;
        mm.blocks[0].ln2.gamma[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        // Abs fallback: LN-γ grads here are O(1e-4); f32 central differences
        // lose ~1 digit at that scale, so a relative check alone is unstable.
        let a = grads.blocks[0].d_ln2.d_gamma[i];
        let err = rel_err(a, g);
        assert!(err < TOL || (a - g).abs() < 2e-4, "parallel d_ln2_gamma[{i}]: analytic {a} vs numeric {g} (err {err})");
    }
    for i in [0, 13, 40] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[1].mha.w_q.w[i] += EPS;
        mm.blocks[1].mha.w_q.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let err = rel_err(grads.blocks[1].d_mha.d_w_q.d_w[i], g);
        assert!(err < TOL, "parallel d_w_q[{i}]: analytic {} vs numeric {g} (err {err})", grads.blocks[1].d_mha.d_w_q.d_w[i]);
    }
}

#[test]
fn layerscale_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x15CA);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 2,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        layerscale: true,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Move the gates off the ~1e-4 identity init so their grads are non-trivial.
    for blk in &mut model.blocks {
        blk.ls1 = random_vec(cfg.d_model, &mut rng);
        blk.ls2 = random_vec(cfg.d_model, &mut rng);
    }

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    for li in 0..2 {
        for c in [0usize, 3, 7] {
            let mut mp = model.clone();
            let mut mm = model.clone();
            mp.blocks[li].ls1[c] += EPS;
            mm.blocks[li].ls1[c] -= EPS;
            let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
            let a = grads.blocks[li].d_ls1[c];
            assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_ls1 blk{li}[{c}]: analytic {a} vs numeric {g}");

            let mut mp = model.clone();
            let mut mm = model.clone();
            mp.blocks[li].ls2[c] += EPS;
            mm.blocks[li].ls2[c] -= EPS;
            let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
            let a = grads.blocks[li].d_ls2[c];
            assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_ls2 blk{li}[{c}]: analytic {a} vs numeric {g}");
        }
    }
    // Downstream attn weight still correct with gates active.
    for i in [0, 31, 60] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[1].mha.w_q.w[i] += EPS;
        mm.blocks[1].mha.w_q.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[1].d_mha.d_w_q.d_w[i];
        assert!(rel_err(a, g) < TOL, "ls d_w_q blk1[{i}]: analytic {a} vs numeric {g}");
    }
}

#[test]
fn swiglu_ffn_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x5417);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        swiglu: true,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    // Gate projection (the SwiGLU-specific path).
    for i in [0, 50, 120] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].ffn.w_gate.as_mut().unwrap().w[i] += EPS;
        mm.blocks[0].ffn.w_gate.as_mut().unwrap().w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_ffn.d_w_gate.d_w[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_w_gate[{i}]: analytic {a} vs numeric {g}");
    }
    // Up (w_fc) and down (w_proj) projections under the gated activation.
    for i in [0, 50, 120] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].ffn.w_fc.w[i] += EPS;
        mm.blocks[0].ffn.w_fc.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_ffn.d_w_fc.d_w[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "swiglu d_w_fc[{i}]: analytic {a} vs numeric {g}");

        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].ffn.w_proj.w[i] += EPS;
        mm.blocks[0].ffn.w_proj.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_ffn.d_w_proj.d_w[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "swiglu d_w_proj[{i}]: analytic {a} vs numeric {g}");
    }
}

#[test]
fn canon_block_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0xCA40);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 2,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        canon_kernel: 3,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // Move taps off the zero (identity) init so the conv path is exercised.
    for blk in &mut model.blocks {
        blk.canon_w = random_vec(cfg.d_model * 3, &mut rng);
    }

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    for li in 0..2 {
        for i in [0usize, 5, 23] {
            let mut mp = model.clone();
            let mut mm = model.clone();
            mp.blocks[li].canon_w[i] += EPS;
            mm.blocks[li].canon_w[i] -= EPS;
            let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
            let a = grads.blocks[li].d_canon[i];
            assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_canon blk{li}[{i}]: analytic {a} vs numeric {g}");
        }
    }
    // Downstream attn weight still correct with the conv active.
    for i in [0, 13, 40] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[1].mha.w_q.w[i] += EPS;
        mm.blocks[1].mha.w_q.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[1].d_mha.d_w_q.d_w[i];
        assert!(rel_err(a, g) < TOL, "canon d_w_q blk1[{i}]: analytic {a} vs numeric {g}");
    }
}

#[test]
fn diff_attn_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0xD1FF);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        diff_attn: true,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);
    // w_o is zero-init (MHA = identity), which nulls the attention path;
    // randomize it so Q/K/V/λ gradients are actually exercised.
    model.blocks[0].mha.w_o.w = random_vec(cfg.d_model * cfg.d_model, &mut rng);

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    // Learned per-head λ.
    for h in [0usize, 1] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].mha.lambda[h] += EPS;
        mm.blocks[0].mha.lambda[h] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_mha.d_lambda[h];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_lambda[{h}]: analytic {a} vs numeric {g}");
    }
    // Q (split into the two maps) and V (shared) projections.
    for i in [0, 20, 50] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].mha.w_q.w[i] += EPS;
        mm.blocks[0].mha.w_q.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_mha.d_w_q.d_w[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "diff d_w_q[{i}]: analytic {a} vs numeric {g}");

        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.blocks[0].mha.w_v.w[i] += EPS;
        mm.blocks[0].mha.w_v.w[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.blocks[0].d_mha.d_w_v.d_w[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "diff d_w_v[{i}]: analytic {a} vs numeric {g}");
    }
}

#[test]
fn sink_tokens_backward_matches_numerical() {
    use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};

    let mut rng = StdRng::seed_from_u64(0x5141);
    let cfg = TransformerConfig {
        vocab: 11,
        d_model: 8,
        n_layer: 1,
        n_head: 2,
        d_ff: 16,
        seq_len_max: 8,
        sink_tokens: 2,
        ..TransformerConfig::default()
    };
    let model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = vec![2, 5, 1, 8, 3];
    let targets: Vec<u32> = vec![5, 1, 8, 3, 0];

    let ce = |m: &DecoderTransformer| -> f32 {
        let (logits, _) = m.forward(&tokens);
        let v = m.config.vocab;
        let t = targets.len();
        let mut total = 0.0_f32;
        for ti in 0..t {
            let row = &logits[ti * v..(ti + 1) * v];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += -(row[targets[ti] as usize] - max - sum.ln());
        }
        total / t as f32
    };

    let (logits, cache) = model.forward(&tokens);
    assert_eq!(logits.len(), targets.len() * cfg.vocab, "sink must not change scored length");
    let mut grads = ModelGrads::zeros(&cfg);
    model.backward(&cache, &logits, &targets, &mut grads);

    // Sink/register rows (2 * 8 = 16 elements).
    for i in [0usize, 7, 15] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.sink.as_mut().unwrap()[i] += EPS;
        mm.sink.as_mut().unwrap()[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.embed.d_sink[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "d_sink[{i}]: analytic {a} vs numeric {g}");
    }
    // Token embedding still correct alongside the sinks.
    for i in [0, 25, 60] {
        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.embed.w_tok[i] += EPS;
        mm.embed.w_tok[i] -= EPS;
        let g = (ce(&mp) - ce(&mm)) / (2.0 * EPS);
        let a = grads.embed.d_w_tok[i];
        assert!(rel_err(a, g) < TOL || (a - g).abs() < 2e-4, "sink d_w_tok[{i}]: analytic {a} vs numeric {g}");
    }
}

// Silence dead_code on Linear* re-exports if no test uses them yet.
#[allow(dead_code)]
fn _types_used() {
    let _: Option<LinearProj> = None;
    let _: Option<LinearGrads> = None;
}
