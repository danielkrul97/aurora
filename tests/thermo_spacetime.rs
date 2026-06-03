//! Gradient + causality checks for the spacetime substrate's causal dilated
//! conv (`docs/13` §21). Same FD convention as `thermo_primitives.rs`.

#![allow(clippy::needless_range_loop)]

use aurora::thermo::spacetime::{st_conv_backward, st_conv_forward, StConvGrads, StConvWeights};
use aurora::thermo::spacetime_model::{StConfig, StGrads, StLM};
use aurora::transformer::loss::cross_entropy_with_grad;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPS: f32 = 1e-3;
const TOL: f32 = 5e-2;
const ABS_TOL: f32 = 1e-3;

fn random_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[track_caller]
fn assert_close(label: &str, analytic: f32, numeric: f32) {
    let abs = (analytic - numeric).abs();
    let rel = abs / analytic.abs().max(numeric.abs()).max(1e-6);
    assert!(abs < ABS_TOL || rel < TOL, "{label}: analytic {analytic} vs numeric {numeric} (abs {abs}, rel {rel})");
}

#[test]
fn st_conv_is_strictly_causal() {
    // Perturbing the input at time t0 must leave every output at t < t0 bit-
    // identical (no leakage from the future), and change t0 itself.
    let mut rng = StdRng::seed_from_u64(0xCA05);
    let (l, c, c_h, c_out, taps) = (12usize, 4usize, 5usize, 4usize, 3usize);
    let w = StConvWeights::new(c, c_h, c_out, taps, 2, &mut rng);
    let s = random_vec(l * c, &mut rng);
    for &dil in &[1usize, 2, 3] {
        let raw0 = st_conv_forward(&w, &s, l, dil).0;
        let t0 = 7usize;
        let mut sp = s.clone();
        for ci in 0..c {
            sp[t0 * c + ci] += 0.5;
        }
        let raw1 = st_conv_forward(&w, &sp, l, dil).0;
        for t in 0..t0 {
            for co in 0..c_out {
                assert!(
                    (raw0[t * c_out + co] - raw1[t * c_out + co]).abs() < 1e-7,
                    "dil {dil}: output at t={t} changed by a perturbation at t0={t0} (acausal leak)"
                );
            }
        }
        let changed: f32 = (0..c_out).map(|co| (raw0[t0 * c_out + co] - raw1[t0 * c_out + co]).abs()).sum();
        assert!(changed > 1e-6, "dil {dil}: output at t0={t0} did not respond to its own input");
    }
}

fn check_grads(dil: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (l, c, c_h, c_out, taps) = (10usize, 4usize, 6usize, 4usize, 3usize);
    let w = StConvWeights::new(c, c_h, c_out, taps, 2, &mut rng);
    let s = random_vec(l * c, &mut rng);
    let upstream = random_vec(l * c_out, &mut rng);

    let (_raw, cache) = st_conv_forward(&w, &s, l, dil);
    let mut g = StConvGrads::zeros(c, c_h, c_out, taps);
    let d_s = st_conv_backward(&w, &cache, &upstream, &mut g, l);

    let loss = |s: &[f32]| dot(&st_conv_forward(&w, s, l, dil).0, &upstream);
    let tag = format!("dil{dil}");
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("{tag} d_s·r[{t}]"), dot(&d_s, &r), num);
    }
    let check_w = |label: &str, sel: fn(&mut StConvWeights) -> &mut Vec<f32>, ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(ana.len(), rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            sel(&mut wp)[k] += EPS * rk;
            sel(&mut wm)[k] -= EPS * rk;
        }
        let num = (dot(&st_conv_forward(&wp, &s, l, dil).0, &upstream)
            - dot(&st_conv_forward(&wm, &s, l, dil).0, &upstream))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check_w(&format!("{tag} d_w1·r"), |w| &mut w.w1, &g.d_w1, &mut rng);
    check_w(&format!("{tag} d_b1·r"), |w| &mut w.b1, &g.d_b1, &mut rng);
    check_w(&format!("{tag} d_w2·r"), |w| &mut w.w2, &g.d_w2, &mut rng);
    check_w(&format!("{tag} d_b2·r"), |w| &mut w.b2, &g.d_b2, &mut rng);
}

#[test]
fn st_conv_backward_matches_numerical() {
    check_grads(1, 0x57C1);
    check_grads(2, 0x57C2);
    check_grads(4, 0x57C4);
}

// ---------- end-to-end StLM (parallel spacetime, K-deep BPTT) ----------

fn st_rollout_loss(m: &StLM, tokens: &[u32], targets: &[u32]) -> f32 {
    let logits = m.forward(tokens).logits;
    let v = m.cfg.vocab;
    let mut l = 0.0;
    for t in 0..tokens.len() {
        l += cross_entropy_with_grad(&logits[t * v..(t + 1) * v], targets[t] as usize).0;
    }
    l / tokens.len() as f32
}

#[test]
fn st_lm_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x57E11);
    let cfg = StConfig { vocab: 8, c: 5, conv_hidden: 7, k_steps: 3, taps: 2, dilation_base: 2, dt: 0.7 };
    let model = StLM::new(cfg, &mut rng);
    let tokens: Vec<u32> = (0..9).map(|_| rng.random_range(0..8u32)).collect();
    let targets: Vec<u32> = (0..9).map(|_| rng.random_range(0..8u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = StGrads::zeros(&model);
    let analytic = model.backward(&cache, &tokens, &targets, &mut g);
    assert!(
        (analytic - st_rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut StLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut mp, k, EPS * rk);
            perturb(&mut mm, k, -EPS * rk);
        }
        let num = (st_rollout_loss(&mp, &tokens, &targets) - st_rollout_loss(&mm, &tokens, &targets)) / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("st conv.w1", model.conv.w1.len(), &|m, k, d| m.conv.w1[k] += d, &g.conv.d_w1, &mut rng);
    check("st conv.w2", model.conv.w2.len(), &|m, k, d| m.conv.w2[k] += d, &g.conv.d_w2, &mut rng);
    check("st embed", model.embed.len(), &|m, k, d| m.embed[k] += d, &g.d_embed, &mut rng);
    check("st ro_w", model.ro_w.len(), &|m, k, d| m.ro_w[k] += d, &g.d_ro_w, &mut rng);
    check("st ro_bias", model.ro_bias.len(), &|m, k, d| m.ro_bias[k] += d, &g.d_ro_bias, &mut rng);
}
