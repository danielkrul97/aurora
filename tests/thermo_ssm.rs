//! Gradient check for the selective-SSM scan (`docs/13` §22). Same FD
//! convention as `thermo_primitives.rs`.

#![allow(clippy::needless_range_loop)]

use aurora::thermo::ssm::{selective_scan_backward, selective_scan_forward};
use aurora::thermo::ssm_model::{SsmConfig, SsmGrads, SsmLM};
use aurora::transformer::loss::cross_entropy_with_grad;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPS: f32 = 1e-3;
const TOL: f32 = 5e-2;
const ABS_TOL: f32 = 1e-3;

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
fn selective_scan_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x5_5_1);
    let (l, c) = (12usize, 5usize);
    // a in (0,1) via sigmoid of random logits; v arbitrary.
    let a: Vec<f32> = (0..l * c).map(|_| 1.0 / (1.0 + (-(rng.random::<f32>() * 4.0 - 2.0)).exp())).collect();
    let v: Vec<f32> = (0..l * c).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let upstream: Vec<f32> = (0..l * c).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    let y = selective_scan_forward(&a, &v, l, c);
    let (d_a, d_v) = selective_scan_backward(&a, &v, &y, &upstream, l, c);

    let loss = |a: &[f32], v: &[f32]| dot(&selective_scan_forward(a, v, l, c), &upstream);

    // d_a — random-projection directional derivatives (a stays interior to (0,1)).
    for t in 0..4 {
        let r: Vec<f32> = (0..l * c).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let ap: Vec<f32> = a.iter().zip(&r).map(|(x, b)| x + EPS * b).collect();
        let am: Vec<f32> = a.iter().zip(&r).map(|(x, b)| x - EPS * b).collect();
        let num = (loss(&ap, &v) - loss(&am, &v)) / (2.0 * EPS);
        assert_close(&format!("d_a·r[{t}]"), dot(&d_a, &r), num);
    }
    // d_v.
    for t in 0..4 {
        let r: Vec<f32> = (0..l * c).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let vp: Vec<f32> = v.iter().zip(&r).map(|(x, b)| x + EPS * b).collect();
        let vm: Vec<f32> = v.iter().zip(&r).map(|(x, b)| x - EPS * b).collect();
        let num = (loss(&a, &vp) - loss(&a, &vm)) / (2.0 * EPS);
        assert_close(&format!("d_v·r[{t}]"), dot(&d_v, &r), num);
    }
}

#[test]
fn selective_scan_is_causal_and_gated() {
    // a→1 holds state (perfect memory); a→0 forgets (y_t = v_t). And a
    // perturbation at t0 must not change outputs before t0.
    let (l, c) = (8usize, 2usize);
    let v: Vec<f32> = (0..l * c).map(|i| (i as f32).sin()).collect();
    let hold = vec![0.999_f32; l * c];
    let forget = vec![0.001_f32; l * c];
    let yh = selective_scan_forward(&hold, &v, l, c);
    let yf = selective_scan_forward(&forget, &v, l, c);
    // forget gate ≈ pass-through of v.
    for i in 0..l * c {
        assert!((yf[i] - v[i]).abs() < 1e-2, "forget gate should pass v at {i}");
    }
    // hold gate ≈ frozen at the first value.
    for t in 1..l {
        for ci in 0..c {
            assert!((yh[t * c + ci] - yh[ci]).abs() < 0.1, "hold gate should freeze state");
        }
    }
    // causality: perturb v at t0, outputs before t0 unchanged.
    let a: Vec<f32> = (0..l * c).map(|_| 0.5).collect();
    let y0 = selective_scan_forward(&a, &v, l, c);
    let mut vp = v.clone();
    let t0 = 5;
    for ci in 0..c {
        vp[t0 * c + ci] += 1.0;
    }
    let y1 = selective_scan_forward(&a, &vp, l, c);
    for t in 0..t0 {
        for ci in 0..c {
            assert!((y0[t * c + ci] - y1[t * c + ci]).abs() < 1e-7, "acausal leak at t={t}");
        }
    }
}

// ---------- end-to-end SsmLM ----------

fn ssm_rollout_loss(m: &SsmLM, tokens: &[u32], targets: &[u32]) -> f32 {
    let logits = m.forward(tokens).logits;
    let v = m.cfg.vocab;
    let mut l = 0.0;
    for t in 0..tokens.len() {
        l += cross_entropy_with_grad(&logits[t * v..(t + 1) * v], targets[t] as usize).0;
    }
    l / tokens.len() as f32
}

#[test]
fn ssm_lm_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x5_5_3);
    let cfg = SsmConfig { vocab: 8, c: 6, n_layers: 2 };
    let model = SsmLM::new(cfg, &mut rng);
    let tokens: Vec<u32> = (0..9).map(|_| rng.random_range(0..8u32)).collect();
    let targets: Vec<u32> = (0..9).map(|_| rng.random_range(0..8u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = SsmGrads::zeros(&model);
    let analytic = model.backward(&cache, &tokens, &targets, &mut g);
    assert!(
        (analytic - ssm_rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut SsmLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut mp, k, EPS * rk);
            perturb(&mut mm, k, -EPS * rk);
        }
        let num = (ssm_rollout_loss(&mp, &tokens, &targets) - ssm_rollout_loss(&mm, &tokens, &targets)) / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("win[0]", model.blocks[0].win.len(), &|m, k, d| m.blocks[0].win[k] += d, &g.blocks[0].d_win, &mut rng);
    check("wg[0]", model.blocks[0].wg.len(), &|m, k, d| m.blocks[0].wg[k] += d, &g.blocks[0].d_wg, &mut rng);
    check("wo[0]", model.blocks[0].wo.len(), &|m, k, d| m.blocks[0].wo[k] += d, &g.blocks[0].d_wo, &mut rng);
    check("bg[0]", model.blocks[0].bg.len(), &|m, k, d| m.blocks[0].bg[k] += d, &g.blocks[0].d_bg, &mut rng);
    check("win[1]", model.blocks[1].win.len(), &|m, k, d| m.blocks[1].win[k] += d, &g.blocks[1].d_win, &mut rng);
    check("embed", model.embed.len(), &|m, k, d| m.embed[k] += d, &g.d_embed, &mut rng);
    check("ro_w", model.ro_w.len(), &|m, k, d| m.ro_w[k] += d, &g.d_ro_w, &mut rng);
    check("ro_bias", model.ro_bias.len(), &|m, k, d| m.ro_bias[k] += d, &g.d_ro_bias, &mut rng);
}
