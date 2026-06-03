//! Numerical gradient checks for thermodynamic-LM primitives.
//!
//! Central-difference epsilon = 1e-3, relative tolerance = 5e-2 — same
//! convention as `transformer_primitives.rs`. Every hand-written backward
//! is checked against finite differences of the forward before it is
//! wired into the rollout.

#![allow(clippy::needless_range_loop)]

use aurora::thermo::conv::{conv_backward, conv_forward, ConvGrads, ConvWeights};
use aurora::thermo::diffusion::{
    diffusion_backward, diffusion_forward, DiffusionGrads, DiffusionWeights,
};
use aurora::thermo::field::{ThermoConfig, CH_E, CH_RHO};
use aurora::thermo::flux::{flux_div_backward, flux_div_forward};
use aurora::thermo::inject::{inject_backward, inject_forward, InjectGrads, InjectWeights};
use aurora::thermo::readout::{readout_backward, readout_forward, ReadoutGrads, ReadoutWeights};
use aurora::thermo::model::{ThermoGrads, ThermoLM};
use aurora::thermo::step::{step_backward, step_forward, StepGrads, StepWeights};
use aurora::thermo::vortex::{build_sink_mask, VortexField};
use aurora::transformer::loss::cross_entropy_with_grad;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPS: f32 = 1e-3;
const TOL: f32 = 5e-2;
/// Absolute floor: near-zero gradients are dominated by central-difference
/// truncation (O(EPS²·f‴)), so the relative test is uninformative there.
/// A real backward bug shows errors comparable to the gradient magnitude,
/// far above this floor.
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
    assert!(
        abs < ABS_TOL || rel < TOL,
        "{label}: analytic {analytic} vs numeric {numeric} (abs {abs}, rel {rel})"
    );
}

// ---------- conv (3×3 + GELU + 1×1) ----------

#[test]
fn conv_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x7E2);
    let cfg = ThermoConfig::toy(8);
    let w = ConvWeights::new(&cfg, &mut rng);
    let s = random_vec(cfg.field_len(), &mut rng);
    let upstream = random_vec(cfg.cells() * cfg.c_out(), &mut rng);

    let (_raw, cache) = conv_forward(&w, &s, &cfg);
    let mut g = ConvGrads::zeros(&cfg);
    let d_s = conv_backward(&w, &cache, &upstream, &mut g, &cfg);

    let loss = |s: &[f32]| dot(&conv_forward(&w, s, &cfg).0, &upstream);

    // d_s: random-projection directional derivatives.
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("d_s·r[{t}]"), dot(&d_s, &r), num);
    }

    // weight gradients: directional derivative along a random direction in
    // each tensor (checks the whole gradient, not a sampled element).
    let check_w = |label: &str, sel: fn(&mut ConvWeights) -> &mut Vec<f32>, ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(ana.len(), rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            sel(&mut wp)[k] += EPS * rk;
            sel(&mut wm)[k] -= EPS * rk;
        }
        let num = (dot(&conv_forward(&wp, &s, &cfg).0, &upstream)
            - dot(&conv_forward(&wm, &s, &cfg).0, &upstream))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check_w("d_w1·r", |w| &mut w.w1, &g.d_w1, &mut rng);
    check_w("d_b1·r", |w| &mut w.b1, &g.d_b1, &mut rng);
    check_w("d_w2·r", |w| &mut w.w2, &g.d_w2, &mut rng);
    check_w("d_b2·r", |w| &mut w.b2, &g.d_b2, &mut rng);
}

// ---------- flux-form divergence ----------

#[test]
fn flux_div_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xF1A);
    let cfg = ThermoConfig::toy(8);
    let n = cfg.cells() * cfg.n_conserved();
    let fx = random_vec(n, &mut rng);
    let fy = random_vec(n, &mut rng);
    let fz = random_vec(n, &mut rng);
    let upstream = random_vec(n, &mut rng);

    let (d_fx, d_fy, d_fz) = flux_div_backward(&upstream, &cfg);
    let loss = |fx: &[f32], fy: &[f32], fz: &[f32]| {
        dot(&flux_div_forward(fx, fy, fz, &cfg), &upstream)
    };

    for t in 0..3 {
        let r = random_vec(n, &mut rng);
        let fxp: Vec<f32> = fx.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let fxm: Vec<f32> = fx.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&fxp, &fy, &fz) - loss(&fxm, &fy, &fz)) / (2.0 * EPS);
        assert_close(&format!("d_fx·r[{t}]"), dot(&d_fx, &r), num);

        let fyp: Vec<f32> = fy.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let fym: Vec<f32> = fy.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&fx, &fyp, &fz) - loss(&fx, &fym, &fz)) / (2.0 * EPS);
        assert_close(&format!("d_fy·r[{t}]"), dot(&d_fy, &r), num);

        let fzp: Vec<f32> = fz.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let fzm: Vec<f32> = fz.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&fx, &fy, &fzp) - loss(&fx, &fy, &fzm)) / (2.0 * EPS);
        assert_close(&format!("d_fz·r[{t}]"), dot(&d_fz, &r), num);
    }
}

#[test]
fn flux_div_conserves_total_exactly() {
    // Structural property: for ANY fluxes, the grid-wide divergence sums
    // to zero per conserved channel (closed walls + telescoping), so
    // applying it to a field leaves each conserved channel's total fixed.
    let mut rng = StdRng::seed_from_u64(0xC04);
    let mut cfg = ThermoConfig::nano(64);
    cfg.d = 4; // exercise the z-flux telescoping, not just one 2D plane
    let nc = cfg.n_conserved();
    let n = cfg.cells() * nc;
    let fx = random_vec(n, &mut rng);
    let fy = random_vec(n, &mut rng);
    let fz = random_vec(n, &mut rng);
    let div = flux_div_forward(&fx, &fy, &fz, &cfg);

    for m in 0..nc {
        let mut sum = 0.0_f64;
        for cell in 0..cfg.cells() {
            sum += f64::from(div[cell * nc + m]);
        }
        // f32 accumulation noise (sum is f64); tolerance is generous
        // relative to the per-cell O(1) magnitudes.
        assert!(sum.abs() < 1e-3, "channel {m} divergence sum {sum} ≠ 0");
    }
}

// ---------- temperature-gated diffusion ----------

#[test]
fn diffusion_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xD1FF);
    let cfg = ThermoConfig::toy(8);
    let mut w = DiffusionWeights::new(&cfg);
    // Randomize gates so the T-coupling path is non-degenerate.
    w.k0 = random_vec(cfg.c, &mut rng).iter().map(|v| 0.2 + 0.1 * v).collect();
    w.a = random_vec(cfg.c, &mut rng);
    w.b = random_vec(cfg.c, &mut rng);
    let s = random_vec(cfg.field_len(), &mut rng);
    let upstream = random_vec(cfg.field_len(), &mut rng);

    let (_delta, cache) = diffusion_forward(&s, &w, &cfg);
    let mut g = DiffusionGrads::zeros(&cfg);
    let d_s = diffusion_backward(&w, &cache, &upstream, &mut g, &cfg);

    let loss = |s: &[f32]| dot(&diffusion_forward(s, &w, &cfg).0, &upstream);

    // d_s — exercises both the ∇² path and the κ(T) coupling on CH_T.
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("d_s·r[{t}]"), dot(&d_s, &r), num);
    }

    // gate-parameter gradients.
    let check = |label: &str, sel: fn(&mut DiffusionWeights) -> &mut Vec<f32>, ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(ana.len(), rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            sel(&mut wp)[k] += EPS * rk;
            sel(&mut wm)[k] -= EPS * rk;
        }
        let num = (dot(&diffusion_forward(&s, &wp, &cfg).0, &upstream)
            - dot(&diffusion_forward(&s, &wm, &cfg).0, &upstream))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("d_k0·r", |w| &mut w.k0, &g.d_k0, &mut rng);
    check("d_a·r", |w| &mut w.a, &g.d_a, &mut rng);
    check("d_b·r", |w| &mut w.b, &g.d_b, &mut rng);
}

// ---------- token injection ----------

#[test]
fn inject_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x1A3C);
    let cfg = ThermoConfig::toy(8);
    let w = InjectWeights::new(&cfg, &mut rng);
    let s = random_vec(cfg.field_len(), &mut rng);
    let upstream = random_vec(cfg.field_len(), &mut rng);
    let token = 3u32;
    let anneal = 0.7_f32;

    let (_out, cache) = inject_forward(&s, token, anneal, &w, &cfg);
    let mut g = InjectGrads::zeros(&cfg);
    let d_s = inject_backward(&cache, &upstream, &w, &mut g, &cfg);

    // d_s is the identity (injection is additive).
    for (i, (&a, &b)) in d_s.iter().zip(&upstream).enumerate() {
        assert_close(&format!("d_s[{i}]"), a, b);
    }

    // d_inj — only the token's row is nonzero; check it directionally.
    let r = random_vec(cfg.vocab * cfg.c, &mut rng);
    let mut wp = w.clone();
    let mut wm = w.clone();
    for (k, &rk) in r.iter().enumerate() {
        wp.inj[k] += EPS * rk;
        wm.inj[k] -= EPS * rk;
    }
    let lp = dot(&inject_forward(&s, token, anneal, &wp, &cfg).0, &upstream);
    let lm = dot(&inject_forward(&s, token, anneal, &wm, &cfg).0, &upstream);
    assert_close("d_inj·r", dot(&g.d_inj, &r), (lp - lm) / (2.0 * EPS));
}

// ---------- moment readout ----------

#[test]
fn readout_backward_matches_numerical() {
    for (label, moments, spectral) in
        [("moments", true, false), ("spectral", false, true), ("both", true, true)]
    {
        let mut rng = StdRng::seed_from_u64(0x4EAD);
        let mut cfg = ThermoConfig::toy(8);
        cfg.readout_moments = moments;
        cfg.readout_spectral = spectral;
        let w = ReadoutWeights::new(&cfg, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let upstream = random_vec(cfg.vocab, &mut rng);

        let (_logits, cache) = readout_forward(&s, &w, &cfg);
        let mut g = ReadoutGrads::zeros(&cfg);
        let d_s = readout_backward(&w, &cache, &upstream, &mut g, &cfg);

        let loss = |s: &[f32]| dot(&readout_forward(s, &w, &cfg).0, &upstream);
        for t in 0..3 {
            let r = random_vec(s.len(), &mut rng);
            let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
            let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
            let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
            assert_close(&format!("{label} d_s·r[{t}]"), dot(&d_s, &r), num);
        }

        let r = random_vec(w.proj.w.len(), &mut rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            wp.proj.w[k] += EPS * rk;
            wm.proj.w[k] -= EPS * rk;
        }
        let num = (dot(&readout_forward(&s, &wp, &cfg).0, &upstream)
            - dot(&readout_forward(&s, &wm, &cfg).0, &upstream))
            / (2.0 * EPS);
        assert_close(&format!("{label} d_proj·r"), dot(&g.d_proj.d_w, &r), num);
    }
}

// ---------- one full sub-step (conv + flux + diffusion composite) ----------

#[test]
fn step_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x57E9);
    let cfg = ThermoConfig::toy(8);
    let w = StepWeights::new(&cfg, &mut rng);
    let s = random_vec(cfg.field_len(), &mut rng);
    let upstream = random_vec(cfg.field_len(), &mut rng);
    let anneal = 0.6_f32;

    let (_out, cache) = step_forward(&s, &w, anneal, &cfg);
    let mut g = StepGrads::zeros(&cfg);
    let d_s = step_backward(&w, &cache, &upstream, &mut g, &cfg);

    let loss = |s: &[f32]| dot(&step_forward(s, &w, anneal, &cfg).0, &upstream);
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("d_s·r[{t}]"), dot(&d_s, &r), num);
    }

    // conv + diffusion parameters through the composite.
    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut StepWeights, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut wp, k, EPS * rk);
            perturb(&mut wm, k, -EPS * rk);
        }
        let num = (dot(&step_forward(&s, &wp, anneal, &cfg).0, &upstream)
            - dot(&step_forward(&s, &wm, anneal, &cfg).0, &upstream))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("d_conv.w1·r", w.conv.w1.len(), &|w, k, d| w.conv.w1[k] += d, &g.conv.d_w1, &mut rng);
    check("d_conv.w2·r", w.conv.w2.len(), &|w, k, d| w.conv.w2[k] += d, &g.conv.d_w2, &mut rng);
    check("d_diff.k0·r", w.diffusion.k0.len(), &|w, k, d| w.diffusion.k0[k] += d, &g.diffusion.d_k0, &mut rng);
    check("d_diff.b·r", w.diffusion.b.len(), &|w, k, d| w.diffusion.b[k] += d, &g.diffusion.d_b, &mut rng);
}

#[test]
fn step_backward_ablations_match_numerical() {
    // Verify the gated backward paths: diffusion off, and conservation off
    // (no flux-form — all channels take free deltas).
    for (label, use_diffusion, conserved) in [
        ("no-diffusion", false, vec![CH_RHO, CH_E]),
        ("no-conserve", true, vec![]),
    ] {
        let mut rng = StdRng::seed_from_u64(0xAB1A);
        let mut cfg = ThermoConfig::toy(8);
        cfg.use_diffusion = use_diffusion;
        cfg.conserved = conserved;
        let w = StepWeights::new(&cfg, &mut rng);
        let s = random_vec(cfg.field_len(), &mut rng);
        let upstream = random_vec(cfg.field_len(), &mut rng);

        let (_out, cache) = step_forward(&s, &w, 0.6, &cfg);
        let mut g = StepGrads::zeros(&cfg);
        let d_s = step_backward(&w, &cache, &upstream, &mut g, &cfg);

        let loss = |s: &[f32]| dot(&step_forward(s, &w, 0.6, &cfg).0, &upstream);
        for t in 0..3 {
            let r = random_vec(s.len(), &mut rng);
            let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
            let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
            let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
            assert_close(&format!("{label} d_s·r[{t}]"), dot(&d_s, &r), num);
        }

        // conv kernel grad (the always-present path through the operator).
        let r = random_vec(w.conv.w1.len(), &mut rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            wp.conv.w1[k] += EPS * rk;
            wm.conv.w1[k] -= EPS * rk;
        }
        let num = (dot(&step_forward(&s, &wp, 0.6, &cfg).0, &upstream)
            - dot(&step_forward(&s, &wm, 0.6, &cfg).0, &upstream))
            / (2.0 * EPS);
        assert_close(&format!("{label} d_conv.w1·r"), dot(&g.conv.d_w1, &r), num);
    }
}

// ---------- vortex advection + horizon sink (the §19 black-hole geometry) ----------

#[test]
fn step_vortex_conserves_total() {
    // The prescribed vortex rides on the same telescoping edges as the learned
    // flux, so it must leave each conserved channel's total exactly fixed
    // (sink off — the sink is the one op that deliberately breaks this).
    let mut rng = StdRng::seed_from_u64(0x71D);
    let mut cfg = ThermoConfig::nano(64);
    cfg.d = 3; // exercise the z-flux vortex, not just the plane
    // Isolate the flux path: κ(T)·∇² with a spatially varying κ (random T)
    // is not itself conservative, so diffusion would mask the vortex property.
    cfg.use_diffusion = false;
    let mut w = StepWeights::new(&cfg, &mut rng);
    w.vortex = Some(VortexField::new(&cfg, 0.5, 0.8));
    let s = random_vec(cfg.field_len(), &mut rng);
    let (out, _c) = step_forward(&s, &w, 1.0, &cfg);
    for &ch in &cfg.conserved {
        let before: f64 = (0..cfg.cells()).map(|cell| f64::from(s[cell * cfg.c + ch])).sum();
        let after: f64 = (0..cfg.cells()).map(|cell| f64::from(out[cell * cfg.c + ch])).sum();
        assert!((before - after).abs() < 2e-3, "channel {ch} total {before} → {after}");
    }
}

#[test]
fn step_backward_vortex_sink_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0x40C5);
    let cfg = ThermoConfig::toy(8);
    let mut w = StepWeights::new(&cfg, &mut rng);
    w.vortex = Some(VortexField::new(&cfg, 0.4, 0.6));
    w.sink = Some(build_sink_mask(&cfg, 0.9, 0.5));
    let s = random_vec(cfg.field_len(), &mut rng);
    let upstream = random_vec(cfg.field_len(), &mut rng);

    let (_out, cache) = step_forward(&s, &w, 0.6, &cfg);
    let mut g = StepGrads::zeros(&cfg);
    let d_s = step_backward(&w, &cache, &upstream, &mut g, &cfg);

    let loss = |s: &[f32]| dot(&step_forward(s, &w, 0.6, &cfg).0, &upstream);
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("vtx d_s·r[{t}]"), dot(&d_s, &r), num);
    }

    // conv.w1 through the vortex+sink composite (catches a wrong sink adjoint).
    let r = random_vec(w.conv.w1.len(), &mut rng);
    let mut wp = w.clone();
    let mut wm = w.clone();
    for (k, &rk) in r.iter().enumerate() {
        wp.conv.w1[k] += EPS * rk;
        wm.conv.w1[k] -= EPS * rk;
    }
    let num = (dot(&step_forward(&s, &wp, 0.6, &cfg).0, &upstream)
        - dot(&step_forward(&s, &wm, 0.6, &cfg).0, &upstream))
        / (2.0 * EPS);
    assert_close("vtx d_conv.w1·r", dot(&g.conv.d_w1, &r), num);
}

#[test]
fn rollout_vortex_backward_matches_numerical() {
    // All four knobs on at once, end to end: vortex flux + sink (in step),
    // ring profile (in inject), horizon gate (in readout). The param grads
    // must match finite differences through the whole fixed geometry.
    let mut rng = StdRng::seed_from_u64(0x70B1E);
    let cfg = ThermoConfig::toy(8);
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);
    model.vortex_beta = 0.4;
    model.vortex_swirl = 0.6;
    model.inject_ring_radius = 1.2;
    model.readout_horizon = 1.0;
    model.sink_radius = 0.9;
    model.sink_strength = 0.4;
    model.rebuild_experimental();
    let tokens: Vec<u32> = (0..4).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..4).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = ThermoGrads::zeros(&cfg);
    let analytic = model.backward(&cache, &targets, &mut g);
    assert!(
        (analytic - rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut ThermoLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut mp, k, EPS * rk);
            perturb(&mut mm, k, -EPS * rk);
        }
        let num = (rollout_loss(&mp, &tokens, &targets) - rollout_loss(&mm, &tokens, &targets))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("vtx conv.w1", model.step.conv.w1.len(), &|m, k, d| m.step.conv.w1[k] += d, &g.step.conv.d_w1, &mut rng);
    check("vtx inject.inj", model.inject.inj.len(), &|m, k, d| m.inject.inj[k] += d, &g.inject.d_inj, &mut rng);
    check("vtx readout.w", model.readout.proj.w.len(), &|m, k, d| m.readout.proj.w[k] += d, &g.readout.d_proj.d_w, &mut rng);
    check("vtx readout.bias", model.readout.bias.len(), &|m, k, d| m.readout.bias[k] += d, &g.readout.d_bias, &mut rng);
}

// ---------- end-to-end rollout BPTT ----------

/// Mean teacher-forced cross-entropy of a window — the quantity `backward`
/// differentiates.
fn rollout_loss(model: &ThermoLM, tokens: &[u32], targets: &[u32]) -> f32 {
    let logits = model.forward(tokens).logits;
    let v = model.config.vocab;
    let mut l = 0.0;
    for ti in 0..tokens.len() {
        let (loss, _) = cross_entropy_with_grad(&logits[ti * v..(ti + 1) * v], targets[ti] as usize);
        l += loss;
    }
    l / tokens.len() as f32
}

#[test]
fn rollout_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xB0B0);
    let cfg = ThermoConfig::toy(8);
    let model = ThermoLM::new(cfg.clone(), &mut rng);
    let tokens: Vec<u32> = (0..3).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..3).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = ThermoGrads::zeros(&cfg);
    let analytic_loss = model.backward(&cache, &targets, &mut g);
    assert!(
        (analytic_loss - rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    // Directional derivative of the loss vs analytic grad, per tensor —
    // exercises the whole inject→K-steps→readout chain end to end.
    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut ThermoLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut mp, k, EPS * rk);
            perturb(&mut mm, k, -EPS * rk);
        }
        let num = (rollout_loss(&mp, &tokens, &targets) - rollout_loss(&mm, &tokens, &targets))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("conv.w1", model.step.conv.w1.len(), &|m, k, d| m.step.conv.w1[k] += d, &g.step.conv.d_w1, &mut rng);
    check("conv.w2", model.step.conv.w2.len(), &|m, k, d| m.step.conv.w2[k] += d, &g.step.conv.d_w2, &mut rng);
    check("diff.b", model.config.c, &|m, k, d| m.step.diffusion.b[k] += d, &g.step.diffusion.d_b, &mut rng);
    check("inject.inj", model.inject.inj.len(), &|m, k, d| m.inject.inj[k] += d, &g.inject.d_inj, &mut rng);
    check("readout.w", model.readout.proj.w.len(), &|m, k, d| m.readout.proj.w[k] += d, &g.readout.d_proj.d_w, &mut rng);
    check("readout.bias", model.readout.bias.len(), &|m, k, d| m.readout.bias[k] += d, &g.readout.d_bias, &mut rng);
}

// ---------- deep supervision (auxiliary intermediate-readout losses) ----------

/// Combined deep-supervision training loss: main final-readout CE plus the
/// `aux_weight`-scaled mean CE over the intermediate sub-step readouts — the
/// exact scalar `backward` differentiates when `aux_weight > 0`.
fn rollout_train_loss_aux(model: &ThermoLM, tokens: &[u32], targets: &[u32]) -> f32 {
    let cache = model.forward(tokens);
    let v = model.config.vocab;
    let (t, k) = (tokens.len(), model.config.k_steps);
    let mut l_main = 0.0;
    for ti in 0..t {
        l_main += cross_entropy_with_grad(&cache.logits[ti * v..(ti + 1) * v], targets[ti] as usize).0;
    }
    l_main /= t as f32;
    let mut l_aux = 0.0;
    if model.aux_weight > 0.0 && k > 1 {
        for ti in 0..t {
            for j in 0..(k - 1) {
                let row = &cache.aux_logits[(ti * (k - 1) + j) * v..(ti * (k - 1) + j + 1) * v];
                l_aux += cross_entropy_with_grad(row, targets[ti] as usize).0;
            }
        }
        l_aux *= model.aux_weight / (t as f32 * (k - 1) as f32);
    }
    l_main + l_aux
}

#[test]
fn rollout_deep_supervision_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xDEE9);
    let mut cfg = ThermoConfig::toy(8);
    cfg.k_steps = 3; // K>1 → two intermediate aux taps (a_1, a_2) per token
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);
    model.aux_weight = 0.4;
    let tokens: Vec<u32> = (0..3).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..3).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = ThermoGrads::zeros(&cfg);
    model.backward(&cache, &targets, &mut g);

    // The whole inject→K-steps→(aux+main readouts) chain must match the
    // finite difference of the combined loss, per tensor.
    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut ThermoLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (kk, &rk) in r.iter().enumerate() {
            perturb(&mut mp, kk, EPS * rk);
            perturb(&mut mm, kk, -EPS * rk);
        }
        let num = (rollout_train_loss_aux(&mp, &tokens, &targets)
            - rollout_train_loss_aux(&mm, &tokens, &targets))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("ds conv.w1", model.step.conv.w1.len(), &|m, kk, d| m.step.conv.w1[kk] += d, &g.step.conv.d_w1, &mut rng);
    check("ds readout.w", model.readout.proj.w.len(), &|m, kk, d| m.readout.proj.w[kk] += d, &g.readout.d_proj.d_w, &mut rng);
    check("ds readout.bias", model.readout.bias.len(), &|m, kk, d| m.readout.bias[kk] += d, &g.readout.d_bias, &mut rng);
    check("ds inject.inj", model.inject.inj.len(), &|m, kk, d| m.inject.inj[kk] += d, &g.inject.d_inj, &mut rng);
}

// ---------- delta readout (logits off field_after − field_before) ----------

#[test]
fn rollout_delta_readout_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xDE17A);
    let cfg = ThermoConfig::toy(8);
    let mut model = ThermoLM::new(cfg.clone(), &mut rng);
    model.delta_readout = true;
    // 4 tokens so the cross-token carry (−d_from_readout into the previous
    // token's field) is exercised, not just a single position.
    let tokens: Vec<u32> = (0..4).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let targets: Vec<u32> = (0..4).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = ThermoGrads::zeros(&cfg);
    let analytic = model.backward(&cache, &targets, &mut g);
    assert!(
        (analytic - rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut ThermoLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (kk, &rk) in r.iter().enumerate() {
            perturb(&mut mp, kk, EPS * rk);
            perturb(&mut mm, kk, -EPS * rk);
        }
        let num = (rollout_loss(&mp, &tokens, &targets) - rollout_loss(&mm, &tokens, &targets))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("delta conv.w1", model.step.conv.w1.len(), &|m, kk, d| m.step.conv.w1[kk] += d, &g.step.conv.d_w1, &mut rng);
    check("delta inject.inj", model.inject.inj.len(), &|m, kk, d| m.inject.inj[kk] += d, &g.inject.d_inj, &mut rng);
    check("delta readout.w", model.readout.proj.w.len(), &|m, kk, d| m.readout.proj.w[kk] += d, &g.readout.d_proj.d_w, &mut rng);
    check("delta readout.bias", model.readout.bias.len(), &|m, kk, d| m.readout.bias[kk] += d, &g.readout.d_bias, &mut rng);
}
