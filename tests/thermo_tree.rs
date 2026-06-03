//! Numerical gradient checks for the tree (hyperbolic) substrate primitives.
//!
//! Same central-difference convention as `thermo_primitives.rs`
//! (EPS=1e-3, rel TOL=5e-2, abs floor 1e-3). The hand-written `tree_conv`
//! backward is checked against finite differences before it is wired into a
//! rollout (`docs/13` §20).

#![allow(clippy::needless_range_loop)]

use aurora::thermo::tree::{
    tree_conv_backward, tree_conv_forward, TreeConvGrads, TreeConvWeights, TreeTopology,
};
use aurora::thermo::tree_model::{TreeConfig, TreeGrads, TreeLM};
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
    assert!(
        abs < ABS_TOL || rel < TOL,
        "{label}: analytic {analytic} vs numeric {numeric} (abs {abs}, rel {rel})"
    );
}

#[test]
fn tree_topology_shapes() {
    // Full binary tree, depth 4 → 15 nodes; heap parent/child wiring.
    let t = TreeTopology::new(2, 4);
    assert_eq!(t.n, 15);
    assert_eq!(t.taps, 4); // self + parent + 2 children
    // root: no parent, two children (1,2).
    assert_eq!(&t.neigh[0..4], &[0, -1, 1, 2]);
    // node 3: parent 1, children 7,8.
    assert_eq!(&t.neigh[3 * 4..3 * 4 + 4], &[3, 1, 7, 8]);
    // node 14 (last): leaf, parent 6, no children.
    assert_eq!(&t.neigh[14 * 4..14 * 4 + 4], &[14, 6, -1, -1]);
    // leaves of depth-4 binary tree = 8 (nodes 7..14).
    assert_eq!(t.leaves, (7u32..15).collect::<Vec<_>>());

    // Ternary depth 3 → (27−1)/2 = 13 nodes, taps = 5.
    let t3 = TreeTopology::new(3, 3);
    assert_eq!(t3.n, 13);
    assert_eq!(t3.taps, 5);
}

fn check_one(arity: usize, depth: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let topo = TreeTopology::new(arity, depth);
    let (c, c_h, c_out) = (5usize, 6usize, 5usize);
    let taps = topo.taps;
    let w = TreeConvWeights::new(c, c_h, c_out, taps, 2, &mut rng);
    let s = random_vec(topo.n * c, &mut rng);
    let upstream = random_vec(topo.n * c_out, &mut rng);

    let (_raw, cache) = tree_conv_forward(&w, &s, &topo);
    let mut g = TreeConvGrads::zeros(c, c_h, c_out, taps);
    let d_s = tree_conv_backward(&w, &cache, &upstream, &mut g, &topo);

    let loss = |s: &[f32]| dot(&tree_conv_forward(&w, s, &topo).0, &upstream);
    let tag = format!("k{arity}d{depth}");

    // d_s — random-projection directional derivatives.
    for t in 0..4 {
        let r = random_vec(s.len(), &mut rng);
        let sp: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a + EPS * b).collect();
        let sm: Vec<f32> = s.iter().zip(&r).map(|(a, b)| a - EPS * b).collect();
        let num = (loss(&sp) - loss(&sm)) / (2.0 * EPS);
        assert_close(&format!("{tag} d_s·r[{t}]"), dot(&d_s, &r), num);
    }

    // weight gradients — directional derivative per tensor.
    let check_w = |label: &str, sel: fn(&mut TreeConvWeights) -> &mut Vec<f32>, ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(ana.len(), rng);
        let mut wp = w.clone();
        let mut wm = w.clone();
        for (k, &rk) in r.iter().enumerate() {
            sel(&mut wp)[k] += EPS * rk;
            sel(&mut wm)[k] -= EPS * rk;
        }
        let num = (dot(&tree_conv_forward(&wp, &s, &topo).0, &upstream)
            - dot(&tree_conv_forward(&wm, &s, &topo).0, &upstream))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check_w(&format!("{tag} d_w1·r"), |w| &mut w.w1, &g.d_w1, &mut rng);
    check_w(&format!("{tag} d_b1·r"), |w| &mut w.b1, &g.d_b1, &mut rng);
    check_w(&format!("{tag} d_w2·r"), |w| &mut w.w2, &g.d_w2, &mut rng);
    check_w(&format!("{tag} d_b2·r"), |w| &mut w.b2, &g.d_b2, &mut rng);
}

#[test]
fn tree_conv_backward_matches_numerical() {
    check_one(2, 4, 0x7233); // binary tree
    check_one(3, 3, 0xC0DE); // ternary (variable child fan-in)
    check_one(1, 5, 0x9A7B); // degenerate path (k=1)
}

// ---------- end-to-end TreeLM rollout BPTT ----------

fn tree_rollout_loss(m: &TreeLM, tokens: &[u32], targets: &[u32]) -> f32 {
    let logits = m.forward(tokens).logits;
    let v = m.tcfg.vocab;
    let mut l = 0.0;
    for ti in 0..tokens.len() {
        l += cross_entropy_with_grad(&logits[ti * v..(ti + 1) * v], targets[ti] as usize).0;
    }
    l / tokens.len() as f32
}

#[test]
fn tree_rollout_backward_matches_numerical() {
    let mut rng = StdRng::seed_from_u64(0xB047);
    let tcfg = TreeConfig { vocab: 8, arity: 2, depth: 3, c: 4, conv_hidden: 5, k_steps: 2, dt: 0.5 };
    let model = TreeLM::new(tcfg, &mut rng);
    let tokens: Vec<u32> = (0..4).map(|_| rng.random_range(0..8u32)).collect();
    let targets: Vec<u32> = (0..4).map(|_| rng.random_range(0..8u32)).collect();

    let cache = model.forward(&tokens);
    let mut g = TreeGrads::zeros(&model);
    let analytic = model.backward(&cache, &targets, &mut g);
    assert!(
        (analytic - tree_rollout_loss(&model, &tokens, &targets)).abs() < 1e-5,
        "forward/backward loss mismatch"
    );

    let mut check = |label: &str, n: usize, perturb: &dyn Fn(&mut TreeLM, usize, f32), ana: &[f32], rng: &mut StdRng| {
        let r = random_vec(n, rng);
        let mut mp = model.clone();
        let mut mm = model.clone();
        for (k, &rk) in r.iter().enumerate() {
            perturb(&mut mp, k, EPS * rk);
            perturb(&mut mm, k, -EPS * rk);
        }
        let num = (tree_rollout_loss(&mp, &tokens, &targets) - tree_rollout_loss(&mm, &tokens, &targets))
            / (2.0 * EPS);
        assert_close(label, dot(ana, &r), num);
    };
    check("tree conv.w1", model.conv.w1.len(), &|m, k, d| m.conv.w1[k] += d, &g.conv.d_w1, &mut rng);
    check("tree conv.w2", model.conv.w2.len(), &|m, k, d| m.conv.w2[k] += d, &g.conv.d_w2, &mut rng);
    check("tree inject.inj", model.inject.inj.len(), &|m, k, d| m.inject.inj[k] += d, &g.inject.d_inj, &mut rng);
    check("tree readout.w", model.readout.proj.w.len(), &|m, k, d| m.readout.proj.w[k] += d, &g.readout.d_proj.d_w, &mut rng);
    check("tree readout.bias", model.readout.bias.len(), &|m, k, d| m.readout.bias[k] += d, &g.readout.d_bias, &mut rng);
}
