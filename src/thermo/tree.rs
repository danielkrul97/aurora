#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]

//! Tree (hyperbolic) substrate — experimental non-traditional geometry for the
//! thermo-LM (see `docs/13` §20).
//!
//! The flat `D×H×W` lattice gives the local operator only a polynomial
//! receptive field (≈K cells after K sub-steps). A **tree** lattice gives an
//! *exponential* one: a `k`-ary tree of `depth` levels reaches all `k^depth`
//! leaves from the root in `depth` hops — the geometry of negative curvature,
//! and of language's recursive/hierarchical structure. We keep the field as a
//! flat `[N×C]` array (heap layout, root = node 0) so `inject` (a profile over
//! the N nodes) and the `pool=1` moment readout reuse the grid code unchanged;
//! only the convolution becomes tree-aware.
//!
//! `TreeConv` mirrors the grid `conv` exactly — `Conv1x1(GELU(GraphConv(s)))`
//! — but the spatial stage aggregates over a node's **tree taps**
//! `{self, parent, child_0..child_{k-1}}` (`taps = k+2`) instead of 27 grid
//! offsets. A missing neighbour (root's parent, a leaf's children) contributes
//! zero, the same "nothing beyond the boundary" convention as the grid's
//! zero-pad. Same `*Weights`/`*Grads`/`*Cache` + forward/backward pattern,
//! validated by finite-difference gradient checks in `tests/thermo_tree.rs`.

use rand::Rng;

/// Heap-laid-out full `k`-ary tree: node `i`'s parent is `(i-1)/k`, its
/// children are `k*i+1 ..= k*i+k`. Precomputes the per-node neighbour table so
/// the conv inner loop is a flat tap walk.
#[derive(Clone, Debug)]
pub struct TreeTopology {
    pub arity: usize,
    pub depth: usize,
    /// Node count = (k^depth − 1)/(k − 1).
    pub n: usize,
    /// Taps per node: self + parent + arity children.
    pub taps: usize,
    /// `[n*taps]` neighbour cell index per tap, `-1` = missing (zero-pad).
    /// Tap 0 = self, tap 1 = parent, taps `2..2+arity` = children.
    pub neigh: Vec<i32>,
    /// Leaf node indices (no children) — the natural input layer.
    pub leaves: Vec<u32>,
}

impl TreeTopology {
    pub fn new(arity: usize, depth: usize) -> Self {
        assert!(arity >= 1 && depth >= 1, "tree needs arity≥1, depth≥1");
        // n = (k^depth − 1)/(k − 1); k = 1 degenerates to a path of length depth.
        let n = if arity == 1 {
            depth
        } else {
            (arity.pow(depth as u32) - 1) / (arity - 1)
        };
        let taps = arity + 2;
        let mut neigh = vec![-1_i32; n * taps];
        let mut leaves = Vec::new();
        for i in 0..n {
            let base = i * taps;
            neigh[base] = i as i32; // self
            neigh[base + 1] = if i == 0 { -1 } else { ((i - 1) / arity) as i32 }; // parent
            let mut is_leaf = true;
            for j in 0..arity {
                let c = arity * i + 1 + j;
                if c < n {
                    neigh[base + 2 + j] = c as i32;
                    is_leaf = false;
                }
            }
            if is_leaf {
                leaves.push(i as u32);
            }
        }
        Self { arity, depth, n, taps, neigh, leaves }
    }
}

#[derive(Clone, Debug)]
pub struct TreeConvWeights {
    /// `[c_h * c * taps]`, index `(co*c + ci)*taps + tap`.
    pub w1: Vec<f32>,
    /// `[c_h]`
    pub b1: Vec<f32>,
    /// `[c_out * c_h]`, index `co*c_h + ci`.
    pub w2: Vec<f32>,
    /// `[c_out]`
    pub b2: Vec<f32>,
    pub c: usize,
    pub c_h: usize,
    pub c_out: usize,
    pub taps: usize,
}

impl TreeConvWeights {
    pub fn new<R: Rng>(c: usize, c_h: usize, c_out: usize, taps: usize, k_steps: usize, rng: &mut R) -> Self {
        let fan1_in = c * taps;
        let b1_bound = (6.0_f32 / (fan1_in + c_h) as f32).sqrt();
        // Gentle-start residual scaling, identical to the grid conv: keep
        // `s + Δt·N_θ(s)` near-identity at init across the K-step rollout.
        let resid_scale = (2.0 * k_steps as f32).sqrt().recip();
        let b2_bound = (6.0_f32 / (c_h + c_out) as f32).sqrt() * resid_scale;
        let unif = |bound: f32, rng: &mut R| (rng.random::<f32>() * 2.0 - 1.0) * bound;
        Self {
            w1: (0..c_h * c * taps).map(|_| unif(b1_bound, rng)).collect(),
            b1: vec![0.0; c_h],
            w2: (0..c_out * c_h).map(|_| unif(b2_bound, rng)).collect(),
            b2: vec![0.0; c_out],
            c,
            c_h,
            c_out,
            taps,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TreeConvGrads {
    pub d_w1: Vec<f32>,
    pub d_b1: Vec<f32>,
    pub d_w2: Vec<f32>,
    pub d_b2: Vec<f32>,
}

impl TreeConvGrads {
    pub fn zeros(c: usize, c_h: usize, c_out: usize, taps: usize) -> Self {
        Self {
            d_w1: vec![0.0; c_h * c * taps],
            d_b1: vec![0.0; c_h],
            d_w2: vec![0.0; c_out * c_h],
            d_b2: vec![0.0; c_out],
        }
    }

    pub fn zero(&mut self) {
        self.d_w1.fill(0.0);
        self.d_b1.fill(0.0);
        self.d_w2.fill(0.0);
        self.d_b2.fill(0.0);
    }
}

#[derive(Clone, Debug, Default)]
pub struct TreeConvCache {
    pub s: Vec<f32>,
    pub h_post: Vec<f32>,
    /// `gelu'(h_pre)` per element, captured in the forward.
    pub gelu_deriv: Vec<f32>,
}

#[inline]
fn w1_idx(co: usize, ci: usize, tap: usize, c: usize, taps: usize) -> usize {
    (co * c + ci) * taps + tap
}

/// `gelu` and its derivative sharing one `tanh` (bit-identical to
/// `linalg::gelu`/`gelu_backward`).
#[inline]
fn gelu_and_deriv(x: f32) -> (f32, f32) {
    let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = k * (x + 0.044_715 * x * x * x);
    let t = inner.tanh();
    let post = 0.5 * x * (1.0 + t);
    let sech2 = 1.0 - t * t;
    let d_inner_dx = k * (1.0 + 3.0 * 0.044_715 * x * x);
    let deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner_dx;
    (post, deriv)
}

/// Forward. `s` is `[n*c]`; returns `raw [n*c_out]` and the cache.
pub fn tree_conv_forward(w: &TreeConvWeights, s: &[f32], topo: &TreeTopology) -> (Vec<f32>, TreeConvCache) {
    let (c, c_h, c_out, taps, n) = (w.c, w.c_h, w.c_out, w.taps, topo.n);
    debug_assert_eq!(s.len(), n * c);
    debug_assert_eq!(taps, topo.taps);

    // GraphConv → h_pre, then GELU → h_post (capturing the derivative).
    let mut h_post = vec![0.0_f32; n * c_h];
    let mut gelu_deriv = vec![0.0_f32; n * c_h];
    for node in 0..n {
        let nb = &topo.neigh[node * taps..(node + 1) * taps];
        for co in 0..c_h {
            let mut acc = w.b1[co];
            for (tap, &src) in nb.iter().enumerate() {
                if src < 0 {
                    continue;
                }
                let sbase = src as usize * c;
                for ci in 0..c {
                    acc += w.w1[w1_idx(co, ci, tap, c, taps)] * s[sbase + ci];
                }
            }
            let (post, deriv) = gelu_and_deriv(acc);
            h_post[node * c_h + co] = post;
            gelu_deriv[node * c_h + co] = deriv;
        }
    }

    // Conv1x1 (per-node channel mix) → raw.
    let mut raw = vec![0.0_f32; n * c_out];
    for node in 0..n {
        let hp = &h_post[node * c_h..(node + 1) * c_h];
        let out = &mut raw[node * c_out..(node + 1) * c_out];
        for co in 0..c_out {
            let mut acc = w.b2[co];
            let wrow = &w.w2[co * c_h..(co + 1) * c_h];
            for ci in 0..c_h {
                acc += wrow[ci] * hp[ci];
            }
            out[co] = acc;
        }
    }

    (raw, TreeConvCache { s: s.to_vec(), h_post, gelu_deriv })
}

/// Backward. `d_raw` is `[n*c_out]`; returns `d_s [n*c]`, accumulates weight
/// grads into `g`.
pub fn tree_conv_backward(
    w: &TreeConvWeights,
    cache: &TreeConvCache,
    d_raw: &[f32],
    g: &mut TreeConvGrads,
    topo: &TreeTopology,
) -> Vec<f32> {
    let (c, c_h, c_out, taps, n) = (w.c, w.c_h, w.c_out, w.taps, topo.n);
    debug_assert_eq!(d_raw.len(), n * c_out);

    // Conv1x1 backward → d_h_post, d_w2, d_b2.
    let mut d_h_post = vec![0.0_f32; n * c_h];
    for node in 0..n {
        let dr = &d_raw[node * c_out..(node + 1) * c_out];
        let hp = &cache.h_post[node * c_h..(node + 1) * c_h];
        let dhp = &mut d_h_post[node * c_h..(node + 1) * c_h];
        for co in 0..c_out {
            let gco = dr[co];
            g.d_b2[co] += gco;
            let w_row = &w.w2[co * c_h..(co + 1) * c_h];
            let dw_row = &mut g.d_w2[co * c_h..(co + 1) * c_h];
            for ci in 0..c_h {
                dw_row[ci] += gco * hp[ci];
                dhp[ci] += gco * w_row[ci];
            }
        }
    }

    // GELU backward via cached derivative.
    let mut d_h_pre = vec![0.0_f32; n * c_h];
    for k in 0..d_h_pre.len() {
        d_h_pre[k] = d_h_post[k] * cache.gelu_deriv[k];
    }

    // GraphConv backward → d_s (scatter to tree neighbours), d_w1, d_b1.
    let mut d_s = vec![0.0_f32; n * c];
    for node in 0..n {
        let nb = &topo.neigh[node * taps..(node + 1) * taps];
        let dhp = &d_h_pre[node * c_h..(node + 1) * c_h];
        for (co, &gco) in dhp.iter().enumerate() {
            g.d_b1[co] += gco;
        }
        for (tap, &src) in nb.iter().enumerate() {
            if src < 0 {
                continue;
            }
            let sbase = src as usize * c;
            for ci in 0..c {
                let sv = cache.s[sbase + ci];
                let mut ds = 0.0_f32;
                for co in 0..c_h {
                    let gco = dhp[co];
                    let wi = w1_idx(co, ci, tap, c, taps);
                    g.d_w1[wi] += gco * sv;
                    ds += gco * w.w1[wi];
                }
                d_s[sbase + ci] += ds;
            }
        }
    }

    d_s
}
