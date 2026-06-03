#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]

//! Spacetime substrate — non-traditional geometry #3 (`docs/13` §21).
//!
//! The recurrent thermo-LM evolves one field token-by-token; long-range info
//! must survive the recurrent carry, the readout sees only the current
//! time-step, and BPTT runs T·K deep. The **spacetime** substrate lays the
//! sequence out along a field axis instead: the field is `[L×C]` (time = the
//! axis), every token injects at its own time-cell, a **causal** operator
//! propagates information forward in time, and each cell reads out *locally*
//! (its own causally-accumulated state → next-token logits). Consequences:
//! the readout is time-local (so temporal geometry is finally visible to the
//! loss, unlike the global spatial pool of §19/§20), and BPTT is only K deep
//! (the K operator steps), not T·K — directly defusing the §10 risk-1 killer.
//!
//! The novel primitive is the **causal dilated 1-D conv**: a cell at time `t`
//! reads only `t, t−d, t−2d, …` (`taps` past-and-present samples, dilation `d`),
//! never the future. Per-sub-step dilation `d = base^k` gives exponential
//! temporal reach (WaveNet-style) — the dilation idea on the axis where reach
//! actually pays. Same `Conv1x1(GELU(CausalConv1d(s)))` shape and
//! `*Weights/*Grads/*Cache` + forward/backward pattern as the grid conv,
//! validated by finite-difference gradient checks in `tests/thermo_spacetime.rs`.

use rand::Rng;

#[derive(Clone, Debug)]
pub struct StConvWeights {
    /// `[c_h * c * taps]`, index `(co*c + ci)*taps + tap`. tap 0 = present
    /// (`t`), tap j = `t − j·dilation` (strictly causal: only the past).
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
    /// Causal time taps (present + `taps−1` past samples).
    pub taps: usize,
}

impl StConvWeights {
    pub fn new<R: Rng>(c: usize, c_h: usize, c_out: usize, taps: usize, k_steps: usize, rng: &mut R) -> Self {
        let fan1_in = c * taps;
        let b1_bound = (6.0_f32 / (fan1_in + c_h) as f32).sqrt();
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
pub struct StConvGrads {
    pub d_w1: Vec<f32>,
    pub d_b1: Vec<f32>,
    pub d_w2: Vec<f32>,
    pub d_b2: Vec<f32>,
}

impl StConvGrads {
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
pub struct StConvCache {
    pub s: Vec<f32>,
    pub h_post: Vec<f32>,
    pub gelu_deriv: Vec<f32>,
    pub dilation: usize,
}

#[inline]
fn w1_idx(co: usize, ci: usize, tap: usize, c: usize, taps: usize) -> usize {
    (co * c + ci) * taps + tap
}

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

/// Forward. `s` is `[l*c]` (time-major); `dilation` ≥ 1. Returns `raw [l*c_out]`
/// and the cache. Strictly causal: tap `j` reads `s[t − j·dilation]`, zero for
/// negative time (no leakage from the future).
pub fn st_conv_forward(w: &StConvWeights, s: &[f32], l: usize, dilation: usize) -> (Vec<f32>, StConvCache) {
    let (c, c_h, c_out, taps) = (w.c, w.c_h, w.c_out, w.taps);
    debug_assert_eq!(s.len(), l * c);
    let d = dilation.max(1);

    let mut h_post = vec![0.0_f32; l * c_h];
    let mut gelu_deriv = vec![0.0_f32; l * c_h];
    for t in 0..l {
        for co in 0..c_h {
            let mut acc = w.b1[co];
            for tap in 0..taps {
                let off = tap * d;
                if off > t {
                    continue; // past the start of the sequence → zero
                }
                let sbase = (t - off) * c;
                for ci in 0..c {
                    acc += w.w1[w1_idx(co, ci, tap, c, taps)] * s[sbase + ci];
                }
            }
            let (post, deriv) = gelu_and_deriv(acc);
            h_post[t * c_h + co] = post;
            gelu_deriv[t * c_h + co] = deriv;
        }
    }

    let mut raw = vec![0.0_f32; l * c_out];
    for t in 0..l {
        let hp = &h_post[t * c_h..(t + 1) * c_h];
        let out = &mut raw[t * c_out..(t + 1) * c_out];
        for co in 0..c_out {
            let mut acc = w.b2[co];
            let wrow = &w.w2[co * c_h..(co + 1) * c_h];
            for ci in 0..c_h {
                acc += wrow[ci] * hp[ci];
            }
            out[co] = acc;
        }
    }

    (raw, StConvCache { s: s.to_vec(), h_post, gelu_deriv, dilation: d })
}

/// Backward. `d_raw` is `[l*c_out]`; returns `d_s [l*c]`, accumulates weight
/// grads. The causal gather's adjoint scatters cell `t`'s gradient back onto
/// `t − j·dilation` (i.e. d_s flows toward the past, mirroring the forward).
pub fn st_conv_backward(
    w: &StConvWeights,
    cache: &StConvCache,
    d_raw: &[f32],
    g: &mut StConvGrads,
    l: usize,
) -> Vec<f32> {
    let (c, c_h, c_out, taps) = (w.c, w.c_h, w.c_out, w.taps);
    debug_assert_eq!(d_raw.len(), l * c_out);
    let d = cache.dilation.max(1);

    // Conv1x1 backward → d_h_post, d_w2, d_b2.
    let mut d_h_post = vec![0.0_f32; l * c_h];
    for t in 0..l {
        let dr = &d_raw[t * c_out..(t + 1) * c_out];
        let hp = &cache.h_post[t * c_h..(t + 1) * c_h];
        let dhp = &mut d_h_post[t * c_h..(t + 1) * c_h];
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

    // GELU backward.
    let mut d_h_pre = vec![0.0_f32; l * c_h];
    for k in 0..d_h_pre.len() {
        d_h_pre[k] = d_h_post[k] * cache.gelu_deriv[k];
    }

    // CausalConv1d backward → d_s, d_w1, d_b1.
    let mut d_s = vec![0.0_f32; l * c];
    for t in 0..l {
        let dhp = &d_h_pre[t * c_h..(t + 1) * c_h];
        for (co, &gco) in dhp.iter().enumerate() {
            g.d_b1[co] += gco;
        }
        for tap in 0..taps {
            let off = tap * d;
            if off > t {
                continue;
            }
            let sbase = (t - off) * c;
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
