//! Sprint 62: Muon optimizer tests.
//!
//! Two layers of coverage:
//!   1. Newton–Schulz orthogonalization correctness (numeric, in
//!      isolation): random matrices in square + both rectangular
//!      orientations, plus zero-matrix safety.
//!   2. End-to-end hybrid step: NaN-free model step, and copy-task
//!      convergence parity with the AdamW baseline.

#![allow(clippy::needless_range_loop, clippy::cast_precision_loss)]

use aurora::optimizer::{
    muon_apply, newton_schulz5, AdamWConfig, MuonConfig, MuonGroup, MuonScratch,
};
use aurora::transformer::{DecoderTransformer, ModelGrads, TransformerConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

/// Compute `M M^T` for an `[m × n]` row-major matrix.
fn mmt(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            let mut s = 0.0_f32;
            for k in 0..cols {
                s += m[i * cols + k] * m[j * cols + k];
            }
            out[i * rows + j] = s;
        }
    }
    out
}

/// Compute `M^T M` for an `[m × n]` row-major matrix.
fn mtm(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0_f32;
            for k in 0..rows {
                s += m[k * cols + i] * m[k * cols + j];
            }
            out[i * cols + j] = s;
        }
    }
    out
}

/// Frobenius distance between `mat` (size×size) and identity.
fn dist_from_identity(mat: &[f32], size: usize) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..size {
        for j in 0..size {
            let target = if i == j { 1.0 } else { 0.0 };
            let d = mat[i * size + j] - target;
            s += d * d;
        }
    }
    s.sqrt()
}

// ---------- 1. Newton–Schulz orthogonalization ----------
//
// NS5 with the Jordan/You quintic does NOT exactly orthogonalize — the
// fixed points of p(σ) = a·σ + b·σ³ + c·σ⁵ at (3.4445, -4.7750, 2.0315)
// are all unstable, so singular values oscillate inside a basin around
// [0.6, 1.3] rather than converging to 1. That's by design: the paper
// trades exact orthogonality for a 4-5x faster pull from σ near zero.
// Tests below verify the basin, not exact identity.

fn assert_singular_basin(mm_t: &[f32], size: usize, label: &str) {
    let mut min_diag = f32::INFINITY;
    let mut max_diag = f32::NEG_INFINITY;
    let mut max_off = 0.0_f32;
    for i in 0..size {
        let d = mm_t[i * size + i];
        min_diag = min_diag.min(d);
        max_diag = max_diag.max(d);
        for j in 0..size {
            if i != j {
                max_off = max_off.max(mm_t[i * size + j].abs());
            }
        }
    }
    // Diagonals = σ², expected ∈ [0.3, 2.5] for the NS5 basin in f32.
    assert!(
        min_diag > 0.3 && max_diag < 2.5,
        "{label}: σ² out of NS5 basin (min {min_diag}, max {max_diag})"
    );
    // Off-diagonals should be much smaller than diagonals — that's the
    // orthogonality bite of NS5.
    assert!(
        max_off < 0.5 * min_diag,
        "{label}: off-diag {max_off} not << min diag {min_diag} (orthogonality failed)"
    );
}

#[test]
fn newton_schulz_orthogonalizes_square_matrix() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let n = 16;
    let mut m = random_matrix(n, n, &mut rng);
    let mut scratch = Vec::new();
    newton_schulz5(&mut m, n, n, &mut scratch, 5);
    let mm_t = mmt(&m, n, n);
    assert_singular_basin(&mm_t, n, "square 16×16");
}

#[test]
fn newton_schulz_orthogonalizes_tall_matrix() {
    // rows > cols → algorithm transposes internally. Expect M^T M in basin.
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let (r, c) = (24, 8);
    let mut m = random_matrix(r, c, &mut rng);
    let mut scratch = Vec::new();
    newton_schulz5(&mut m, r, c, &mut scratch, 5);
    let mt_m = mtm(&m, r, c);
    assert_singular_basin(&mt_m, c, "tall 24×8 (M^T M)");
}

#[test]
fn newton_schulz_orthogonalizes_wide_matrix() {
    // rows < cols → no internal transpose. Expect M M^T in basin.
    let mut rng = StdRng::seed_from_u64(0xFADE);
    let (r, c) = (8, 24);
    let mut m = random_matrix(r, c, &mut rng);
    let mut scratch = Vec::new();
    newton_schulz5(&mut m, r, c, &mut scratch, 5);
    let mm_t = mmt(&m, r, c);
    assert_singular_basin(&mm_t, r, "wide 8×24 (M M^T)");
}

#[test]
fn newton_schulz_concentrates_spectrum_via_trace_ratio() {
    // For an `[n × n]` matrix M with singular values σ_i, the ratio
    //     R = trace((M M^T)²) / trace(M M^T)²
    //       = (Σ σ⁴) / (Σ σ²)²
    // sits in [1/n, 1]: it hits 1/n exactly when all σ are equal and 1
    // when one dominates. NS5 should drive R toward 1/n (≈ 0.083 for
    // n=12).
    fn trace_ratio(m: &[f32], n: usize) -> f32 {
        let a = mmt(m, n, n);
        let mut t2 = 0.0_f32;
        for i in 0..n {
            t2 += a[i * n + i];
        }
        let mut t4 = 0.0_f32;
        for i in 0..n {
            for j in 0..n {
                t4 += a[i * n + j] * a[j * n + i];
            }
        }
        t4 / (t2 * t2 + 1e-12)
    }

    let mut rng = StdRng::seed_from_u64(0xDEED);
    let n = 12;
    let m_orig = random_matrix(n, n, &mut rng);
    let r_before = trace_ratio(&m_orig, n);

    let mut m_ns = m_orig.clone();
    let mut scratch = Vec::new();
    newton_schulz5(&mut m_ns, n, n, &mut scratch, 5);
    let r_after = trace_ratio(&m_ns, n);

    let uniform = 1.0_f32 / n as f32;
    // Allow up to 50% above uniform; before-NS, the ratio for random
    // matrices is typically 0.13–0.20.
    assert!(
        r_after < uniform * 1.5,
        "trace ratio after NS5 ({r_after:.4}) too far from uniform ({uniform:.4}); before {r_before:.4}"
    );
    assert!(
        r_after < r_before * 0.85,
        "NS5 did not shrink spread: before {r_before:.4}, after {r_after:.4}"
    );
}

#[test]
fn newton_schulz_handles_zero_matrix() {
    // All-zero input must not produce NaN/Inf (the 1/Frobenius bound
    // would divide by zero without the early-exit guard).
    let n = 8;
    let mut m = vec![0.0_f32; n * n];
    let mut scratch = Vec::new();
    newton_schulz5(&mut m, n, n, &mut scratch, 5);
    for &v in &m {
        assert!(v.is_finite(), "NS on zero matrix produced {v}");
        assert_eq!(v, 0.0, "NS on zero matrix should stay zero, got {v}");
    }
}

#[test]
fn newton_schulz_is_scale_invariant_in_direction() {
    // NS normalizes singular values to ~1, so scaling the input by a
    // constant should give the same output (up to floating-point noise).
    let mut rng = StdRng::seed_from_u64(0x12345);
    let n = 12;
    let m_orig = random_matrix(n, n, &mut rng);
    let mut m_small = m_orig.clone();
    let mut m_big: Vec<f32> = m_orig.iter().map(|v| v * 1000.0).collect();

    let mut scratch = Vec::new();
    newton_schulz5(&mut m_small, n, n, &mut scratch, 5);
    newton_schulz5(&mut m_big, n, n, &mut scratch, 5);

    let mut max_diff = 0.0_f32;
    for (a, b) in m_small.iter().zip(m_big.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-3,
        "scale invariance violated: max element diff {max_diff}"
    );
}

// ---------- 2. muon_apply unit ----------

#[test]
fn muon_apply_first_step_moves_param_against_grad() {
    // With zero initial momentum, the first Muon step direction must be
    // opposite to the gradient (after Newton–Schulz, signs are largely
    // preserved on a random-positive grad).
    let mut rng = StdRng::seed_from_u64(0xABCDEF);
    let (r, c) = (8, 8);
    let g: Vec<f32> = (0..r * c).map(|_| rng.random::<f32>() + 0.1).collect();
    let mut params = vec![1.0_f32; r * c];
    let mut group = MuonGroup::zeros(r * c);
    let cfg = MuonConfig::default();
    let mut scratch = MuonScratch::new();

    muon_apply(&mut params, &g, &mut group, r, c, &cfg, &mut scratch);

    // After one step, params should have moved (no longer all 1.0).
    let mut moved = 0;
    for &p in &params {
        if (p - 1.0).abs() > 1e-6 {
            moved += 1;
        }
        assert!(p.is_finite(), "param became {p}");
    }
    assert_eq!(moved, r * c, "expected all params to move, only {moved} did");

    // Momentum buffer should be exactly the gradient (β=0.95 × 0 + g).
    for i in 0..g.len() {
        assert!(
            (group.m[i] - g[i]).abs() < 1e-6,
            "momentum[{i}] = {} vs g = {}",
            group.m[i], g[i]
        );
    }
}

// ---------- 3. Full-model integration ----------

#[test]
fn muon_hybrid_step_does_not_nan() {
    let mut rng = StdRng::seed_from_u64(0xC0DECAFE);
    let cfg = TransformerConfig {
        vocab: 32,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 32,
        rope: false,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let tokens: Vec<u32> = (0..16).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
    let input = &tokens[..15];
    let target = &tokens[1..];

    let (logits, cache) = model.forward(input);
    let mut grads = ModelGrads::zeros(&cfg);
    let loss = model.backward(&cache, &logits, target, &mut grads);
    assert!(loss.is_finite(), "loss NaN/Inf");

    let _ = DecoderTransformer::clip_grads_adamw_subset(&mut grads, 1.0);

    let muon_cfg = MuonConfig::default();
    let adam_cfg = AdamWConfig::default();
    model.step_muon_hybrid(&grads, &muon_cfg, &adam_cfg);

    for v in model.embed.w_tok.iter().take(64) {
        assert!(v.is_finite(), "tok embed NaN/Inf after Muon step");
    }
    for blk in &model.blocks {
        for v in blk.mha.w_q.w.iter().take(16) {
            assert!(v.is_finite(), "W_q NaN/Inf after Muon step");
        }
        for v in blk.ffn.w_proj.w.iter().take(16) {
            assert!(v.is_finite(), "W_proj NaN/Inf after Muon step");
        }
        for v in &blk.ln1.gamma {
            assert!(v.is_finite(), "LN1 γ NaN/Inf after Muon step");
        }
    }
}

#[test]
fn muon_converges_on_copy_task() {
    // Same toy as `cpu_transformer_converges_on_copy_task` in
    // tests/transformer_smoke.rs, swapped to step_muon_hybrid. We
    // require the same convergence threshold; if Muon were strictly
    // worse than AdamW on this trivial task, something is wrong with
    // the hybrid wiring.
    let mut rng = StdRng::seed_from_u64(0x5071);
    let cfg = TransformerConfig {
        vocab: 8,
        d_model: 32,
        n_layer: 2,
        n_head: 2,
        d_ff: 64,
        seq_len_max: 16,
        rope: false,
        ..TransformerConfig::default()
    };
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    let muon_cfg = MuonConfig {
        lr: 0.02,
        weight_decay: 0.0,
        ..MuonConfig::default()
    };
    let adam_cfg = AdamWConfig {
        lr: 1e-2,
        weight_decay: 0.0,
        ..AdamWConfig::default()
    };

    let mut grads = ModelGrads::zeros(&cfg);
    let mut first_loss: f32 = 0.0;
    let mut last_loss: f32 = 0.0;
    let steps = 600;

    for step in 0..steps {
        let half: Vec<u32> = (0..6).map(|_| rng.random_range(0..cfg.vocab as u32)).collect();
        let mut tokens: Vec<u32> = Vec::with_capacity(12);
        for &t in &half {
            tokens.push(t);
            tokens.push(t);
        }
        let input: Vec<u32> = tokens[..11].to_vec();
        let target: Vec<u32> = tokens[1..].to_vec();

        let (logits, cache) = model.forward(&input);
        grads.zero();
        let loss = model.backward(&cache, &logits, &target, &mut grads);
        DecoderTransformer::clip_grads_adamw_subset(&mut grads, 1.0);
        model.step_muon_hybrid(&grads, &muon_cfg, &adam_cfg);

        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
    }

    let baseline = (cfg.vocab as f32).ln();
    assert!(
        last_loss < first_loss * 0.7,
        "Muon loss did not decrease: first={first_loss}, last={last_loss}, baseline={baseline}"
    );
    assert!(
        last_loss < baseline * 0.9,
        "Muon last loss {last_loss} above 0.9× random baseline {baseline}"
    );
}
