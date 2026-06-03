#![allow(clippy::needless_range_loop, clippy::must_use_candidate, clippy::many_single_char_names)]

//! Selective state-space scan — the core of the §22 selective-SSM experiment
//! (deep-research top pick: Mamba-style input-dependent recurrence).
//!
//! Our spacetime model mixes time with a STATIC (input-independent) causal
//! conv; the research verdict is that static/LTI mixing fails associative
//! recall (≈82% of the LM gap), and the fix is a CONTENT-DEPENDENT temporal
//! gate. This primitive is exactly that: a per-channel linear recurrence whose
//! forget gate `a_t` is computed from the input (so the model chooses what to
//! remember vs forget per token):
//! ```text
//! h_t = a_t · h_{t-1} + (1 − a_t) · v_t      (content-dependent EMA)
//! y_t = h_t
//! ```
//! `a_t ∈ (0,1)` (a sigmoid gate supplied by the caller) keeps the recurrence
//! contracting → bounded gradients, unlike the deep nonlinear field rollout.
//! Backward is the adjoint reverse-scan: `dh_t = dy_t + a_{t+1}·dh_{t+1}`,
//! `dv_t = (1−a_t)·dh_t`, `da_t = (h_{t-1}−v_t)·dh_t`. O(L·C), no tape.
//! Validated by finite-difference gradient checks in `tests/thermo_ssm.rs`.

/// Forward selective scan. `a` (forget gate, already in (0,1)) and `v` (value)
/// are `[l*c]` time-major (`t*c + ci`). Returns `y = h` `[l*c]`.
pub fn selective_scan_forward(a: &[f32], v: &[f32], l: usize, c: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), l * c);
    debug_assert_eq!(v.len(), l * c);
    let mut y = vec![0.0_f32; l * c];
    let mut h = vec![0.0_f32; c];
    for t in 0..l {
        let base = t * c;
        for ci in 0..c {
            let at = a[base + ci];
            h[ci] = at * h[ci] + (1.0 - at) * v[base + ci];
            y[base + ci] = h[ci];
        }
    }
    y
}

/// Backward (adjoint reverse-scan). Needs the forward output `y` (= the carried
/// states). Returns `(d_a, d_v)`, both `[l*c]`.
pub fn selective_scan_backward(
    a: &[f32],
    v: &[f32],
    y: &[f32],
    dy: &[f32],
    l: usize,
    c: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_a = vec![0.0_f32; l * c];
    let mut d_v = vec![0.0_f32; l * c];
    let mut dh = vec![0.0_f32; c]; // carries a_{t+1}·dh_{t+1} into step t
    for t in (0..l).rev() {
        let base = t * c;
        for ci in 0..c {
            let idx = base + ci;
            let at = a[idx];
            let dht = dh[ci] + dy[idx]; // dL/dh_t
            let h_prev = if t > 0 { y[idx - c] } else { 0.0 };
            d_v[idx] = (1.0 - at) * dht;
            d_a[idx] = (h_prev - v[idx]) * dht;
            dh[ci] = at * dht; // contribution h_t → h_{t-1}
        }
    }
    (d_a, d_v)
}
