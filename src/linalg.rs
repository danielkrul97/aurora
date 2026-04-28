#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Naïve f32 matrix-vector primitives shared by the surrogate-gradient
//! SNN modules. f32-specialized; generic abstraction is premature here.

#[must_use]
pub fn matvec(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), rows * cols);
    debug_assert_eq!(v.len(), cols);
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let row_offset = r * cols;
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += m[row_offset + c] * v[c];
        }
        out[r] = acc;
    }
    out
}

#[must_use]
pub fn matvec_transpose(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), rows * cols);
    debug_assert_eq!(v.len(), rows);
    let mut out = vec![0.0_f32; cols];
    for r in 0..rows {
        let scale = v[r];
        if scale == 0.0 {
            continue;
        }
        let row_offset = r * cols;
        for c in 0..cols {
            out[c] += m[row_offset + c] * scale;
        }
    }
    out
}
