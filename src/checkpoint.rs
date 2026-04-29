#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

//! Weight checkpointing for `MultiLayerLnnSnn`.
//!
//! Binary format (little-endian, magic `AURORA-LNN01`):
//! ```text
//! magic       : [u8; 12] = b"AURORA-LNN01"
//! _pad        : [u8; 4]
//! n_input     : u32
//! n_output    : u32
//! n_layers    : u32
//! epoch       : u32
//! per layer:
//!   n_in      : u32
//!   n_hidden  : u32
//!   w_f       : f32[n_hidden * (n_in + n_hidden)]
//!   w_g       : f32[n_hidden * (n_in + n_hidden)]
//!   w_h       : f32[n_hidden * (n_in + n_hidden)]
//!   b_f       : f32[n_hidden]
//!   b_g       : f32[n_hidden]
//!   b_h       : f32[n_hidden]
//!   log_tau   : f32[n_hidden]
//! w_out       : f32[n_output * top_hidden]
//! b_o         : f32[n_output]
//! ```
//!
//! Adam state is intentionally NOT serialized — resuming uses fresh
//! moments (the optimizer warmup over a few hundred steps is acceptable).
//!
//! Breaking change vs. legacy `AURORA01`/`AURORA02` SNN format. Old
//! checkpoints are not loadable; LNN architecture has different
//! parameter layout (`w_f, w_g, w_h, log_tau` vs. `w_in, w_rec,
//! w_gate_in, w_gate_rec, alpha`).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::lnn::MultiLayerLnnSnn;

const MAGIC: &[u8; 12] = b"AURORA-LNN01";

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub net: MultiLayerLnnSnn,
    pub epoch: u32,
}

/// Save a network plus the number of completed epochs.
///
/// # Errors
/// Propagates the underlying I/O error if the file cannot be written.
pub fn save<P: AsRef<Path>>(path: P, net: &MultiLayerLnnSnn, epoch: u32) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    w.write_all(&[0u8; 4])?;
    write_u32(&mut w, net.n_input as u32)?;
    write_u32(&mut w, net.n_output as u32)?;
    write_u32(&mut w, net.layers.len() as u32)?;
    write_u32(&mut w, epoch)?;

    for layer in &net.layers {
        write_u32(&mut w, layer.n_in as u32)?;
        write_u32(&mut w, layer.n_hidden as u32)?;
        write_f32_slice(&mut w, &layer.w_f)?;
        write_f32_slice(&mut w, &layer.w_g)?;
        write_f32_slice(&mut w, &layer.w_h)?;
        write_f32_slice(&mut w, &layer.b_f)?;
        write_f32_slice(&mut w, &layer.b_g)?;
        write_f32_slice(&mut w, &layer.b_h)?;
        write_f32_slice(&mut w, &layer.log_tau)?;
    }
    write_f32_slice(&mut w, &net.w_out)?;
    write_f32_slice(&mut w, &net.b_o)?;
    w.flush()?;
    Ok(())
}

/// Load a checkpoint. The returned network has fresh Adam state.
///
/// # Errors
/// Returns an error if the file is missing, truncated, or has the wrong magic.
pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Checkpoint> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 12];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad checkpoint magic — expected AURORA-LNN01",
        ));
    }
    let mut pad = [0u8; 4];
    r.read_exact(&mut pad)?;

    let n_input = read_u32(&mut r)? as usize;
    let n_output = read_u32(&mut r)? as usize;
    let n_layers = read_u32(&mut r)? as usize;
    let epoch = read_u32(&mut r)?;

    let mut hidden_dims = Vec::with_capacity(n_layers);
    let mut layer_blobs: Vec<[Vec<f32>; 7]> = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_in = read_u32(&mut r)? as usize;
        let n_hidden = read_u32(&mut r)? as usize;
        hidden_dims.push(n_hidden);
        let n_z = n_in + n_hidden;
        let w_f = read_f32_vec(&mut r, n_hidden * n_z)?;
        let w_g = read_f32_vec(&mut r, n_hidden * n_z)?;
        let w_h = read_f32_vec(&mut r, n_hidden * n_z)?;
        let b_f = read_f32_vec(&mut r, n_hidden)?;
        let b_g = read_f32_vec(&mut r, n_hidden)?;
        let b_h = read_f32_vec(&mut r, n_hidden)?;
        let log_tau = read_f32_vec(&mut r, n_hidden)?;
        layer_blobs.push([w_f, w_g, w_h, b_f, b_g, b_h, log_tau]);
    }
    let top_hidden = *hidden_dims.last().expect("≥1 layer");
    let w_out = read_f32_vec(&mut r, n_output * top_hidden)?;
    let b_o = read_f32_vec(&mut r, n_output)?;

    let mut rng = StdRng::seed_from_u64(0);
    let mut net = MultiLayerLnnSnn::new(n_input, &hidden_dims, n_output, &mut rng);
    for (i, blob) in layer_blobs.into_iter().enumerate() {
        let [w_f, w_g, w_h, b_f, b_g, b_h, log_tau] = blob;
        let layer = &mut net.layers[i];
        layer.w_f = w_f;
        layer.w_g = w_g;
        layer.w_h = w_h;
        layer.b_f = b_f;
        layer.b_g = b_g;
        layer.b_h = b_h;
        layer.log_tau = log_tau;
    }
    net.w_out = w_out;
    net.b_o = b_o;

    Ok(Checkpoint { net, epoch })
}

// --- low-level binary helpers -------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32_slice<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    // SAFETY: f32 has the same alignment+size as 4 raw bytes; the slice
    // is only borrowed for the duration of write_all.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    w.write_all(bytes)
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0_f32; n];
    // SAFETY: f32 layout is 4 bytes; the borrow lives only for read_exact.
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * 4) };
    r.read_exact(bytes)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::AdamConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn save_load_roundtrip_preserves_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let net = MultiLayerLnnSnn::new(16, &[8, 8], 16, &mut rng);
        let path = std::env::temp_dir().join("aurora_lnn_ckpt_test.bin");
        save(&path, &net, 7).unwrap();
        let ckpt = load(&path).unwrap();
        assert_eq!(ckpt.epoch, 7);
        assert_eq!(ckpt.net.n_input, net.n_input);
        assert_eq!(ckpt.net.n_output, net.n_output);
        assert_eq!(ckpt.net.layers.len(), net.layers.len());
        for (a, b) in ckpt.net.layers.iter().zip(&net.layers) {
            assert_eq!(a.w_f, b.w_f);
            assert_eq!(a.w_g, b.w_g);
            assert_eq!(a.w_h, b.w_h);
            assert_eq!(a.b_f, b.b_f);
            assert_eq!(a.b_g, b.b_g);
            assert_eq!(a.b_h, b.b_h);
            assert_eq!(a.log_tau, b.log_tau);
        }
        assert_eq!(ckpt.net.w_out, net.w_out);
        assert_eq!(ckpt.net.b_o, net.b_o);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_load_after_training_step() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut net = MultiLayerLnnSnn::new(8, &[4], 8, &mut rng);
        let inputs: Vec<u32> = (0..16).map(|t| (t % 8) as u32).collect();
        let targets: Vec<u32> = (1..17).map(|t| (t % 8) as u32).collect();
        let cache = net.forward_lm_tokens(&inputs);
        net.backward_lm_tokens(&cache, &inputs, &targets);
        net.step_adam(&AdamConfig::default());
        let path = std::env::temp_dir().join("aurora_lnn_trained_test.bin");
        save(&path, &net, 1).unwrap();
        let ckpt = load(&path).unwrap();
        assert_eq!(ckpt.epoch, 1);
        for (a, b) in ckpt.net.layers.iter().zip(&net.layers) {
            for (x, y) in a.log_tau.iter().zip(&b.log_tau) {
                assert!((x - y).abs() < 1e-6);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("aurora_bad_magic.bin");
        std::fs::write(&path, b"NOTAURORA0000").unwrap();
        let r = load(&path);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&path);
    }
}
