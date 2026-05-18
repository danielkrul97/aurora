#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

//! Weight checkpointing for `crate::hdc::HdcLm`.
//!
//! Binary format (little-endian, magic `AURORA-HDC01`):
//! ```text
//! magic       : [u8; 12] = b"AURORA-HDC01"
//! _pad        : [u8; 4]
//! dim         : u32
//! vocab       : u32
//! epoch       : u32
//! _reserved   : u32
//! codebook    : f32[vocab * dim]   (frozen ±1, but stored as f32 for simplicity)
//! w_out       : f32[vocab * dim]
//! b_o         : f32[vocab]
//! ```
//!
//! Adam state is intentionally not serialized — resuming uses fresh
//! moments (the optimizer warm-up over a few hundred steps is acceptable).
//!
//! Breaking change vs. legacy `AURORA-LNN01` / `AURORA01..02` formats.
//! HDC has a fundamentally different parameter layout (codebook + flat
//! linear readout vs. per-layer LNN matrices), so old checkpoints are
//! not loadable.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::hdc::HdcLm;

const MAGIC: &[u8; 12] = b"AURORA-HDC01";

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub net: HdcLm,
    pub epoch: u32,
}

/// Save a network plus the number of completed epochs.
///
/// # Errors
/// Propagates the underlying I/O error if the file cannot be written.
pub fn save<P: AsRef<Path>>(path: P, net: &HdcLm, epoch: u32) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    w.write_all(&[0u8; 4])?;
    write_u32(&mut w, net.dim as u32)?;
    write_u32(&mut w, net.vocab as u32)?;
    write_u32(&mut w, epoch)?;
    write_u32(&mut w, 0)?; // reserved

    write_f32_slice(&mut w, &net.codebook.items)?;
    write_f32_slice(&mut w, &net.readout.w)?;
    write_f32_slice(&mut w, &net.readout.b)?;
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
            "bad checkpoint magic — expected AURORA-HDC01",
        ));
    }
    let mut pad = [0u8; 4];
    r.read_exact(&mut pad)?;

    let dim = read_u32(&mut r)? as usize;
    let vocab = read_u32(&mut r)? as usize;
    let epoch = read_u32(&mut r)?;
    let _reserved = read_u32(&mut r)?;

    let codebook = read_f32_vec(&mut r, vocab * dim)?;
    let w_out = read_f32_vec(&mut r, vocab * dim)?;
    let b_o = read_f32_vec(&mut r, vocab)?;

    let mut rng = StdRng::seed_from_u64(0);
    let mut net = HdcLm::new(dim, vocab, &mut rng);
    net.codebook.items = codebook;
    net.readout.w = w_out;
    net.readout.b = b_o;

    Ok(Checkpoint { net, epoch })
}

// --- low-level binary helpers -------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32_slice<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
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
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * 4) };
    r.read_exact(bytes)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HdcState;
    use crate::optimizer::AdamConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn save_load_roundtrip_preserves_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let net = HdcLm::new(64, 32, &mut rng);
        let path = std::env::temp_dir().join("aurora_hdc_ckpt_test.bin");
        save(&path, &net, 7).unwrap();
        let ckpt = load(&path).unwrap();
        assert_eq!(ckpt.epoch, 7);
        assert_eq!(ckpt.net.dim, net.dim);
        assert_eq!(ckpt.net.vocab, net.vocab);
        assert_eq!(ckpt.net.codebook.items, net.codebook.items);
        assert_eq!(ckpt.net.readout.w, net.readout.w);
        assert_eq!(ckpt.net.readout.b, net.readout.b);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_load_after_training_step() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut net = HdcLm::new(32, 8, &mut rng);
        let inputs_owned: Vec<Vec<u32>> = (0..2)
            .map(|b| (0..16).map(|t| ((t + b) % 8) as u32).collect())
            .collect();
        let targets_owned: Vec<Vec<u32>> = (0..2)
            .map(|b| (0..16).map(|t| ((t + b + 1) % 8) as u32).collect())
            .collect();
        let inputs: Vec<&[u32]> = inputs_owned.iter().map(Vec::as_slice).collect();
        let targets: Vec<&[u32]> = targets_owned.iter().map(Vec::as_slice).collect();
        let mut states: Vec<HdcState> = (0..2).map(|_| net.fresh_state()).collect();
        let _ = net.forward_backward_batch_tokens_stateful(&inputs, &targets, &mut states);
        net.clip_grads(1.0);
        net.step_adam(&AdamConfig::default());
        let path = std::env::temp_dir().join("aurora_hdc_trained_test.bin");
        save(&path, &net, 1).unwrap();
        let ckpt = load(&path).unwrap();
        assert_eq!(ckpt.epoch, 1);
        for (a, b) in ckpt.net.readout.w.iter().zip(&net.readout.w) {
            assert!((a - b).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("aurora_hdc_bad_magic.bin");
        std::fs::write(&path, b"NOTAURORA0000").unwrap();
        let r = load(&path);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&path);
    }
}
