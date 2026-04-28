#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

//! Sprint 28: weight checkpointing for `MultiLayerGatedSnn`.
//!
//! Binary format (little-endian):
//! ```text
//! magic       : [u8; 8] = b"AURORA01"
//! n_input     : u32
//! n_output    : u32
//! n_layers    : u32
//! threshold   : f32
//! alpha       : f32
//! epoch       : u32   (training metadata: completed epochs)
//! _pad        : u32
//! per layer:
//!   n_in      : u32
//!   n_hidden  : u32
//!   w_in      : f32[n_hidden * n_in]
//!   w_rec     : f32[n_hidden * n_hidden]
//!   w_gate_in : f32[n_hidden * n_in]
//!   w_gate_rec: f32[n_hidden * n_hidden]
//!   b_h       : f32[n_hidden]
//!   b_gate    : f32[n_hidden]
//! w_out       : f32[n_output * top_hidden]
//! b_o         : f32[n_output]
//! ```
//!
//! Adam state is intentionally NOT serialized — resuming uses fresh
//! moments. The slight optimizer warmup is acceptable; storing m1/m2
//! would roughly triple checkpoint size for a feature most of the time.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::snn_gated_multi::MultiLayerGatedSnn;
use crate::surrogate::SpikeFn;

const MAGIC: &[u8; 8] = b"AURORA01";

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub net: MultiLayerGatedSnn,
    pub epoch: u32,
}

/// Save a network plus the number of completed epochs.
///
/// # Errors
/// Propagates the underlying I/O error if the file cannot be written.
pub fn save<P: AsRef<Path>>(path: P, net: &MultiLayerGatedSnn, epoch: u32) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    write_u32(&mut w, net.n_input as u32)?;
    write_u32(&mut w, net.n_output as u32)?;
    write_u32(&mut w, net.layers.len() as u32)?;
    write_f32(&mut w, net.spike_fn.threshold)?;
    write_f32(&mut w, net.spike_fn.alpha)?;
    write_u32(&mut w, epoch)?;
    write_u32(&mut w, 0)?;

    for layer in &net.layers {
        write_u32(&mut w, layer.n_in as u32)?;
        write_u32(&mut w, layer.n_hidden as u32)?;
        write_f32_slice(&mut w, &layer.w_in)?;
        write_f32_slice(&mut w, &layer.w_rec)?;
        write_f32_slice(&mut w, &layer.w_gate_in)?;
        write_f32_slice(&mut w, &layer.w_gate_rec)?;
        write_f32_slice(&mut w, &layer.b_h)?;
        write_f32_slice(&mut w, &layer.b_gate)?;
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

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad checkpoint magic",
        ));
    }
    let n_input = read_u32(&mut r)? as usize;
    let n_output = read_u32(&mut r)? as usize;
    let n_layers = read_u32(&mut r)? as usize;
    let threshold = read_f32(&mut r)?;
    let alpha = read_f32(&mut r)?;
    let epoch = read_u32(&mut r)?;
    let _pad = read_u32(&mut r)?;

    let mut hidden_dims = Vec::with_capacity(n_layers);
    let mut layer_blobs: Vec<[Vec<f32>; 6]> = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_in = read_u32(&mut r)? as usize;
        let n_hidden = read_u32(&mut r)? as usize;
        hidden_dims.push(n_hidden);
        let w_in = read_f32_vec(&mut r, n_hidden * n_in)?;
        let w_rec = read_f32_vec(&mut r, n_hidden * n_hidden)?;
        let w_gate_in = read_f32_vec(&mut r, n_hidden * n_in)?;
        let w_gate_rec = read_f32_vec(&mut r, n_hidden * n_hidden)?;
        let b_h = read_f32_vec(&mut r, n_hidden)?;
        let b_gate = read_f32_vec(&mut r, n_hidden)?;
        layer_blobs.push([w_in, w_rec, w_gate_in, w_gate_rec, b_h, b_gate]);
    }
    let top_hidden = *hidden_dims.last().expect("≥1 layer");
    let w_out = read_f32_vec(&mut r, n_output * top_hidden)?;
    let b_o = read_f32_vec(&mut r, n_output)?;

    let mut rng = StdRng::seed_from_u64(0);
    let mut net = MultiLayerGatedSnn::new(
        n_input,
        &hidden_dims,
        n_output,
        SpikeFn { threshold, alpha },
        &mut rng,
    );
    for (i, blob) in layer_blobs.into_iter().enumerate() {
        let [w_in, w_rec, w_gate_in, w_gate_rec, b_h, b_gate] = blob;
        let layer = &mut net.layers[i];
        layer.w_in = w_in;
        layer.w_rec = w_rec;
        layer.w_gate_in = w_gate_in;
        layer.w_gate_rec = w_gate_rec;
        layer.b_h = b_h;
        layer.b_gate = b_gate;
    }
    net.w_out = w_out;
    net.b_o = b_o;

    Ok(Checkpoint { net, epoch })
}

// --- low-level binary helpers -------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32_slice<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    // SAFETY: f32 has the same alignment+size as 4 raw bytes; the slice
    // is only borrowed for the duration of write_all.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
    };
    w.write_all(bytes)
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0_f32; n];
    // SAFETY: f32 layout is 4 bytes; the borrow lives only for read_exact.
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * 4)
    };
    r.read_exact(bytes)?;
    Ok(out)
}
