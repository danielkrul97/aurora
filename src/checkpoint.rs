#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::too_many_lines
)]

//! AURORA04 checkpoint format for the decoder-only transformer.
//!
//! Binary layout (little-endian):
//! ```text
//! magic       : [u8; 8] = b"AURORA04"
//! vocab       : u32
//! d_model     : u32
//! n_layer     : u32
//! n_head      : u32
//! d_ff        : u32
//! seq_len_max : u32
//! flags       : u32   (bit 0 = rope; w_pos is stored either way)
//! epoch       : u32
//! step        : u64
//!
//! w_tok       : f32[vocab * d_model]
//! w_pos       : f32[seq_len_max * d_model]
//! per block (× n_layer):
//!   ln1_gamma : f32[d_model]
//!   ln1_beta  : f32[d_model]
//!   w_q       : f32[d_model * d_model]
//!   w_k       : f32[d_model * d_model]
//!   w_v       : f32[d_model * d_model]
//!   w_o       : f32[d_model * d_model]
//!   ln2_gamma : f32[d_model]
//!   ln2_beta  : f32[d_model]
//!   w_fc      : f32[d_ff * d_model]
//!   w_proj    : f32[d_model * d_ff]
//! final_gamma : f32[d_model]
//! final_beta  : f32[d_model]
//! ```
//!
//! AdamW moments are NOT serialized — resume uses fresh state. The
//! warmup cost is negligible vs the moment-buffer size (~2× model bytes).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::transformer::{DecoderTransformer, TransformerConfig};

const MAGIC: &[u8; 8] = b"AURORA04";

const FLAG_ROPE: u32 = 1;

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub model: DecoderTransformer,
    pub epoch: u32,
    pub step: u64,
}

/// Save model + training progress to disk.
pub fn save<P: AsRef<Path>>(
    path: P,
    model: &DecoderTransformer,
    epoch: u32,
    step: u64,
) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    write_u32(&mut w, model.config.vocab as u32)?;
    write_u32(&mut w, model.config.d_model as u32)?;
    write_u32(&mut w, model.config.n_layer as u32)?;
    write_u32(&mut w, model.config.n_head as u32)?;
    write_u32(&mut w, model.config.d_ff as u32)?;
    write_u32(&mut w, model.config.seq_len_max as u32)?;
    write_u32(&mut w, if model.config.rope { FLAG_ROPE } else { 0 })?;
    write_u32(&mut w, epoch)?;
    write_u64(&mut w, step)?;

    write_f32_slice(&mut w, &model.embed.w_tok)?;
    write_f32_slice(&mut w, &model.embed.w_pos)?;
    for blk in &model.blocks {
        write_f32_slice(&mut w, &blk.ln1.gamma)?;
        write_f32_slice(&mut w, &blk.ln1.beta)?;
        write_f32_slice(&mut w, &blk.mha.w_q.w)?;
        write_f32_slice(&mut w, &blk.mha.w_k.w)?;
        write_f32_slice(&mut w, &blk.mha.w_v.w)?;
        write_f32_slice(&mut w, &blk.mha.w_o.w)?;
        write_f32_slice(&mut w, &blk.ln2.gamma)?;
        write_f32_slice(&mut w, &blk.ln2.beta)?;
        write_f32_slice(&mut w, &blk.ffn.w_fc.w)?;
        write_f32_slice(&mut w, &blk.ffn.w_proj.w)?;
    }
    write_f32_slice(&mut w, &model.final_ln.gamma)?;
    write_f32_slice(&mut w, &model.final_ln.beta)?;
    w.flush()
}

/// Load a checkpoint. Returned model has fresh AdamW moments.
pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Checkpoint> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad checkpoint magic — expected AURORA04",
        ));
    }
    let vocab = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let n_layer = read_u32(&mut r)? as usize;
    let n_head = read_u32(&mut r)? as usize;
    let d_ff = read_u32(&mut r)? as usize;
    let seq_len_max = read_u32(&mut r)? as usize;
    let flags = read_u32(&mut r)?;
    let epoch = read_u32(&mut r)?;
    let step = read_u64(&mut r)?;

    let cfg = TransformerConfig {
        vocab,
        d_model,
        n_layer,
        n_head,
        d_ff,
        seq_len_max,
        rope: flags & FLAG_ROPE != 0,
    };

    let mut rng = StdRng::seed_from_u64(0);
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    model.embed.w_tok = read_f32_vec(&mut r, vocab * d_model)?;
    model.embed.w_pos = read_f32_vec(&mut r, seq_len_max * d_model)?;
    for blk in &mut model.blocks {
        blk.ln1.gamma = read_f32_vec(&mut r, d_model)?;
        blk.ln1.beta = read_f32_vec(&mut r, d_model)?;
        blk.mha.w_q.w = read_f32_vec(&mut r, d_model * d_model)?;
        blk.mha.w_k.w = read_f32_vec(&mut r, d_model * d_model)?;
        blk.mha.w_v.w = read_f32_vec(&mut r, d_model * d_model)?;
        blk.mha.w_o.w = read_f32_vec(&mut r, d_model * d_model)?;
        blk.ln2.gamma = read_f32_vec(&mut r, d_model)?;
        blk.ln2.beta = read_f32_vec(&mut r, d_model)?;
        blk.ffn.w_fc.w = read_f32_vec(&mut r, d_ff * d_model)?;
        blk.ffn.w_proj.w = read_f32_vec(&mut r, d_model * d_ff)?;
    }
    model.final_ln.gamma = read_f32_vec(&mut r, d_model)?;
    model.final_ln.beta = read_f32_vec(&mut r, d_model)?;
    model.step = step;

    Ok(Checkpoint {
        model,
        epoch,
        step,
    })
}

// --- low-level binary helpers ------------------------------------------

pub fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn write_f32_slice<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
    };
    w.write_all(bytes)
}
pub fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
pub fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
pub fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
pub fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0_f32; n];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * 4)
    };
    r.read_exact(bytes)?;
    Ok(out)
}
