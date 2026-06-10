#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::too_many_lines
)]

//! AURORA06 checkpoint format for the decoder-only transformer.
//!
//! Binary layout (little-endian):
//! ```text
//! magic        : [u8; 8] = b"AURORA06"
//! vocab        : u32
//! d_model      : u32
//! n_layer      : u32   (logical depth; tie_layers stores 1 block)
//! n_head       : u32
//! d_ff         : u32
//! seq_len_max  : u32
//! flags        : u32   (bit 0 = rope; bit 1 = rmsnorm; bit 2 = relu2;
//!                       bit 3 = qk_norm; bit 4 = untied_head;
//!                       bit 5 = value_resid; bit 6 = tie_layers.
//!                       w_pos and LN β are stored either way)
//! softcap      : f32   (0.0 = off)
//! mtp_weight   : f32   (0.0 = off; > 0 ⇒ w_head2 tensor present)
//! local_window : u32   (0 = full attention)
//! global_layers: u32
//! ngram_rows   : u32   (0 = off; > 0 ⇒ w_ngram tensor present)
//! epoch        : u32
//! step         : u64
//!
//! w_tok       : f32[vocab * d_model]
//! w_pos       : f32[seq_len_max * d_model]
//! w_head      : f32[vocab * d_model]   (only when untied_head)
//! w_head2     : f32[vocab * d_model]   (only when mtp_weight > 0)
//! w_ngram     : f32[ngram_rows * d_model] (only when ngram_rows > 0)
//! lambda_v    : f32[2 * blocks]        (only when value_resid)
//! per block (× n_layer, or × 1 when tie_layers):
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

const MAGIC: &[u8; 8] = b"AURORA06";

const FLAG_ROPE: u32 = 1;
const FLAG_RMSNORM: u32 = 1 << 1;
const FLAG_RELU2: u32 = 1 << 2;
const FLAG_QK_NORM: u32 = 1 << 3;
const FLAG_UNTIED_HEAD: u32 = 1 << 4;
const FLAG_VALUE_RESID: u32 = 1 << 5;
const FLAG_TIE_LAYERS: u32 = 1 << 6;

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
    let mut flags = 0u32;
    if model.config.rope {
        flags |= FLAG_ROPE;
    }
    if model.config.rmsnorm {
        flags |= FLAG_RMSNORM;
    }
    if model.config.relu2 {
        flags |= FLAG_RELU2;
    }
    if model.config.qk_norm {
        flags |= FLAG_QK_NORM;
    }
    if model.config.untied_head {
        flags |= FLAG_UNTIED_HEAD;
    }
    if model.config.value_resid {
        flags |= FLAG_VALUE_RESID;
    }
    if model.config.tie_layers {
        flags |= FLAG_TIE_LAYERS;
    }
    write_u32(&mut w, flags)?;
    write_f32(&mut w, model.config.softcap)?;
    write_f32(&mut w, model.config.mtp_weight)?;
    write_u32(&mut w, model.config.local_window as u32)?;
    write_u32(&mut w, model.config.global_layers as u32)?;
    write_u32(&mut w, model.config.ngram_rows as u32)?;
    write_u32(&mut w, epoch)?;
    write_u64(&mut w, step)?;

    write_f32_slice(&mut w, &model.embed.w_tok)?;
    write_f32_slice(&mut w, &model.embed.w_pos)?;
    if let Some(w_head) = model.w_head.as_ref() {
        write_f32_slice(&mut w, w_head)?;
    }
    if let Some(w_head2) = model.w_head2.as_ref() {
        write_f32_slice(&mut w, w_head2)?;
    }
    if model.config.ngram_rows > 0 {
        write_f32_slice(&mut w, &model.embed.w_ngram)?;
    }
    if model.config.value_resid {
        let lambdas: Vec<f32> = model.blocks.iter().flat_map(|b| b.mha.lambda_v).collect();
        write_f32_slice(&mut w, &lambdas)?;
    }
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
            "bad checkpoint magic — expected AURORA06",
        ));
    }
    let vocab = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let n_layer = read_u32(&mut r)? as usize;
    let n_head = read_u32(&mut r)? as usize;
    let d_ff = read_u32(&mut r)? as usize;
    let seq_len_max = read_u32(&mut r)? as usize;
    let flags = read_u32(&mut r)?;
    let softcap = read_f32(&mut r)?;
    let mtp_weight = read_f32(&mut r)?;
    let local_window = read_u32(&mut r)? as usize;
    let global_layers = read_u32(&mut r)? as usize;
    let ngram_rows = read_u32(&mut r)? as usize;
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
        rmsnorm: flags & FLAG_RMSNORM != 0,
        relu2: flags & FLAG_RELU2 != 0,
        qk_norm: flags & FLAG_QK_NORM != 0,
        untied_head: flags & FLAG_UNTIED_HEAD != 0,
        softcap,
        zero_ffn_proj: false,
        mtp_weight,
        value_resid: flags & FLAG_VALUE_RESID != 0,
        tie_layers: flags & FLAG_TIE_LAYERS != 0,
        local_window,
        global_layers,
        ngram_rows,
    };

    let mut rng = StdRng::seed_from_u64(0);
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    model.embed.w_tok = read_f32_vec(&mut r, vocab * d_model)?;
    model.embed.w_pos = read_f32_vec(&mut r, seq_len_max * d_model)?;
    if cfg.untied_head {
        model.w_head = Some(read_f32_vec(&mut r, vocab * d_model)?);
    }
    if cfg.mtp_weight > 0.0 {
        model.w_head2 = Some(read_f32_vec(&mut r, vocab * d_model)?);
    }
    if cfg.ngram_rows > 0 {
        model.embed.w_ngram = read_f32_vec(&mut r, cfg.ngram_rows * d_model)?;
    }
    if cfg.value_resid {
        let lam = read_f32_vec(&mut r, 2 * model.blocks.len())?;
        for (bi, blk) in model.blocks.iter_mut().enumerate() {
            blk.mha.lambda_v = [lam[2 * bi], lam[2 * bi + 1]];
        }
    }
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
