#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::too_many_lines
)]

//! AURORA07 checkpoint format for the decoder-only transformer.
//!
//! Binary layout (little-endian):
//! ```text
//! magic       : [u8; 8] = b"AURORA07"
//! vocab       : u32
//! d_model     : u32
//! n_layer     : u32
//! n_head      : u32
//! d_ff        : u32
//! seq_len_max : u32
//! flags       : u32   (bit 0 = rope; 1 = rmsnorm; 2 = relu2; 3 = qk_norm;
//!                      4 = untied_head; 5 = swiglu; 6 = layerscale;
//!                      7 = parallel_block; 8 = diff_attn; 9 = head_bias)
//! softcap     : f32   (0.0 = off)
//! canon_kernel: u32   (V6+; 0 = off)
//! sink_tokens : u32   (V7+; 0 = off)
//! epoch       : u32
//! step        : u64
//!
//! w_tok       : f32[vocab * d_model]
//! w_pos       : f32[seq_len_max * d_model]
//! w_head      : f32[vocab * d_model]            (only when untied_head)
//! head_bias   : f32[vocab]                      (V7+, only when head_bias)
//! sink        : f32[sink_tokens * d_model]      (V7+, only when sink_tokens>0)
//! per block (× n_layer):
//!   ln1_gamma : f32[d_model]
//!   ln1_beta  : f32[d_model]
//!   w_q/w_k/w_v/w_o : f32[d_model * d_model] each
//!   ln2_gamma : f32[d_model]
//!   ln2_beta  : f32[d_model]
//!   w_fc      : f32[d_ff * d_model]
//!   w_proj    : f32[d_model * d_ff]
//!   canon_w   : f32[d_model * canon_kernel]     (V6+, only when canon)
//!   w_gate    : f32[d_ff * d_model]             (V7+, only when swiglu)
//!   ls1/ls2   : f32[d_model] each               (V7+, only when layerscale)
//!   lambda    : f32[n_head]                     (V7+, only when diff_attn)
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

// V7 adds the swiglu/layerscale/parallel_block/diff_attn/head_bias flags
// (bits 5–9), a `sink_tokens: u32` header field, and the matching
// conditional tensors (per-block w_gate/ls1/ls2/lambda, global head_bias
// and sink). V6 added `canon_kernel: u32` + per-block canon_w. V5/V6 are
// still read (newer features default off); writes are always V7.
// (embed_lr_mult is a training-time LR knob, not serialized.)
const MAGIC_V5: &[u8; 8] = b"AURORA05";
const MAGIC_V6: &[u8; 8] = b"AURORA06";
const MAGIC_V7: &[u8; 8] = b"AURORA07";

const FLAG_ROPE: u32 = 1;
const FLAG_RMSNORM: u32 = 1 << 1;
const FLAG_RELU2: u32 = 1 << 2;
const FLAG_QK_NORM: u32 = 1 << 3;
const FLAG_UNTIED_HEAD: u32 = 1 << 4;
const FLAG_SWIGLU: u32 = 1 << 5;
const FLAG_LAYERSCALE: u32 = 1 << 6;
const FLAG_PARALLEL: u32 = 1 << 7;
const FLAG_DIFF_ATTN: u32 = 1 << 8;
const FLAG_HEAD_BIAS: u32 = 1 << 9;

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

    w.write_all(MAGIC_V7)?;
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
    if model.config.swiglu {
        flags |= FLAG_SWIGLU;
    }
    if model.config.layerscale {
        flags |= FLAG_LAYERSCALE;
    }
    if model.config.parallel_block {
        flags |= FLAG_PARALLEL;
    }
    if model.config.diff_attn {
        flags |= FLAG_DIFF_ATTN;
    }
    if model.config.head_bias {
        flags |= FLAG_HEAD_BIAS;
    }
    write_u32(&mut w, flags)?;
    write_f32(&mut w, model.config.softcap)?;
    write_u32(&mut w, model.config.canon_kernel as u32)?;
    write_u32(&mut w, model.config.sink_tokens as u32)?;
    write_u32(&mut w, epoch)?;
    write_u64(&mut w, step)?;

    write_f32_slice(&mut w, &model.embed.w_tok)?;
    write_f32_slice(&mut w, &model.embed.w_pos)?;
    if let Some(w_head) = model.w_head.as_ref() {
        write_f32_slice(&mut w, w_head)?;
    }
    if let Some(hb) = model.head_bias.as_ref() {
        write_f32_slice(&mut w, hb)?;
    }
    if let Some(sink) = model.sink.as_ref() {
        write_f32_slice(&mut w, sink)?;
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
        if model.config.canon_kernel > 0 {
            write_f32_slice(&mut w, &blk.canon_w)?;
        }
        if let Some(wg) = blk.ffn.w_gate.as_ref() {
            write_f32_slice(&mut w, &wg.w)?;
        }
        if model.config.layerscale {
            write_f32_slice(&mut w, &blk.ls1)?;
            write_f32_slice(&mut w, &blk.ls2)?;
        }
        if model.config.diff_attn {
            write_f32_slice(&mut w, &blk.mha.lambda)?;
        }
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
    let v6 = &magic == MAGIC_V6;
    let v7 = &magic == MAGIC_V7;
    if &magic != MAGIC_V5 && !v6 && !v7 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad checkpoint magic — expected AURORA05, AURORA06, or AURORA07",
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
    let canon_kernel = if v6 || v7 { read_u32(&mut r)? as usize } else { 0 };
    let sink_tokens = if v7 { read_u32(&mut r)? as usize } else { 0 };
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
        canon_kernel,
        swiglu: flags & FLAG_SWIGLU != 0,
        layerscale: flags & FLAG_LAYERSCALE != 0,
        parallel_block: flags & FLAG_PARALLEL != 0,
        diff_attn: flags & FLAG_DIFF_ATTN != 0,
        head_bias: flags & FLAG_HEAD_BIAS != 0,
        sink_tokens,
        ..TransformerConfig::default()
    };

    let mut rng = StdRng::seed_from_u64(0);
    let mut model = DecoderTransformer::new(cfg.clone(), &mut rng);

    model.embed.w_tok = read_f32_vec(&mut r, vocab * d_model)?;
    model.embed.w_pos = read_f32_vec(&mut r, seq_len_max * d_model)?;
    if cfg.untied_head {
        model.w_head = Some(read_f32_vec(&mut r, vocab * d_model)?);
    }
    if cfg.head_bias {
        model.head_bias = Some(read_f32_vec(&mut r, vocab)?);
    }
    if sink_tokens > 0 {
        model.sink = Some(read_f32_vec(&mut r, sink_tokens * d_model)?);
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
        if canon_kernel > 0 {
            blk.canon_w = read_f32_vec(&mut r, d_model * canon_kernel)?;
        }
        if let Some(wg) = blk.ffn.w_gate.as_mut() {
            wg.w = read_f32_vec(&mut r, d_ff * d_model)?;
        }
        if cfg.layerscale {
            blk.ls1 = read_f32_vec(&mut r, d_model)?;
            blk.ls2 = read_f32_vec(&mut r, d_model)?;
        }
        if cfg.diff_attn {
            blk.mha.lambda = read_f32_vec(&mut r, n_head)?;
        }
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
