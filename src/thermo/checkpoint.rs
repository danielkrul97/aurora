#![allow(clippy::cast_possible_truncation, clippy::missing_errors_doc, clippy::missing_panics_doc)]

//! THERMO01 checkpoint format for the thermodynamic LM.
//!
//! Mirrors the AURORA03 transformer checkpoint style: little-endian magic
//! + config dims + raw f32 parameter slices, in a fixed order. AdamW
//! moments are NOT serialized — resume starts with fresh optimizer state
//! (same convention as AURORA03). Reuses the binary helpers from
//! `crate::checkpoint`.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::checkpoint::{read_f32_vec, read_u32, read_u64, write_f32_slice, write_u32, write_u64};
use crate::thermo::field::ThermoConfig;
use crate::thermo::model::ThermoLM;

const MAGIC: &[u8; 8] = b"THERMO01";

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub model: ThermoLM,
    pub epoch: u32,
    pub step: u64,
}

pub fn save<P: AsRef<Path>>(path: P, model: &ThermoLM, epoch: u32, step: u64) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    let c = &model.config;

    w.write_all(MAGIC)?;
    write_u32(&mut w, c.vocab as u32)?;
    write_u32(&mut w, c.h as u32)?;
    write_u32(&mut w, c.w as u32)?;
    write_u32(&mut w, c.c as u32)?;
    write_u32(&mut w, c.conv_hidden as u32)?;
    write_u32(&mut w, c.k_steps as u32)?;
    write_u32(&mut w, c.pool as u32)?;
    write_f32_slice(&mut w, &[c.dt, c.t_init, c.anneal_tmax, c.anneal_alpha])?;
    write_u32(&mut w, c.conserved.len() as u32)?;
    for &ci in &c.conserved {
        write_u32(&mut w, ci as u32)?;
    }
    write_u32(&mut w, u32::from(c.readout_moments))?;
    write_u32(&mut w, u32::from(c.readout_spectral))?;
    write_u32(&mut w, c.readout_modes as u32)?;
    write_u32(&mut w, u32::from(c.use_diffusion))?;
    write_u32(&mut w, epoch)?;
    write_u64(&mut w, step)?;

    write_f32_slice(&mut w, &model.step.conv.w1)?;
    write_f32_slice(&mut w, &model.step.conv.b1)?;
    write_f32_slice(&mut w, &model.step.conv.w2)?;
    write_f32_slice(&mut w, &model.step.conv.b2)?;
    write_f32_slice(&mut w, &model.step.diffusion.k0)?;
    write_f32_slice(&mut w, &model.step.diffusion.a)?;
    write_f32_slice(&mut w, &model.step.diffusion.b)?;
    write_f32_slice(&mut w, &model.inject.inj)?;
    write_f32_slice(&mut w, &model.readout.proj.w)?;
    write_f32_slice(&mut w, &model.readout.bias)?;
    w.flush()
}

pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Checkpoint> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad checkpoint magic — expected THERMO01"));
    }
    let vocab = read_u32(&mut r)? as usize;
    let h = read_u32(&mut r)? as usize;
    let w = read_u32(&mut r)? as usize;
    let c = read_u32(&mut r)? as usize;
    let conv_hidden = read_u32(&mut r)? as usize;
    let k_steps = read_u32(&mut r)? as usize;
    let pool = read_u32(&mut r)? as usize;
    let floats = read_f32_vec(&mut r, 4)?;
    let n_cons = read_u32(&mut r)? as usize;
    let mut conserved = Vec::with_capacity(n_cons);
    for _ in 0..n_cons {
        conserved.push(read_u32(&mut r)? as usize);
    }
    let readout_moments = read_u32(&mut r)? != 0;
    let readout_spectral = read_u32(&mut r)? != 0;
    let readout_modes = read_u32(&mut r)? as usize;
    let use_diffusion = read_u32(&mut r)? != 0;
    let epoch = read_u32(&mut r)?;
    let step = read_u64(&mut r)?;

    let cfg = ThermoConfig {
        vocab,
        h,
        w,
        c,
        conv_hidden,
        conserved,
        k_steps,
        pool,
        dt: floats[0],
        t_init: floats[1],
        anneal_tmax: floats[2],
        anneal_alpha: floats[3],
        readout_moments,
        readout_spectral,
        readout_modes,
        use_diffusion,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = ThermoLM::new(cfg, &mut rng);

    model.step.conv.w1 = read_f32_vec(&mut r, model.step.conv.w1.len())?;
    model.step.conv.b1 = read_f32_vec(&mut r, model.step.conv.b1.len())?;
    model.step.conv.w2 = read_f32_vec(&mut r, model.step.conv.w2.len())?;
    model.step.conv.b2 = read_f32_vec(&mut r, model.step.conv.b2.len())?;
    model.step.diffusion.k0 = read_f32_vec(&mut r, model.config.c)?;
    model.step.diffusion.a = read_f32_vec(&mut r, model.config.c)?;
    model.step.diffusion.b = read_f32_vec(&mut r, model.config.c)?;
    model.inject.inj = read_f32_vec(&mut r, model.inject.inj.len())?;
    model.readout.proj.w = read_f32_vec(&mut r, model.readout.proj.w.len())?;
    model.readout.bias = read_f32_vec(&mut r, model.readout.bias.len())?;
    model.step_count = step;

    Ok(Checkpoint { model, epoch, step })
}
