//! MNIST IDX file loader (binary format).
//!
//! Default search path is `data/mnist/`; download via `scripts/get_mnist.sh`.
//! Files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
//!        `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub const IMAGE_SIDE: usize = 28;
pub const IMAGE_PIXELS: usize = IMAGE_SIDE * IMAGE_SIDE;

const MAGIC_IMAGES: u32 = 0x0000_0803;
const MAGIC_LABELS: u32 = 0x0000_0801;

/// # Errors
/// Returns an error if the file is missing, has wrong magic, or the
/// declared element count doesn't match the file size.
pub fn load_images<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f64>>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header truncated",
        ));
    }
    let magic = read_u32_be(&buf, 0);
    if magic != MAGIC_IMAGES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected magic 0x{MAGIC_IMAGES:08X}, got 0x{magic:08X}"),
        ));
    }
    let n = read_u32_be(&buf, 4) as usize;
    let rows = read_u32_be(&buf, 8) as usize;
    let cols = read_u32_be(&buf, 12) as usize;
    let pixels = rows * cols;
    if buf.len() != 16 + n * pixels {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("size mismatch: header says {n} × {rows}×{cols}"),
        ));
    }
    let images = (0..n)
        .map(|i| {
            let start = 16 + i * pixels;
            buf[start..start + pixels]
                .iter()
                .map(|&b| f64::from(b) / 255.0)
                .collect()
        })
        .collect();
    Ok(images)
}

/// # Errors
/// Returns an error if the file is missing, has wrong magic, or its
/// length doesn't match the declared label count.
pub fn load_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header truncated",
        ));
    }
    let magic = read_u32_be(&buf, 0);
    if magic != MAGIC_LABELS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected magic 0x{MAGIC_LABELS:08X}, got 0x{magic:08X}"),
        ));
    }
    let n = read_u32_be(&buf, 4) as usize;
    if buf.len() != 8 + n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("size mismatch: header says {n} labels"),
        ));
    }
    Ok(buf[8..].to_vec())
}

fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}
