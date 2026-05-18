//! Typed wrappers over `wgpu::Buffer`. Each `Buffer` is `bytemuck::Pod`
//! cast from `f32` slices on upload and read back to `Vec<f32>` on
//! download.
//!
//! `GpuBuffer` is row-major `[rows][cols]` — same layout as our CPU
//! `Vec<f32>` matrices, so upload/download are straight copies.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::GpuContext;

pub struct GpuBuffer {
    pub(crate) inner: Arc<wgpu::Buffer>,
    pub len: usize,
}

impl GpuBuffer {
    /// Storage buffer initialized from a CPU slice. Usage =
    /// `STORAGE | COPY_DST | COPY_SRC` so the same buffer can be read
    /// back later.
    ///
    /// # Panics
    /// Panics if the device cannot allocate the buffer (out of memory).
    #[must_use]
    pub fn from_slice(ctx: &GpuContext, label: &str, data: &[f32]) -> Self {
        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            inner: Arc::new(buf),
            len: data.len(),
        }
    }

    /// Same as `from_slice` but for `u32` payloads (token IDs).
    #[must_use]
    pub fn from_u32_slice(ctx: &GpuContext, label: &str, data: &[u32]) -> Self {
        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            inner: Arc::new(buf),
            len: data.len(),
        }
    }

    /// Empty zero-initialized storage buffer.
    #[must_use]
    pub fn zeros(ctx: &GpuContext, label: &str, len: usize) -> Self {
        let bytes = vec![0.0_f32; len];
        Self::from_slice(ctx, label, &bytes)
    }

    /// Allocate an uninitialized storage buffer (no host upload). The
    /// contents are undefined; caller must either write before read or
    /// zero via a GPU kernel (e.g. `ZeroBufferKernel`). Saves the host
    /// allocation + PCIe upload that `zeros()` incurs for large buffers.
    #[must_use]
    pub fn empty(ctx: &GpuContext, label: &str, len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            inner: Arc::new(buf),
            len,
        }
    }

    /// Upload from CPU. Length must match.
    pub fn upload(&self, ctx: &GpuContext, data: &[f32]) {
        debug_assert_eq!(data.len(), self.len);
        ctx.queue
            .write_buffer(&self.inner, 0, bytemuck::cast_slice(data));
    }

    /// Download to CPU. Issues a staging-buffer copy + map + readback.
    /// Synchronous via pollster.
    ///
    /// # Panics
    /// Panics if the GPU readback fails (device lost, out of memory,
    /// staging-buffer mapping error).
    #[must_use]
    pub fn download(&self, ctx: &GpuContext) -> Vec<f32> {
        let bytes = (self.len * std::mem::size_of::<f32>()) as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora gpu download staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora gpu download encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, bytes);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .expect("map_async sender dropped")
            .expect("map_async failed");

        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out
    }
}
