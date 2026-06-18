//! Opt-in per-region GPU timing via wgpu timestamp queries.
//!
//! Pure diagnostic — never on the training hot path. No-ops gracefully
//! when the adapter lacks `TIMESTAMP_QUERY`. Usage: `begin_frame`, then
//! `mark(enc, "label")` between dispatch groups, `resolve(enc)` once
//! before submit, and after submit read `spans(ctx)` — span `i` is the
//! GPU time from marker `i` to marker `i+1`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::gpu::context::GpuContext;

pub struct GpuProfiler {
    enabled: bool,
    qset: Option<wgpu::QuerySet>,
    resolve_buf: Option<wgpu::Buffer>,
    read_buf: Option<wgpu::Buffer>,
    period_ns: f32,
    capacity: u32,
    labels: Vec<String>,
    next: u32,
}

impl GpuProfiler {
    /// Allocate the query set + resolve/read buffers. If the adapter has
    /// no `TIMESTAMP_QUERY` the profiler is a no-op (`enabled() == false`).
    #[must_use]
    pub fn new(ctx: &GpuContext, capacity: u32) -> Self {
        let enabled = ctx.adapter.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        );
        if !enabled {
            return Self {
                enabled: false,
                qset: None,
                resolve_buf: None,
                read_buf: None,
                period_ns: 0.0,
                capacity: 0,
                labels: Vec::new(),
                next: 0,
            };
        }
        let qset = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("aurora profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        let bytes = u64::from(capacity) * 8;
        let resolve_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora profiler resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora profiler read"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            enabled: true,
            qset: Some(qset),
            resolve_buf: Some(resolve_buf),
            read_buf: Some(read_buf),
            period_ns: ctx.queue.get_timestamp_period(),
            capacity,
            labels: Vec::new(),
            next: 0,
        }
    }

    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn begin_frame(&mut self) {
        self.next = 0;
        self.labels.clear();
    }

    /// Write a timestamp at the current encoder position, labeled `label`.
    pub fn mark(&mut self, enc: &mut wgpu::CommandEncoder, label: &str) {
        if !self.enabled || self.next >= self.capacity {
            return;
        }
        if let Some(qset) = &self.qset {
            enc.write_timestamp(qset, self.next);
            self.labels.push(label.to_string());
            self.next += 1;
        }
    }

    /// Encode the resolve of all written timestamps. Call once on the
    /// encoder right before `submit`.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.next == 0 {
            return;
        }
        if let (Some(qset), Some(resolve), Some(read)) =
            (&self.qset, &self.resolve_buf, &self.read_buf)
        {
            enc.resolve_query_set(qset, 0..self.next, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, read, 0, u64::from(self.next) * 8);
        }
    }

    /// Read back the captured spans as `(label, milliseconds)`. Call after
    /// the encoder carrying `resolve` was submitted. Blocks on the GPU.
    #[must_use]
    pub fn spans(&self, ctx: &GpuContext) -> Vec<(String, f64)> {
        if !self.enabled || self.next < 2 {
            return Vec::new();
        }
        let read = self.read_buf.as_ref().unwrap();
        let slice = read.slice(..u64::from(self.next) * 8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map sender dropped").expect("map failed");
        let mapped = slice.get_mapped_range();
        let ticks: Vec<u64> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        read.unmap();

        let mut out = Vec::with_capacity(ticks.len().saturating_sub(1));
        for i in 0..ticks.len().saturating_sub(1) {
            let dt = ticks[i + 1].saturating_sub(ticks[i]);
            let ms = f64::from(self.period_ns) * dt as f64 / 1.0e6;
            out.push((self.labels[i].clone(), ms));
        }
        out
    }
}

/// Diagnostic probe: time a device-side buffer copy of `floats` f32s.
/// Returns the measured GPU milliseconds, or `None` if the adapter lacks
/// timestamp support. Validates the whole query → resolve → readback path
/// on the running device.
#[must_use]
pub fn probe_copy_ms(ctx: &GpuContext, floats: u64) -> Option<f64> {
    let mut prof = GpuProfiler::new(ctx, 4);
    if !prof.enabled() {
        return None;
    }
    let bytes = floats * 4;
    let src = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("profiler probe src"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("profiler probe dst"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    prof.begin_frame();
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("profiler probe"),
        });
    prof.mark(&mut enc, "copy");
    enc.copy_buffer_to_buffer(&src, 0, &dst, 0, bytes);
    prof.mark(&mut enc, "end");
    prof.resolve(&mut enc);
    ctx.queue.submit(Some(enc.finish()));
    prof.spans(ctx).first().map(|(_, ms)| *ms)
}
