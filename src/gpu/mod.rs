//! GPU infrastructure for the bipolar HDC stack. wgpu over
//! Vulkan/Metal/DX12 backend.
//!
//! `GpuContext` holds device + queue. `buffers::GpuBuffer` is a typed
//! wrapper over `wgpu::Buffer` that handles upload/download for `f32`
//! slices. `hdc_gpu::HdcGpu` mirrors the CPU `crate::hdc::HdcLm` API
//! but keeps the codebook, readout weights, and Adam state resident in
//! VRAM.
//!
//! Gated by the `gpu` cargo feature.

#![allow(
    clippy::redundant_clone,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

pub mod buffers;
pub mod context;
pub mod hdc_gpu;
pub mod kernels_common;
pub mod kernels_hdc;

pub use context::GpuContext;
pub use hdc_gpu::HdcGpu;
