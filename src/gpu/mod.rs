//! GPU infrastructure for LNN-CfC. wgpu over Vulkan/Metal/DX12 backend.
//!
//! `GpuContext` holds device + queue. `buffers::GpuBuffer` is a typed
//! wrapper over `wgpu::Buffer` that handles upload/download for `f32`
//! slices. `lnn_gpu::MultiLayerLnnSnnGpu` mirrors the CPU
//! `crate::lnn::MultiLayerLnnSnn` API but keeps weights, activations,
//! and Adam state resident in VRAM.
//!
//! Gated by the `gpu` cargo feature. Without the feature, only the CPU
//! path in `crate::lnn` is compiled.

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
pub mod kernels_lnn;
pub mod lnn_gpu;

pub use context::GpuContext;
pub use lnn_gpu::MultiLayerLnnSnnGpu;
