//! GPU infrastructure for `GatedRecurrentSnn` (Sprint 13, foundation).
//!
//! Backend: wgpu over Vulkan/Metal/DX12 — cross-platform, no CUDA
//! toolchain required. WGSL compute shaders for sparse matvec and the
//! gated cell. CPU↔GPU parity is asserted via integration tests in
//! `tests/gpu_parity.rs`; see also `Sprint 13` in
//! `docs/10_implementacni_plan.md` for scope and exit criteria.
//!
//! This module is gated by the `gpu` cargo feature. The CPU code path
//! has no dependency on wgpu and continues to work without it.
//!
//! Current status: foundation. `GpuContext` initializes a wgpu device
//! and queue; `kernels::sparse_matvec_onehot` runs the sparse one-hot
//! matvec used in the LM forward path. Backward, full forward/backward
//! integration, and Adam port are tracked in Sprint 13 followups.

#![allow(
    clippy::redundant_clone,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate
)]

pub mod buffers;
pub mod context;
pub mod kernels;
pub mod snn_gated_gpu;

pub use context::GpuContext;
pub use snn_gated_gpu::GatedRecurrentSnnGpu;
