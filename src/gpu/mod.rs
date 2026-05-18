//! GPU infrastructure for the decoder-only transformer.
//!
//! Backend: wgpu over Vulkan/Metal/DX12 — cross-platform, no CUDA
//! toolchain required. Gated by the `gpu` cargo feature; CPU code path
//! has no dependency on wgpu and continues to work without it.

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

pub mod attention;
pub mod buffers;
pub mod context;
pub mod elementwise;
pub mod embedding;
pub mod gelu;
pub mod kv_cache;
pub mod layer_norm;
pub mod matmul;
pub mod optimizer;
pub mod reduce;
pub mod softmax_ce;
pub mod transformer;

pub use attention::{CausalMhaBackwardKernel, CausalMhaForwardKernel};
pub use context::GpuContext;
pub use elementwise::{ScaleBufferKernel, VectorAddKernel, ZeroBufferKernel};
pub use embedding::{EmbedScatterAddKernel, TokenPosEmbedKernel};
pub use gelu::{GeluBackwardKernel, GeluKernel};
pub use kv_cache::{KvBlockGpu, KvCacheGpu, MhaKvCacheKernel, SingleTokenEmbedKernel};
pub use layer_norm::{LayerNormBackwardKernel, LayerNormForwardKernel};
pub use matmul::{cpu_gemm, cpu_gemm_transpose_b, GemmKernel, GemmTransposeBKernel};
pub use optimizer::AdamWKernel;
pub use reduce::{
    EmbedScatterReduceKernel, GradNormSqReduceKernel, OuterProductAccumKernel, RowReduceAddKernel,
};
pub use softmax_ce::SoftmaxCrossEntropyKernel;
pub use transformer::DecoderTransformerGpu;
