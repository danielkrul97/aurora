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
pub mod muon;
pub mod optimizer;
pub mod reduce;
pub mod softmax_ce;
pub mod spacetime_gpu;
pub mod thermo_conv;
pub mod thermo_model;
pub mod thermo_step;
pub mod thermo_stencil;
pub mod transformer;

pub use attention::{CausalMhaBackwardKernel, CausalMhaForwardKernel};
pub use context::GpuContext;
pub use elementwise::{VectorAdd3Kernel, VectorAddKernel, ZeroBufferKernel};
pub use embedding::{EmbedScatterAddKernel, TokenPosEmbedKernel};
pub use gelu::{GeluBackwardKernel, GeluKernel};
pub use kv_cache::{KvBlockGpu, KvCacheGpu, MhaKvCacheKernel, SingleTokenEmbedKernel};
pub use layer_norm::{LayerNormBackwardKernel, LayerNormForwardKernel};
pub use matmul::{cpu_gemm, cpu_gemm_transpose_b, GemmKernel, GemmTransposeBKernel};
pub use muon::{
    MuonApplyKernel, MuonCombineBKernel, MuonFrobInvKernel, MuonLoadStoreKernel,
    MuonMomentumKernel, MuonStepXKernel,
};
pub use optimizer::AdamWKernel;
pub use reduce::{
    EmbedScatterReduceKernel, GradNormFinalizeKernel, GradNormSqReduceKernel,
    OuterProductAccumKernel, OuterProductSplitKKernel, RowReduceAddKernel, ScaleByBufferKernel,
};
pub use softmax_ce::SoftmaxCrossEntropyKernel;
pub use spacetime_gpu::StLmGpu;
pub use thermo_model::ThermoGpu;
pub use thermo_conv::{
    ThermoConv1x1Kernel, ThermoConv3x3Dw1Kernel, ThermoConv3x3Dw1SplitKKernel,
    ThermoConv3x3DsKernel, ThermoConv3x3GeluCacheKernel, ThermoConv3x3GeluKernel,
    ThermoConv3x3Kernel, ThermoConvBwdHiddenKernel, ThermoIm2colDw1Kernel,
};
pub use thermo_stencil::{
    ThermoAvgPoolBackwardKernel, ThermoAvgPoolKernel, ThermoDiffusionBwdDsKernel,
    ThermoDiffusionBwdParamKernel, ThermoDiffusionBwdPreKernel, ThermoDiffusionForwardKernel,
    ThermoFeatureAssembleKernel, ThermoFeatureDisassembleKernel, ThermoFluxDivBackwardKernel,
    ThermoFluxDivKernel, ThermoInjectBackwardBatchedKernel, ThermoInjectBackwardKernel,
    ThermoInjectBatchedKernel, ThermoInjectKernel, ThermoSpectralBackwardKernel,
    ThermoSpectralKernel,
};
pub use thermo_step::{
    ExtractFluxKernel, StepAssembleKernel, StepTransportBwdKernel, StepTransportFwdKernel,
    ThermoBiasRowAddKernel, ThermoDisassembleKernel, ThermoFluxScatterKernel, ThermoGpuForward,
    ThermoScaleKernel,
};
pub use transformer::DecoderTransformerGpu;
