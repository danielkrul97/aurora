//! Decoder-only transformer LM.
//!
//! Pre-norm GPT-style: token + positional embedding, N transformer
//! blocks (causal MHA + FFN, each with residual and LayerNorm), final
//! LayerNorm, tied LM head.

pub mod attention;
pub mod block;
pub mod embedding;
pub mod ffn;
pub mod kv_cache;
pub mod loss;
pub mod model;
pub mod schedule;

pub use attention::{
    causal_mha_backward, causal_mha_forward, MhaCache, MhaGrads, MhaWeights,
};
pub use block::{
    block_backward, block_forward, TransformerBlock, TransformerBlockCache, TransformerBlockGrads,
};
pub use embedding::{
    embed_backward, embed_forward, tied_lm_head_backward, tied_lm_head_forward, EmbedCache,
    EmbedGrads, EmbedWeights,
};
pub use ffn::{ffn_backward, ffn_forward, FfnCache, FfnGrads, FfnWeights};
pub use kv_cache::{generate_step, KvBlock, KvCache};
pub use loss::{argmax, cross_entropy_with_grad, softmax};
pub use model::{DecoderTransformer, ForwardCache, ModelGrads, TransformerConfig};
pub use schedule::cosine_warmup;
