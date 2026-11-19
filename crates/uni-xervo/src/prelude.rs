//! Convenience re-exports for the common uni-xervo surface.
//!
//! `use uni_xervo::prelude::*;` brings the runtime, catalog/config types, every
//! task trait (so their methods are in scope on `Arc<dyn …>` handles), the
//! common input/result types, and the host-side scoring helpers into scope.
//! Provider types stay in [`crate::provider`] — import the concrete providers
//! you register explicitly.

// Runtime + catalog/config.
#[doc(inline)]
pub use crate::api::{ModelAliasSpec, ModelTask, RetryConfig, WarmupPolicy};
#[doc(inline)]
pub use crate::error::{Result, RuntimeError};
#[doc(inline)]
pub use crate::inspect::{
    AliasSnapshot, LoadState, ProviderDescription, ProviderSnapshot, VerificationStatus,
    VerifyErrorKind, VerifyFailure, VerifyOutcome,
};
pub use crate::runtime::{ModelRuntime, ModelRuntimeBuilder};

// Provider + capability surface.
#[doc(inline)]
pub use crate::traits::{ModelProvider, ProviderCapabilities, ProviderHealth};

// Metadata supertrait + every task trait (method-in-scope on handles).
#[doc(inline)]
pub use crate::traits::{
    AudioEmbeddingModel, DocumentExtractionModel, EmbeddingModel, GeneratorModel,
    HybridEmbeddingModel, ImageEmbeddingModel, ModelInfo, MultiVectorEmbeddingModel,
    MultimodalEmbeddingModel, NlpModel, OcrModel, RawTensorModel, RerankerModel,
    SparseEmbeddingModel, TranscriptionModel,
};

// Inputs, options, and result types.
#[doc(inline)]
pub use crate::traits::{
    AudioInput, ContentBlock, DocExtractOptions, DocExtractResult, DocOutputFormat, EmbedResult,
    GenerationOptions, GenerationResult, HeadSet, HybridEmbedResult, ImageInput, Message, Modality,
    MultiVectorEmbedResult, MultimodalBlock, MultimodalInput, NlpRequest, NlpResult, NlpTasks,
    ScoredDoc, SparseEmbedResult, SparseVector, TensorBatch, TensorSpec, TensorValue, TokenUsage,
    TranscribeOptions, TranscribeResult,
};

// Host-side scoring helpers for sparse / multi-vector outputs.
#[doc(inline)]
pub use crate::score::{colbert_rerank, max_sim, sparse_dot};
