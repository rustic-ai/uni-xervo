//! Unified Rust runtime for local and remote embedding, reranking, and generation models.
//!
//! Uni-Xervo provides a single, provider-agnostic API for loading and running ML models
//! across a wide range of backends — from local inference engines (Candle, ONNX Runtime,
//! mistral.rs) to remote API services (OpenAI, Gemini, Anthropic, Cohere, Mistral,
//! Voyage AI, Vertex AI, Azure OpenAI, llama.cpp `llama-server`), and raw ONNX graphs.
//!
//! # Key concepts
//!
//! - **[`ModelRuntime`](runtime::ModelRuntime)** — the central runtime that owns providers
//!   and manages a catalog of model aliases.
//! - **[`ModelAliasSpec`](api::ModelAliasSpec)** — a declarative specification that maps a
//!   human-readable alias (e.g. `"embed/default"`) to a concrete provider + model pair.
//! - **Providers** — pluggable backends that implement [`ModelProvider`](traits::ModelProvider).
//!   Each provider advertises the tasks it supports and knows how to load models.
//! - **Traits** — [`EmbeddingModel`](traits::EmbeddingModel),
//!   [`RerankerModel`](traits::RerankerModel), and
//!   [`GeneratorModel`](traits::GeneratorModel) are the task-specific interfaces returned
//!   by the runtime. The retrieval surface adds
//!   [`SparseEmbeddingModel`](traits::SparseEmbeddingModel),
//!   [`MultiVectorEmbeddingModel`](traits::MultiVectorEmbeddingModel), and
//!   [`HybridEmbeddingModel`](traits::HybridEmbeddingModel) (single forward pass
//!   over a multi-output graph), and the multimodal extension surface adds
//!   [`ImageEmbeddingModel`](traits::ImageEmbeddingModel),
//!   [`AudioEmbeddingModel`](traits::AudioEmbeddingModel),
//!   [`MultimodalEmbeddingModel`](traits::MultimodalEmbeddingModel),
//!   [`NlpModel`](traits::NlpModel),
//!   [`DocumentExtractionModel`](traits::DocumentExtractionModel),
//!   [`TranscriptionModel`](traits::TranscriptionModel), and
//!   [`OcrModel`](traits::OcrModel) — resolved via the matching methods on
//!   [`ModelRuntime`](runtime::ModelRuntime) (`sparse_embedder`,
//!   `multi_vector_embedder`, `hybrid_embedder`, `image_embedder`,
//!   `multimodal_embedder`, `nlp_model`, `document_extractor`, `transcriber`,
//!   `ocr_model`). `local/onnx` implements the sparse / multi-vector / hybrid /
//!   image / NLP / OCR tasks and `remote/cohere` + `remote/gemini` implement
//!   multimodal embedding; `audio_embedder` has no bundled provider yet.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use uni_xervo::api::{ModelAliasSpec, ModelTask};
//! use uni_xervo::runtime::ModelRuntime;
//! # #[cfg(feature = "provider-candle")]
//! use uni_xervo::provider::candle::LocalCandleProvider;
//!
//! # #[cfg(feature = "provider-candle")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let spec = ModelAliasSpec {
//!     alias: "embed/local".into(),
//!     task: ModelTask::Embed,
//!     provider_id: "local/candle".into(),
//!     model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
//!     revision: None,
//!     warmup: Default::default(),
//!     required: true,
//!     timeout: None,
//!     load_timeout: None,
//!     retry: None,
//!     options: serde_json::Value::Null,
//! };
//!
//! let runtime = ModelRuntime::builder()
//!     .register_provider(LocalCandleProvider::new())
//!     .catalog(vec![spec])
//!     .build()
//!     .await?;
//!
//! let model = runtime.embedding("embed/local").await?;
//! let embeddings = model.embed(&["Hello, world!"]).await?;
//! println!("dim: {}", embeddings.vectors[0].len());
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod cache;
// Shared document-VLM output parsers, used by any provider that runs such a
// model (`local/onnx`, `local/mistralrs`).
#[cfg(any(
    feature = "provider-onnx",
    feature = "provider-onnx-dynamic",
    feature = "provider-mistralrs"
))]
mod doc_parse;
pub mod error;
mod options_validation;
pub mod prelude;
pub mod provider;
pub mod reliability;
pub mod runtime;
pub mod score;
pub mod traits;

#[cfg(test)]
mod mock;
