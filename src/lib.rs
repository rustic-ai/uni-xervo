//! Unified Rust runtime for local and remote embedding, reranking, and generation models.
//!
//! Uni-Xervo provides a single, provider-agnostic API for loading and running ML models
//! across a wide range of backends — from local inference engines (Candle, FastEmbed,
//! mistral.rs) to remote API services (OpenAI, Gemini, Anthropic, Cohere, Mistral,
//! Voyage AI, Vertex AI, Azure OpenAI).
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
//!   by the runtime.
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
//! let embeddings = model.embed(vec!["Hello, world!"]).await?;
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod cache;
pub mod error;
mod options_validation;
pub mod provider;
pub mod reliability;
pub mod runtime;
pub mod traits;

#[cfg(test)]
mod mock;
