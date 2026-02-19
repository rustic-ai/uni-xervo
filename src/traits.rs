//! Core traits that every provider and model implementation must satisfy.

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::Result;
use async_trait::async_trait;
use std::any::Any;

/// Advertised capabilities of a [`ModelProvider`].
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    /// The set of [`ModelTask`] variants this provider can handle.
    pub supported_tasks: Vec<ModelTask>,
}

/// Health status reported by a provider.
#[derive(Debug, Clone)]
pub enum ProviderHealth {
    /// The provider is fully operational.
    Healthy,
    /// The provider is operational but experiencing partial issues.
    Degraded(String),
    /// The provider cannot serve requests.
    Unhealthy(String),
}

/// A pluggable backend that knows how to load models for one or more
/// [`ModelTask`] types.
///
/// Providers are registered with [`ModelRuntimeBuilder::register_provider`](crate::runtime::ModelRuntimeBuilder::register_provider)
/// and are identified by their [`provider_id`](ModelProvider::provider_id)
/// (e.g. `"local/candle"`, `"remote/openai"`).
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Unique identifier for this provider (e.g. `"local/candle"`, `"remote/openai"`).
    fn provider_id(&self) -> &'static str;

    /// Return the set of tasks this provider supports.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Load (or connect to) a model described by `spec` and return a type-erased
    /// handle.
    ///
    /// The returned [`LoadedModelHandle`] is expected to contain an
    /// `Arc<dyn EmbeddingModel>`, `Arc<dyn RerankerModel>`, or
    /// `Arc<dyn GeneratorModel>` depending on the task.
    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle>;

    /// Report the current health of this provider.
    async fn health(&self) -> ProviderHealth;

    /// Optional one-time warmup hook called during runtime startup.
    ///
    /// Use this for provider-wide initialization such as setting up API clients
    /// or pre-caching shared resources. The default implementation is a no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// A type-erased, reference-counted handle to a loaded model instance.
///
/// Providers wrap their concrete model (e.g. `Arc<dyn EmbeddingModel>`) inside
/// this `Arc<dyn Any + Send + Sync>` so the runtime can store them uniformly.
/// The runtime later downcasts the handle back to the expected trait object.
pub type LoadedModelHandle = std::sync::Arc<dyn Any + Send + Sync>;

/// A model that produces dense vector embeddings from text.
#[async_trait]
pub trait EmbeddingModel: Send + Sync + Any {
    /// Embed a batch of text strings into dense vectors.
    ///
    /// Returns one `Vec<f32>` per input text, each with [`dimensions()`](EmbeddingModel::dimensions)
    /// elements.
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>>;

    /// The dimensionality of the embedding vectors produced by this model.
    fn dimensions(&self) -> u32;

    /// The underlying model identifier (e.g. a HuggingFace repo ID or API model name).
    fn model_id(&self) -> &str;

    /// Optional warmup hook (e.g. load weights into memory on first access).
    /// The default is a no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// A single scored document returned by a [`RerankerModel`].
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    /// Zero-based index into the original `docs` slice passed to
    /// [`RerankerModel::rerank`].
    pub index: usize,
    /// Relevance score assigned by the reranker (higher is more relevant).
    pub score: f32,
    /// The document text, if the provider returns it. May be `None`.
    pub text: Option<String>,
}

/// A model that re-scores documents against a query for relevance ranking.
#[async_trait]
pub trait RerankerModel: Send + Sync {
    /// Rerank `docs` by relevance to `query`, returning scored results
    /// (typically sorted by descending score).
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>>;

    /// Optional warmup hook. The default is a no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// Sampling and length parameters for text generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate. Provider default if `None`.
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random).
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
}

/// The output of a text generation call.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated text.
    pub text: String,
    /// Token usage statistics, if reported by the provider.
    pub usage: Option<TokenUsage>,
}

/// Token counts for a generation request.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    /// Number of tokens in the prompt / input.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Sum of prompt and completion tokens.
    pub total_tokens: usize,
}

/// A model that generates text from a conversational message history.
///
/// Messages are passed as a flat `&[String]` slice where even-indexed entries
/// (0, 2, 4, ...) are user turns and odd-indexed entries are assistant turns.
#[async_trait]
pub trait GeneratorModel: Send + Sync {
    /// Generate a response given a conversation history and sampling options.
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult>;

    /// Optional warmup hook. The default is a no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}
