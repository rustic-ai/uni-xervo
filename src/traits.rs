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

// ---------------------------------------------------------------------------
// Multimodal message types
// ---------------------------------------------------------------------------

/// The role of a message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    /// System-level instructions.
    System,
    /// A user turn.
    User,
    /// An assistant (model) turn.
    Assistant,
}

/// Image data that can be passed as part of a [`ContentBlock`].
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// Raw image bytes with a MIME type (e.g. `"image/png"`).
    Bytes { data: Vec<u8>, media_type: String },
    /// A URL pointing to an image.
    Url(String),
}

/// A single block of content within a [`Message`].
#[derive(Debug, Clone)]
pub enum ContentBlock {
    /// Plain text content.
    Text(String),
    /// An image (for vision models).
    Image(ImageInput),
}

/// A single message in a conversation, containing one or more content blocks.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender.
    pub role: MessageRole,
    /// The content blocks that make up this message.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a user message with a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Create an assistant message with a single text block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Create a system message with a single text block.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    /// Extract the concatenated text from all [`ContentBlock::Text`] blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ---------------------------------------------------------------------------
// Generation options and results
// ---------------------------------------------------------------------------

/// Sampling and length parameters for text generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate. Provider default if `None`.
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random).
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Desired image width (for diffusion models; ignored by text/vision).
    pub width: Option<u32>,
    /// Desired image height (for diffusion models; ignored by text/vision).
    pub height: Option<u32>,
}

/// An image produced by a generation call (e.g. from a diffusion model).
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Raw image bytes (e.g. PNG).
    pub data: Vec<u8>,
    /// MIME type (e.g. `"image/png"`).
    pub media_type: String,
}

/// Audio output produced by a speech model.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// PCM sample data.
    pub pcm_data: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: usize,
    /// Number of audio channels.
    pub channels: usize,
}

/// The output of a generation call.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated text (may be empty for image/audio-only results).
    pub text: String,
    /// Token usage statistics, if reported by the provider.
    pub usage: Option<TokenUsage>,
    /// Generated images (non-empty for diffusion models).
    pub images: Vec<GeneratedImage>,
    /// Generated audio (present for speech models).
    pub audio: Option<AudioOutput>,
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

/// A model that generates text, images, or audio from a conversational
/// message history.
///
/// Messages carry explicit roles via [`Message`] and may contain multimodal
/// content (text and images). The output [`GenerationResult`] is a union:
/// text, images, and audio fields — consumers check what is populated.
#[async_trait]
pub trait GeneratorModel: Send + Sync {
    /// Generate a response given a conversation history and sampling options.
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult>;

    /// Optional warmup hook. The default is a no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}
