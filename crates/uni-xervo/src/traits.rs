//! Core traits that every provider and model implementation must satisfy.

pub mod asr;
pub mod docs;
pub mod hybrid;
pub mod multimodal;
pub mod multivector;
pub mod nlp;
pub mod raw_tensor_model;
pub mod sparse;

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::Result;
use async_trait::async_trait;
use std::any::Any;

pub use raw_tensor_model::{
    DimSize, RawTensorModel, TensorBatch, TensorDtype, TensorSpec, TensorValue,
};

pub use asr::{
    TranscribeOptions, TranscribeResult, TranscribeSegment, TranscribeWord, TranscriptionModel,
};
pub use docs::{
    DocBlock, DocBlockKind, DocExtractOptions, DocExtractResult, DocOutputFormat,
    DocumentExtractionModel, OcrBlock, OcrModel, OcrResult,
};
pub use hybrid::{HeadSet, HybridEmbedResult, HybridEmbeddingModel};
pub use multimodal::{
    AudioEmbeddingModel, AudioInput, ImageEmbeddingModel, Modality, MultimodalBlock,
    MultimodalEmbeddingModel, MultimodalInput,
};
pub use multivector::{MultiVectorEmbedResult, MultiVectorEmbeddingModel};
pub use nlp::{
    DepLink, ExtractionSchema, FactModality, FactPolarity, LabelSpec, NerEntity, NlpCalibration,
    NlpCalibrationOverrides, NlpEntity, NlpFact, NlpLabelMaps, NlpModel, NlpRequest, NlpResult,
    NlpSentence, NlpTasks, NlpToken, NlpWindow, SpeechAct, SrlFrame, SrlRole, Turn, WindowFrame,
    WindowSentence, WordAnalysis, WordSpan,
};
pub use sparse::{SparseEmbedResult, SparseEmbeddingModel, SparseVector};

/// Advertised capabilities of a [`ModelProvider`].
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    /// The set of [`ModelTask`] variants this provider can handle.
    pub supported_tasks: Vec<ModelTask>,
}

/// Health status reported by a provider.
///
/// `#[non_exhaustive]` so a finer-grained status can be added later.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// `Arc<dyn EmbeddingModel>`, `Arc<dyn RerankerModel>`,
    /// `Arc<dyn GeneratorModel>`, or `Arc<dyn RawTensorModel>` depending on the task.
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

/// Metadata common to every loaded model handle.
///
/// Supertrait of every task trait ([`EmbeddingModel`], [`RerankerModel`], …) so a
/// caller holding any typed handle can identify the model and inspect its
/// requested execution backends uniformly.
pub trait ModelInfo: Send + Sync {
    /// The underlying model identifier (e.g. a HuggingFace repo ID or API model name).
    fn model_id(&self) -> &str;

    /// Names of the ONNX Runtime execution providers this model's session
    /// attached, in priority order. Every `local/onnx` model reports it once
    /// loaded; it is empty for remote and non-ONNX models, and for
    /// `local/onnx` document extraction, which runs no ORT session yet.
    ///
    /// A provider is listed only if it was requested (or in the build's
    /// default list), compiled into this binary, and registered with the
    /// session. So `["cuda", "cpu"]` reads `["cpu"]` when the build lacks
    /// `gpu-cuda`, when the CUDA library or driver fails to load, or when the
    /// session fails to build on CUDA (no usable device) and is rebuilt on
    /// CPU. Each of those logs a warning.
    ///
    /// Attached is not the same as used for every node: ORT still places
    /// operators an accelerator cannot run on CPU, and no API exposes that
    /// split. Dense embedding, reranking (cross-encoder and generative) and
    /// the sparse, multi-vector and hybrid embedders also run a short
    /// inference at load, and fall back to `["cpu"]` if it fails on the
    /// accelerator; the other `local/onnx` models (NLP, OCR, image embedding,
    /// raw) do not, so an accelerator that attaches but fails at inference is
    /// not reflected for them.
    ///
    /// Nothing listed after `cpu` is reported: ORT gives each node to the first
    /// provider that claims it and CPU claims every standard operator, so
    /// `["cpu", "coreml"]` reports `["cpu"]`.
    ///
    /// A model built from several sessions reports the providers all of them
    /// attached: OCR with a detector lists only those both stages have, and a
    /// kniv v7 model starts from its `plain` graph and narrows if a graph
    /// built later attached fewer. So for kniv v7 the list can shrink after
    /// load, on the first request that needs a new graph; it never grows.
    fn active_execution_providers(&self) -> Vec<String> {
        Vec::new()
    }
}

/// A model that produces dense vector embeddings from text.
#[async_trait]
pub trait EmbeddingModel: ModelInfo {
    /// Embed a batch of text strings into dense vectors.
    ///
    /// Returns an [`EmbedResult`] whose `vectors` holds one `Vec<f32>` per input
    /// text, each with [`dimensions()`](EmbeddingModel::dimensions) elements.
    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult>;

    /// The dimensionality of the embedding vectors produced by this model.
    fn dimensions(&self) -> u32;

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
pub trait RerankerModel: ModelInfo {
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

/// Result of a dense-embedding call, carrying the vectors plus optional usage.
///
/// Returned by [`EmbeddingModel`], [`ImageEmbeddingModel`],
/// [`AudioEmbeddingModel`], and [`MultimodalEmbeddingModel`] — every dense
/// embedder shares this shape so remote providers (OpenAI / Cohere / Gemini /
/// etc.) can report per-call token usage.
///
/// `usage` is `Some` when the provider reports it (remote APIs), and `None`
/// for local providers where the concept does not apply.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// One dense vector per input, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Token usage reported by the provider, if any.
    pub usage: Option<TokenUsage>,
}

/// A model that generates text, images, or audio from a conversational
/// message history.
///
/// Messages carry explicit roles via [`Message`] and may contain multimodal
/// content (text and images). The output [`GenerationResult`] is a union:
/// text, images, and audio fields — consumers check what is populated.
#[async_trait]
pub trait GeneratorModel: ModelInfo {
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
