use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    ContentBlock, EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel,
    LoadedModelHandle, Message, MessageRole, ModelProvider, ProviderCapabilities, ProviderHealth,
    TokenUsage,
};
use async_trait::async_trait;
use mistralrs::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, GgufModelBuilder, IsqType, Model, ModelDType,
    PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextModelBuilder,
};
use serde::Deserialize;
use std::sync::Arc;

/// Local inference provider using the mistral.rs engine.
///
/// Supports HuggingFace models with optional ISQ (in-situ quantization)
/// for both embedding and text generation tasks.
pub struct LocalMistralRsProvider;

impl LocalMistralRsProvider {
    pub fn new() -> Self {
        Self
    }

    /// Set `HF_HOME` to our unified cache root before the first mistralrs load.
    ///
    /// mistralrs-core stores its HF cache handle in a process-global `OnceLock<Cache>`
    /// (`GLOBAL_HF_CACHE`) that is initialised exactly once — from `HF_HOME` at the
    /// time of the first model load.  The per-builder `from_hf_cache_path()` API feeds
    /// into the same `get_or_init` call and is therefore silently ignored on every load
    /// after the first one.
    ///
    /// Setting `HF_HOME` here (before any builder `.build()` call) ensures the
    /// `OnceLock` captures our directory.  Subsequent calls are no-ops because the env
    /// var is already set and `OnceLock` is already initialised.
    fn init_hf_cache() {
        let cache_root = crate::cache::resolve_provider_cache_root("mistralrs");
        // SAFETY: single-threaded with respect to the first mistralrs load; the
        // OnceLock guarantees only the first initialisation matters.
        unsafe {
            std::env::set_var("HF_HOME", &cache_root);
        }
    }
}

impl Default for LocalMistralRsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for LocalMistralRsProvider {
    fn provider_id(&self) -> &'static str {
        "local/mistralrs"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate],
        }
    }

    async fn warmup(&self) -> Result<()> {
        Self::init_hf_cache();
        Ok(())
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        // Best-effort: set HF_HOME before the first mistralrs OnceLock init.
        // No-op if warmup() already ran or if a previous load already set it.
        Self::init_hf_cache();

        let has_options = match &spec.options {
            serde_json::Value::Null => false,
            serde_json::Value::Object(map) => !map.is_empty(),
            _ => true,
        };

        let opts: MistralRsOptions = if has_options {
            serde_json::from_value(spec.options.clone())
                .map_err(|e| RuntimeError::Config(format!("Invalid mistralrs options: {}", e)))?
        } else {
            MistralRsOptions::default()
        };

        match spec.task {
            ModelTask::Embed => self.load_embedding(spec, &opts).await,
            ModelTask::Generate => self.load_generator(spec, &opts).await,
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "mistralrs provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

impl LocalMistralRsProvider {
    async fn load_embedding(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        tracing::info!(model_id = %spec.model_id, "Loading mistralrs embedding model");

        // When gguf_files is set, model_id is treated as the GGUF directory path.
        let model = if let Some(files) = &opts.gguf_files {
            if opts.dtype.is_some() {
                tracing::debug!("dtype option ignored for GGUF models");
            }
            let mut builder = GgufModelBuilder::new(spec.model_id.clone(), files.clone());

            if let Some(ref chat_tmpl) = opts.chat_template {
                builder = builder.with_chat_template(chat_tmpl.clone());
            }
            if let Some(ref tok_json) = opts.tokenizer_json {
                builder = builder.with_tokenizer_json(tok_json.clone());
            }
            builder = builder.with_logging();

            builder.build().await.map_err(|e| {
                RuntimeError::Load(format!(
                    "Failed to build mistralrs GGUF embedding model: {}",
                    e
                ))
            })?
        } else {
            let mut builder = EmbeddingModelBuilder::new(&spec.model_id);

            let dtype = resolve_model_dtype(opts)?;
            builder = builder.with_dtype(dtype);

            if let Some(ref isq_str) = opts.isq {
                let isq = parse_isq_type(isq_str)?;
                builder = builder.with_isq(isq);
            }

            if opts.force_cpu {
                builder = builder.with_force_cpu();
            }

            if let Some(ref rev) = spec.revision {
                builder = builder.with_hf_revision(rev);
            }

            if let Some(max_seqs) = opts.max_num_seqs {
                builder = builder.with_max_num_seqs(max_seqs);
            }

            if let Some(ref tok_json) = opts.tokenizer_json {
                builder = builder.with_tokenizer_json(tok_json);
            }

            builder = builder.with_logging();

            builder.build().await.map_err(|e| {
                RuntimeError::Load(format!("Failed to build mistralrs embedding model: {}", e))
            })?
        };

        let dimensions = match opts.embedding_dimensions {
            Some(d) => d,
            None => {
                tracing::info!("Probing embedding dimensions with test input");
                let probe = model.generate_embedding("probe").await.map_err(|e| {
                    RuntimeError::Load(format!("Failed to probe embedding dimensions: {}", e))
                })?;
                validate_embeddings(std::slice::from_ref(&probe)).map_err(|e| {
                    RuntimeError::Load(format!(
                        "Probe returned invalid values: {e}. Try dtype: \"f32\""
                    ))
                })?;
                probe.len() as u32
            }
        };

        tracing::info!(
            model_id = %spec.model_id,
            dimensions,
            "mistralrs embedding model loaded"
        );

        let service = MistralRsEmbeddingService {
            model,
            model_id: spec.model_id.clone(),
            dimensions,
        };

        let handle: Arc<dyn EmbeddingModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn load_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        let pipeline = opts.pipeline.as_deref().unwrap_or("text");
        match pipeline {
            "text" => self.load_text_generator(spec, opts).await,
            "vision" => self.load_vision_generator(spec, opts).await,
            "diffusion" => self.load_diffusion_generator(spec, opts).await,
            "speech" => self.load_speech_generator(spec, opts).await,
            _ => Err(RuntimeError::Config(format!(
                "Unknown pipeline '{}'. Valid: text, vision, diffusion, speech",
                pipeline
            ))),
        }
    }

    async fn load_text_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        tracing::info!(model_id = %spec.model_id, "Loading mistralrs text generator model");

        let model = if let Some(files) = &opts.gguf_files {
            if opts.dtype.is_some() {
                tracing::debug!("dtype option ignored for GGUF models");
            }
            let mut builder = GgufModelBuilder::new(spec.model_id.clone(), files.clone());

            if let Some(ref chat_tmpl) = opts.chat_template {
                builder = builder.with_chat_template(chat_tmpl.clone());
            }
            if let Some(ref tok_json) = opts.tokenizer_json {
                builder = builder.with_tokenizer_json(tok_json.clone());
            }
            if opts.paged_attention {
                builder = builder
                    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                    .map_err(|e| {
                        RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
                    })?;
            }
            builder = builder.with_logging();

            builder.build().await.map_err(|e| {
                RuntimeError::Load(format!(
                    "Failed to build mistralrs GGUF generator model: {}",
                    e
                ))
            })?
        } else {
            let mut builder = TextModelBuilder::new(&spec.model_id);

            let dtype = resolve_model_dtype(opts)?;
            builder = builder.with_dtype(dtype);

            if let Some(ref isq_str) = opts.isq {
                let isq = parse_isq_type(isq_str)?;
                builder = builder.with_isq(isq);
            }

            if opts.force_cpu {
                builder = builder.with_force_cpu();
            }

            if let Some(ref rev) = spec.revision {
                builder = builder.with_hf_revision(rev);
            }

            if opts.paged_attention {
                builder = builder
                    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                    .map_err(|e| {
                        RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
                    })?;
            }

            if let Some(ref chat_tmpl) = opts.chat_template {
                builder = builder.with_chat_template(chat_tmpl);
            }

            if let Some(ref tok_json) = opts.tokenizer_json {
                builder = builder.with_tokenizer_json(tok_json);
            }

            if let Some(max_seqs) = opts.max_num_seqs {
                builder = builder.with_max_num_seqs(max_seqs);
            }

            builder = builder.with_logging();

            builder.build().await.map_err(|e| {
                RuntimeError::Load(format!("Failed to build mistralrs generator model: {}", e))
            })?
        };

        tracing::info!(model_id = %spec.model_id, "mistralrs generator model loaded");

        let service = MistralRsGeneratorService {
            model,
            model_id: spec.model_id.clone(),
        };

        let handle: Arc<dyn GeneratorModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn load_vision_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        use mistralrs::VisionModelBuilder;

        if opts.gguf_files.is_some() {
            return Err(RuntimeError::Config(
                "GGUF is not supported for the vision pipeline".to_string(),
            ));
        }

        tracing::info!(model_id = %spec.model_id, "Loading mistralrs vision generator model");

        let mut builder = VisionModelBuilder::new(&spec.model_id);
        let dtype = resolve_model_dtype(opts)?;
        builder = builder.with_dtype(dtype);

        if let Some(ref isq_str) = opts.isq {
            let isq = parse_isq_type(isq_str)?;
            builder = builder.with_isq(isq);
        }
        if opts.force_cpu {
            builder = builder.with_force_cpu();
        }
        if let Some(ref rev) = spec.revision {
            builder = builder.with_hf_revision(rev);
        }
        if opts.paged_attention {
            builder = builder
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                .map_err(|e| {
                    RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
                })?;
        }
        if let Some(ref chat_tmpl) = opts.chat_template {
            builder = builder.with_chat_template(chat_tmpl);
        }
        if let Some(ref tok_json) = opts.tokenizer_json {
            builder = builder.with_tokenizer_json(tok_json);
        }
        if let Some(max_seqs) = opts.max_num_seqs {
            builder = builder.with_max_num_seqs(max_seqs);
        }
        builder = builder.with_logging();

        let model = builder.build().await.map_err(|e| {
            RuntimeError::Load(format!("Failed to build mistralrs vision model: {}", e))
        })?;

        tracing::info!(model_id = %spec.model_id, "mistralrs vision model loaded");

        let service = MistralRsVisionService {
            model,
            model_id: spec.model_id.clone(),
        };
        let handle: Arc<dyn GeneratorModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn load_diffusion_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        use mistralrs::{DiffusionLoaderType, DiffusionModelBuilder};

        let loader_type = match opts.diffusion_loader_type.as_deref().unwrap_or("flux") {
            "flux" => DiffusionLoaderType::Flux,
            "flux_offloaded" => DiffusionLoaderType::FluxOffloaded,
            other => {
                return Err(RuntimeError::Config(format!(
                    "Unknown diffusion_loader_type '{}'. Valid: flux, flux_offloaded",
                    other
                )));
            }
        };

        tracing::info!(model_id = %spec.model_id, "Loading mistralrs diffusion model");

        let mut builder = DiffusionModelBuilder::new(&spec.model_id, loader_type);
        if opts.force_cpu {
            builder = builder.with_force_cpu();
        }
        let dtype = resolve_model_dtype(opts)?;
        builder = builder.with_dtype(dtype);
        builder = builder.with_logging();

        let model = builder.build().await.map_err(|e| {
            RuntimeError::Load(format!("Failed to build mistralrs diffusion model: {}", e))
        })?;

        tracing::info!(model_id = %spec.model_id, "mistralrs diffusion model loaded");

        let service = MistralRsDiffusionService {
            model,
            #[allow(dead_code)]
            model_id: spec.model_id.clone(),
        };
        let handle: Arc<dyn GeneratorModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn load_speech_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        use mistralrs::{SpeechLoaderType, SpeechModelBuilder};

        let loader_type = match opts.speech_loader_type.as_deref().unwrap_or("dia") {
            "dia" => SpeechLoaderType::Dia,
            other => {
                return Err(RuntimeError::Config(format!(
                    "Unknown speech_loader_type '{}'. Valid: dia",
                    other
                )));
            }
        };

        tracing::info!(model_id = %spec.model_id, "Loading mistralrs speech model");

        let mut builder = SpeechModelBuilder::new(&spec.model_id, loader_type);
        if opts.force_cpu {
            builder = builder.with_force_cpu();
        }
        let dtype = resolve_model_dtype(opts)?;
        builder = builder.with_dtype(dtype);
        builder = builder.with_logging();

        let model = builder.build().await.map_err(|e| {
            RuntimeError::Load(format!("Failed to build mistralrs speech model: {}", e))
        })?;

        tracing::info!(model_id = %spec.model_id, "mistralrs speech model loaded");

        let service = MistralRsSpeechService {
            model,
            #[allow(dead_code)]
            model_id: spec.model_id.clone(),
        };
        let handle: Arc<dyn GeneratorModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct MistralRsOptions {
    /// ISQ quantization type, e.g. "Q4K", "Q8_0"
    isq: Option<String>,
    /// Force CPU inference (default: false)
    #[serde(default)]
    force_cpu: bool,
    /// Enable paged attention (default: false)
    #[serde(default)]
    paged_attention: bool,
    /// Maximum number of sequences for batching
    max_num_seqs: Option<usize>,
    /// Override chat template
    chat_template: Option<String>,
    /// Override tokenizer JSON path
    tokenizer_json: Option<String>,
    /// Override embedding dimensions (probed at load if absent)
    embedding_dimensions: Option<u32>,
    /// List of GGUF filenames (enables GGUF mode)
    gguf_files: Option<Vec<String>>,
    /// Model data type: "auto", "f16", "bf16", "f32"
    dtype: Option<String>,
    /// Pipeline type: "text" (default), "vision", "diffusion", "speech"
    pipeline: Option<String>,
    /// Diffusion loader type: "flux", "flux_offloaded"
    diffusion_loader_type: Option<String>,
    /// Speech loader type: "dia"
    speech_loader_type: Option<String>,
}

// ---------------------------------------------------------------------------
// ISQ type parsing
// ---------------------------------------------------------------------------

fn parse_isq_type(s: &str) -> Result<IsqType> {
    match s.to_uppercase().as_str() {
        "Q4_0" => Ok(IsqType::Q4_0),
        "Q4_1" => Ok(IsqType::Q4_1),
        "Q5_0" => Ok(IsqType::Q5_0),
        "Q5_1" => Ok(IsqType::Q5_1),
        "Q8_0" => Ok(IsqType::Q8_0),
        "Q8_1" => Ok(IsqType::Q8_1),
        "Q2K" => Ok(IsqType::Q2K),
        "Q3K" => Ok(IsqType::Q3K),
        "Q4K" => Ok(IsqType::Q4K),
        "Q5K" => Ok(IsqType::Q5K),
        "Q6K" => Ok(IsqType::Q6K),
        "Q8K" => Ok(IsqType::Q8K),
        "HQQ4" => Ok(IsqType::HQQ4),
        "HQQ8" => Ok(IsqType::HQQ8),
        "F8E4M3" => Ok(IsqType::F8E4M3),
        "AFQ8" => Ok(IsqType::AFQ8),
        "AFQ6" => Ok(IsqType::AFQ6),
        "AFQ4" => Ok(IsqType::AFQ4),
        "AFQ3" => Ok(IsqType::AFQ3),
        "AFQ2" => Ok(IsqType::AFQ2),
        other => Err(RuntimeError::Config(format!(
            "Unknown ISQ type '{}'. Valid types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, \
             Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, HQQ4, HQQ8, F8E4M3, AFQ2-AFQ8",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Model dtype parsing
// ---------------------------------------------------------------------------

fn parse_model_dtype(s: &str) -> Result<ModelDType> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(ModelDType::Auto),
        "f16" => Ok(ModelDType::F16),
        "bf16" => Ok(ModelDType::BF16),
        "f32" => Ok(ModelDType::F32),
        other => Err(RuntimeError::Config(format!(
            "Unknown dtype '{}'. Valid values: auto, f16, bf16, f32",
            other
        ))),
    }
}

fn resolve_model_dtype(opts: &MistralRsOptions) -> Result<ModelDType> {
    if let Some(ref s) = opts.dtype {
        return parse_model_dtype(s);
    }
    if opts.force_cpu || !has_gpu_support() {
        tracing::info!("No GPU detected or force_cpu=true; defaulting dtype to F32");
        Ok(ModelDType::F32)
    } else {
        Ok(ModelDType::Auto)
    }
}

#[allow(unexpected_cfgs)]
fn has_gpu_support() -> bool {
    cfg!(any(feature = "gpu-cuda", feature = "gpu-metal"))
}

// ---------------------------------------------------------------------------
// Embedding validation
// ---------------------------------------------------------------------------

fn validate_embeddings(embeddings: &[Vec<f32>]) -> Result<()> {
    for (i, vec) in embeddings.iter().enumerate() {
        let nan_count = vec.iter().filter(|v| v.is_nan()).count();
        let inf_count = vec.iter().filter(|v| v.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            return Err(RuntimeError::InferenceError(format!(
                "Embedding vector {} contains invalid values ({} NaN, {} Inf out of {} dims). \
                 This typically happens with F16 on CPU. Set options: {{\"dtype\": \"f32\"}}.",
                i,
                nan_count,
                inf_count,
                vec.len()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding service
// ---------------------------------------------------------------------------

struct MistralRsEmbeddingService {
    model: Model,
    model_id: String,
    dimensions: u32,
}

#[async_trait]
impl EmbeddingModel for MistralRsEmbeddingService {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let request =
            EmbeddingRequestBuilder::new().add_prompts(texts.iter().map(|s| s.to_string()));

        let embeddings = self.model.generate_embeddings(request).await.map_err(|e| {
            RuntimeError::InferenceError(format!("Embedding inference failed: {}", e))
        })?;

        validate_embeddings(&embeddings)?;

        Ok(embeddings)
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Generator service
// ---------------------------------------------------------------------------

struct MistralRsGeneratorService {
    model: Model,
    #[allow(dead_code)] // kept for diagnostics/logging
    model_id: String,
}

#[async_trait]
impl GeneratorModel for MistralRsGeneratorService {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let mut request = RequestBuilder::new();

        for msg in messages {
            let role = match msg.role {
                MessageRole::System => TextMessageRole::System,
                MessageRole::User => TextMessageRole::User,
                MessageRole::Assistant => TextMessageRole::Assistant,
            };
            request = request.add_message(role, msg.text());
        }

        // Apply sampling parameters
        let has_sampling = options.temperature.is_some()
            || options.top_p.is_some()
            || options.max_tokens.is_some();

        if has_sampling {
            if let Some(temp) = options.temperature {
                request = request.set_sampler_temperature(temp as f64);
            }
            if let Some(top_p) = options.top_p {
                request = request.set_sampler_topp(top_p as f64);
            }
            if let Some(max_tokens) = options.max_tokens {
                request = request.set_sampler_max_len(max_tokens);
            }
        } else {
            request = request.set_deterministic_sampler();
        }

        let response = self.model.send_chat_request(request).await.map_err(|e| {
            RuntimeError::InferenceError(format!("Generation inference failed: {}", e))
        })?;

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        let usage = TokenUsage {
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
        };

        Ok(GenerationResult {
            text,
            usage: Some(usage),
            images: vec![],
            audio: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Vision service
// ---------------------------------------------------------------------------

struct MistralRsVisionService {
    model: Model,
    #[allow(dead_code)]
    model_id: String,
}

#[async_trait]
impl GeneratorModel for MistralRsVisionService {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let mut request = RequestBuilder::new();

        for msg in messages {
            let role = match msg.role {
                MessageRole::System => TextMessageRole::System,
                MessageRole::User => TextMessageRole::User,
                MessageRole::Assistant => TextMessageRole::Assistant,
            };

            // Collect images from this message
            let images: Vec<image::DynamicImage> = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Image(img_input) => {
                        let bytes = match img_input {
                            crate::traits::ImageInput::Bytes { data, .. } => data.clone(),
                            crate::traits::ImageInput::Url(_url) => {
                                tracing::warn!(
                                    "URL-based image input not yet supported in vision pipeline"
                                );
                                return None;
                            }
                        };
                        match image::load_from_memory(&bytes) {
                            Ok(img) => Some(img),
                            Err(e) => {
                                tracing::error!("Failed to decode image: {}", e);
                                None
                            }
                        }
                    }
                    _ => None,
                })
                .collect();

            let text = msg.text();

            if images.is_empty() {
                request = request.add_message(role, text);
            } else {
                request = request
                    .add_image_message(role, text, images, &self.model)
                    .map_err(|e| {
                        RuntimeError::InferenceError(format!("Failed to add vision message: {}", e))
                    })?;
            }
        }

        // Apply sampling parameters
        let has_sampling = options.temperature.is_some()
            || options.top_p.is_some()
            || options.max_tokens.is_some();

        if has_sampling {
            if let Some(temp) = options.temperature {
                request = request.set_sampler_temperature(temp as f64);
            }
            if let Some(top_p) = options.top_p {
                request = request.set_sampler_topp(top_p as f64);
            }
            if let Some(max_tokens) = options.max_tokens {
                request = request.set_sampler_max_len(max_tokens);
            }
        } else {
            request = request.set_deterministic_sampler();
        }

        let response =
            self.model.send_chat_request(request).await.map_err(|e| {
                RuntimeError::InferenceError(format!("Vision inference failed: {}", e))
            })?;

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        let usage = TokenUsage {
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
        };

        Ok(GenerationResult {
            text,
            usage: Some(usage),
            images: vec![],
            audio: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Diffusion service
// ---------------------------------------------------------------------------

struct MistralRsDiffusionService {
    model: Model,
    #[allow(dead_code)]
    model_id: String,
}

#[async_trait]
impl GeneratorModel for MistralRsDiffusionService {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        use mistralrs::DiffusionGenerationParams;

        // Extract the text prompt from messages
        let prompt = messages
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|b| match b {
                ContentBlock::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let height = options.height.unwrap_or(720) as usize;
        let width = options.width.unwrap_or(1280) as usize;

        let response = self
            .model
            .generate_image(
                prompt,
                mistralrs::ImageGenerationResponseFormat::B64Json,
                DiffusionGenerationParams { height, width },
            )
            .await
            .map_err(|e| {
                RuntimeError::InferenceError(format!("Diffusion inference failed: {}", e))
            })?;

        // The response is a base64-encoded image
        let b64 = response.data[0].b64_json.as_deref().ok_or_else(|| {
            RuntimeError::InferenceError("Diffusion response missing b64_json data".to_string())
        })?;
        let image_data = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
            .map_err(|e| {
                RuntimeError::InferenceError(format!("Failed to decode diffusion output: {}", e))
            })?;

        Ok(GenerationResult {
            text: String::new(),
            usage: None,
            images: vec![crate::traits::GeneratedImage {
                data: image_data,
                media_type: "image/png".to_string(),
            }],
            audio: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Speech service
// ---------------------------------------------------------------------------

struct MistralRsSpeechService {
    model: Model,
    #[allow(dead_code)]
    model_id: String,
}

#[async_trait]
impl GeneratorModel for MistralRsSpeechService {
    async fn generate(
        &self,
        messages: &[Message],
        _options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Extract the text prompt from messages
        let prompt = messages
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|b| match b {
                ContentBlock::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let (pcm_data, sample_rate, channels) =
            self.model.generate_speech(prompt).await.map_err(|e| {
                RuntimeError::InferenceError(format!("Speech inference failed: {}", e))
            })?;

        Ok(GenerationResult {
            text: String::new(),
            usage: None,
            images: vec![],
            audio: Some(crate::traits::AudioOutput {
                pcm_data: (*pcm_data).clone(),
                sample_rate,
                channels,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // validate_embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn validate_embeddings_valid() {
        let vecs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(validate_embeddings(&vecs).is_ok());
    }

    #[test]
    fn validate_embeddings_empty() {
        assert!(validate_embeddings(&[]).is_ok());
    }

    #[test]
    fn validate_embeddings_nan() {
        let vecs = vec![vec![1.0, f32::NAN, 3.0]];
        let err = validate_embeddings(&vecs).unwrap_err();
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn validate_embeddings_inf() {
        let vecs = vec![vec![1.0, f32::INFINITY, 3.0]];
        let err = validate_embeddings(&vecs).unwrap_err();
        assert!(err.to_string().contains("Inf"));
    }

    #[test]
    fn validate_embeddings_all_nan() {
        let vecs = vec![vec![f32::NAN, f32::NAN, f32::NAN]];
        let err = validate_embeddings(&vecs).unwrap_err();
        assert!(err.to_string().contains("3 NaN"));
    }

    // -----------------------------------------------------------------------
    // parse_model_dtype
    // -----------------------------------------------------------------------

    #[test]
    fn parse_model_dtype_valid() {
        assert!(matches!(parse_model_dtype("auto"), Ok(ModelDType::Auto)));
        assert!(matches!(parse_model_dtype("f16"), Ok(ModelDType::F16)));
        assert!(matches!(parse_model_dtype("bf16"), Ok(ModelDType::BF16)));
        assert!(matches!(parse_model_dtype("f32"), Ok(ModelDType::F32)));
    }

    #[test]
    fn parse_model_dtype_case_insensitive() {
        assert!(matches!(parse_model_dtype("F16"), Ok(ModelDType::F16)));
        assert!(matches!(parse_model_dtype("BF16"), Ok(ModelDType::BF16)));
        assert!(matches!(parse_model_dtype("Auto"), Ok(ModelDType::Auto)));
    }

    #[test]
    fn parse_model_dtype_invalid() {
        let err = parse_model_dtype("int8").unwrap_err();
        assert!(err.to_string().contains("Unknown dtype"));
    }

    // -----------------------------------------------------------------------
    // resolve_model_dtype
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_model_dtype_explicit_overrides_force_cpu() {
        let opts = MistralRsOptions {
            dtype: Some("f16".to_string()),
            force_cpu: true,
            ..Default::default()
        };
        assert!(matches!(resolve_model_dtype(&opts), Ok(ModelDType::F16)));
    }

    #[test]
    fn resolve_model_dtype_force_cpu_defaults_f32() {
        let opts = MistralRsOptions {
            force_cpu: true,
            ..Default::default()
        };
        assert!(matches!(resolve_model_dtype(&opts), Ok(ModelDType::F32)));
    }

    #[test]
    fn resolve_model_dtype_no_gpu_defaults_f32() {
        // Without gpu-cuda or gpu-metal features, has_gpu_support() returns false.
        let opts = MistralRsOptions::default();
        if !has_gpu_support() {
            assert!(matches!(resolve_model_dtype(&opts), Ok(ModelDType::F32)));
        }
    }
}
