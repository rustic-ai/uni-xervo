use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use mistralrs::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, GgufModelBuilder, IsqType, Model,
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
        tracing::info!(model_id = %spec.model_id, "Loading mistralrs generator model");

        let model = if let Some(files) = &opts.gguf_files {
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
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let mut request = RequestBuilder::new();

        // Map messages to alternating User/Assistant roles.
        // Even-indexed messages (0, 2, 4, ...) are User, odd-indexed are Assistant.
        for (i, msg) in messages.iter().enumerate() {
            let role = if i % 2 == 0 {
                TextMessageRole::User
            } else {
                TextMessageRole::Assistant
            };
            request = request.add_message(role, msg);
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
        })
    }
}
