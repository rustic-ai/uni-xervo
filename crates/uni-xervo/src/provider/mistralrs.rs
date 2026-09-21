use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    ContentBlock, DocExtractOptions, DocExtractResult, DocumentExtractionModel, EmbedResult,
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, ImageInput,
    LoadedModelHandle, Message, MessageRole, ModelInfo, ModelProvider, ProviderCapabilities,
    ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, EmbeddingModelBuilder, EmbeddingRequestBuilder,
    GgufModelBuilder, IsqType, Model, ModelDType, PagedAttentionMetaBuilder, RequestBuilder,
    TextMessageRole, TextModelBuilder, UqffEmbeddingModelBuilder, UqffMultimodalModelBuilder,
    UqffTextModelBuilder,
};
use serde::Deserialize;
use std::path::PathBuf;
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
            supported_tasks: vec![
                ModelTask::Embed,
                ModelTask::Generate,
                ModelTask::DocumentExtract,
            ],
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
            ModelTask::DocumentExtract => self.load_document_extractor(spec, &opts).await,
            ModelTask::Raw => Err(RuntimeError::CapabilityMismatch(
                "mistralrs provider does not support task Raw".to_string(),
            )),
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
            let mut builder = if let Some(files) = &opts.uqff_files {
                let paths: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
                UqffEmbeddingModelBuilder::new(&spec.model_id, paths).into_inner()
            } else {
                EmbeddingModelBuilder::new(&spec.model_id)
            };

            let dtype = resolve_model_dtype(opts)?;
            builder = builder.with_dtype(dtype);

            if opts.uqff_files.is_none()
                && let Some(ref isq_str) = opts.isq
            {
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

            if let Some(setting) = build_device_map_setting_text(opts) {
                builder = builder.with_device_mapping(setting);
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
                let paged_cfg = PagedAttentionMetaBuilder::default().build().map_err(|e| {
                    RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
                })?;
                builder = builder.with_paged_attn(paged_cfg);
            }
            builder = builder.with_logging();

            builder.build().await.map_err(|e| {
                RuntimeError::Load(format!(
                    "Failed to build mistralrs GGUF generator model: {}",
                    e
                ))
            })?
        } else {
            let mut builder = if let Some(files) = &opts.uqff_files {
                let paths: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
                UqffTextModelBuilder::new(&spec.model_id, paths).into_inner()
            } else {
                TextModelBuilder::new(&spec.model_id)
            };

            let dtype = resolve_model_dtype(opts)?;
            builder = builder.with_dtype(dtype);

            if opts.uqff_files.is_none()
                && let Some(ref isq_str) = opts.isq
            {
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
                let paged_cfg = PagedAttentionMetaBuilder::default().build().map_err(|e| {
                    RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
                })?;
                builder = builder.with_paged_attn(paged_cfg);
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

            if let Some(setting) = build_device_map_setting_text(opts) {
                builder = builder.with_device_mapping(setting);
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

    async fn build_vision_service(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<Arc<dyn GeneratorModel>> {
        use mistralrs::MultimodalModelBuilder;

        if opts.gguf_files.is_some() {
            return Err(RuntimeError::Config(
                "GGUF is not supported for the vision pipeline".to_string(),
            ));
        }

        tracing::info!(model_id = %spec.model_id, "Loading mistralrs vision generator model");

        let mut builder = if let Some(files) = &opts.uqff_files {
            let paths: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
            UqffMultimodalModelBuilder::new(&spec.model_id, paths).into_inner()
        } else {
            MultimodalModelBuilder::new(&spec.model_id)
        };
        let dtype = resolve_model_dtype(opts)?;
        builder = builder.with_dtype(dtype);

        if opts.uqff_files.is_none()
            && let Some(ref isq_str) = opts.isq
        {
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
            let paged_cfg = PagedAttentionMetaBuilder::default().build().map_err(|e| {
                RuntimeError::Load(format!("Failed to configure paged attention: {}", e))
            })?;
            builder = builder.with_paged_attn(paged_cfg);
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
        if let Some(setting) = build_device_map_setting_multimodal(opts) {
            builder = builder.with_device_mapping(setting);
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
        Ok(Arc::new(service) as Arc<dyn GeneratorModel>)
    }

    async fn load_vision_generator(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        let handle = self.build_vision_service(spec, opts).await?;
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn load_document_extractor(
        &self,
        spec: &ModelAliasSpec,
        opts: &MistralRsOptions,
    ) -> Result<LoadedModelHandle> {
        // Document extraction (olmOCR-2 and similar) always runs on the vision
        // pipeline, regardless of any `pipeline` option the caller set.
        let generator = self.build_vision_service(spec, opts).await?;
        // Read from the typed options rather than re-reading `spec.options`:
        // one source of truth, so the struct and the accepted schema cannot
        // drift apart again.
        let style_str = opts.style.as_deref().unwrap_or("olmocr");
        let style = crate::doc_parse::style_from_str(style_str).ok_or_else(|| {
            RuntimeError::Config(format!(
                "Document extractor '{}' has unknown `style` value '{style_str}'; \
                 expected one of: granite-docling, mineru, olmocr",
                spec.alias
            ))
        })?;
        let extractor = MistralRsDocumentExtractor {
            generator,
            model_id: spec.model_id.clone(),
            style,
        };
        let handle: Arc<dyn DocumentExtractionModel> = Arc::new(extractor);
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
    /// UQFF (mistralrs pre-quantized) filenames or paths.
    ///
    /// For HuggingFace UQFF repos, only the first shard needs to be named
    /// (e.g. "q4k-0.uqff"); remaining shards are auto-discovered. The
    /// quantization variant is selected by which file is named (q4k vs q5k
    /// vs afq8 etc.). UQFF skips the bf16-then-quantize load flow and is
    /// the practical path for fitting larger multimodal models on small
    /// GPUs. Mutually exclusive with `gguf_files` and `isq` (validation
    /// rejects the combinations). Honored on text, vision, and embedding
    /// pipelines.
    uqff_files: Option<Vec<String>>,
    /// Model data type: "auto", "f16", "bf16", "f32"
    dtype: Option<String>,
    /// Pipeline type: "text" (default), "vision", "diffusion", "speech"
    pipeline: Option<String>,
    /// Diffusion loader type: "flux", "flux_offloaded"
    diffusion_loader_type: Option<String>,
    /// Speech loader type: "dia"
    speech_loader_type: Option<String>,
    /// Override auto-device-mapper max sequence length (default 4096).
    /// Lowering this lets more layers fit on small GPUs, since the planner
    /// reserves KV-cache headroom proportional to this value.
    /// Honored by text, vision, and embedding pipelines.
    max_seq_len: Option<usize>,
    /// Override auto-device-mapper max batch size (default 1).
    /// Honored by text, vision, and embedding pipelines.
    max_batch_size: Option<usize>,
    /// Override max image shape (height, width) for the multimodal planner
    /// (default [1024, 1024]). Lowering this frees VRAM for layer placement
    /// on small GPUs. Vision pipeline only.
    max_image_shape: Option<[usize; 2]>,
    /// Override max number of images per request (default 1).
    /// Vision pipeline only.
    max_num_images: Option<usize>,
    /// Document-extraction output parser: "granite-docling", "mineru", or
    /// "olmocr" (the default). Selects how the generated text is parsed back
    /// into [`DocBlock`](crate::traits::DocBlock)s. Consumed by
    /// `load_document_extractor`; ignored by every other task.
    ///
    /// This field exists because the struct is `deny_unknown_fields`: without
    /// it, any catalog entry carrying `style` fails to deserialize at `load`
    /// time, for *every* task, before the task dispatch is even reached.
    style: Option<String>,
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
    if opts.force_cpu {
        tracing::info!("force_cpu=true; defaulting dtype to F32");
        Ok(ModelDType::F32)
    } else if !has_gpu_support() {
        tracing::info!("GPU feature not enabled (gpu-cuda/gpu-metal); defaulting dtype to F32");
        Ok(ModelDType::F32)
    } else {
        Ok(ModelDType::Auto)
    }
}

#[allow(unexpected_cfgs)]
fn has_gpu_support() -> bool {
    cfg!(any(feature = "gpu-cuda", feature = "gpu-metal"))
}

/// Build a text-style `DeviceMapSetting` if any text-relevant override is set.
///
/// Returns `None` when the user has not opted in, so callers leave the
/// builder at its default (mistralrs picks `default_text` internally).
fn build_device_map_setting_text(opts: &MistralRsOptions) -> Option<DeviceMapSetting> {
    if opts.max_seq_len.is_none() && opts.max_batch_size.is_none() {
        return None;
    }
    Some(DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
        max_seq_len: opts
            .max_seq_len
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN),
        max_batch_size: opts
            .max_batch_size
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE),
    }))
}

/// Build a multimodal `DeviceMapSetting` if any text- or multimodal-relevant
/// override is set.
fn build_device_map_setting_multimodal(opts: &MistralRsOptions) -> Option<DeviceMapSetting> {
    if opts.max_seq_len.is_none()
        && opts.max_batch_size.is_none()
        && opts.max_image_shape.is_none()
        && opts.max_num_images.is_none()
    {
        return None;
    }
    let max_image_shape = opts.max_image_shape.map(|[h, w]| (h, w)).unwrap_or((
        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
    ));
    Some(DeviceMapSetting::Auto(AutoDeviceMapParams::Multimodal {
        max_seq_len: opts
            .max_seq_len
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN),
        max_batch_size: opts
            .max_batch_size
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE),
        max_image_shape,
        max_num_images: opts
            .max_num_images
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
    }))
}

/// Extract the text of the last user message, which is the most relevant
/// prompt for single-shot pipelines like diffusion and speech.
fn extract_last_user_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .filter(|m| m.role == MessageRole::User)
        .flat_map(|m| m.content.iter())
        .find_map(|b| match b {
            ContentBlock::Text(t) => Some(t.clone()),
            _ => None,
        })
        .unwrap_or_default()
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
    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult> {
        if texts.is_empty() {
            return Ok(EmbedResult {
                vectors: vec![],
                usage: None,
            });
        }

        let request =
            EmbeddingRequestBuilder::new().add_prompts(texts.iter().map(|s| s.to_string()));

        let embeddings = self.model.generate_embeddings(request).await.map_err(|e| {
            RuntimeError::InferenceError(format!("Embedding inference failed: {}", e))
        })?;

        validate_embeddings(&embeddings)?;

        Ok(EmbedResult {
            vectors: embeddings,
            usage: None,
        })
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }
}

impl ModelInfo for MistralRsEmbeddingService {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Generator service
// ---------------------------------------------------------------------------

struct MistralRsGeneratorService {
    model: Model,
    model_id: String,
}

impl ModelInfo for MistralRsGeneratorService {
    fn model_id(&self) -> &str {
        &self.model_id
    }
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
    model_id: String,
}

impl ModelInfo for MistralRsVisionService {
    fn model_id(&self) -> &str {
        &self.model_id
    }
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
            let mut images: Vec<image::DynamicImage> = Vec::new();
            for block in &msg.content {
                if let ContentBlock::Image(img_input) = block {
                    let bytes = match img_input {
                        crate::traits::ImageInput::Bytes { data, .. } => data.clone(),
                        crate::traits::ImageInput::Url(_url) => {
                            return Err(RuntimeError::Config(
                                "URL-based image input not yet supported in vision pipeline"
                                    .to_string(),
                            ));
                        }
                    };
                    let img = image::load_from_memory(&bytes).map_err(|e| {
                        RuntimeError::InferenceError(format!("Failed to decode image: {}", e))
                    })?;
                    images.push(img);
                }
            }

            let text = msg.text();

            if images.is_empty() {
                request = request.add_message(role, text);
            } else {
                request = request.add_image_message(role, text, images);
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
// Document extraction service (olmOCR-2 and similar, on the vision pipeline)
// ---------------------------------------------------------------------------

/// The olmOCR-2 prompt — image-only, no document-anchoring text.
///
/// olmOCR-2 dropped olmOCR-1's anchor-text requirement; this is the static
/// no-anchoring instruction. The model is given a single rendered page and
/// returns Markdown with a YAML front-matter header.
const OLMOCR_PROMPT: &str = "Attached is one page of a document that you must process. \
Just return the plain text representation of this document as if you were reading it naturally. \
Convert equations to LateX and tables to HTML. \
If there are any figures or charts, label them with the following markdown syntax \
`![Alt text...](page_startx_starty_width_height.png)`. \
Return your output as markdown, with a front matter section on top specifying values for the \
primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.";

/// First-pass sampling temperature for document extraction.
///
/// olmOCR's reference pipeline starts near-greedy and only escalates on retry;
/// this is the deterministic-leaning first pass. Temperature-escalating retries
/// are a future refinement layered above this single pass.
const DOC_EXTRACT_TEMPERATURE: f32 = 0.1;

/// Maximum tokens to generate per page (olmOCR's reference cap).
const DOC_EXTRACT_MAX_TOKENS: usize = 8000;

/// Document extractor wrapping a mistral.rs vision generator (e.g. olmOCR-2).
struct MistralRsDocumentExtractor {
    generator: Arc<dyn GeneratorModel>,
    model_id: String,
    style: crate::doc_parse::DocStyle,
}

impl ModelInfo for MistralRsDocumentExtractor {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl DocumentExtractionModel for MistralRsDocumentExtractor {
    async fn extract(
        &self,
        pages: Vec<ImageInput>,
        _options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>> {
        let mut results = Vec::with_capacity(pages.len());
        // One page per request: olmOCR-2 is single-image, and this sidesteps the
        // known multi-image KV-cache issue in the vision pipeline.
        for page in pages {
            let messages = [Message {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text(OLMOCR_PROMPT.to_string()),
                    ContentBlock::Image(page),
                ],
            }];
            let options = GenerationOptions {
                max_tokens: Some(DOC_EXTRACT_MAX_TOKENS),
                temperature: Some(DOC_EXTRACT_TEMPERATURE),
                top_p: None,
                width: None,
                height: None,
            };
            let generated = self.generator.generate(&messages, options).await?;
            results.push(crate::doc_parse::parse_by_style(
                self.style,
                &generated.text,
            ));
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Diffusion service
// ---------------------------------------------------------------------------

struct MistralRsDiffusionService {
    model: Model,
    model_id: String,
}

impl ModelInfo for MistralRsDiffusionService {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl GeneratorModel for MistralRsDiffusionService {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        use mistralrs::DiffusionGenerationParams;

        // Extract the text prompt from the last user message
        let prompt = extract_last_user_prompt(messages);

        let height = options.height.unwrap_or(720) as usize;
        let width = options.width.unwrap_or(1280) as usize;

        let response = self
            .model
            .generate_image(
                prompt,
                mistralrs::ImageGenerationResponseFormat::B64Json,
                DiffusionGenerationParams { height, width },
                None,
            )
            .await
            .map_err(|e| {
                RuntimeError::InferenceError(format!("Diffusion inference failed: {}", e))
            })?;

        // The response is a base64-encoded image
        let first = response.data.first().ok_or_else(|| {
            RuntimeError::InferenceError("Diffusion response returned no image data".to_string())
        })?;
        let b64 = first.b64_json.as_deref().ok_or_else(|| {
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
    model_id: String,
}

impl ModelInfo for MistralRsSpeechService {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl GeneratorModel for MistralRsSpeechService {
    async fn generate(
        &self,
        messages: &[Message],
        _options: GenerationOptions,
    ) -> Result<GenerationResult> {
        // Extract the text prompt from the last user message
        let prompt = extract_last_user_prompt(messages);

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
    // Options deserialization
    // -----------------------------------------------------------------------

    /// `MistralRsOptions` is `deny_unknown_fields`, so every key the provider
    /// accepts must exist as a field. `style` is read by
    /// `load_document_extractor`, and the catalog validator permits it — but
    /// it was missing here, which made any spec carrying it fail to load.
    ///
    /// This is the exact options JSON used by
    /// `tests/mistralrs_doc_extract_expensive_test.rs`. That test is
    /// `#[ignore]`d behind `EXPENSIVE_TESTS`, so the drift went unnoticed; a
    /// builder-level test would not catch it either, because `WarmupPolicy::Lazy`
    /// means `build()` runs option *validation* but never `load()`.
    #[test]
    fn options_accept_style_field() {
        let opts: MistralRsOptions =
            serde_json::from_value(serde_json::json!({"isq": "Q4K", "style": "olmocr"}))
                .expect("options carrying `style` must deserialize");
        assert_eq!(opts.style.as_deref(), Some("olmocr"));
        assert_eq!(opts.isq.as_deref(), Some("Q4K"));
    }

    /// Absent `style` still means the olmOCR default, and genuinely unknown
    /// keys must still be rejected — the fix adds a field, it does not weaken
    /// `deny_unknown_fields`.
    #[test]
    fn options_style_defaults_and_unknown_keys_still_rejected() {
        let opts: MistralRsOptions = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(opts.style, None);

        let err = serde_json::from_value::<MistralRsOptions>(
            serde_json::json!({"not_a_real_option": true}),
        )
        .err()
        .expect("unknown keys must still be rejected");
        assert!(err.to_string().contains("not_a_real_option"), "{err}");
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

    mod extract_last_user_prompt_tests {
        use super::*;

        #[test]
        fn returns_last_user_text() {
            let messages = vec![
                Message::user("first"),
                Message::assistant("reply"),
                Message::user("second"),
            ];
            assert_eq!(extract_last_user_prompt(&messages), "second");
        }

        #[test]
        fn skips_system_and_assistant() {
            let messages = vec![
                Message::system("system prompt"),
                Message::assistant("assistant reply"),
            ];
            assert_eq!(extract_last_user_prompt(&messages), "");
        }

        #[test]
        fn empty_messages_returns_empty() {
            assert_eq!(extract_last_user_prompt(&[]), "");
        }
    }
}

#[cfg(test)]
mod doc_extract_tests {
    use super::*;
    use crate::traits::{DocBlockKind, DocOutputFormat};

    /// A vision generator stub that returns scripted text and asserts the
    /// per-request contract (one message carrying the olmOCR prompt + exactly
    /// one page image).
    struct MockVisionGen {
        reply: String,
    }

    impl ModelInfo for MockVisionGen {
        fn model_id(&self) -> &str {
            "mock/vision"
        }
    }

    #[async_trait]
    impl GeneratorModel for MockVisionGen {
        async fn generate(
            &self,
            messages: &[Message],
            _options: GenerationOptions,
        ) -> Result<GenerationResult> {
            assert_eq!(messages.len(), 1, "one message per request");
            let image_count = messages[0]
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::Image(_)))
                .count();
            assert_eq!(image_count, 1, "exactly one page image per request");
            let has_prompt = messages[0]
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("front matter")));
            assert!(has_prompt, "the olmOCR prompt must be included");
            Ok(GenerationResult {
                text: self.reply.clone(),
                usage: None,
                images: vec![],
                audio: None,
            })
        }
    }

    fn page() -> ImageInput {
        ImageInput::Bytes {
            data: vec![0u8; 4],
            media_type: "image/png".to_string(),
        }
    }

    fn options() -> DocExtractOptions {
        DocExtractOptions {
            output: DocOutputFormat::Markdown,
            include_tables: true,
            include_formulas: true,
            include_bboxes: true,
        }
    }

    #[tokio::test]
    async fn olmocr_extractor_builds_prompt_and_parses_output() {
        let generator: Arc<dyn GeneratorModel> = Arc::new(MockVisionGen {
            reply: "# Title\n\nA paragraph.".to_string(),
        });
        let extractor = MistralRsDocumentExtractor {
            generator,
            model_id: "allenai/olmOCR-2-7B-1025".to_string(),
            style: crate::doc_parse::DocStyle::OlmOcr,
        };

        let results = extractor.extract(vec![page()], options()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(results[0].blocks[0].content, "Title");
    }

    #[tokio::test]
    async fn olmocr_extractor_processes_each_page_singly() {
        let generator: Arc<dyn GeneratorModel> = Arc::new(MockVisionGen {
            reply: "body".to_string(),
        });
        let extractor = MistralRsDocumentExtractor {
            generator,
            model_id: "olmocr".to_string(),
            style: crate::doc_parse::DocStyle::OlmOcr,
        };

        // Two pages -> two results; the per-request assertions guarantee each
        // generate() call carried exactly one image.
        let results = extractor
            .extract(vec![page(), page()], options())
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }
}
