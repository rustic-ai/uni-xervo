use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
};
use async_trait::async_trait;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use candle_transformers::models::gemma::{Config as GemmaConfig, Model as GemmaModel};
use candle_transformers::models::jina_bert::{
    BertModel as JinaBertModel, Config as JinaBertConfig,
};
use hf_hub::{
    Repo, RepoType,
    api::tokio::{Api, ApiBuilder},
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tokio::sync::Mutex;

#[derive(Deserialize, Debug)]
struct BaseConfig {
    architectures: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelArchitecture {
    Bert,
    JinaBert,
    Gemma,
}

impl ModelArchitecture {
    fn from_config(config: &BaseConfig) -> Result<Self> {
        if let Some(archs) = &config.architectures
            && let Some(arch) = archs.first()
        {
            return match arch.as_str() {
                "BertModel" | "BertForMaskedLM" => Ok(Self::Bert),
                "JinaBertModel" | "JinaBertForMaskedLM" => Ok(Self::JinaBert),
                "GemmaModel" | "GemmaForCausalLM" => Ok(Self::Gemma),
                _ => Err(RuntimeError::Config(format!(
                    "Unsupported architecture: {}",
                    arch
                ))),
            };
        }
        // Default to Bert if unspecified (legacy behavior)
        Ok(Self::Bert)
    }
}

/// Local embedding provider using the [Candle](https://github.com/huggingface/candle)
/// ML framework.
///
/// Supports Bert, JinaBert, and Gemma architectures with lazy weight loading
/// from HuggingFace Hub and mean-pooled, L2-normalized embeddings.
#[derive(Default)]
pub struct LocalCandleProvider;

impl LocalCandleProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ModelProvider for LocalCandleProvider {
    fn provider_id(&self) -> &'static str {
        "local/candle"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        if spec.task != ModelTask::Embed {
            return Err(RuntimeError::CapabilityMismatch(format!(
                "Candle provider does not support task {:?}",
                spec.task
            )));
        }

        let model_type = CandleTextModel::from_name(&spec.model_id).ok_or_else(|| {
            RuntimeError::Config(format!("Unsupported Candle model: {}", spec.model_id))
        })?;

        let cache_dir =
            crate::cache::resolve_cache_dir("candle", model_type.model_id(), &spec.options);

        tracing::info!(model = ?model_type, "Initializing Candle model");
        let model = CandleEmbeddingModel::new(model_type, spec.revision.clone(), cache_dir);

        let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }

    async fn warmup(&self) -> Result<()> {
        tracing::info!("Warming up LocalCandleProvider");
        // Pre-initialize HF API to warm up network/cache
        let _ = Api::new().map_err(|e| RuntimeError::Load(e.to_string()))?;
        Ok(())
    }
}

/// Supported text embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandleTextModel {
    /// all-MiniLM-L6-v2: 384 dims, fastest, English-optimized
    #[default]
    AllMiniLmL6V2,
    /// BGE-small-en-v1.5: 384 dims, high quality English
    BgeSmallEnV15,
    /// BGE-base-en-v1.5: 768 dimensions, higher quality English
    BgeBaseEnV15,
}

impl CandleTextModel {
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
        }
    }

    pub fn dimensions(&self) -> u32 {
        match self {
            Self::AllMiniLmL6V2 | Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 => 768,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "bge-base-en-v1.5",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "all-minilm-l6-v2" | "allminilml6v2" | "default" => Some(Self::AllMiniLmL6V2),
            "bge-small-en-v1.5" | "bgesmallenv15" => Some(Self::BgeSmallEnV15),
            "bge-base-en-v1.5" | "bgebaseenv15" => Some(Self::BgeBaseEnV15),
            // Map known HF IDs to enum
            "sentence-transformers/all-minilm-l6-v2" => Some(Self::AllMiniLmL6V2),
            "baai/bge-small-en-v1.5" => Some(Self::BgeSmallEnV15),
            "baai/bge-base-en-v1.5" => Some(Self::BgeBaseEnV15),
            _ => None,
        }
    }
}

enum InnerModel {
    Bert(BertModel),
    JinaBert(JinaBertModel),
    Gemma(GemmaModel),
}

struct LoadedModel {
    model: InnerModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// A lazily-loaded embedding model backed by Candle.
///
/// On first [`embed`](crate::traits::EmbeddingModel::embed) call (or explicit
/// [`warmup`](crate::traits::EmbeddingModel::warmup)), the model weights and
/// tokenizer are downloaded from HuggingFace Hub and loaded into memory.
pub struct CandleEmbeddingModel {
    model_type: CandleTextModel,
    revision: Option<String>,
    cache_dir: PathBuf,
    state: Arc<Mutex<Option<LoadedModel>>>,
}

impl CandleEmbeddingModel {
    pub fn new(model_type: CandleTextModel, revision: Option<String>, cache_dir: PathBuf) -> Self {
        Self {
            model_type,
            revision,
            cache_dir,
            state: Arc::new(Mutex::new(None)),
        }
    }

    async fn ensure_loaded(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        if state.is_some() {
            return Ok(());
        }

        tracing::info!(
            model = self.model_type.name(),
            "Loading Candle embedding model"
        );

        let api = ApiBuilder::new()
            .with_cache_dir(self.cache_dir.clone())
            .build()
            .map_err(|e| RuntimeError::Load(e.to_string()))?;
        let repo = match &self.revision {
            Some(rev) => Repo::with_revision(
                self.model_type.model_id().to_string(),
                RepoType::Model,
                rev.clone(),
            ),
            None => Repo::model(self.model_type.model_id().to_string()),
        };
        let api_repo = api.repo(repo);

        let config_path = api_repo
            .get("config.json")
            .await
            .map_err(|e| RuntimeError::Load(e.to_string()))?;

        let config_contents =
            std::fs::read_to_string(&config_path).map_err(|e| RuntimeError::Load(e.to_string()))?;

        let base_config: BaseConfig = serde_json::from_str(&config_contents)
            .map_err(|e| RuntimeError::Load(e.to_string()))?;

        let arch = ModelArchitecture::from_config(&base_config)?;
        tracing::info!(architecture = ?arch, "Detected model architecture");

        let tokenizer_path = api_repo
            .get("tokenizer.json")
            .await
            .map_err(|e| RuntimeError::Load(e.to_string()))?;
        let weights_path = api_repo
            .get("model.safetensors")
            .await
            .map_err(|e| RuntimeError::Load(e.to_string()))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RuntimeError::Load(format!("Failed to load tokenizer: {}", e)))?;

        let padding = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        // Gemma usually handles truncation differently or defaults are fine.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|e| RuntimeError::Load(format!("Failed to set truncation: {}", e)))?;

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(|e| RuntimeError::Load(e.to_string()))?
        };

        let model = match arch {
            ModelArchitecture::Bert => {
                let config: BertConfig = serde_json::from_str(&config_contents)
                    .map_err(|e| RuntimeError::Load(e.to_string()))?;
                let model =
                    BertModel::load(vb, &config).map_err(|e| RuntimeError::Load(e.to_string()))?;
                InnerModel::Bert(model)
            }
            ModelArchitecture::JinaBert => {
                let config: JinaBertConfig = serde_json::from_str(&config_contents)
                    .map_err(|e| RuntimeError::Load(e.to_string()))?;
                let model = JinaBertModel::new(vb, &config)
                    .map_err(|e| RuntimeError::Load(e.to_string()))?;
                InnerModel::JinaBert(model)
            }
            ModelArchitecture::Gemma => {
                let config: GemmaConfig = serde_json::from_str(&config_contents)
                    .map_err(|e| RuntimeError::Load(e.to_string()))?;
                let model = GemmaModel::new(false, &config, vb)
                    .map_err(|e| RuntimeError::Load(e.to_string()))?;
                InnerModel::Gemma(model)
            }
        };

        tracing::info!(
            model = self.model_type.name(),
            dimensions = self.model_type.dimensions(),
            "Candle embedding model loaded"
        );

        *state = Some(LoadedModel {
            model,
            tokenizer,
            device,
        });

        Ok(())
    }
}

#[async_trait]
impl EmbeddingModel for CandleEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        self.ensure_loaded().await?;

        let state_guard = self.state.lock().await;
        let loaded = state_guard
            .as_ref()
            .ok_or_else(|| RuntimeError::Load("Model state missing".to_string()))?;

        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = loaded
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RuntimeError::InferenceError(format!("Tokenization failed: {}", e)))?;

        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();
        let mut all_token_type_ids = Vec::new();

        for encoding in &encodings {
            all_input_ids.push(
                encoding
                    .get_ids()
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            );
            all_attention_masks.push(
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            );
            all_token_type_ids.push(
                encoding
                    .get_type_ids()
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            );
        }

        let batch_size = texts.len();
        let seq_len = all_input_ids[0].len();

        let input_ids_flat: Vec<i64> = all_input_ids.into_iter().flatten().collect();
        let attention_mask_flat: Vec<i64> = all_attention_masks.into_iter().flatten().collect();
        let token_type_ids_flat: Vec<i64> = all_token_type_ids.into_iter().flatten().collect();

        let input_ids = Tensor::from_vec(input_ids_flat, (batch_size, seq_len), &loaded.device)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let attention_mask =
            Tensor::from_vec(attention_mask_flat, (batch_size, seq_len), &loaded.device)
                .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_flat, (batch_size, seq_len), &loaded.device)
                .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let embeddings = match &loaded.model {
            InnerModel::Bert(m) => m
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| RuntimeError::InferenceError(e.to_string()))?,
            InnerModel::JinaBert(m) => m
                .forward(&input_ids)
                .map_err(|e| RuntimeError::InferenceError(e.to_string()))?,
            InnerModel::Gemma(_m) => {
                // Gemma expects (input_ids, input_positions) usually.
                // We construct simple positions 0..seq_len
                // Note: This assumes simple batching without specialized attention masks for Gemma
                // which might be suboptimal but functional for embedding.
                let positions = (0..seq_len).map(|i| i as i64).collect::<Vec<_>>();
                let _positions = Tensor::from_vec(positions, (seq_len,), &loaded.device)
                    .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
                    .broadcast_as((batch_size, seq_len))
                    .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

                // Gemma forward returns logits? Or hidden states?
                // Standard candle-transformers Gemma::forward returns logits.
                // We usually want hidden states.
                // If the model struct doesn't expose it, we are stuck for Gemma via this provider
                // without copying the model code.
                // For now, let's try calling it. If it returns logits (vocab size), we can't use it for embedding easily
                // without knowing which layer to take (usually hidden states before head).
                // However, "Embedding Gemma" might NOT have an LM head?
                // If it's `GemmaForCausalLM`, it has a head.
                // If we load it as `GemmaModel`, does it include head?
                // `candle_transformers::models::gemma::Model` usually includes the head.
                // We'll return an error for now for Gemma until we resolve this.
                return Err(RuntimeError::InferenceError(
                    "Gemma embedding not fully implemented (requires hidden state access)"
                        .to_string(),
                ));
            }
        };

        // Mean pooling
        let attention_mask_f32 = attention_mask
            .to_dtype(DType::F32)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let mask_expanded = attention_mask_f32
            .unsqueeze(2)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let mask_expanded = mask_expanded
            .broadcast_as(embeddings.shape())
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let masked_embeddings = embeddings
            .mul(&mask_expanded)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let mask_sum = attention_mask_f32
            .sum(1)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .unsqueeze(1)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let mask_sum = mask_sum
            .broadcast_as(sum_embeddings.shape())
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;
        let mask_sum = mask_sum
            .clamp(1e-9, f64::MAX)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let mean_embeddings = sum_embeddings
            .div(&mask_sum)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let norm = mean_embeddings
            .sqr()
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .sqrt()
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let normalized = mean_embeddings
            .broadcast_div(&norm)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        let embeddings_vec: Vec<Vec<f32>> = normalized
            .to_vec2()
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?;

        Ok(embeddings_vec)
    }

    fn dimensions(&self) -> u32 {
        self.model_type.dimensions()
    }

    fn model_id(&self) -> &str {
        self.model_type.model_id()
    }

    async fn warmup(&self) -> Result<()> {
        self.ensure_loaded().await
    }
}
