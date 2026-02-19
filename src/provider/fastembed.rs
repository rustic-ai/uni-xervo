use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
};
use anyhow::anyhow;
use async_trait::async_trait;
use fastembed::{InitOptions, TextEmbedding};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::oneshot;

/// Local embedding provider using [FastEmbed](https://github.com/Anush008/fastembed-rs)
/// (ONNX Runtime).
///
/// Supports a wide range of embedding models. Inference is offloaded to a
/// dedicated thread with an enlarged stack to accommodate ONNX Runtime's
/// requirements.
pub struct LocalFastEmbedProvider;

impl LocalFastEmbedProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalFastEmbedProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for LocalFastEmbedProvider {
    fn provider_id(&self) -> &'static str {
        "local/fastembed"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        if spec.task != ModelTask::Embed {
            return Err(RuntimeError::CapabilityMismatch(format!(
                "FastEmbed provider does not support task {:?}",
                spec.task
            )));
        }

        let model_name = spec.model_id.clone();
        let cache_dir = crate::cache::resolve_cache_dir("fastembed", &model_name, &spec.options);

        // Offload initialization to a blocking thread because it can refer to onnxruntime which might be heavy
        // fastembed init might block.
        let service =
            tokio::task::spawn_blocking(move || FastEmbedService::new(&model_name, &cache_dir))
                .await
                .map_err(|e| RuntimeError::Load(format!("Join error: {}", e)))?
                .map_err(|e| RuntimeError::Load(e.to_string()))?;

        let handle: Arc<dyn EmbeddingModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// Stack size for embedding threads.
const EMBEDDING_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Wrapper around a [`TextEmbedding`] instance that implements
/// [`EmbeddingModel`].
///
/// Each inference call spawns a short-lived worker thread with a larger stack
/// to satisfy ONNX Runtime's stack requirements.
pub struct FastEmbedService {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: u32,
}

impl FastEmbedService {
    pub fn new(model_name: &str, cache_dir: &Path) -> anyhow::Result<Self> {
        let model_enum = match model_name {
            "AllMiniLML6V2" | "all-MiniLM-L6-v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
            "AllMiniLML6V2Q" => fastembed::EmbeddingModel::AllMiniLML6V2Q,
            "AllMiniLML12V2" => fastembed::EmbeddingModel::AllMiniLML12V2,
            "AllMiniLML12V2Q" => fastembed::EmbeddingModel::AllMiniLML12V2Q,
            "AllMpnetBaseV2" | "all-mpnet-base-v2" => fastembed::EmbeddingModel::AllMpnetBaseV2,
            "BGEBaseENV15" | "bge-base-en-v1.5" => fastembed::EmbeddingModel::BGEBaseENV15,
            "BGEBaseENV15Q" => fastembed::EmbeddingModel::BGEBaseENV15Q,
            "BGELargeENV15" | "bge-large-en-v1.5" => fastembed::EmbeddingModel::BGELargeENV15,
            "BGELargeENV15Q" => fastembed::EmbeddingModel::BGELargeENV15Q,
            "BGESmallENV15" | "bge-small-en-v1.5" => fastembed::EmbeddingModel::BGESmallENV15,
            "BGESmallENV15Q" => fastembed::EmbeddingModel::BGESmallENV15Q,
            "NomicEmbedTextV1" => fastembed::EmbeddingModel::NomicEmbedTextV1,
            "NomicEmbedTextV15" | "nomic-embed-text-v1.5" => {
                fastembed::EmbeddingModel::NomicEmbedTextV15
            }
            "NomicEmbedTextV15Q" => fastembed::EmbeddingModel::NomicEmbedTextV15Q,
            "ParaphraseMLMiniLML12V2" => fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMiniLML12V2Q" => fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            "ParaphraseMLMpnetBaseV2" => fastembed::EmbeddingModel::ParaphraseMLMpnetBaseV2,
            "BGESmallZHV15" => fastembed::EmbeddingModel::BGESmallZHV15,
            "BGELargeZHV15" => fastembed::EmbeddingModel::BGELargeZHV15,
            "BGEM3" => fastembed::EmbeddingModel::BGEM3,
            "ModernBertEmbedLarge" => fastembed::EmbeddingModel::ModernBertEmbedLarge,
            "MultilingualE5Small" | "multilingual-e5-small" => {
                fastembed::EmbeddingModel::MultilingualE5Small
            }
            "MultilingualE5Base" | "multilingual-e5-base" => {
                fastembed::EmbeddingModel::MultilingualE5Base
            }
            "MultilingualE5Large" | "multilingual-e5-large" => {
                fastembed::EmbeddingModel::MultilingualE5Large
            }
            "MxbaiEmbedLargeV1" | "mxbai-embed-large-v1" => {
                fastembed::EmbeddingModel::MxbaiEmbedLargeV1
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported FastEmbed model: {}. Please check fastembed docs for supported models.",
                    model_name
                ));
            }
        };

        let mut options = InitOptions::new(model_enum.clone());
        options = options.with_cache_dir(cache_dir.to_path_buf());

        let model = TextEmbedding::try_new(options)
            .map_err(|e| anyhow!("Failed to initialize FastEmbed model: {}", e))?;

        // Determine dimensions
        let dimensions = match model_enum {
            fastembed::EmbeddingModel::AllMiniLML6V2
            | fastembed::EmbeddingModel::AllMiniLML6V2Q
            | fastembed::EmbeddingModel::AllMiniLML12V2
            | fastembed::EmbeddingModel::AllMiniLML12V2Q
            | fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2
            | fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2Q
            | fastembed::EmbeddingModel::BGESmallENV15
            | fastembed::EmbeddingModel::BGESmallENV15Q
            | fastembed::EmbeddingModel::MultilingualE5Small => 384,

            fastembed::EmbeddingModel::BGESmallZHV15 => 512,

            fastembed::EmbeddingModel::AllMpnetBaseV2
            | fastembed::EmbeddingModel::ParaphraseMLMpnetBaseV2
            | fastembed::EmbeddingModel::BGEBaseENV15
            | fastembed::EmbeddingModel::BGEBaseENV15Q
            | fastembed::EmbeddingModel::NomicEmbedTextV1
            | fastembed::EmbeddingModel::NomicEmbedTextV15
            | fastembed::EmbeddingModel::NomicEmbedTextV15Q
            | fastembed::EmbeddingModel::MultilingualE5Base => 768,

            fastembed::EmbeddingModel::BGELargeENV15
            | fastembed::EmbeddingModel::BGELargeENV15Q
            | fastembed::EmbeddingModel::BGELargeZHV15
            | fastembed::EmbeddingModel::BGEM3
            | fastembed::EmbeddingModel::ModernBertEmbedLarge
            | fastembed::EmbeddingModel::MultilingualE5Large
            | fastembed::EmbeddingModel::MxbaiEmbedLargeV1 => 1024,

            _ => {
                // Fallback for new models or quantized variants not explicitly listed
                // We could log a warning here or return a default.
                // Assuming 768 is a safe-ish bet for unknown models or 1024 for "Large" ones.
                // Better approach: Since we can't easily probe without loading, we might just
                // assume a default and let the user override via config if needed.
                // But for now, to satisfy exhaustiveness:
                1024
            }
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: model_name.to_string(),
            dimensions,
        })
    }
}

#[async_trait]
impl EmbeddingModel for FastEmbedService {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model = self.model.clone();

        let (tx, rx) = oneshot::channel();

        // Spawn a dedicated thread with larger stack for ONNX Runtime
        thread::Builder::new()
            .name("fastembed-worker".to_string())
            .stack_size(EMBEDDING_THREAD_STACK_SIZE)
            .spawn(move || {
                let result = model
                    .lock()
                    .map_err(|_| anyhow!("Failed to lock embedding model"))
                    .and_then(|mut guard| {
                        guard
                            .embed(texts_vec, None)
                            .map_err(|e| anyhow!("FastEmbed error: {}", e))
                    });
                let _ = tx.send(result);
            })
            .map_err(|e| {
                RuntimeError::InferenceError(format!("Failed to spawn embedding thread: {}", e))
            })?;

        let result = rx
            .await
            .map_err(|_| RuntimeError::InferenceError("Embedding thread panicked".to_string()))?;

        result.map_err(|e| RuntimeError::InferenceError(e.to_string()))
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_name
    }
}
