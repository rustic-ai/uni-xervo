//! The core runtime that manages providers, catalogs, and loaded model instances.

mod hybrid_adapter;

use crate::api::{ModelAliasSpec, ModelRuntimeKey};
use crate::error::{Result, RuntimeError};
use crate::options_validation::validate_provider_options;
use crate::reliability::{
    InstrumentedAudioEmbeddingModel, InstrumentedDocumentExtractionModel,
    InstrumentedEmbeddingModel, InstrumentedGeneratorModel, InstrumentedHybridEmbeddingModel,
    InstrumentedImageEmbeddingModel, InstrumentedMultiVectorEmbeddingModel,
    InstrumentedMultimodalEmbeddingModel, InstrumentedNlpModel, InstrumentedOcrModel,
    InstrumentedRawTensorModel, InstrumentedRerankerModel, InstrumentedSparseEmbeddingModel,
    InstrumentedTranscriptionModel,
};
use crate::traits::{
    AudioEmbeddingModel, DocumentExtractionModel, EmbeddingModel, GeneratorModel,
    HybridEmbeddingModel, ImageEmbeddingModel, LoadedModelHandle, ModelProvider,
    MultiVectorEmbeddingModel, MultimodalEmbeddingModel, NlpModel, OcrModel, RawTensorModel,
    RerankerModel, SparseEmbeddingModel, TranscriptionModel,
};
use dashmap::DashMap;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Per-alias cache of fully-instrumented typed handles.
///
/// Eliminates repeated spec lookups, key hashing, registry reads, and
/// wrapper allocations on hot paths where the same alias is resolved
/// many times (e.g. per-turn inference in a pipeline).
///
/// Uses [`DashMap`] for lock-free concurrent reads on the hot path.
/// Entries are populated on first successful resolve via
/// `entry().or_insert_with(...)`, so concurrent first-callers all observe
/// the same `Arc` and the wrapper is allocated exactly once per alias.
/// Once populated, an entry is never mutated.
#[derive(Default)]
struct HandleCache {
    embeddings: DashMap<String, Arc<dyn EmbeddingModel>>,
    rerankers: DashMap<String, Arc<dyn RerankerModel>>,
    generators: DashMap<String, Arc<dyn GeneratorModel>>,
    raw_tensor_models: DashMap<String, Arc<dyn RawTensorModel>>,
    // Multimodal extension surface — added in Phase 6, instrumentation
    // filled in by Phase 7.
    image_embedders: DashMap<String, Arc<dyn ImageEmbeddingModel>>,
    audio_embedders: DashMap<String, Arc<dyn AudioEmbeddingModel>>,
    multimodal_embedders: DashMap<String, Arc<dyn MultimodalEmbeddingModel>>,
    sparse_embedders: DashMap<String, Arc<dyn SparseEmbeddingModel>>,
    multi_vector_embedders: DashMap<String, Arc<dyn MultiVectorEmbeddingModel>>,
    hybrid_embedders: DashMap<String, Arc<dyn HybridEmbeddingModel>>,
    nlp_models: DashMap<String, Arc<dyn NlpModel>>,
    document_extractors: DashMap<String, Arc<dyn DocumentExtractionModel>>,
    transcribers: DashMap<String, Arc<dyn TranscriptionModel>>,
    ocr_models: DashMap<String, Arc<dyn OcrModel>>,
}

/// Default load timeout applied when [`ModelAliasSpec::load_timeout`] is `None`.
const DEFAULT_LOAD_TIMEOUT_SECS: u64 = 600;

/// The central runtime that owns registered providers and a catalog of model
/// aliases.
///
/// Obtain an instance via [`ModelRuntime::builder()`] and the
/// [`ModelRuntimeBuilder`].  Once built, use [`embedding`](Self::embedding),
/// [`reranker`](Self::reranker), or [`generator`](Self::generator) to obtain
/// typed, instrumented model handles.
///
/// Models are loaded lazily on first access (unless configured for eager or
/// background warmup) and cached in an internal registry so that subsequent
/// requests for the same model are served instantly.
pub struct ModelRuntime {
    providers: HashMap<String, Box<dyn ModelProvider>>,
    registry: Arc<ModelRegistry>,
    catalog: RwLock<HashMap<String, ModelAliasSpec>>,
    handle_cache: HandleCache,
}

/// Internal registry that caches loaded model instances and coordinates
/// concurrent load requests to prevent duplicate work.
#[derive(Default)]
pub struct ModelRegistry {
    instances: RwLock<HashMap<ModelRuntimeKey, LoadedModelHandle>>,
    /// Per-key mutexes to prevent concurrent loads of the same model.
    loader_locks: Mutex<HashMap<ModelRuntimeKey, Arc<Mutex<()>>>>,
}

impl ModelRuntime {
    /// Create a new [`ModelRuntimeBuilder`] for configuring and constructing a
    /// runtime.
    pub fn builder() -> ModelRuntimeBuilder {
        ModelRuntimeBuilder::default()
    }

    /// Register a new model alias at runtime.
    pub async fn register(&self, spec: ModelAliasSpec) -> Result<()> {
        spec.validate()?;
        if !self.providers.contains_key(&spec.provider_id) {
            return Err(RuntimeError::Config(format!(
                "Unknown provider '{}' for alias '{}'",
                spec.provider_id, spec.alias
            )));
        }
        validate_provider_options(&spec.provider_id, spec.task, &spec.options)?;
        let mut catalog = self.catalog.write().await;
        if catalog.contains_key(&spec.alias) {
            return Err(RuntimeError::Config(format!(
                "Alias '{}' already exists",
                spec.alias
            )));
        }
        catalog.insert(spec.alias.clone(), spec);
        Ok(())
    }

    /// Check if an alias exists in the catalog.
    pub async fn contains_alias(&self, alias: &str) -> bool {
        let catalog = self.catalog.read().await;
        catalog.contains_key(alias)
    }

    /// Look up a spec by alias, returning an error if not found.
    async fn lookup_spec(&self, alias: &str) -> Result<ModelAliasSpec> {
        let catalog = self.catalog.read().await;
        catalog
            .get(alias)
            .cloned()
            .ok_or_else(|| RuntimeError::AliasNotFound {
                alias: alias.to_string(),
            })
    }

    /// Pre-load and cache every model in the catalog.
    ///
    /// Models already loaded are skipped. Fails fast on the first error.
    /// Call this during application startup to avoid cold-start latency on
    /// first inference.
    pub async fn prefetch_all(&self) -> Result<()> {
        let specs: Vec<ModelAliasSpec> = {
            let catalog = self.catalog.read().await;
            catalog.values().cloned().collect()
        };
        for spec in specs {
            tracing::info!(alias = %spec.alias, "Prefetching model");
            self.resolve_and_load_internal(&spec).await?;
        }
        Ok(())
    }

    /// Pre-load and cache specific aliases.
    ///
    /// Returns an error immediately if an alias is not found in the catalog
    /// or if any model fails to load. Models already loaded are skipped.
    pub async fn prefetch(&self, aliases: &[&str]) -> Result<()> {
        for alias in aliases {
            let spec = self.lookup_spec(alias).await?;
            tracing::info!(alias = %alias, "Prefetching model");
            self.resolve_and_load_internal(&spec).await?;
        }
        Ok(())
    }

    /// Resolve, load (if necessary), and return an instrumented [`EmbeddingModel`]
    /// handle for the given alias.
    ///
    /// The returned handle is cached per alias so that repeated calls skip
    /// spec lookup, key hashing, and wrapper allocation.
    pub async fn embedding(&self, alias: &str) -> Result<Arc<dyn EmbeddingModel>> {
        if let Some(cached) = self.handle_cache.embeddings.get(alias) {
            return Ok(cached.clone());
        }

        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn EmbeddingModel>>() {
            let cached = self
                .handle_cache
                .embeddings
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn EmbeddingModel> = Arc::new(InstrumentedEmbeddingModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }

        // As in `sparse_embedder`: an `embed_hybrid` alias can serve a dense
        // query from its dense head.
        if let Some(model) = handle.downcast_ref::<Arc<dyn HybridEmbeddingModel>>() {
            let dimensions = hybrid_adapter::head_width_of(
                model,
                crate::traits::HeadSet::DENSE,
                alias,
                &spec.provider_id,
                "EmbeddingModel",
            )?;
            let view: Arc<dyn EmbeddingModel> = Arc::new(hybrid_adapter::HybridAsDense {
                inner: model.clone(),
                dimensions,
            });
            let cached = self
                .handle_cache
                .embeddings
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn EmbeddingModel> = Arc::new(InstrumentedEmbeddingModel {
                        inner: view,
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }

        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement EmbeddingModel",
            alias
        )))
    }

    /// Resolve a dense text [`EmbeddingModel`] by alias.
    ///
    /// Agent-noun alias for [`embedding`](Self::embedding), matching the
    /// `image_embedder` / `sparse_embedder` / `multi_vector_embedder` naming.
    pub async fn embedder(&self, alias: &str) -> Result<Arc<dyn EmbeddingModel>> {
        self.embedding(alias).await
    }

    /// Resolve, load (if necessary), and return an instrumented [`RerankerModel`]
    /// handle for the given alias.
    ///
    /// The returned handle is cached per alias so that repeated calls skip
    /// spec lookup, key hashing, and wrapper allocation.
    pub async fn reranker(&self, alias: &str) -> Result<Arc<dyn RerankerModel>> {
        if let Some(cached) = self.handle_cache.rerankers.get(alias) {
            return Ok(cached.clone());
        }

        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn RerankerModel>>() {
            let cached = self
                .handle_cache
                .rerankers
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn RerankerModel> = Arc::new(InstrumentedRerankerModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement RerankerModel",
            alias
        )))
    }

    /// Resolve, load (if necessary), and return an instrumented [`GeneratorModel`]
    /// handle for the given alias.
    ///
    /// The returned handle is cached per alias so that repeated calls skip
    /// spec lookup, key hashing, and wrapper allocation.
    pub async fn generator(&self, alias: &str) -> Result<Arc<dyn GeneratorModel>> {
        if let Some(cached) = self.handle_cache.generators.get(alias) {
            return Ok(cached.clone());
        }

        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn GeneratorModel>>() {
            let cached = self
                .handle_cache
                .generators
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn GeneratorModel> = Arc::new(InstrumentedGeneratorModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement GeneratorModel",
            alias
        )))
    }

    /// Resolve, load (if necessary), and return an instrumented [`RawTensorModel`]
    /// handle for the given alias.
    ///
    /// The returned handle is cached per alias so that repeated calls skip
    /// spec lookup, key hashing, and wrapper allocation.
    pub async fn raw_tensor_model(&self, alias: &str) -> Result<Arc<dyn RawTensorModel>> {
        if let Some(cached) = self.handle_cache.raw_tensor_models.get(alias) {
            return Ok(cached.clone());
        }

        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn RawTensorModel>>() {
            let cached = self
                .handle_cache
                .raw_tensor_models
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn RawTensorModel> = Arc::new(InstrumentedRawTensorModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }

        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "RawTensorModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`ImageEmbeddingModel`] handle for the given alias.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use uni_xervo::runtime::ModelRuntime;
    /// # use uni_xervo::traits::ImageInput;
    /// # async fn run(runtime: Arc<ModelRuntime>) -> uni_xervo::error::Result<()> {
    /// let embedder = runtime.image_embedder("embed/siglip").await?;
    /// let image = ImageInput::Bytes {
    ///     data: std::fs::read("photo.png").unwrap(),
    ///     media_type: "image/png".to_string(),
    /// };
    /// let result = embedder.embed(vec![image]).await?;
    /// println!("dimension: {}", result.vectors[0].len());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model fails to load, or
    /// the provider does not implement image embedding.
    pub async fn image_embedder(&self, alias: &str) -> Result<Arc<dyn ImageEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.image_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn ImageEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .image_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn ImageEmbeddingModel> =
                        Arc::new(InstrumentedImageEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "ImageEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`AudioEmbeddingModel`] handle for the given alias.
    pub async fn audio_embedder(&self, alias: &str) -> Result<Arc<dyn AudioEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.audio_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn AudioEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .audio_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn AudioEmbeddingModel> =
                        Arc::new(InstrumentedAudioEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "AudioEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`MultimodalEmbeddingModel`] handle for the given alias.
    pub async fn multimodal_embedder(
        &self,
        alias: &str,
    ) -> Result<Arc<dyn MultimodalEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.multimodal_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn MultimodalEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .multimodal_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn MultimodalEmbeddingModel> =
                        Arc::new(InstrumentedMultimodalEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "MultimodalEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`SparseEmbeddingModel`] handle for the given alias.
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model fails to load, or
    /// the provider does not implement sparse embedding.
    pub async fn sparse_embedder(&self, alias: &str) -> Result<Arc<dyn SparseEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.sparse_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn SparseEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .sparse_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn SparseEmbeddingModel> =
                        Arc::new(InstrumentedSparseEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        // An `embed_hybrid` alias stores `Arc<dyn HybridEmbeddingModel>`, so the
        // downcast above misses even though the graph has a sparse head. Serve
        // it through a single-head view rather than making the caller register a
        // second alias for the same weights.
        if let Some(model) = handle.downcast_ref::<Arc<dyn HybridEmbeddingModel>>() {
            let vocab_size = hybrid_adapter::head_width_of(
                model,
                crate::traits::HeadSet::SPARSE,
                alias,
                &spec.provider_id,
                "SparseEmbeddingModel",
            )?;
            let view: Arc<dyn SparseEmbeddingModel> = Arc::new(hybrid_adapter::HybridAsSparse {
                inner: model.clone(),
                vocab_size,
            });
            let cached = self
                .handle_cache
                .sparse_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    // Instrument outside the view so timeouts, retries and
                    // metrics are attributed to this alias and task.
                    let wrapper: Arc<dyn SparseEmbeddingModel> =
                        Arc::new(InstrumentedSparseEmbeddingModel {
                            inner: view,
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "SparseEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`MultiVectorEmbeddingModel`] handle for the given alias.
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model fails to load, or
    /// the provider does not implement multi-vector embedding.
    pub async fn multi_vector_embedder(
        &self,
        alias: &str,
    ) -> Result<Arc<dyn MultiVectorEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.multi_vector_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn MultiVectorEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .multi_vector_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn MultiVectorEmbeddingModel> =
                        Arc::new(InstrumentedMultiVectorEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        // Same single-head view as `sparse_embedder`, for the ColBERT head.
        if let Some(model) = handle.downcast_ref::<Arc<dyn HybridEmbeddingModel>>() {
            let dimensions = hybrid_adapter::head_width_of(
                model,
                crate::traits::HeadSet::MULTI_VECTOR,
                alias,
                &spec.provider_id,
                "MultiVectorEmbeddingModel",
            )?;
            let view: Arc<dyn MultiVectorEmbeddingModel> =
                Arc::new(hybrid_adapter::HybridAsMultiVector {
                    inner: model.clone(),
                    dimensions,
                });
            let cached = self
                .handle_cache
                .multi_vector_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn MultiVectorEmbeddingModel> =
                        Arc::new(InstrumentedMultiVectorEmbeddingModel {
                            inner: view,
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "MultiVectorEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`HybridEmbeddingModel`] handle for the given alias.
    ///
    /// The handle serves dense, sparse, and multi-vector heads from a single
    /// forward pass; select which to materialize with a
    /// [`HeadSet`](crate::traits::HeadSet). Only multi-output graphs with a
    /// hybrid preset (e.g. `BGEM3Hybrid`) resolve here — single-head models use
    /// the per-task resolvers.
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model has no hybrid preset,
    /// the model fails to load, or the loaded handle lacks the hybrid capability.
    pub async fn hybrid_embedder(&self, alias: &str) -> Result<Arc<dyn HybridEmbeddingModel>> {
        if let Some(cached) = self.handle_cache.hybrid_embedders.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn HybridEmbeddingModel>>() {
            let cached = self
                .handle_cache
                .hybrid_embedders
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn HybridEmbeddingModel> =
                        Arc::new(InstrumentedHybridEmbeddingModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "HybridEmbeddingModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented [`NlpModel`]
    /// handle for the given alias.
    pub async fn nlp_model(&self, alias: &str) -> Result<Arc<dyn NlpModel>> {
        if let Some(cached) = self.handle_cache.nlp_models.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn NlpModel>>() {
            let cached = self
                .handle_cache
                .nlp_models
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn NlpModel> = Arc::new(InstrumentedNlpModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "NlpModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`DocumentExtractionModel`] handle for the given alias.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use uni_xervo::runtime::ModelRuntime;
    /// # use uni_xervo::traits::{DocExtractOptions, DocOutputFormat, ImageInput};
    /// # async fn run(runtime: Arc<ModelRuntime>) -> uni_xervo::error::Result<()> {
    /// let extractor = runtime.document_extractor("docext/olmocr").await?;
    /// let page = ImageInput::Bytes {
    ///     data: std::fs::read("page.png").unwrap(),
    ///     media_type: "image/png".to_string(),
    /// };
    /// let options = DocExtractOptions {
    ///     output: DocOutputFormat::Markdown,
    ///     include_tables: true,
    ///     include_formulas: true,
    ///     include_bboxes: false,
    /// };
    /// let pages = extractor.extract(vec![page], options).await?;
    /// println!("{}", pages[0].plain_markdown);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model fails to load, or
    /// the provider does not implement document extraction.
    pub async fn document_extractor(
        &self,
        alias: &str,
    ) -> Result<Arc<dyn DocumentExtractionModel>> {
        if let Some(cached) = self.handle_cache.document_extractors.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn DocumentExtractionModel>>() {
            let cached = self
                .handle_cache
                .document_extractors
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn DocumentExtractionModel> =
                        Arc::new(InstrumentedDocumentExtractionModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "DocumentExtractionModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented
    /// [`TranscriptionModel`] handle for the given alias.
    pub async fn transcriber(&self, alias: &str) -> Result<Arc<dyn TranscriptionModel>> {
        if let Some(cached) = self.handle_cache.transcribers.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn TranscriptionModel>>() {
            let cached = self
                .handle_cache
                .transcribers
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn TranscriptionModel> =
                        Arc::new(InstrumentedTranscriptionModel {
                            inner: model.clone(),
                            alias: alias.to_string(),
                            provider_id: spec.provider_id.clone(),
                            timeout: spec.timeout.map(std::time::Duration::from_secs),
                            retry: spec.retry.clone(),
                        });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "TranscriptionModel".to_string(),
        })
    }

    /// Resolve, load (if necessary), and return an instrumented [`OcrModel`]
    /// handle for the given alias.
    ///
    /// # Examples
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use uni_xervo::runtime::ModelRuntime;
    /// # use uni_xervo::traits::ImageInput;
    /// # async fn run(runtime: Arc<ModelRuntime>) -> uni_xervo::error::Result<()> {
    /// let ocr = runtime.ocr_model("ocr/ppocr-en").await?;
    /// let image = ImageInput::Bytes {
    ///     data: std::fs::read("scan.png").unwrap(),
    ///     media_type: "image/png".to_string(),
    /// };
    /// let results = ocr.recognize(vec![image]).await?;
    /// println!("{}", results[0].plain_text);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the alias is unknown, the model fails to load, or
    /// the provider does not implement OCR.
    pub async fn ocr_model(&self, alias: &str) -> Result<Arc<dyn OcrModel>> {
        if let Some(cached) = self.handle_cache.ocr_models.get(alias) {
            return Ok(cached.clone());
        }
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn OcrModel>>() {
            let cached = self
                .handle_cache
                .ocr_models
                .entry(alias.to_string())
                .or_insert_with(|| {
                    let wrapper: Arc<dyn OcrModel> = Arc::new(InstrumentedOcrModel {
                        inner: model.clone(),
                        alias: alias.to_string(),
                        provider_id: spec.provider_id.clone(),
                        timeout: spec.timeout.map(std::time::Duration::from_secs),
                        retry: spec.retry.clone(),
                    });
                    wrapper
                })
                .clone();
            return Ok(cached);
        }
        Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: spec.provider_id,
            capability: "OcrModel".to_string(),
        })
    }

    #[tracing::instrument(skip(self, spec), fields(provider, model))]
    async fn resolve_and_load_internal(
        &self,
        spec: &ModelAliasSpec,
    ) -> Result<Arc<dyn Any + Send + Sync>> {
        let key = ModelRuntimeKey::new(spec);

        // Fast path: already loaded
        {
            let registry = self.registry.instances.read().await;
            if let Some(handle) = registry.get(&key) {
                return Ok(handle.clone());
            }
        }

        // Slow path: coordinate loading
        let lock = {
            let mut locks = self.registry.loader_locks.lock().await;
            locks
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        // Acquire loader lock for this key
        let _guard = lock.lock().await;

        // Double-check after acquiring the loader lock
        {
            let registry = self.registry.instances.read().await;
            if let Some(handle) = registry.get(&key) {
                let result = Ok(handle.clone());
                let mut locks = self.registry.loader_locks.lock().await;
                locks.remove(&key);
                return result;
            }
        }

        let load_timeout =
            std::time::Duration::from_secs(spec.load_timeout.unwrap_or(DEFAULT_LOAD_TIMEOUT_SECS));

        let result = match tokio::time::timeout(load_timeout, async {
            let provider = self.providers.get(&spec.provider_id).ok_or_else(|| {
                RuntimeError::ProviderNotFound(format!("Provider '{}' not found", spec.provider_id))
            })?;

            tracing::info!(alias = %spec.alias, provider = %spec.provider_id, "Loading model instance");
            let start = std::time::Instant::now();
            let handle_result = provider.load(spec).await;
            let duration = start.elapsed().as_secs_f64();

            metrics::histogram!("model_load.duration_seconds").record(duration);

            let handle = match handle_result {
                Ok(h) => {
                    metrics::counter!("model_load.total", "status" => "success").increment(1);
                    h
                }
                Err(e) => {
                    metrics::counter!("model_load.total", "status" => "failure").increment(1);
                    tracing::error!(alias = %spec.alias, error = %e, "Model load failed");
                    return Err(e);
                }
            };

            // Model warmup. Dispatch covers every task trait — including the
            // multimodal surface, which previously fell through to no warmup at
            // all. The trait default is a no-op, so adding arms only enables
            // eager weight loading for providers that override `warmup`.
            if let Some(model) = handle.downcast_ref::<Arc<dyn EmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn RerankerModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn GeneratorModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn RawTensorModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn ImageEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn AudioEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn MultimodalEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn SparseEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn MultiVectorEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn HybridEmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn NlpModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn DocumentExtractionModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn TranscriptionModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn OcrModel>>() {
                model.warmup().await?;
            }

            {
                let mut registry = self.registry.instances.write().await;
                registry.insert(key.clone(), handle.clone());
            }

            Ok(handle)
        })
        .await
        {
            Ok(res) => res,
            Err(_) => {
                metrics::counter!("model_load.total", "status" => "failure").increment(1);
                tracing::error!(
                    alias = %spec.alias,
                    provider = %spec.provider_id,
                    timeout_secs = load_timeout.as_secs(),
                    "Model load timed out"
                );
                Err(RuntimeError::Timeout)
            }
        };

        // Bound loader lock map growth by removing this key once the load path completes.
        // Existing waiters hold cloned lock Arcs, so this is safe.
        {
            let mut locks = self.registry.loader_locks.lock().await;
            locks.remove(&key);
        }

        result
    }
}

/// Builder for constructing a [`ModelRuntime`] with registered providers,
/// a model catalog, and a warmup policy.
///
/// ```rust,no_run
/// # use uni_xervo::runtime::ModelRuntime;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = ModelRuntime::builder()
///     // .register_provider(...)
///     // .catalog(...)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct ModelRuntimeBuilder {
    providers: HashMap<String, Box<dyn ModelProvider>>,
    catalog: Vec<ModelAliasSpec>,
    warmup_policy: crate::api::WarmupPolicy,
}

impl ModelRuntimeBuilder {
    /// Register a provider. The provider's
    /// [`provider_id`](crate::traits::ModelProvider::provider_id) is used as
    /// the lookup key; registering a second provider with the same ID
    /// replaces the first.
    pub fn register_provider<P: ModelProvider + 'static>(mut self, provider: P) -> Self {
        self.providers
            .insert(provider.provider_id().to_string(), Box::new(provider));
        self
    }

    /// Set the model catalog from a pre-built vector of specs.
    pub fn catalog(mut self, catalog: Vec<ModelAliasSpec>) -> Self {
        self.catalog = catalog;
        self
    }

    /// Load catalog from a JSON string (array of model alias specs).
    pub fn catalog_from_str(mut self, s: &str) -> Result<Self> {
        self.catalog = crate::api::catalog_from_str(s)?;
        Ok(self)
    }

    /// Load catalog from a JSON file (array of model alias specs).
    pub fn catalog_from_file(mut self, path: impl AsRef<std::path::Path>) -> Result<Self> {
        self.catalog = crate::api::catalog_from_file(path)?;
        Ok(self)
    }

    /// Set the global warmup policy applied to providers during
    /// [`build`](Self::build).
    pub fn warmup_policy(mut self, policy: crate::api::WarmupPolicy) -> Self {
        self.warmup_policy = policy;
        self
    }

    /// Validate the catalog, execute the warmup policy, and return the
    /// constructed [`ModelRuntime`].
    ///
    /// Returns an error if any spec references an unknown provider, contains
    /// invalid options, or if a required eager warmup fails.
    pub async fn build(self) -> Result<Arc<ModelRuntime>> {
        let mut catalog_map = HashMap::new();
        for spec in self.catalog {
            spec.validate()?;
            if !self.providers.contains_key(&spec.provider_id) {
                return Err(RuntimeError::Config(format!(
                    "Unknown provider '{}' for alias '{}'",
                    spec.provider_id, spec.alias
                )));
            }
            validate_provider_options(&spec.provider_id, spec.task, &spec.options)?;
            if catalog_map.insert(spec.alias.clone(), spec).is_some() {
                return Err(RuntimeError::Config(
                    "Duplicate alias in catalog".to_string(),
                ));
            }
        }

        let runtime = Arc::new(ModelRuntime {
            providers: self.providers,
            registry: Arc::new(ModelRegistry::default()),
            catalog: RwLock::new(catalog_map),
            handle_cache: HandleCache::default(),
        });

        // Provider Warmup Phase
        match self.warmup_policy {
            crate::api::WarmupPolicy::Eager => {
                for (id, provider) in &runtime.providers {
                    tracing::info!(provider = %id, "Eagerly warming up provider");
                    provider.warmup().await.map_err(|e| {
                        RuntimeError::Load(format!("Failed to warmup provider {}: {}", id, e))
                    })?;
                }
            }
            crate::api::WarmupPolicy::Background => {
                for id in runtime.providers.keys() {
                    tracing::info!(provider = %id, "Scheduling background provider warmup");
                    // We have the Arc<ModelRuntime> already.
                    let rt = runtime.clone();
                    let provider_id = id.clone();
                    tokio::spawn(async move {
                        if let Some(provider) = rt.providers.get(&provider_id)
                            && let Err(e) = provider.warmup().await
                        {
                            tracing::error!(provider = %provider_id, error = %e, "Background provider warmup failed");
                        }
                    });
                }
            }
            crate::api::WarmupPolicy::Lazy => {
                tracing::debug!("Lazy provider warmup (no-op)");
            }
        }

        // Model Warmup Phase
        let mut warmup_tasks = Vec::new();

        let specs: Vec<ModelAliasSpec> = {
            let catalog = runtime.catalog.read().await;
            catalog.values().cloned().collect()
        };

        for spec in specs {
            match spec.warmup {
                crate::api::WarmupPolicy::Eager => {
                    tracing::info!(alias = %spec.alias, "Eagerly warming up model");
                    if let Err(e) = runtime.resolve_and_load_internal(&spec).await {
                        if spec.required {
                            return Err(e);
                        }
                        tracing::error!(
                            alias = %spec.alias,
                            provider = %spec.provider_id,
                            error = %e,
                            "Optional eager model warmup failed; continuing startup"
                        );
                    }
                }
                crate::api::WarmupPolicy::Background => {
                    tracing::info!(alias = %spec.alias, "Scheduling background warmup");
                    let rt = runtime.clone();
                    let spec_clone = spec.clone();
                    // Spawn background task
                    warmup_tasks.push(tokio::spawn(async move {
                        if let Err(e) = rt.resolve_and_load_internal(&spec_clone).await {
                            tracing::error!(alias = %spec_clone.alias, error = %e, "Background warmup failed");
                        }
                    }));
                }
                crate::api::WarmupPolicy::Lazy => {
                    tracing::debug!(alias = %spec.alias, "Lazy warmup (no-op)");
                }
            }
        }

        // We don't await background tasks here, they run detached.
        // Eager tasks are already awaited.

        Ok(runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ModelTask;
    use crate::mock::{MockProvider, make_spec};

    #[tokio::test]
    async fn loader_lock_entries_cleaned_after_successful_load() {
        let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
        let runtime = ModelRuntime::builder()
            .register_provider(MockProvider::embed_only())
            .catalog(vec![spec])
            .build()
            .await
            .unwrap();

        let _ = runtime.embedding("embed/test").await.unwrap();

        let locks = runtime.registry.loader_locks.lock().await;
        assert!(
            locks.is_empty(),
            "loader lock map should be empty after load"
        );
    }

    #[tokio::test]
    async fn loader_lock_entries_cleaned_after_failed_load() {
        let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");
        spec.warmup = crate::api::WarmupPolicy::Lazy;
        let runtime = ModelRuntime::builder()
            .register_provider(MockProvider::failing())
            .catalog(vec![spec])
            .build()
            .await
            .unwrap();

        let err = runtime.embedding("embed/test").await;
        assert!(err.is_err());

        let locks = runtime.registry.loader_locks.lock().await;
        assert!(
            locks.is_empty(),
            "loader lock map should be empty after failure"
        );
    }

    #[tokio::test]
    async fn loader_lock_entries_cleaned_after_load_timeout() {
        let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
        spec.warmup = crate::api::WarmupPolicy::Lazy;
        spec.load_timeout = Some(1);

        let runtime = ModelRuntime::builder()
            .register_provider(MockProvider::embed_only().with_load_delay(2_000))
            .catalog(vec![spec])
            .build()
            .await
            .unwrap();

        let err = runtime.embedding("embed/test").await;
        assert!(matches!(err, Err(RuntimeError::Timeout)));

        let locks = runtime.registry.loader_locks.lock().await;
        assert!(
            locks.is_empty(),
            "loader lock map should be empty after load timeout"
        );
    }

    // -----------------------------------------------------------------------
    // An EmbedHybrid alias serves the single-head accessors (issue #49)
    // -----------------------------------------------------------------------

    use crate::traits::HeadSet;

    /// Register one `EmbedHybrid` alias, the way a hybrid retrieval pipeline
    /// does, and resolve it through each single-head accessor.
    async fn hybrid_runtime(provider: MockProvider) -> Arc<ModelRuntime> {
        let spec = make_spec(
            "embed/hybrid",
            ModelTask::EmbedHybrid,
            "mock/hybrid",
            "aapot/bge-m3-onnx",
        );
        ModelRuntime::builder()
            .register_provider(provider)
            .catalog(vec![spec])
            .build()
            .await
            .unwrap()
    }

    fn hybrid_provider() -> MockProvider {
        MockProvider::new("mock/hybrid", vec![ModelTask::EmbedHybrid])
    }

    /// The reported bug: both query channels failed with
    /// `ProviderCapabilityMissing` while ingest wrote the columns fine.
    #[tokio::test]
    async fn hybrid_alias_serves_sparse_and_multi_vector() {
        let runtime = hybrid_runtime(hybrid_provider()).await;

        let sparse = runtime
            .sparse_embedder("embed/hybrid")
            .await
            .expect("a hybrid alias must satisfy a sparse query");
        assert_eq!(sparse.vocab_size(), 250002);
        let out = sparse.embed(&["a", "b"]).await.unwrap();
        assert_eq!(out.vectors.len(), 2);

        let mv = runtime
            .multi_vector_embedder("embed/hybrid")
            .await
            .expect("a hybrid alias must satisfy a multi-vector query");
        assert_eq!(mv.dimensions(), 1024);
        let out = mv.embed(&["a", "b"]).await.unwrap();
        assert_eq!(out.vectors.len(), 2);
    }

    /// Not in the original report: the dense accessor has the same shape, so a
    /// hybrid alias must serve a dense query too.
    #[tokio::test]
    async fn hybrid_alias_serves_dense() {
        let runtime = hybrid_runtime(hybrid_provider()).await;

        let dense = runtime
            .embedding("embed/hybrid")
            .await
            .expect("a hybrid alias must satisfy a dense query");
        assert_eq!(dense.dimensions(), 1024);
        let out = dense.embed(&["a"]).await.unwrap();
        assert_eq!(out.vectors.len(), 1);
        assert_eq!(out.vectors[0].len(), 1024);
    }

    /// The hybrid handle itself must keep working unchanged.
    #[tokio::test]
    async fn hybrid_alias_still_serves_the_hybrid_accessor() {
        let runtime = hybrid_runtime(hybrid_provider()).await;

        let hybrid = runtime.hybrid_embedder("embed/hybrid").await.unwrap();
        let out = hybrid.embed(&["a"], HeadSet::ALL).await.unwrap();
        assert!(out.dense.is_some() && out.sparse.is_some() && out.multi_vector.is_some());
    }

    /// A head the graph does not expose must still fail — but say why, rather
    /// than claiming the provider has no sparse capability at all.
    #[tokio::test]
    async fn hybrid_alias_without_the_head_fails_with_a_specific_message() {
        let provider = hybrid_provider().with_hybrid_heads(HeadSet::DENSE);
        let runtime = hybrid_runtime(provider).await;

        let err = runtime
            .sparse_embedder("embed/hybrid")
            .await
            .err()
            .expect("a hybrid model with no sparse head cannot serve sparse");
        let msg = err.to_string();
        assert!(msg.contains("does not expose that head"), "{msg}");
        assert!(msg.contains("embed/hybrid"), "{msg}");
    }

    /// A hybrid implementation that doesn't report widths cannot be adapted,
    /// because the single-head traits must answer `vocab_size` / `dimensions`.
    #[tokio::test]
    async fn hybrid_alias_without_widths_is_rejected() {
        let runtime = hybrid_runtime(hybrid_provider().without_hybrid_widths()).await;

        let err = runtime
            .sparse_embedder("embed/hybrid")
            .await
            .err()
            .expect("no reported width means no faithful vocab_size");
        assert!(err.to_string().contains("does not report its width"));
    }

    /// The per-alias handle cache must hold the adapted view, so the second
    /// resolve is served from cache rather than rebuilt.
    #[tokio::test]
    async fn adapted_hybrid_handles_are_cached_per_alias() {
        let runtime = hybrid_runtime(hybrid_provider()).await;

        let first = runtime.sparse_embedder("embed/hybrid").await.unwrap();
        let second = runtime.sparse_embedder("embed/hybrid").await.unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "expected the cached adapter, not a fresh one"
        );
    }

    /// A non-hybrid alias that genuinely lacks the capability must keep its
    /// original error — the fallback must not mask real mismatches.
    #[tokio::test]
    async fn non_hybrid_alias_still_reports_capability_missing() {
        let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
        let runtime = ModelRuntime::builder()
            .register_provider(MockProvider::embed_only())
            .catalog(vec![spec])
            .build()
            .await
            .unwrap();

        let err = runtime.sparse_embedder("embed/test").await.err().unwrap();
        assert!(matches!(
            err,
            RuntimeError::ProviderCapabilityMissing { .. }
        ));
        assert!(err.to_string().contains("SparseEmbeddingModel"));
    }
}
