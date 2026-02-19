//! The core runtime that manages providers, catalogs, and loaded model instances.

use crate::api::{ModelAliasSpec, ModelRuntimeKey};
use crate::error::{Result, RuntimeError};
use crate::options_validation::validate_provider_options;
use crate::reliability::{
    InstrumentedEmbeddingModel, InstrumentedGeneratorModel, InstrumentedRerankerModel,
};
use crate::traits::{
    EmbeddingModel, GeneratorModel, LoadedModelHandle, ModelProvider, RerankerModel,
};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

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
            .ok_or_else(|| RuntimeError::Config(format!("Alias '{}' not found", alias)))
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
    pub async fn embedding(&self, alias: &str) -> Result<Arc<dyn EmbeddingModel>> {
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn EmbeddingModel>>() {
            let instrumented = InstrumentedEmbeddingModel {
                inner: model.clone(),
                alias: alias.to_string(),
                provider_id: spec.provider_id.clone(),
                timeout: spec.timeout.map(std::time::Duration::from_secs),
                retry: spec.retry.clone(),
            };
            return Ok(Arc::new(instrumented));
        }

        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement EmbeddingModel",
            alias
        )))
    }

    /// Resolve, load (if necessary), and return an instrumented [`RerankerModel`]
    /// handle for the given alias.
    pub async fn reranker(&self, alias: &str) -> Result<Arc<dyn RerankerModel>> {
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn RerankerModel>>() {
            let instrumented = InstrumentedRerankerModel {
                inner: model.clone(),
                alias: alias.to_string(),
                provider_id: spec.provider_id.clone(),
                timeout: spec.timeout.map(std::time::Duration::from_secs),
                retry: spec.retry.clone(),
            };
            return Ok(Arc::new(instrumented));
        }
        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement RerankerModel",
            alias
        )))
    }

    /// Resolve, load (if necessary), and return an instrumented [`GeneratorModel`]
    /// handle for the given alias.
    pub async fn generator(&self, alias: &str) -> Result<Arc<dyn GeneratorModel>> {
        let spec = self.lookup_spec(alias).await?;
        let handle = self.resolve_and_load_internal(&spec).await?;
        if let Some(model) = handle.downcast_ref::<Arc<dyn GeneratorModel>>() {
            let instrumented = InstrumentedGeneratorModel {
                inner: model.clone(),
                alias: alias.to_string(),
                provider_id: spec.provider_id.clone(),
                timeout: spec.timeout.map(std::time::Duration::from_secs),
                retry: spec.retry.clone(),
            };
            return Ok(Arc::new(instrumented));
        }
        Err(RuntimeError::CapabilityMismatch(format!(
            "Model for alias '{}' does not implement GeneratorModel",
            alias
        )))
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

            // Model warmup
            if let Some(model) = handle.downcast_ref::<Arc<dyn EmbeddingModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn RerankerModel>>() {
                model.warmup().await?;
            } else if let Some(model) = handle.downcast_ref::<Arc<dyn GeneratorModel>>() {
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
}
