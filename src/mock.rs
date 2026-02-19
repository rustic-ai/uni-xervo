#![allow(dead_code)]

//! Mock implementations for testing
//!
//! This module provides mock model implementations and providers for testing purposes.
//! All types are gated with `#[cfg(test)]`.

use crate::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use crate::error::{Result, RuntimeError};
use crate::runtime::ModelRuntime;
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth, RerankerModel, ScoredDoc, TokenUsage,
};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Mock embedding model with configurable behavior
pub struct MockEmbeddingModel {
    dimensions: u32,
    model_id: String,
    fail_on_embed: bool,
    fail_count: AtomicU32,
    embed_delay_ms: u64,
    call_count: AtomicU32,
    warmup_count: Arc<AtomicU32>,
}

impl MockEmbeddingModel {
    pub fn new(dimensions: u32, model_id: String) -> Self {
        Self {
            dimensions,
            model_id,
            fail_on_embed: false,
            fail_count: AtomicU32::new(0),
            embed_delay_ms: 0,
            call_count: AtomicU32::new(0),
            warmup_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn with_fail_count(mut self, count: u32) -> Self {
        self.fail_count = AtomicU32::new(count);
        self
    }

    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.embed_delay_ms = delay_ms;
        self
    }

    pub fn with_warmup_tracker(mut self, tracker: Arc<AtomicU32>) -> Self {
        self.warmup_count = tracker;
        self
    }

    pub fn with_failure(mut self, fail: bool) -> Self {
        self.fail_on_embed = fail;
        self
    }

    pub fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn warmup_count(&self) -> u32 {
        self.warmup_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.embed_delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.embed_delay_ms)).await;
        }

        if self.fail_on_embed {
            return Err(RuntimeError::InferenceError(
                "Mock embedding failure".to_string(),
            ));
        }

        // Handle fail_count
        let current_fails = self.fail_count.load(Ordering::SeqCst);
        if current_fails > 0 {
            self.fail_count.fetch_sub(1, Ordering::SeqCst);
            return Err(RuntimeError::RateLimited); // RateLimited is retryable
        }

        // Return deterministic vectors
        let embeddings = texts
            .iter()
            .map(|_| vec![0.1; self.dimensions as usize])
            .collect();

        Ok(embeddings)
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn warmup(&self) -> Result<()> {
        self.warmup_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Mock reranker model with configurable behavior
pub struct MockRerankerModel {
    fail_on_rerank: bool,
    call_count: AtomicU32,
    warmup_count: AtomicU32,
}

impl MockRerankerModel {
    pub fn new() -> Self {
        Self {
            fail_on_rerank: false,
            call_count: AtomicU32::new(0),
            warmup_count: AtomicU32::new(0),
        }
    }

    pub fn with_failure(mut self, fail: bool) -> Self {
        self.fail_on_rerank = fail;
        self
    }

    pub fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn warmup_count(&self) -> u32 {
        self.warmup_count.load(Ordering::SeqCst)
    }
}

impl Default for MockRerankerModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RerankerModel for MockRerankerModel {
    async fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.fail_on_rerank {
            return Err(RuntimeError::InferenceError(
                "Mock reranker failure".to_string(),
            ));
        }

        // Return scored docs with descending scores
        let scored_docs = docs
            .iter()
            .enumerate()
            .map(|(i, text)| ScoredDoc {
                index: i,
                score: 1.0 / (i + 1) as f32,
                text: Some(text.to_string()),
            })
            .collect();

        Ok(scored_docs)
    }

    async fn warmup(&self) -> Result<()> {
        self.warmup_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Mock generator model with configurable behavior
pub struct MockGeneratorModel {
    response_text: String,
    fail_on_generate: bool,
    call_count: AtomicU32,
    warmup_count: AtomicU32,
}

impl MockGeneratorModel {
    pub fn new(response_text: String) -> Self {
        Self {
            response_text,
            fail_on_generate: false,
            call_count: AtomicU32::new(0),
            warmup_count: AtomicU32::new(0),
        }
    }

    pub fn with_failure(mut self, fail: bool) -> Self {
        self.fail_on_generate = fail;
        self
    }

    pub fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }

    pub fn warmup_count(&self) -> u32 {
        self.warmup_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl GeneratorModel for MockGeneratorModel {
    async fn generate(
        &self,
        messages: &[String],
        _options: GenerationOptions,
    ) -> Result<GenerationResult> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        if self.fail_on_generate {
            return Err(RuntimeError::InferenceError(
                "Mock generator failure".to_string(),
            ));
        }

        Ok(GenerationResult {
            text: self.response_text.clone(),
            usage: Some(TokenUsage {
                prompt_tokens: messages.join(" ").split_whitespace().count(),
                completion_tokens: self.response_text.split_whitespace().count(),
                total_tokens: messages.join(" ").split_whitespace().count()
                    + self.response_text.split_whitespace().count(),
            }),
        })
    }

    async fn warmup(&self) -> Result<()> {
        self.warmup_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Mock provider with configurable behavior
pub struct MockProvider {
    provider_id: &'static str,
    supported_tasks: Vec<ModelTask>,
    health: ProviderHealth,
    load_count: AtomicU32,
    warmup_count: AtomicU32,
    load_delay_ms: u64,
    model_delay_ms: u64,
    model_fail_count: u32,
    fail_on_load: bool,
    model_warmup_tracker: Option<Arc<AtomicU32>>,
}

impl MockProvider {
    pub fn new(provider_id: &'static str, supported_tasks: Vec<ModelTask>) -> Self {
        Self {
            provider_id,
            supported_tasks,
            health: ProviderHealth::Healthy,
            load_count: AtomicU32::new(0),
            warmup_count: AtomicU32::new(0),
            load_delay_ms: 0,
            model_delay_ms: 0,
            model_fail_count: 0,
            fail_on_load: false,
            model_warmup_tracker: None,
        }
    }

    pub fn with_model_fail_count(mut self, count: u32) -> Self {
        self.model_fail_count = count;
        self
    }

    pub fn with_model_delay(mut self, delay_ms: u64) -> Self {
        self.model_delay_ms = delay_ms;
        self
    }

    pub fn with_model_warmup_tracker(mut self, tracker: Arc<AtomicU32>) -> Self {
        self.model_warmup_tracker = Some(tracker);
        self
    }

    pub fn embed_only() -> Self {
        Self::new("mock/embed", vec![ModelTask::Embed])
    }

    pub fn generate_only() -> Self {
        Self::new("mock/generate", vec![ModelTask::Generate])
    }

    pub fn rerank_only() -> Self {
        Self::new("mock/rerank", vec![ModelTask::Rerank])
    }

    pub fn failing() -> Self {
        let mut provider = Self::new("mock/failing", vec![ModelTask::Embed]);
        provider.fail_on_load = true;
        provider
    }

    pub fn with_health(mut self, health: ProviderHealth) -> Self {
        self.health = health;
        self
    }

    pub fn with_load_delay(mut self, delay_ms: u64) -> Self {
        self.load_delay_ms = delay_ms;
        self
    }

    pub fn load_count(&self) -> u32 {
        self.load_count.load(Ordering::SeqCst)
    }

    pub fn warmup_count(&self) -> u32 {
        self.warmup_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    fn provider_id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: self.supported_tasks.clone(),
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        self.load_count.fetch_add(1, Ordering::SeqCst);

        if self.load_delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.load_delay_ms)).await;
        }

        if self.fail_on_load {
            return Err(RuntimeError::Load("Mock load failure".to_string()));
        }

        if !self.supported_tasks.contains(&spec.task) {
            return Err(RuntimeError::CapabilityMismatch(format!(
                "Mock provider does not support task {:?}",
                spec.task
            )));
        }

        // Use correct double-Arc wrapping pattern
        match spec.task {
            ModelTask::Embed => {
                let mut model = MockEmbeddingModel::new(384, spec.model_id.clone());
                if self.model_delay_ms > 0 {
                    model = model.with_delay(self.model_delay_ms);
                }
                if self.model_fail_count > 0 {
                    model = model.with_fail_count(self.model_fail_count);
                }
                if let Some(tracker) = &self.model_warmup_tracker {
                    model = model.with_warmup_tracker(tracker.clone());
                }
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let model = MockRerankerModel::new();
                let handle: Arc<dyn RerankerModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = MockGeneratorModel::new("Mock response".to_string());
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
        }
    }

    async fn health(&self) -> ProviderHealth {
        self.health.clone()
    }

    async fn warmup(&self) -> Result<()> {
        self.warmup_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Helper function to create a simple spec
pub fn make_spec(
    alias: &str,
    task: ModelTask,
    provider_id: &str,
    model_id: &str,
) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: provider_id.to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Object(serde_json::Map::new()),
    }
}

/// Create a runtime with a mock embedding provider and single alias
pub async fn runtime_with_embed() -> Result<Arc<ModelRuntime>> {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
}

/// Create a runtime with a mock generator provider and single alias
pub async fn runtime_with_generator() -> Result<Arc<ModelRuntime>> {
    let provider = MockProvider::generate_only();
    let spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/generate",
        "test-model",
    );

    ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
}

/// Create a runtime with a mock reranker provider and single alias
pub async fn runtime_with_reranker() -> Result<Arc<ModelRuntime>> {
    let provider = MockProvider::rerank_only();
    let spec = make_spec(
        "rerank/test",
        ModelTask::Rerank,
        "mock/rerank",
        "test-model",
    );

    ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
}
