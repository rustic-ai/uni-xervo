use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{RemoteProviderBase, check_http_status, resolve_api_key};
use crate::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
    RerankerModel, ScoredDoc,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

/// Remote provider that calls the [Voyage AI API](https://docs.voyageai.com/reference/embeddings-api)
/// for embedding and reranking. Does not support text generation.
///
/// Requires the `VOYAGE_API_KEY` environment variable (or a custom env var
/// name via the `api_key_env` option).
pub struct RemoteVoyageAIProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteVoyageAIProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteVoyageAIProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn insert_test_breaker(&self, key: crate::api::ModelRuntimeKey, age: std::time::Duration) {
        self.base.insert_test_breaker(key, age);
    }

    #[cfg(test)]
    fn breaker_count(&self) -> usize {
        self.base.breaker_count()
    }

    #[cfg(test)]
    fn force_cleanup_now_for_test(&self) {
        self.base.force_cleanup_now_for_test();
    }
}

#[async_trait]
impl ModelProvider for RemoteVoyageAIProvider {
    fn provider_id(&self) -> &'static str {
        "remote/voyageai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Rerank],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "VOYAGE_API_KEY")?;

        match spec.task {
            ModelTask::Embed => {
                let model = VoyageAIEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let model = VoyageAIRerankerModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn RerankerModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Voyage AI provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

struct VoyageAIEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl EmbeddingModel for VoyageAIEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let response = self
                    .client
                    .post("https://api.voyageai.com/v1/embeddings")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&json!({
                        "input": texts,
                        "model": self.model_id
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Voyage AI", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let mut embeddings = Vec::new();
                if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
                    for item in data {
                        if let Some(embedding) = item.get("embedding").and_then(|e| e.as_array()) {
                            let vec: Vec<f32> = embedding
                                .iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            embeddings.push(vec);
                        }
                    }
                }
                Ok(embeddings)
            })
            .await
    }

    fn dimensions(&self) -> u32 {
        match self.model_id.as_str() {
            "voyage-large-2" => 1536,
            _ => 1024,
        }
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

struct VoyageAIRerankerModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl RerankerModel for VoyageAIRerankerModel {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        let query = query.to_string();
        let docs: Vec<String> = docs.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let response = self
                    .client
                    .post("https://api.voyageai.com/v1/reranking")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&json!({
                        "query": query,
                        "documents": docs,
                        "model": self.model_id,
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Voyage AI", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let data = body.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
                    RuntimeError::ApiError("Invalid rerank response format".to_string())
                })?;

                let mut results = Vec::new();
                for item in data {
                    let index = item.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let score = item
                        .get("relevance_score")
                        .and_then(|s| s.as_f64())
                        .unwrap_or(0.0) as f32;
                    results.push(ScoredDoc {
                        index,
                        score,
                        text: None,
                    });
                }
                Ok(results)
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ModelRuntimeKey;
    use crate::provider::remote_common::RemoteProviderBase;
    use crate::traits::ModelProvider;
    use std::time::Duration;

    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn spec(alias: &str, task: ModelTask, model_id: &str) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: alias.to_string(),
            task,
            provider_id: "remote/voyageai".to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: crate::api::WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn breaker_reused_for_same_runtime_key() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("VOYAGE_API_KEY", "test-key") };

        let provider = RemoteVoyageAIProvider::new();
        let s1 = spec("embed/a", ModelTask::Embed, "voyage-3");
        let s2 = spec("embed/b", ModelTask::Embed, "voyage-3");

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("VOYAGE_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_isolated_by_task_and_model() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("VOYAGE_API_KEY", "test-key") };

        let provider = RemoteVoyageAIProvider::new();
        let embed = spec("embed/a", ModelTask::Embed, "voyage-3");
        let rerank = spec("rerank/a", ModelTask::Rerank, "rerank-2");

        let _ = provider.load(&embed).await.unwrap();
        let _ = provider.load(&rerank).await.unwrap();

        assert_eq!(provider.breaker_count(), 2);

        unsafe { std::env::remove_var("VOYAGE_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("VOYAGE_API_KEY", "test-key") };

        let provider = RemoteVoyageAIProvider::new();
        let stale = spec("embed/stale", ModelTask::Embed, "voyage-3");
        let fresh = spec("rerank/fresh", ModelTask::Rerank, "rerank-2");
        provider.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        provider.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(provider.breaker_count(), 2);

        provider.force_cleanup_now_for_test();
        let _ = provider.load(&fresh).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("VOYAGE_API_KEY") };
    }

    #[tokio::test]
    async fn generate_capability_mismatch() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("VOYAGE_API_KEY", "test-key") };

        let provider = RemoteVoyageAIProvider::new();
        let s = spec("gen/a", ModelTask::Generate, "voyage-3");
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not support task")
        );

        unsafe { std::env::remove_var("VOYAGE_API_KEY") };
    }
}
