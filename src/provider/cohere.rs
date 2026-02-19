use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{RemoteProviderBase, check_http_status, resolve_api_key};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth, RerankerModel, ScoredDoc, TokenUsage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

/// Remote provider that calls the [Cohere API](https://docs.cohere.com/reference/about)
/// for embedding, text generation (chat), and reranking.
///
/// Requires the `CO_API_KEY` environment variable (or a custom env var name
/// via the `api_key_env` option).
pub struct RemoteCohereProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteCohereProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteCohereProvider {
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
impl ModelProvider for RemoteCohereProvider {
    fn provider_id(&self) -> &'static str {
        "remote/cohere"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate, ModelTask::Rerank],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "CO_API_KEY")?;

        let input_type = spec
            .options
            .get("input_type")
            .and_then(|v| v.as_str())
            .unwrap_or("search_document")
            .to_string();

        match spec.task {
            ModelTask::Embed => {
                let model = CohereEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    api_key,
                    input_type,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = CohereGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let model = CohereRerankerModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn RerankerModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

struct CohereEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
    input_type: String,
}

#[async_trait]
impl EmbeddingModel for CohereEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let response = self
                    .client
                    .post("https://api.cohere.com/v2/embed")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&json!({
                        "texts": texts,
                        "model": self.model_id,
                        "input_type": self.input_type,
                        "embedding_types": ["float"]
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Cohere", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let float_embeddings = body
                    .get("embeddings")
                    .and_then(|e| e.get("float"))
                    .and_then(|f| f.as_array())
                    .ok_or_else(|| {
                        RuntimeError::ApiError(
                            "Invalid Cohere embedding response format".to_string(),
                        )
                    })?;

                let mut result = Vec::new();
                for embedding in float_embeddings {
                    if let Some(values) = embedding.as_array() {
                        let vec: Vec<f32> = values
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect();
                        result.push(vec);
                    }
                }
                Ok(result)
            })
            .await
    }

    fn dimensions(&self) -> u32 {
        match self.model_id.as_str() {
            "embed-english-light-v3.0" | "embed-multilingual-light-v3.0" => 384,
            _ => 1024,
        }
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

struct CohereGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl GeneratorModel for CohereGeneratorModel {
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let messages: Vec<serde_json::Value> = messages
            .iter()
            .enumerate()
            .map(|(i, content)| {
                let role = if i % 2 == 0 { "user" } else { "assistant" };
                json!({ "role": role, "content": content })
            })
            .collect();

        self.cb
            .call(move || async move {
                let mut body = json!({
                    "model": self.model_id,
                    "messages": messages,
                });

                if let Some(max_tokens) = options.max_tokens {
                    body["max_tokens"] = json!(max_tokens);
                }
                if let Some(temperature) = options.temperature {
                    body["temperature"] = json!(temperature);
                }
                if let Some(top_p) = options.top_p {
                    body["p"] = json!(top_p);
                }

                let response = self
                    .client
                    .post("https://api.cohere.com/v2/chat")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Cohere", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let text = body
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                let usage = body.get("usage").map(|u| {
                    let input = u
                        .get("tokens")
                        .and_then(|t| t.get("input_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let output = u
                        .get("tokens")
                        .and_then(|t| t.get("output_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    TokenUsage {
                        prompt_tokens: input as usize,
                        completion_tokens: output as usize,
                        total_tokens: (input + output) as usize,
                    }
                });

                Ok(GenerationResult { text, usage })
            })
            .await
    }
}

struct CohereRerankerModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl RerankerModel for CohereRerankerModel {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        let query = query.to_string();
        let docs: Vec<String> = docs.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let response = self
                    .client
                    .post("https://api.cohere.com/v2/rerank")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&json!({
                        "query": query,
                        "documents": docs,
                        "model": self.model_id,
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Cohere", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let results_json =
                    body.get("results")
                        .and_then(|r| r.as_array())
                        .ok_or_else(|| {
                            RuntimeError::ApiError("Invalid rerank response format".to_string())
                        })?;

                let mut results = Vec::new();
                for item in results_json {
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
            provider_id: "remote/cohere".to_string(),
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
        unsafe { std::env::set_var("CO_API_KEY", "test-key") };

        let provider = RemoteCohereProvider::new();
        let s1 = spec("embed/a", ModelTask::Embed, "embed-english-v3.0");
        let s2 = spec("embed/b", ModelTask::Embed, "embed-english-v3.0");

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("CO_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_isolated_by_task_and_model() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("CO_API_KEY", "test-key") };

        let provider = RemoteCohereProvider::new();
        let embed = spec("embed/a", ModelTask::Embed, "embed-english-v3.0");
        let gen_spec = spec("chat/a", ModelTask::Generate, "command-r-plus");
        let rerank = spec("rerank/a", ModelTask::Rerank, "rerank-english-v3.0");

        let _ = provider.load(&embed).await.unwrap();
        let _ = provider.load(&gen_spec).await.unwrap();
        let _ = provider.load(&rerank).await.unwrap();

        assert_eq!(provider.breaker_count(), 3);

        unsafe { std::env::remove_var("CO_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("CO_API_KEY", "test-key") };

        let provider = RemoteCohereProvider::new();
        let stale = spec("embed/stale", ModelTask::Embed, "embed-english-v3.0");
        let fresh = spec("chat/fresh", ModelTask::Generate, "command-r-plus");
        provider.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        provider.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(provider.breaker_count(), 2);

        provider.force_cleanup_now_for_test();
        let _ = provider.load(&fresh).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("CO_API_KEY") };
    }

    #[tokio::test]
    async fn supports_all_three_tasks() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("CO_API_KEY", "test-key") };

        let provider = RemoteCohereProvider::new();

        let embed = spec("embed/a", ModelTask::Embed, "embed-english-v3.0");
        assert!(provider.load(&embed).await.is_ok());

        let gen_spec = spec("gen/a", ModelTask::Generate, "command-r-plus");
        assert!(provider.load(&gen_spec).await.is_ok());

        let rerank = spec("rerank/a", ModelTask::Rerank, "rerank-english-v3.0");
        assert!(provider.load(&rerank).await.is_ok());

        unsafe { std::env::remove_var("CO_API_KEY") };
    }
}
