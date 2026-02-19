use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{RemoteProviderBase, check_http_status, resolve_api_key};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

/// Remote provider that calls the [Mistral AI API](https://docs.mistral.ai/api/)
/// for embedding and text generation (chat completions).
///
/// Requires the `MISTRAL_API_KEY` environment variable (or a custom env var
/// name via the `api_key_env` option).
pub struct RemoteMistralProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteMistralProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteMistralProvider {
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
impl ModelProvider for RemoteMistralProvider {
    fn provider_id(&self) -> &'static str {
        "remote/mistral"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "MISTRAL_API_KEY")?;

        match spec.task {
            ModelTask::Embed => {
                let model = MistralEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = MistralGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Mistral provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

struct MistralEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl EmbeddingModel for MistralEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let response = self
                    .client
                    .post("https://api.mistral.ai/v1/embeddings")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&json!({
                        "model": self.model_id,
                        "input": texts
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Mistral", response)?
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
        // All current Mistral embedding models use 1024 dimensions.
        1024
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

struct MistralGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl GeneratorModel for MistralGeneratorModel {
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
                    body["top_p"] = json!(top_p);
                }

                let response = self
                    .client
                    .post("https://api.mistral.ai/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Mistral", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let text = body["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();

                let usage = body.get("usage").map(|u| TokenUsage {
                    prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as usize,
                    total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as usize,
                });

                Ok(GenerationResult { text, usage })
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
            provider_id: "remote/mistral".to_string(),
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
        unsafe { std::env::set_var("MISTRAL_API_KEY", "test-key") };

        let provider = RemoteMistralProvider::new();
        let s1 = spec("embed/a", ModelTask::Embed, "mistral-embed");
        let s2 = spec("embed/b", ModelTask::Embed, "mistral-embed");

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("MISTRAL_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_isolated_by_task_and_model() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MISTRAL_API_KEY", "test-key") };

        let provider = RemoteMistralProvider::new();
        let embed = spec("embed/a", ModelTask::Embed, "mistral-embed");
        let gen_spec = spec("chat/a", ModelTask::Generate, "mistral-small-latest");

        let _ = provider.load(&embed).await.unwrap();
        let _ = provider.load(&gen_spec).await.unwrap();

        assert_eq!(provider.breaker_count(), 2);

        unsafe { std::env::remove_var("MISTRAL_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MISTRAL_API_KEY", "test-key") };

        let provider = RemoteMistralProvider::new();
        let stale = spec("embed/stale", ModelTask::Embed, "mistral-embed");
        let fresh = spec("chat/fresh", ModelTask::Generate, "mistral-small-latest");
        provider.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        provider.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(provider.breaker_count(), 2);

        provider.force_cleanup_now_for_test();
        let _ = provider.load(&fresh).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("MISTRAL_API_KEY") };
    }

    #[tokio::test]
    async fn rerank_capability_mismatch() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MISTRAL_API_KEY", "test-key") };

        let provider = RemoteMistralProvider::new();
        let s = spec("rerank/a", ModelTask::Rerank, "mistral-embed");
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not support task")
        );

        unsafe { std::env::remove_var("MISTRAL_API_KEY") };
    }
}
