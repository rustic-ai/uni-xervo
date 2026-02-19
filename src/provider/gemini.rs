use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{
    RemoteProviderBase, build_google_generate_payload, check_http_status, resolve_api_key,
};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

/// Remote provider that calls the [Google Gemini API](https://ai.google.dev/api)
/// for embedding (`batchEmbedContents`) and text generation (`generateContent`).
///
/// Requires the `GEMINI_API_KEY` environment variable (or a custom env var name
/// via the `api_key_env` option).
pub struct RemoteGeminiProvider {
    base: RemoteProviderBase,
}

impl RemoteGeminiProvider {
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

impl Default for RemoteGeminiProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

#[async_trait]
impl ModelProvider for RemoteGeminiProvider {
    fn provider_id(&self) -> &'static str {
        "remote/gemini"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "GEMINI_API_KEY")?;

        match spec.task {
            ModelTask::Embed => {
                let model = GeminiEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = GeminiGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Gemini provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// Embedding model backed by the Gemini batch embedding API.
pub struct GeminiEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl EmbeddingModel for GeminiEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
                    self.model_id, self.api_key
                );

                let requests: Vec<_> = texts
                    .iter()
                    .map(|t| {
                        json!({
                            "model": format!("models/{}", self.model_id),
                            "content": { "parts": [{ "text": t }] }
                        })
                    })
                    .collect();

                let response = self
                    .client
                    .post(&url)
                    .json(&json!({ "requests": requests }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Gemini", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let embeddings_json = body
                    .get("embeddings")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        RuntimeError::ApiError("Invalid response format".to_string())
                    })?;

                let mut result = Vec::new();
                for item in embeddings_json {
                    let values = item
                        .get("values")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| {
                            RuntimeError::ApiError("Missing values in embedding".to_string())
                        })?;

                    let vec: Vec<f32> = values
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    result.push(vec);
                }
                Ok(result)
            })
            .await
    }

    fn dimensions(&self) -> u32 {
        // All current Gemini embedding models use 768 dimensions.
        768
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Text generation model backed by the Gemini `generateContent` API.
pub struct GeminiGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
}

#[async_trait]
impl GeneratorModel for GeminiGeneratorModel {
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let messages: Vec<String> = messages.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                    self.model_id, self.api_key
                );

                let payload = build_google_generate_payload(&messages, &options);

                let response = self
                    .client
                    .post(&url)
                    .json(&payload)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Gemini", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let candidates = body
                    .get("candidates")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| RuntimeError::ApiError("No candidates returned".to_string()))?;

                let first_candidate = candidates
                    .first()
                    .ok_or_else(|| RuntimeError::ApiError("Empty candidates".to_string()))?;

                let content_parts = first_candidate
                    .get("content")
                    .and_then(|c| c.get("parts"))
                    .and_then(|p| p.as_array())
                    .ok_or_else(|| RuntimeError::ApiError("Invalid content format".to_string()))?;

                let text = content_parts
                    .first()
                    .and_then(|p| p.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                Ok(GenerationResult {
                    text,
                    usage: None,
                })
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
            provider_id: "remote/gemini".to_string(),
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
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let s1 = spec("embed/a", ModelTask::Embed, "embedding-001");
        let s2 = spec("embed/b", ModelTask::Embed, "embedding-001");

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_isolated_by_task_and_model() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let embed = spec("embed/a", ModelTask::Embed, "embedding-001");
        let gen_spec = spec("chat/a", ModelTask::Generate, "gemini-pro");

        let _ = provider.load(&embed).await.unwrap();
        let _ = provider.load(&gen_spec).await.unwrap();

        assert_eq!(provider.breaker_count(), 2);

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let stale = spec("embed/stale", ModelTask::Embed, "embedding-001");
        let fresh = spec("embed/fresh", ModelTask::Embed, "embedding-002");
        provider.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        provider.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(provider.breaker_count(), 2);

        provider.force_cleanup_now_for_test();
        let _ = provider.load(&fresh).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }

    #[test]
    fn generation_payload_alternates_roles() {
        let messages = vec![
            "user question".to_string(),
            "assistant answer".to_string(),
            "user follow-up".to_string(),
        ];
        let payload = build_google_generate_payload(&messages, &GenerationOptions::default());
        let contents = payload["contents"].as_array().unwrap();

        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
    }

    #[test]
    fn generation_payload_includes_generation_options() {
        let messages = vec!["hello".to_string()];
        let payload = build_google_generate_payload(
            &messages,
            &GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(0.7),
                top_p: Some(0.9),
            },
        );

        assert_eq!(payload["generationConfig"]["maxOutputTokens"], 64);
        let temperature = payload["generationConfig"]["temperature"].as_f64().unwrap();
        let top_p = payload["generationConfig"]["topP"].as_f64().unwrap();
        assert!((temperature - 0.7).abs() < 1e-6);
        assert!((top_p - 0.9).abs() < 1e-6);
    }
}
