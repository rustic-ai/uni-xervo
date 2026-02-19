use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{
    RemoteProviderBase, build_google_generate_payload, check_http_status,
};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle,
    ModelProvider, ProviderCapabilities, ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

fn options_map<'a>(
    provider_id: &str,
    options: &'a serde_json::Value,
) -> Result<Option<&'a serde_json::Map<String, serde_json::Value>>> {
    match options {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(map) => Ok(Some(map)),
        _ => Err(RuntimeError::Config(format!(
            "Options for provider '{}' must be a JSON object or null",
            provider_id
        ))),
    }
}

fn option_string(
    provider_id: &str,
    map: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Result<Option<String>> {
    let Some(map) = map else {
        return Ok(None);
    };
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    let s = value.as_str().ok_or_else(|| {
        RuntimeError::Config(format!(
            "Option '{}' for provider '{}' must be a string",
            key, provider_id
        ))
    })?;
    Ok(Some(s.to_string()))
}

fn option_u32(
    provider_id: &str,
    map: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Result<Option<u32>> {
    let Some(map) = map else {
        return Ok(None);
    };
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    let n = value.as_u64().ok_or_else(|| {
        RuntimeError::Config(format!(
            "Option '{}' for provider '{}' must be a positive integer",
            key, provider_id
        ))
    })?;
    if n == 0 {
        return Err(RuntimeError::Config(format!(
            "Option '{}' for provider '{}' must be greater than 0",
            key, provider_id
        )));
    }
    let n_u32 = u32::try_from(n).map_err(|_| {
        RuntimeError::Config(format!(
            "Option '{}' for provider '{}' is out of range for u32",
            key, provider_id
        ))
    })?;
    Ok(Some(n_u32))
}

/// Resolved and validated Vertex AI configuration extracted from a
/// [`ModelAliasSpec`]'s options and environment variables.
#[derive(Clone)]
struct VertexAiResolvedOptions {
    token: String,
    project_id: String,
    location: String,
    publisher: String,
    embedding_dimensions: Option<u32>,
}

impl VertexAiResolvedOptions {
    fn from_spec(spec: &ModelAliasSpec) -> Result<Self> {
        let provider_id = "remote/vertexai";
        let map = options_map(provider_id, &spec.options)?;

        let token_env = option_string(provider_id, map, "api_token_env")?
            .unwrap_or_else(|| "VERTEX_AI_TOKEN".to_string());
        let token = std::env::var(&token_env)
            .map_err(|_| RuntimeError::Config(format!("{} env var not set", token_env)))?;

        let project_id = if let Some(project_id) = option_string(provider_id, map, "project_id")? {
            project_id
        } else {
            std::env::var("VERTEX_AI_PROJECT").map_err(|_| {
                RuntimeError::Config(
                    "project_id option not set and VERTEX_AI_PROJECT env var not set".to_string(),
                )
            })?
        };

        let location =
            option_string(provider_id, map, "location")?.unwrap_or_else(|| "us-central1".into());
        let publisher =
            option_string(provider_id, map, "publisher")?.unwrap_or_else(|| "google".into());
        let embedding_dimensions = option_u32(provider_id, map, "embedding_dimensions")?;

        Ok(Self {
            token,
            project_id,
            location,
            publisher,
            embedding_dimensions,
        })
    }
}

/// Remote provider that calls the [Google Vertex AI](https://cloud.google.com/vertex-ai/docs)
/// prediction and generation endpoints for embedding and text generation.
///
/// Requires the `VERTEX_AI_TOKEN` environment variable (or a custom env var
/// via `api_token_env`) and either the `project_id` option or the
/// `VERTEX_AI_PROJECT` env var.
pub struct RemoteVertexAIProvider {
    base: RemoteProviderBase,
}

impl RemoteVertexAIProvider {
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

impl Default for RemoteVertexAIProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

#[async_trait]
impl ModelProvider for RemoteVertexAIProvider {
    fn provider_id(&self) -> &'static str {
        "remote/vertexai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let resolved = VertexAiResolvedOptions::from_spec(spec)?;

        match spec.task {
            ModelTask::Embed => {
                let model = VertexAiEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    options: resolved.clone(),
                    dimensions: resolved.embedding_dimensions.unwrap_or(768),
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = VertexAiGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    options: resolved,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Vertex AI provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// Embedding model backed by the Vertex AI prediction API.
pub struct VertexAiEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    options: VertexAiResolvedOptions,
    dimensions: u32,
}

impl VertexAiEmbeddingModel {
    fn endpoint_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}/models/{}:predict",
            self.options.location,
            self.options.project_id,
            self.options.location,
            self.options.publisher,
            self.model_id
        )
    }
}

#[async_trait]
impl EmbeddingModel for VertexAiEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let instances: Vec<_> = texts.iter().map(|t| json!({ "content": t })).collect();
                let response = self
                    .client
                    .post(self.endpoint_url())
                    .header("Authorization", format!("Bearer {}", self.options.token))
                    .json(&json!({ "instances": instances }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Vertex AI", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let predictions = body
                    .get("predictions")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        RuntimeError::ApiError("Invalid response: missing predictions".to_string())
                    })?;

                let mut result = Vec::new();
                for item in predictions {
                    let values_opt = item
                        .get("embeddings")
                        .and_then(|e| e.get("values").and_then(|v| v.as_array()))
                        .or_else(|| {
                            item.get("embeddings")
                                .and_then(|e| e.as_array())
                                .or_else(|| item.get("values").and_then(|v| v.as_array()))
                        });

                    let values = values_opt.ok_or_else(|| {
                        RuntimeError::ApiError(
                            "Invalid embedding format in Vertex AI response".to_string(),
                        )
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
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Text generation model backed by the Vertex AI `generateContent` endpoint.
pub struct VertexAiGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    options: VertexAiResolvedOptions,
}

impl VertexAiGeneratorModel {
    fn endpoint_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}/models/{}:generateContent",
            self.options.location,
            self.options.project_id,
            self.options.location,
            self.options.publisher,
            self.model_id
        )
    }
}

#[async_trait]
impl GeneratorModel for VertexAiGeneratorModel {
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let messages: Vec<String> = messages.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let payload = build_google_generate_payload(&messages, &options);
                let response = self
                    .client
                    .post(self.endpoint_url())
                    .header("Authorization", format!("Bearer {}", self.options.token))
                    .json(&payload)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Vertex AI", response)?
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

                let usage = body.get("usageMetadata").map(|u| TokenUsage {
                    prompt_tokens: u["promptTokenCount"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0) as usize,
                    total_tokens: u["totalTokenCount"].as_u64().unwrap_or(0) as usize,
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

    fn spec(
        alias: &str,
        task: ModelTask,
        model_id: &str,
        options: serde_json::Value,
    ) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: alias.to_string(),
            task,
            provider_id: "remote/vertexai".to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: crate::api::WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options,
        }
    }

    #[tokio::test]
    async fn breaker_reused_for_same_runtime_key() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe {
            std::env::set_var("VERTEX_AI_TOKEN", "test-token");
            std::env::set_var("VERTEX_AI_PROJECT", "test-project");
        }

        let provider = RemoteVertexAIProvider::new();
        let s1 = spec(
            "embed/a",
            ModelTask::Embed,
            "text-embedding-005",
            serde_json::Value::Null,
        );
        let s2 = spec(
            "embed/b",
            ModelTask::Embed,
            "text-embedding-005",
            serde_json::Value::Null,
        );

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        // SAFETY: protected by ENV_LOCK
        unsafe {
            std::env::remove_var("VERTEX_AI_TOKEN");
            std::env::remove_var("VERTEX_AI_PROJECT");
        }
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe {
            std::env::set_var("VERTEX_AI_TOKEN", "test-token");
            std::env::set_var("VERTEX_AI_PROJECT", "test-project");
        }

        let provider = RemoteVertexAIProvider::new();
        let stale = spec(
            "embed/stale",
            ModelTask::Embed,
            "text-embedding-005",
            serde_json::Value::Null,
        );
        let fresh = spec(
            "embed/fresh",
            ModelTask::Embed,
            "text-embedding-004",
            serde_json::Value::Null,
        );
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
        unsafe {
            std::env::remove_var("VERTEX_AI_TOKEN");
            std::env::remove_var("VERTEX_AI_PROJECT");
        }
    }

    #[tokio::test]
    async fn load_fails_when_project_is_missing() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe {
            std::env::set_var("VERTEX_AI_TOKEN", "test-token");
            std::env::remove_var("VERTEX_AI_PROJECT");
        }

        let provider = RemoteVertexAIProvider::new();
        let s = spec(
            "embed/a",
            ModelTask::Embed,
            "text-embedding-005",
            serde_json::Value::Null,
        );
        let err = provider.load(&s).await.unwrap_err();
        assert!(err.to_string().contains("VERTEX_AI_PROJECT"));

        // SAFETY: protected by ENV_LOCK
        unsafe {
            std::env::remove_var("VERTEX_AI_TOKEN");
        }
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
