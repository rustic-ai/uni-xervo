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

/// Remote provider that calls the [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
/// for embedding and text generation.
///
/// Requires the `AZURE_OPENAI_API_KEY` environment variable (or a custom env
/// var name via the `api_key_env` option) and the `resource_name` option.
pub struct RemoteAzureOpenAIProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteAzureOpenAIProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteAzureOpenAIProvider {
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

/// Resolved Azure OpenAI configuration extracted from a [`ModelAliasSpec`]'s
/// options and environment variables.
#[derive(Clone)]
struct AzureResolvedOptions {
    api_key: String,
    resource_name: String,
    api_version: String,
}

impl AzureResolvedOptions {
    fn from_spec(spec: &ModelAliasSpec) -> Result<Self> {
        let api_key = resolve_api_key(&spec.options, "api_key_env", "AZURE_OPENAI_API_KEY")?;

        let resource_name = spec
            .options
            .get("resource_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                RuntimeError::Config(
                    "Option 'resource_name' is required for Azure OpenAI provider".to_string(),
                )
            })?
            .to_string();

        let api_version = spec
            .options
            .get("api_version")
            .and_then(|v| v.as_str())
            .unwrap_or("2024-10-21")
            .to_string();

        Ok(Self {
            api_key,
            resource_name,
            api_version,
        })
    }

    fn embed_url(&self, deployment: &str) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/embeddings?api-version={}",
            self.resource_name, deployment, self.api_version
        )
    }

    fn chat_url(&self, deployment: &str) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource_name, deployment, self.api_version
        )
    }
}

#[async_trait]
impl ModelProvider for RemoteAzureOpenAIProvider {
    fn provider_id(&self) -> &'static str {
        "remote/azure-openai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let resolved = AzureResolvedOptions::from_spec(spec)?;

        match spec.task {
            ModelTask::Embed => {
                let model = AzureOpenAIEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    deployment: spec.model_id.clone(),
                    options: resolved,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = AzureOpenAIGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    deployment: spec.model_id.clone(),
                    options: resolved,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Azure OpenAI provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

struct AzureOpenAIEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    deployment: String,
    options: AzureResolvedOptions,
}

#[async_trait]
impl EmbeddingModel for AzureOpenAIEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let url = self.options.embed_url(&self.deployment);

                let response = self
                    .client
                    .post(&url)
                    .header("api-key", &self.options.api_key)
                    .json(&json!({
                        "input": texts
                    }))
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Azure OpenAI", response)?
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
        // Azure deployments may use various embedding models;
        // default to 1536 (text-embedding-ada-002 / text-embedding-3-small).
        1536
    }

    fn model_id(&self) -> &str {
        &self.deployment
    }
}

struct AzureOpenAIGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    deployment: String,
    options: AzureResolvedOptions,
}

#[async_trait]
impl GeneratorModel for AzureOpenAIGeneratorModel {
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
                let url = self.options.chat_url(&self.deployment);

                let mut body = json!({
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
                    .post(&url)
                    .header("api-key", &self.options.api_key)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Azure OpenAI", response)?
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

    fn spec_with_opts(
        alias: &str,
        task: ModelTask,
        model_id: &str,
        options: serde_json::Value,
    ) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: alias.to_string(),
            task,
            provider_id: "remote/azure-openai".to_string(),
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

    fn default_opts() -> serde_json::Value {
        json!({ "resource_name": "my-resource" })
    }

    #[tokio::test]
    async fn breaker_reused_for_same_runtime_key() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("AZURE_OPENAI_API_KEY", "test-key") };

        let provider = RemoteAzureOpenAIProvider::new();
        let s1 = spec_with_opts(
            "embed/a",
            ModelTask::Embed,
            "text-embedding-ada-002",
            default_opts(),
        );
        let s2 = spec_with_opts(
            "embed/b",
            ModelTask::Embed,
            "text-embedding-ada-002",
            default_opts(),
        );

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("AZURE_OPENAI_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("AZURE_OPENAI_API_KEY", "test-key") };

        let provider = RemoteAzureOpenAIProvider::new();
        let stale = spec_with_opts(
            "embed/stale",
            ModelTask::Embed,
            "text-embedding-ada-002",
            default_opts(),
        );
        let fresh = spec_with_opts("chat/fresh", ModelTask::Generate, "gpt-4o", default_opts());
        provider.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        provider.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(provider.breaker_count(), 2);

        provider.force_cleanup_now_for_test();
        let _ = provider.load(&fresh).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("AZURE_OPENAI_API_KEY") };
    }

    #[tokio::test]
    async fn load_fails_without_resource_name() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("AZURE_OPENAI_API_KEY", "test-key") };

        let provider = RemoteAzureOpenAIProvider::new();
        let s = spec_with_opts(
            "embed/a",
            ModelTask::Embed,
            "text-embedding-ada-002",
            serde_json::Value::Null,
        );
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("resource_name"));

        unsafe { std::env::remove_var("AZURE_OPENAI_API_KEY") };
    }

    #[tokio::test]
    async fn rerank_capability_mismatch() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("AZURE_OPENAI_API_KEY", "test-key") };

        let provider = RemoteAzureOpenAIProvider::new();
        let s = spec_with_opts(
            "rerank/a",
            ModelTask::Rerank,
            "text-embedding-ada-002",
            default_opts(),
        );
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not support task")
        );

        unsafe { std::env::remove_var("AZURE_OPENAI_API_KEY") };
    }

    #[test]
    fn azure_url_construction() {
        let opts = AzureResolvedOptions {
            api_key: "key".to_string(),
            resource_name: "my-resource".to_string(),
            api_version: "2024-10-21".to_string(),
        };

        assert_eq!(
            opts.embed_url("text-embedding-ada-002"),
            "https://my-resource.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21"
        );

        assert_eq!(
            opts.chat_url("gpt-4o"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }
}
