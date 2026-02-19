use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{RemoteProviderBase, check_http_status, resolve_api_key};
use crate::traits::{
    GenerationOptions, GenerationResult, GeneratorModel, LoadedModelHandle, ModelProvider,
    ProviderCapabilities, ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;

/// Remote provider that calls the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
/// for text generation. Does not support embedding or reranking.
///
/// Requires the `ANTHROPIC_API_KEY` environment variable (or a custom env var
/// name via the `api_key_env` option).
pub struct RemoteAnthropicProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteAnthropicProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteAnthropicProvider {
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
impl ModelProvider for RemoteAnthropicProvider {
    fn provider_id(&self) -> &'static str {
        "remote/anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Generate],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "ANTHROPIC_API_KEY")?;

        let anthropic_version = spec
            .options
            .get("anthropic_version")
            .and_then(|v| v.as_str())
            .unwrap_or("2023-06-01")
            .to_string();

        match spec.task {
            ModelTask::Generate => {
                let model = AnthropicGeneratorModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                    anthropic_version,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Anthropic provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

struct AnthropicGeneratorModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
    anthropic_version: String,
}

fn build_anthropic_payload(
    model_id: &str,
    messages: &[serde_json::Value],
    options: &GenerationOptions,
) -> serde_json::Value {
    let max_tokens = options.max_tokens.unwrap_or(1024);

    let mut body = json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": messages,
    });

    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = options.top_p {
        body["top_p"] = json!(top_p);
    }

    body
}

#[async_trait]
impl GeneratorModel for AnthropicGeneratorModel {
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
                let body = build_anthropic_payload(&self.model_id, &messages, &options);

                let response = self
                    .client
                    .post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", &self.anthropic_version)
                    .header("content-type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let body: serde_json::Value = check_http_status("Anthropic", response)?
                    .json()
                    .await
                    .map_err(|e| RuntimeError::ApiError(e.to_string()))?;

                let text = body
                    .get("content")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                let usage = body.get("usage").map(|u| TokenUsage {
                    prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as usize,
                    total_tokens: (u["input_tokens"].as_u64().unwrap_or(0)
                        + u["output_tokens"].as_u64().unwrap_or(0))
                        as usize,
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
            provider_id: "remote/anthropic".to_string(),
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
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };

        let provider = RemoteAnthropicProvider::new();
        let s1 = spec("gen/a", ModelTask::Generate, "claude-sonnet-4-5-20250929");
        let s2 = spec("gen/b", ModelTask::Generate, "claude-sonnet-4-5-20250929");

        let _ = provider.load(&s1).await.unwrap();
        let _ = provider.load(&s2).await.unwrap();

        assert_eq!(provider.breaker_count(), 1);

        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };

        let provider = RemoteAnthropicProvider::new();
        let stale = spec(
            "gen/stale",
            ModelTask::Generate,
            "claude-sonnet-4-5-20250929",
        );
        let fresh = spec(
            "gen/fresh",
            ModelTask::Generate,
            "claude-haiku-3-5-20241022",
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

        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[tokio::test]
    async fn embed_capability_mismatch() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };

        let provider = RemoteAnthropicProvider::new();
        let s = spec("embed/a", ModelTask::Embed, "claude-sonnet-4-5-20250929");
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not support task")
        );

        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[tokio::test]
    async fn rerank_capability_mismatch() {
        let _lock = ENV_LOCK.lock().await;
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };

        let provider = RemoteAnthropicProvider::new();
        let s = spec("rerank/a", ModelTask::Rerank, "claude-sonnet-4-5-20250929");
        let result = provider.load(&s).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not support task")
        );

        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[test]
    fn payload_defaults_max_tokens_to_1024() {
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let payload = build_anthropic_payload(
            "claude-sonnet-4-5-20250929",
            &messages,
            &GenerationOptions::default(),
        );
        assert_eq!(payload["max_tokens"], 1024);
    }

    #[test]
    fn payload_uses_explicit_max_tokens() {
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let payload = build_anthropic_payload(
            "claude-sonnet-4-5-20250929",
            &messages,
            &GenerationOptions {
                max_tokens: Some(512),
                temperature: None,
                top_p: None,
            },
        );
        assert_eq!(payload["max_tokens"], 512);
    }
}
