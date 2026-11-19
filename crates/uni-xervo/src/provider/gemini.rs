use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{
    RemoteProviderBase, build_google_generate_payload, check_http_status, resolve_api_key,
};
use crate::traits::{
    AudioInput, EmbedResult, EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel,
    ImageInput, LoadedModelHandle, Message, Modality, ModelProvider, MultimodalBlock,
    MultimodalEmbeddingModel, MultimodalInput, ProviderCapabilities, ProviderHealth, TokenUsage,
};
use async_trait::async_trait;
use base64::Engine;
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
            supported_tasks: vec![
                ModelTask::Embed,
                ModelTask::Generate,
                ModelTask::EmbedMultimodal,
            ],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let cb = self.base.circuit_breaker_for(spec);
        let api_key = resolve_api_key(&spec.options, "api_key_env", "GEMINI_API_KEY")?;
        let api_version = spec
            .options
            .get("api_version")
            .and_then(|v| v.as_str())
            .unwrap_or("v1beta")
            .to_string();
        let embedding_dimensions = spec
            .options
            .get("embedding_dimensions")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        match spec.task {
            ModelTask::Embed => {
                let default_dims = match spec.model_id.as_str() {
                    "gemini-embedding-001" | "gemini-embedding-exp-03-07" => 3072,
                    _ => 768,
                };
                let model = GeminiEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: cb.clone(),
                    model_id: spec.model_id.clone(),
                    api_key,
                    api_version: api_version.clone(),
                    dimensions: embedding_dimensions.unwrap_or(default_dims),
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
                    api_version,
                };
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::EmbedMultimodal => {
                // Gemini Embedding 2 returns 3072-dim by default.
                let default_dims = 3072;
                let model = GeminiMultimodalEmbeddingModel {
                    client: self.base.client.clone(),
                    cb,
                    model_id: spec.model_id.clone(),
                    api_key,
                    api_version,
                    dimensions: embedding_dimensions.unwrap_or(default_dims),
                };
                let handle: Arc<dyn MultimodalEmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Raw => Err(RuntimeError::CapabilityMismatch(
                "Gemini provider does not support task Raw".to_string(),
            )),
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "Gemini provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }

    fn describe(&self, spec: &ModelAliasSpec) -> crate::inspect::ProviderDescription {
        let version =
            crate::provider::remote_common::option_or(&spec.options, "api_version", "v1beta");
        crate::inspect::ProviderDescription::new()
            .with_endpoint(format!(
                "https://generativelanguage.googleapis.com/{version}/models/{}",
                spec.model_id
            ))
            .with_credential_env(crate::provider::remote_common::credential_env_name(
                &spec.options,
                "api_key_env",
                "GEMINI_API_KEY",
            ))
    }
}

/// Embedding model backed by the Gemini batch embedding API.
pub struct GeminiEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
    api_version: String,
    dimensions: u32,
}

#[async_trait]
impl EmbeddingModel for GeminiEmbeddingModel {
    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult> {
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.cb
            .call(move || async move {
                let url = format!(
                    "https://generativelanguage.googleapis.com/{}/models/{}:batchEmbedContents?key={}",
                    self.api_version, self.model_id, self.api_key
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

                let mut vectors = Vec::new();
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
                    vectors.push(vec);
                }

                // Gemini does not report usage on the batchEmbedContents endpoint.
                Ok(EmbedResult {
                    vectors,
                    usage: None,
                })
            })
            .await
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }
}

impl crate::traits::ModelInfo for GeminiEmbeddingModel {
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
    api_version: String,
}

impl crate::traits::ModelInfo for GeminiGeneratorModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl GeneratorModel for GeminiGeneratorModel {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let messages: Vec<Message> = messages.to_vec();

        self.cb
            .call(move || async move {
                let url = format!(
                    "https://generativelanguage.googleapis.com/{}/models/{}:generateContent?key={}",
                    self.api_version, self.model_id, self.api_key
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

                let usage = body.get("usageMetadata").map(|u| TokenUsage {
                    prompt_tokens: u["promptTokenCount"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0) as usize,
                    total_tokens: u["totalTokenCount"].as_u64().unwrap_or(0) as usize,
                });

                Ok(GenerationResult {
                    text,
                    usage,
                    images: vec![],
                    audio: None,
                })
            })
            .await
    }
}

/// Multimodal embedding model backed by Gemini Embedding 2's
/// `batchEmbedContents` endpoint with multimodal `parts`.
pub struct GeminiMultimodalEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    api_key: String,
    api_version: String,
    dimensions: u32,
}

impl GeminiMultimodalEmbeddingModel {
    /// Convert one [`MultimodalInput`] into a Gemini `content.parts` array.
    fn input_to_parts(input: &MultimodalInput) -> Vec<serde_json::Value> {
        input
            .blocks
            .iter()
            .map(|block| match block {
                MultimodalBlock::Text(text) => json!({ "text": text }),
                MultimodalBlock::Image(ImageInput::Url(url)) => json!({
                    "file_data": { "file_uri": url }
                }),
                MultimodalBlock::Image(ImageInput::Bytes { data, media_type }) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                    json!({
                        "inline_data": { "mime_type": media_type, "data": b64 }
                    })
                }
                MultimodalBlock::Audio(AudioInput::Bytes { data, media_type }) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                    json!({
                        "inline_data": { "mime_type": media_type, "data": b64 }
                    })
                }
                MultimodalBlock::Audio(AudioInput::Pcm {
                    sample_rate,
                    samples,
                    ..
                }) => {
                    // Build a minimal 16-bit PCM WAV in memory so the API
                    // receives a self-describing container. Mono assumption
                    // matches the typical embedding pipeline; multi-channel
                    // callers should encode upstream and use Bytes.
                    let wav_bytes = encode_pcm_to_wav(*sample_rate, samples);
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
                    json!({
                        "inline_data": { "mime_type": "audio/wav", "data": b64 }
                    })
                }
            })
            .collect()
    }
}

/// Encode a slice of f32 PCM samples into a 16-bit mono WAV byte buffer.
///
/// Kept inline because Gemini is currently the only consumer; a shared
/// audio-encode helper can land alongside PR-4 (whisper-cpp) which needs
/// the inverse operation.
fn encode_pcm_to_wav(sample_rate: u32, samples: &[f32]) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let channels: u16 = 1;
    let byte_rate = sample_rate * (bits_per_sample as u32) * (channels as u32) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = (samples.len() as u32) * (bits_per_sample as u32) * (channels as u32) / 8;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let s = (clamped * i16::MAX as f32) as i16;
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

#[async_trait]
impl MultimodalEmbeddingModel for GeminiMultimodalEmbeddingModel {
    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<EmbedResult> {
        let requests: Vec<serde_json::Value> = inputs
            .iter()
            .map(|input| {
                json!({
                    "model": format!("models/{}", self.model_id),
                    "content": { "parts": Self::input_to_parts(input) }
                })
            })
            .collect();

        self.cb
            .call(move || async move {
                let url = format!(
                    "https://generativelanguage.googleapis.com/{}/models/{}:batchEmbedContents?key={}",
                    self.api_version, self.model_id, self.api_key
                );
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
                        RuntimeError::ApiError(
                            "Invalid Gemini multimodal embedding response format".to_string(),
                        )
                    })?;

                let mut vectors = Vec::with_capacity(embeddings_json.len());
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
                    vectors.push(vec);
                }

                // Gemini does not report usage on the embed endpoints.
                Ok(EmbedResult {
                    vectors,
                    usage: None,
                })
            })
            .await
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    fn supported_modalities(&self) -> &[Modality] {
        const GEMINI_2_MODALITIES: &[Modality] = &[
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
        ];
        GEMINI_2_MODALITIES
    }
}

impl crate::traits::ModelInfo for GeminiMultimodalEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model_id
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
        use crate::traits::Message;
        let messages = vec![
            Message::user("user question"),
            Message::assistant("assistant answer"),
            Message::user("user follow-up"),
        ];
        let payload = build_google_generate_payload(&messages, &GenerationOptions::default());
        let contents = payload["contents"].as_array().unwrap();

        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
    }

    #[test]
    fn generation_payload_includes_generation_options() {
        use crate::traits::Message;
        let messages = vec![Message::user("hello")];
        let payload = build_google_generate_payload(
            &messages,
            &GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(0.7),
                top_p: Some(0.9),
                ..Default::default()
            },
        );

        assert_eq!(payload["generationConfig"]["maxOutputTokens"], 64);
        let temperature = payload["generationConfig"]["temperature"].as_f64().unwrap();
        let top_p = payload["generationConfig"]["topP"].as_f64().unwrap();
        assert!((temperature - 0.7).abs() < 1e-6);
        assert!((top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn generation_payload_extracts_system_instruction() {
        use crate::traits::Message;
        let messages = vec![Message::system("you are helpful"), Message::user("hello")];
        let payload = build_google_generate_payload(&messages, &GenerationOptions::default());

        // System message should be extracted into system_instruction
        let si = &payload["system_instruction"];
        assert_eq!(si["parts"][0]["text"], "you are helpful");

        // Contents should only have the user message
        let contents = payload["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn generation_payload_no_system_instruction_without_system_messages() {
        use crate::traits::Message;
        let messages = vec![Message::user("hello"), Message::assistant("hi")];
        let payload = build_google_generate_payload(&messages, &GenerationOptions::default());

        // No system_instruction field should be present
        assert!(payload.get("system_instruction").is_none());

        let contents = payload["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
    }

    fn spec_with_opts(
        alias: &str,
        task: ModelTask,
        model_id: &str,
        options: serde_json::Value,
    ) -> ModelAliasSpec {
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
            options,
        }
    }

    #[tokio::test]
    async fn default_embedding_dimensions() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let s = spec("embed/dim", ModelTask::Embed, "embedding-001");
        let handle = provider.load(&s).await.unwrap();
        let model = handle.downcast_ref::<Arc<dyn EmbeddingModel>>().unwrap();
        assert_eq!(model.dimensions(), 768);

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }

    #[tokio::test]
    async fn custom_embedding_dimensions() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let s = spec_with_opts(
            "embed/dim-custom",
            ModelTask::Embed,
            "embedding-001",
            json!({ "embedding_dimensions": 256 }),
        );
        let handle = provider.load(&s).await.unwrap();
        let model = handle.downcast_ref::<Arc<dyn EmbeddingModel>>().unwrap();
        assert_eq!(model.dimensions(), 256);

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }

    #[tokio::test]
    async fn api_version_option_accepted() {
        let _lock = ENV_LOCK.lock().await;
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("GEMINI_API_KEY", "test-key") };

        let provider = RemoteGeminiProvider::new();
        let s = spec_with_opts(
            "embed/v1",
            ModelTask::Embed,
            "embedding-001",
            json!({ "api_version": "v1" }),
        );
        let handle = provider.load(&s).await;
        assert!(handle.is_ok());

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("GEMINI_API_KEY") };
    }
}
