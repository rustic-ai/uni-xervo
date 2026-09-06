//! Remote embedding provider for a llama.cpp [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
//!
//! `llama-server` exposes an OpenAI-compatible `POST /v1/embeddings` endpoint,
//! but it differs from hosted embedding APIs in one way that matters for
//! production ingestion: **it never truncates embedding input**. Encoder
//! models such as BGE (BERT) are non-causal, so the whole prompt has to fit in
//! a single physical batch (`--ubatch-size`) and inside the context window
//! (`--ctx-size`); anything longer is rejected with an HTTP error instead of
//! being cut to size.
//!
//! This provider therefore runs a deterministic, tokenizer-aware pre-pass
//! before every embedding call:
//!
//! 1. Texts that provably fit are sent as plain strings. Every WordPiece token
//!    consumes at least one character, so a text with at most
//!    `max_input_tokens - 2` characters cannot exceed the budget once the two
//!    special tokens (`[CLS]`/`[SEP]`) are added.
//! 2. Longer texts are sent to the server's native `POST /tokenize` endpoint
//!    with `add_special: true`, which returns the exact id sequence the server
//!    would embed, special tokens included.
//! 3. If that sequence exceeds `max_input_tokens`, the provider keeps the
//!    first `max_input_tokens - 1` ids and re-appends the trailing special
//!    token, then sends the **token array** (not text) to `/v1/embeddings`.
//!    llama.cpp passes integer arrays through verbatim, so the server embeds
//!    exactly the bounded sequence — no lossy detokenize round trip.
//!
//! Batches preserve input order and produce exactly one vector per input.
//!
//! # Options
//!
//! | key | type | required | meaning |
//! |-----|------|----------|---------|
//! | `base_url` | string | yes | OpenAI-compatible root including `/v1`, e.g. `http://127.0.0.1:8080/v1` |
//! | `tokenizer_base_url` | string | no | server root for `/tokenize`; defaults to `base_url` with a trailing `/v1` removed |
//! | `max_input_tokens` | integer ≥ 3 | yes | total token budget **including** special tokens (512 for BGE) |
//! | `embedding_dimensions` | integer > 0 | yes | expected vector width (384 for BGE Small) |
//! | `api_key_env` | string | no | env var holding a bearer token; omit when the server runs without `--api-key` |
//! | `request_timeout_secs` | integer > 0 | no | per-HTTP-request timeout, default 60 |
//!
//! `max_input_tokens` must not exceed the server's `--ubatch-size` or
//! `--ctx-size`; if it does, the server's rejection surfaces as
//! [`RuntimeError::InferenceError`] naming the option.

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::{RemoteProviderBase, parse_openai_embeddings_response};
use crate::traits::{
    EmbedResult, EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities,
    ProviderHealth,
};
use async_trait::async_trait;
use futures::future::try_join_all;
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

/// Provider id used in [`ModelAliasSpec::provider_id`].
pub const PROVIDER_ID: &str = "remote/llamacpp";

/// Default per-request HTTP timeout when `request_timeout_secs` is unset.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;

/// Smallest meaningful `max_input_tokens`: two special tokens plus one
/// content token.
pub const MIN_MAX_INPUT_TOKENS: usize = 3;

/// Remote provider for a llama.cpp `llama-server`. Supports
/// [`ModelTask::Embed`] only.
pub struct RemoteLlamaCppProvider {
    base: RemoteProviderBase,
}

impl Default for RemoteLlamaCppProvider {
    fn default() -> Self {
        Self {
            base: RemoteProviderBase::new(),
        }
    }
}

impl RemoteLlamaCppProvider {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn insert_test_breaker(&self, key: crate::api::ModelRuntimeKey, age: Duration) {
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
impl ModelProvider for RemoteLlamaCppProvider {
    fn provider_id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Embed => {
                let cfg = LlamaCppConfig::from_options(&spec.options)?;
                let model = LlamaCppEmbeddingModel {
                    client: self.base.client.clone(),
                    cb: self.base.circuit_breaker_for(spec),
                    model_id: spec.model_id.clone(),
                    cfg,
                };
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            other => Err(RuntimeError::CapabilityMismatch(format!(
                "llama.cpp provider does not support task {:?}",
                other
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved, validated provider configuration for one alias.
#[derive(Debug, Clone)]
pub(crate) struct LlamaCppConfig {
    /// OpenAI-compatible root including `/v1`, no trailing slash.
    pub base_url: String,
    /// Server root for `/tokenize`, no trailing slash.
    pub tokenizer_base_url: String,
    /// Total token budget including special tokens.
    pub max_input_tokens: usize,
    /// Expected embedding width.
    pub dimensions: u32,
    /// Bearer token, if `api_key_env` was configured.
    pub api_key: Option<String>,
    /// Per-HTTP-request timeout.
    pub request_timeout: Duration,
}

impl LlamaCppConfig {
    /// Parse the alias `options`. The catalog validator already enforces the
    /// same rules, but `ModelProvider::load` is public, so this is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        let cfg_err = |msg: String| RuntimeError::Config(msg);
        let required = |key: &str| {
            cfg_err(format!(
                "Option '{}' for provider '{}' is required",
                key, PROVIDER_ID
            ))
        };

        let base_url = options
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| required("base_url"))?;

        let tokenizer_base_url = match options.get("tokenizer_base_url").and_then(|v| v.as_str()) {
            Some(raw) if !raw.trim().is_empty() => raw.trim().trim_end_matches('/').to_string(),
            _ => resolve_tokenizer_base_url(&base_url),
        };

        let max_input_tokens = options
            .get("max_input_tokens")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| required("max_input_tokens"))? as usize;
        if max_input_tokens < MIN_MAX_INPUT_TOKENS {
            return Err(cfg_err(format!(
                "Option 'max_input_tokens' for provider '{}' must be at least {} \
                 (two special tokens plus one content token)",
                PROVIDER_ID, MIN_MAX_INPUT_TOKENS
            )));
        }

        let dimensions = options
            .get("embedding_dimensions")
            .and_then(|v| v.as_u64())
            .filter(|d| *d > 0 && *d <= u32::MAX as u64)
            .ok_or_else(|| required("embedding_dimensions"))? as u32;

        let api_key = match options.get("api_key_env").and_then(|v| v.as_str()) {
            None => None,
            Some(env_name) => Some(
                std::env::var(env_name)
                    .map_err(|_| cfg_err(format!("{} env var not set", env_name)))?,
            ),
        };

        let request_timeout = Duration::from_secs(
            options
                .get("request_timeout_secs")
                .and_then(|v| v.as_u64())
                .filter(|s| *s > 0)
                .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS),
        );

        Ok(Self {
            base_url,
            tokenizer_base_url,
            max_input_tokens,
            dimensions,
            api_key,
            request_timeout,
        })
    }
}

/// Derive the server root for `/tokenize` from an OpenAI-style `base_url`:
/// strip trailing slashes, then exactly one trailing `/v1` path segment.
///
/// ```text
/// http://h:8080/v1      -> http://h:8080
/// http://h:8080/v1/     -> http://h:8080
/// http://h:8080/api/v1  -> http://h:8080/api
/// http://h:8080         -> http://h:8080   (unchanged)
/// http://h:8080/v10     -> http://h:8080/v10 (unchanged)
/// ```
pub(crate) fn resolve_tokenizer_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    match trimmed.strip_suffix("/v1") {
        Some(root) if !root.is_empty() => root.to_string(),
        _ => trimmed.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Input planning (pure)
// ---------------------------------------------------------------------------

/// Wire form of one element of the `/v1/embeddings` `input` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbedInput {
    /// Send the original text; the server adds special tokens itself.
    Text(String),
    /// Send a pre-bounded token id sequence; the server embeds it verbatim.
    Tokens(Vec<u32>),
}

impl EmbedInput {
    pub(crate) fn to_json(&self) -> Value {
        match self {
            EmbedInput::Text(t) => Value::String(t.clone()),
            EmbedInput::Tokens(ids) => json!(ids),
        }
    }
}

/// `true` when `text` might exceed the budget and must be tokenized to know.
///
/// Every WordPiece token consumes at least one character, so a text with at
/// most `max_input_tokens - 2` characters (two slots reserved for the special
/// tokens) provably fits without a tokenizer round trip.
pub(crate) fn needs_tokenize(text: &str, max_input_tokens: usize) -> bool {
    text.chars().count() > max_input_tokens.saturating_sub(2)
}

/// Decide the wire form for `text` given its full `add_special: true` token
/// sequence. Fits → original text. Too long → the first
/// `max_input_tokens - 1` ids plus the original trailing special token.
pub(crate) fn plan_input(text: &str, tokens: &[u32], max_input_tokens: usize) -> EmbedInput {
    if tokens.is_empty() || tokens.len() <= max_input_tokens {
        return EmbedInput::Text(text.to_string());
    }
    let keep = max_input_tokens.saturating_sub(1).max(1);
    let mut bounded: Vec<u32> = tokens[..keep].to_vec();
    bounded.push(*tokens.last().expect("non-empty"));
    EmbedInput::Tokens(bounded)
}

// ---------------------------------------------------------------------------
// Error mapping (pure)
// ---------------------------------------------------------------------------

/// Classification of a llama.cpp error body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LlamaCppErrorKind {
    /// Prompt longer than the physical batch (`--ubatch-size`); HTTP 500 in
    /// current builds.
    InputTooLarge,
    /// Prompt longer than the context window (`--ctx-size`).
    ExceedsContext,
    /// Anything else.
    Other,
}

const ERROR_SNIPPET_MAX_CHARS: usize = 200;

/// Parse `{"error":{"code","message","type"}}` leniently and classify it.
/// Returns the kind plus a human-readable message (or a truncated raw snippet
/// when the body is not that shape).
pub(crate) fn classify_error_body(body: &str) -> (LlamaCppErrorKind, String) {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let err_obj = parsed.as_ref().and_then(|v| v.get("error"));
    let message = err_obj
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| body.chars().take(ERROR_SNIPPET_MAX_CHARS).collect());
    let err_type = err_obj
        .and_then(|e| e.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let lower = message.to_ascii_lowercase();
    let kind = if lower.contains("too large to process") {
        LlamaCppErrorKind::InputTooLarge
    } else if err_type == "exceed_context_size"
        || lower.contains("exceeds the available context size")
    {
        LlamaCppErrorKind::ExceedsContext
    } else {
        LlamaCppErrorKind::Other
    };
    (kind, message)
}

/// Map a non-2xx llama.cpp response to a [`RuntimeError`].
///
/// Input-size rejections are deterministic client-side configuration problems
/// (the server's `--ubatch-size`/`--ctx-size` is smaller than
/// `max_input_tokens`), so they become [`RuntimeError::InferenceError`] — not
/// retryable and, see [`LlamaCppEmbeddingModel::embed`], not counted against
/// the circuit breaker — regardless of the HTTP status the server chose.
pub(crate) fn map_llamacpp_status(
    status: u16,
    body: &str,
    max_input_tokens: usize,
) -> RuntimeError {
    let (kind, message) = classify_error_body(body);
    match kind {
        LlamaCppErrorKind::InputTooLarge | LlamaCppErrorKind::ExceedsContext => {
            RuntimeError::InferenceError(format!(
                "llama.cpp server rejected the input length ({}); max_input_tokens={} is \
                 larger than the server allows — lower max_input_tokens or start llama-server \
                 with --ubatch-size and --ctx-size of at least that many tokens",
                message, max_input_tokens
            ))
        }
        LlamaCppErrorKind::Other => match status {
            429 => RuntimeError::RateLimited,
            401 | 403 => RuntimeError::Unauthorized,
            500..=599 => RuntimeError::Unavailable,
            _ => RuntimeError::ApiError(format!("llama.cpp API error: {}: {}", status, message)),
        },
    }
}

/// Map a reqwest transport error.
fn map_reqwest_error(e: reqwest::Error) -> RuntimeError {
    if e.is_timeout() {
        RuntimeError::Timeout
    } else {
        RuntimeError::ApiError(e.to_string())
    }
}

/// Check every vector has the configured width.
pub(crate) fn validate_dimensions(vectors: &[Vec<f32>], expected: u32) -> Result<()> {
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != expected as usize {
            return Err(RuntimeError::ApiError(format!(
                "llama.cpp returned a {}-dimensional vector at index {}, expected {} \
                 (check the embedding_dimensions option)",
                v.len(),
                i,
                expected
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding model
// ---------------------------------------------------------------------------

/// Embedding model backed by a llama.cpp server.
pub struct LlamaCppEmbeddingModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    cfg: LlamaCppConfig,
}

impl LlamaCppEmbeddingModel {
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.cfg.api_key {
            Some(key) => req.header("Authorization", format!("Bearer {}", key)),
            None => req,
        }
    }

    /// POST JSON, apply timeout/auth, and map non-2xx responses.
    async fn post_json(&self, url: String, payload: &Value) -> Result<Value> {
        let response = self
            .apply_auth(self.client.post(url))
            .timeout(self.cfg.request_timeout)
            .json(payload)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !(200..300).contains(&status) {
            return Err(map_llamacpp_status(
                status,
                &text,
                self.cfg.max_input_tokens,
            ));
        }
        serde_json::from_str(&text).map_err(|e| {
            RuntimeError::ApiError(format!("llama.cpp returned malformed JSON: {}", e))
        })
    }

    /// Call `/tokenize` with special tokens so the count matches what the
    /// server would embed for the plain string.
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let body = self
            .post_json(
                format!("{}/tokenize", self.cfg.tokenizer_base_url),
                &json!({
                    // `model` is required in router mode so the request reaches
                    // the tokenizer of the model that will do the embedding;
                    // single-model servers ignore it.
                    "model": self.model_id,
                    "content": text,
                    "add_special": true,
                    "parse_special": false,
                }),
            )
            .await?;
        let tokens = body
            .get("tokens")
            .and_then(|t| t.as_array())
            .ok_or_else(|| {
                RuntimeError::ApiError(
                    "llama.cpp /tokenize response malformed: missing 'tokens' array".to_string(),
                )
            })?;
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                t.as_u64()
                    .filter(|v| *v <= u32::MAX as u64)
                    .map(|v| v as u32)
                    .ok_or_else(|| {
                        RuntimeError::ApiError(format!(
                            "llama.cpp /tokenize response malformed: token {} is not an integer id",
                            i
                        ))
                    })
            })
            .collect()
    }

    /// Decide the wire form of every input, tokenizing only those that might
    /// exceed the budget. `try_join_all` preserves input order.
    async fn plan_inputs(&self, texts: &[String]) -> Result<Vec<EmbedInput>> {
        let max = self.cfg.max_input_tokens;
        try_join_all(texts.iter().map(|text| async move {
            if !needs_tokenize(text, max) {
                return Ok(EmbedInput::Text(text.clone()));
            }
            let tokens = self.tokenize(text).await?;
            Ok(plan_input(text, &tokens, max))
        }))
        .await
    }

    async fn post_embeddings(&self, inputs: &[EmbedInput]) -> Result<Value> {
        let input: Vec<Value> = inputs.iter().map(EmbedInput::to_json).collect();
        self.post_json(
            format!("{}/embeddings", self.cfg.base_url),
            &json!({ "model": self.model_id, "input": input }),
        )
        .await
    }

    async fn embed_pipeline(&self, texts: &[String]) -> Result<EmbedResult> {
        let inputs = self.plan_inputs(texts).await?;
        let body = self.post_embeddings(&inputs).await?;
        let result = parse_openai_embeddings_response("llama.cpp", &body, Some(texts.len()))?;
        validate_dimensions(&result.vectors, self.cfg.dimensions)?;
        Ok(result)
    }
}

#[async_trait]
impl EmbeddingModel for LlamaCppEmbeddingModel {
    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult> {
        if texts.is_empty() {
            return Ok(EmbedResult {
                vectors: Vec::new(),
                usage: None,
            });
        }
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // One breaker call covers tokenize + embed. Deterministic input-size
        // rejections (`InferenceError`, produced only by `map_llamacpp_status`
        // in this module) are a configuration problem, not a server outage, so
        // they are returned as `Ok(Err(..))` from the closure — the breaker
        // sees a success and stays closed — and flattened here.
        self.cb
            .call(move || async move {
                match self.embed_pipeline(&texts).await {
                    Err(e @ RuntimeError::InferenceError(_)) => Ok(Err(e)),
                    other => other.map(Ok),
                }
            })
            .await?
    }

    fn dimensions(&self) -> u32 {
        self.cfg.dimensions
    }
}

impl crate::traits::ModelInfo for LlamaCppEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Tests (no HTTP)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ModelRuntimeKey;
    use crate::provider::remote_common::RemoteProviderBase;

    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn options() -> Value {
        json!({
            "base_url": "http://127.0.0.1:8080/v1",
            "max_input_tokens": 512,
            "embedding_dimensions": 384
        })
    }

    fn spec(alias: &str, task: ModelTask, model_id: &str, options: Value) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: alias.to_string(),
            task,
            provider_id: PROVIDER_ID.to_string(),
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

    // --- provider / breaker -------------------------------------------------

    #[tokio::test]
    async fn provider_id_and_capabilities() {
        let p = RemoteLlamaCppProvider::new();
        assert_eq!(p.provider_id(), "remote/llamacpp");
        assert_eq!(p.capabilities().supported_tasks, vec![ModelTask::Embed]);
        assert!(matches!(p.health().await, ProviderHealth::Healthy));
    }

    #[tokio::test]
    async fn load_rejects_non_embed_tasks() {
        let p = RemoteLlamaCppProvider::new();
        for task in [ModelTask::Rerank, ModelTask::Generate, ModelTask::Raw] {
            let err = p
                .load(&spec("x/y", task, "bge", options()))
                .await
                .expect_err("must fail");
            assert!(
                matches!(err, RuntimeError::CapabilityMismatch(_)),
                "{err:?}"
            );
        }
    }

    #[tokio::test]
    async fn load_exposes_configured_dimensions_and_model_id() {
        let p = RemoteLlamaCppProvider::new();
        let handle = p
            .load(&spec("embed/a", ModelTask::Embed, "bge-small", options()))
            .await
            .unwrap();
        let model = handle.downcast_ref::<Arc<dyn EmbeddingModel>>().unwrap();
        assert_eq!(model.dimensions(), 384);
        assert_eq!(model.model_id(), "bge-small");
    }

    #[tokio::test]
    async fn breaker_reused_for_same_runtime_key() {
        let p = RemoteLlamaCppProvider::new();
        let _ = p
            .load(&spec("embed/a", ModelTask::Embed, "bge", options()))
            .await
            .unwrap();
        let _ = p
            .load(&spec("embed/b", ModelTask::Embed, "bge", options()))
            .await
            .unwrap();
        assert_eq!(p.breaker_count(), 1);
    }

    #[tokio::test]
    async fn breaker_isolated_by_model() {
        let p = RemoteLlamaCppProvider::new();
        let _ = p
            .load(&spec("embed/a", ModelTask::Embed, "bge", options()))
            .await
            .unwrap();
        let _ = p
            .load(&spec("embed/b", ModelTask::Embed, "nomic", options()))
            .await
            .unwrap();
        assert_eq!(p.breaker_count(), 2);
    }

    #[tokio::test]
    async fn breaker_cleanup_evicts_stale_entries() {
        let p = RemoteLlamaCppProvider::new();
        let stale = spec("embed/stale", ModelTask::Embed, "old", options());
        let fresh = spec("embed/fresh", ModelTask::Embed, "new", options());
        p.insert_test_breaker(
            ModelRuntimeKey::new(&stale),
            RemoteProviderBase::BREAKER_TTL + Duration::from_secs(5),
        );
        p.insert_test_breaker(ModelRuntimeKey::new(&fresh), Duration::from_secs(1));
        assert_eq!(p.breaker_count(), 2);
        p.force_cleanup_now_for_test();
        let _ = p.load(&fresh).await.unwrap();
        assert_eq!(p.breaker_count(), 1);
    }

    // --- config -------------------------------------------------------------

    #[test]
    fn config_happy_path_and_defaults() {
        let cfg = LlamaCppConfig::from_options(&options()).unwrap();
        assert_eq!(cfg.base_url, "http://127.0.0.1:8080/v1");
        assert_eq!(cfg.tokenizer_base_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.max_input_tokens, 512);
        assert_eq!(cfg.dimensions, 384);
        assert!(cfg.api_key.is_none());
        assert_eq!(
            cfg.request_timeout,
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn config_strips_trailing_slash_and_honours_explicit_tokenizer_url_and_timeout() {
        let cfg = LlamaCppConfig::from_options(&json!({
            "base_url": "http://h:1/v1/",
            "tokenizer_base_url": "http://tok:2/",
            "max_input_tokens": 3,
            "embedding_dimensions": 4,
            "request_timeout_secs": 7
        }))
        .unwrap();
        assert_eq!(cfg.base_url, "http://h:1/v1");
        assert_eq!(cfg.tokenizer_base_url, "http://tok:2");
        assert_eq!(cfg.request_timeout, Duration::from_secs(7));
    }

    #[test]
    fn config_requires_base_url_max_tokens_and_dimensions() {
        for missing in ["base_url", "max_input_tokens", "embedding_dimensions"] {
            let mut o = options();
            o.as_object_mut().unwrap().remove(missing);
            let err = LlamaCppConfig::from_options(&o).expect_err("must fail");
            match err {
                RuntimeError::Config(msg) => assert!(msg.contains(missing), "{msg}"),
                other => panic!("expected Config, got {other:?}"),
            }
        }
        assert!(LlamaCppConfig::from_options(&Value::Null).is_err());
    }

    #[test]
    fn config_rejects_tiny_max_input_tokens() {
        let mut o = options();
        o["max_input_tokens"] = json!(2);
        let err = LlamaCppConfig::from_options(&o).err().unwrap();
        assert!(err.to_string().contains("at least 3"), "{err}");
    }

    #[tokio::test]
    async fn config_api_key_env_present_and_missing() {
        let _lock = ENV_LOCK.lock().await;
        let mut o = options();
        o["api_key_env"] = json!("UNI_XERVO_LLAMACPP_TEST_KEY");

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("UNI_XERVO_LLAMACPP_TEST_KEY") };
        let err = LlamaCppConfig::from_options(&o).err().unwrap();
        assert!(
            err.to_string().contains("UNI_XERVO_LLAMACPP_TEST_KEY"),
            "{err}"
        );

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("UNI_XERVO_LLAMACPP_TEST_KEY", "sekrit") };
        let cfg = LlamaCppConfig::from_options(&o).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sekrit"));
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("UNI_XERVO_LLAMACPP_TEST_KEY") };
    }

    #[test]
    fn tokenizer_base_url_derivation() {
        assert_eq!(
            resolve_tokenizer_base_url("http://h:8080/v1"),
            "http://h:8080"
        );
        assert_eq!(
            resolve_tokenizer_base_url("http://h:8080/v1/"),
            "http://h:8080"
        );
        assert_eq!(
            resolve_tokenizer_base_url("http://h:8080/api/v1"),
            "http://h:8080/api"
        );
        assert_eq!(resolve_tokenizer_base_url("http://h:8080"), "http://h:8080");
        assert_eq!(
            resolve_tokenizer_base_url("http://h:8080/v10"),
            "http://h:8080/v10"
        );
        assert_eq!(resolve_tokenizer_base_url("/v1"), "/v1");
    }

    // --- input planning -----------------------------------------------------

    #[test]
    fn needs_tokenize_boundaries_use_char_count() {
        assert!(!needs_tokenize("", 10));
        assert!(!needs_tokenize("12345678", 10)); // == max - 2
        assert!(needs_tokenize("123456789", 10)); // == max - 1
        // 8 CJK chars are 24 bytes but only 8 chars.
        assert!(!needs_tokenize("你好世界你好世界", 10));
        assert!(needs_tokenize("你好世界你好世界你", 10));
        // Degenerate budgets never underflow.
        assert!(needs_tokenize("a", 1));
        assert!(!needs_tokenize("", 0));
    }

    #[test]
    fn plan_input_fits_sends_text() {
        let tokens: Vec<u32> = (0..10).collect();
        assert_eq!(
            plan_input("t", &tokens, 10),
            EmbedInput::Text("t".to_string())
        );
        assert_eq!(plan_input("t", &[], 10), EmbedInput::Text("t".to_string()));
    }

    #[test]
    fn plan_input_truncates_keeping_leading_ids_and_trailing_special() {
        // [CLS]=101, content 1..=9, [SEP]=102 → 11 tokens, budget 10.
        let mut tokens = vec![101u32];
        tokens.extend(1..=9);
        tokens.push(102);
        match plan_input("t", &tokens, 10) {
            EmbedInput::Tokens(ids) => {
                assert_eq!(ids.len(), 10);
                assert_eq!(ids[0], 101);
                assert_eq!(&ids[1..9], &[1, 2, 3, 4, 5, 6, 7, 8]);
                assert_eq!(*ids.last().unwrap(), 102);
            }
            other => panic!("expected Tokens, got {other:?}"),
        }
    }

    #[test]
    fn embed_input_json_shapes() {
        assert_eq!(
            EmbedInput::Text("hi".into()).to_json(),
            Value::String("hi".into())
        );
        assert_eq!(EmbedInput::Tokens(vec![1, 2]).to_json(), json!([1, 2]));
    }

    // --- error mapping ------------------------------------------------------

    #[test]
    fn classify_error_bodies() {
        let too_large = r#"{"error":{"code":500,"message":"input is too large to process. increase the physical batch size","type":"server_error"}}"#;
        let (k, m) = classify_error_body(too_large);
        assert_eq!(k, LlamaCppErrorKind::InputTooLarge);
        assert!(m.starts_with("input is too large"));

        let ctx_type =
            r#"{"error":{"code":400,"message":"whatever","type":"exceed_context_size"}}"#;
        assert_eq!(
            classify_error_body(ctx_type).0,
            LlamaCppErrorKind::ExceedsContext
        );
        let ctx_msg = r#"{"error":{"message":"prompt exceeds the available context size. increase context size"}}"#;
        assert_eq!(
            classify_error_body(ctx_msg).0,
            LlamaCppErrorKind::ExceedsContext
        );

        let (k, m) = classify_error_body("<html>502 bad gateway</html>");
        assert_eq!(k, LlamaCppErrorKind::Other);
        assert_eq!(m, "<html>502 bad gateway</html>");

        let long = "x".repeat(1000);
        assert_eq!(
            classify_error_body(&long).1.chars().count(),
            ERROR_SNIPPET_MAX_CHARS
        );
    }

    #[test]
    fn status_mapping() {
        let too_large = r#"{"error":{"message":"input is too large to process. increase the physical batch size"}}"#;
        let e = map_llamacpp_status(500, too_large, 512);
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");
        assert!(!e.is_retryable());
        assert!(e.to_string().contains("max_input_tokens=512"), "{e}");

        let e = map_llamacpp_status(
            400,
            r#"{"error":{"type":"exceed_context_size","message":"x"}}"#,
            512,
        );
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");

        assert!(matches!(
            map_llamacpp_status(500, "boom", 512),
            RuntimeError::Unavailable
        ));
        assert!(matches!(
            map_llamacpp_status(503, "", 512),
            RuntimeError::Unavailable
        ));
        assert!(matches!(
            map_llamacpp_status(429, "", 512),
            RuntimeError::RateLimited
        ));
        assert!(matches!(
            map_llamacpp_status(401, "", 512),
            RuntimeError::Unauthorized
        ));
        assert!(matches!(
            map_llamacpp_status(403, "", 512),
            RuntimeError::Unauthorized
        ));
        match map_llamacpp_status(404, r#"{"error":{"message":"no route"}}"#, 512) {
            RuntimeError::ApiError(m) => {
                assert!(m.contains("404") && m.contains("no route"), "{m}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dimension_validation() {
        assert!(validate_dimensions(&[vec![0.0; 4], vec![1.0; 4]], 4).is_ok());
        let err = validate_dimensions(&[vec![0.0; 4], vec![1.0; 3]], 4)
            .err()
            .unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("3-dimensional") && msg.contains("index 1"),
            "{msg}"
        );
        assert!(msg.contains("embedding_dimensions"), "{msg}");
    }
}
