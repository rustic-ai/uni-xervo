//! Remote provider for a llama.cpp [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
//!
//! `llama-server` speaks a near-complete OpenAI-compatible API over one loaded
//! model, plus a handful of native endpoints of its own. This module is the
//! provider shell: the option plumbing every task shares, the HTTP seam, and
//! the error classifier. Each task lives in its own sibling module:
//!
//! | task | module | endpoint |
//! |------|--------|----------|
//! | [`ModelTask::Embed`] | [`embed`] | `POST /v1/embeddings` + native `/tokenize` |
//! | [`ModelTask::Generate`] | [`chat`] | `POST /v1/chat/completions` |
//! | [`ModelTask::Rerank`] | [`rerank`] | `POST /v1/rerank` |
//!
//! # Shared options
//!
//! | key | type | required | meaning |
//! |-----|------|----------|---------|
//! | `base_url` | string | yes | OpenAI-compatible root including `/v1`, e.g. `http://127.0.0.1:8080/v1` |
//! | `api_key_env` | string | no | env var holding a bearer token; omit when the server runs without `--api-key` |
//! | `request_timeout_secs` | integer > 0 | no | per-HTTP-request timeout, default 60 |
//!
//! Per-task options are documented on the module that reads them.
//!
//! # Input length
//!
//! `llama-server` never silently truncates. A prompt that does not fit the
//! physical batch (`--ubatch-size`) or the context window (`--ctx-size`) is
//! rejected with an HTTP error, which [`map_llamacpp_status`] turns into a
//! non-retryable [`RuntimeError::InferenceError`] naming the remedy. That is a
//! deterministic configuration problem rather than a server outage, so these
//! rejections are also kept off the circuit breaker — see
//! [`embed::LlamaCppEmbeddingModel::embed`].

mod chat;
mod embed;
mod rerank;

pub use chat::LlamaCppGeneratorModel;
pub use embed::LlamaCppEmbeddingModel;
pub use rerank::LlamaCppRerankerModel;

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::RemoteProviderBase;
use crate::traits::{
    EmbeddingModel, GeneratorModel, LoadedModelHandle, ModelProvider, ProviderCapabilities,
    ProviderHealth, RerankerModel,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Provider id used in [`ModelAliasSpec::provider_id`].
pub const PROVIDER_ID: &str = "remote/llamacpp";

/// Default per-request HTTP timeout when `request_timeout_secs` is unset.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;

/// Smallest meaningful `max_input_tokens`: two special tokens plus one
/// content token.
pub const MIN_MAX_INPUT_TOKENS: usize = 3;

/// Remote provider for a llama.cpp `llama-server`.
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
            supported_tasks: vec![ModelTask::Embed, ModelTask::Generate, ModelTask::Rerank],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Embed => {
                let model = embed::LlamaCppEmbeddingModel::new(
                    self.base.client.clone(),
                    self.base.circuit_breaker_for(spec),
                    spec.model_id.clone(),
                    embed::EmbedConfig::from_options(&spec.options)?,
                );
                let handle: Arc<dyn EmbeddingModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Generate => {
                let model = chat::LlamaCppGeneratorModel::new(
                    self.base.client.clone(),
                    self.base.circuit_breaker_for(spec),
                    spec.model_id.clone(),
                    chat::ChatConfig::from_options(&spec.options)?,
                );
                let handle: Arc<dyn GeneratorModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let model = rerank::LlamaCppRerankerModel::new(
                    self.base.client.clone(),
                    self.base.circuit_breaker_for(spec),
                    spec.model_id.clone(),
                    rerank::RerankConfig::from_options(&spec.options)?,
                );
                let handle: Arc<dyn RerankerModel> = Arc::new(model);
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
// Shared configuration
// ---------------------------------------------------------------------------

/// The part of an alias's `options` that every task reads.
#[derive(Debug, Clone)]
pub(crate) struct LlamaCppCommon {
    /// OpenAI-compatible root including `/v1`, no trailing slash.
    pub base_url: String,
    /// Bearer token, if `api_key_env` was configured.
    pub api_key: Option<String>,
    /// Per-HTTP-request timeout.
    pub request_timeout: Duration,
}

/// Build the "option is required" error for `key`.
pub(crate) fn required_option(key: &str) -> RuntimeError {
    RuntimeError::Config(format!(
        "Option '{}' for provider '{}' is required",
        key, PROVIDER_ID
    ))
}

impl LlamaCppCommon {
    /// Parse the options every task shares. The catalog validator already
    /// enforces the same rules, but `ModelProvider::load` is public, so this
    /// is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        let base_url = options
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| required_option("base_url"))?;

        let api_key = match options.get("api_key_env").and_then(|v| v.as_str()) {
            None => None,
            Some(env_name) => Some(
                std::env::var(env_name)
                    .map_err(|_| RuntimeError::Config(format!("{} env var not set", env_name)))?,
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
            api_key,
            request_timeout,
        })
    }
}

/// Derive the server root for native (non-`/v1`) endpoints from an
/// OpenAI-style `base_url`: strip trailing slashes, then exactly one trailing
/// `/v1` path segment.
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
// HTTP seam
// ---------------------------------------------------------------------------

/// Attach the bearer token when one is configured.
fn apply_auth(common: &LlamaCppCommon, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match &common.api_key {
        Some(key) => req.header("Authorization", format!("Bearer {}", key)),
        None => req,
    }
}

/// POST JSON to `url`, apply timeout and auth, and map non-2xx responses.
///
/// `token_budget` is the alias's configured input budget when the task has one
/// (embedding's `max_input_tokens`); it only shapes the advice in an
/// input-too-long error. Tasks with no such option pass `None`.
pub(crate) async fn post_json(
    client: &Client,
    common: &LlamaCppCommon,
    url: String,
    payload: &Value,
    token_budget: Option<usize>,
) -> Result<Value> {
    let response = apply_auth(common, client.post(url))
        .timeout(common.request_timeout)
        .json(payload)
        .send()
        .await
        .map_err(map_reqwest_error)?;

    let status = response.status().as_u16();
    let text = response.text().await.map_err(map_reqwest_error)?;
    if !(200..300).contains(&status) {
        return Err(map_llamacpp_status(status, &text, token_budget));
    }
    serde_json::from_str(&text)
        .map_err(|e| RuntimeError::ApiError(format!("llama.cpp returned malformed JSON: {}", e)))
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
    /// The endpoint exists but this server process was not started with the
    /// flag that enables it — `/rerank` without `--reranking`, for instance.
    /// HTTP 501 in current builds.
    NotSupported,
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
    } else if err_type == "not_supported_error" || lower.contains("does not support") {
        LlamaCppErrorKind::NotSupported
    } else {
        LlamaCppErrorKind::Other
    };
    (kind, message)
}

/// Map a non-2xx llama.cpp response to a [`RuntimeError`].
///
/// Two families of rejection are deterministic configuration problems rather
/// than outages, so both become [`RuntimeError::InferenceError`] — not
/// retryable and not counted against the circuit breaker — regardless of the
/// HTTP status the server chose:
///
/// - **Input size**: the server's `--ubatch-size`/`--ctx-size` is smaller than
///   the request. `token_budget` names the offending option when the task has
///   one; without it the advice can only point at the server's own flags.
/// - **Endpoint not enabled**: the route exists but the process was started
///   without the flag that activates it, which current builds report as HTTP
///   501. Retrying cannot help — the server has to be restarted — and the
///   message already names the flag, so it is passed through verbatim.
pub(crate) fn map_llamacpp_status(
    status: u16,
    body: &str,
    token_budget: Option<usize>,
) -> RuntimeError {
    let (kind, message) = classify_error_body(body);
    match kind {
        LlamaCppErrorKind::InputTooLarge | LlamaCppErrorKind::ExceedsContext => {
            match token_budget {
                Some(max_input_tokens) => RuntimeError::InferenceError(format!(
                    "llama.cpp server rejected the input length ({}); max_input_tokens={} is \
                     larger than the server allows — lower max_input_tokens or start \
                     llama-server with --ubatch-size and --ctx-size of at least that many tokens",
                    message, max_input_tokens
                )),
                None => RuntimeError::InferenceError(format!(
                    "llama.cpp server rejected the input length ({}); the request does not fit \
                     the server's limits — send less input, or start llama-server with a larger \
                     --ctx-size and --ubatch-size",
                    message
                )),
            }
        }
        LlamaCppErrorKind::NotSupported => RuntimeError::InferenceError(format!(
            "llama.cpp server rejected the request: {}",
            message
        )),
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

// ---------------------------------------------------------------------------
// Tests (no HTTP)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ModelRuntimeKey;
    use serde_json::json;

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
        assert_eq!(
            p.capabilities().supported_tasks,
            vec![ModelTask::Embed, ModelTask::Generate, ModelTask::Rerank]
        );
        assert!(matches!(p.health().await, ProviderHealth::Healthy));
    }

    #[tokio::test]
    async fn load_rejects_unsupported_tasks() {
        let p = RemoteLlamaCppProvider::new();
        for task in [ModelTask::Raw, ModelTask::EmbedImage] {
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
    async fn load_generate_yields_a_generator_handle() {
        let p = RemoteLlamaCppProvider::new();
        // Generation needs no embed-only options.
        let handle = p
            .load(&spec(
                "gen/a",
                ModelTask::Generate,
                "smolvlm",
                json!({ "base_url": "http://127.0.0.1:8080/v1" }),
            ))
            .await
            .unwrap();
        let model = handle.downcast_ref::<Arc<dyn GeneratorModel>>().unwrap();
        assert_eq!(model.model_id(), "smolvlm");
    }

    #[tokio::test]
    async fn load_rerank_yields_a_reranker_handle() {
        let p = RemoteLlamaCppProvider::new();
        // Reranking needs no embed-only options either.
        let handle = p
            .load(&spec(
                "rerank/a",
                ModelTask::Rerank,
                "jina-reranker-v1-tiny-en",
                json!({ "base_url": "http://127.0.0.1:8080/v1" }),
            ))
            .await
            .unwrap();
        let model = handle.downcast_ref::<Arc<dyn RerankerModel>>().unwrap();
        assert_eq!(model.model_id(), "jina-reranker-v1-tiny-en");
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

    // --- shared config ------------------------------------------------------

    #[test]
    fn common_config_happy_path_and_defaults() {
        let cfg = LlamaCppCommon::from_options(&options()).unwrap();
        assert_eq!(cfg.base_url, "http://127.0.0.1:8080/v1");
        assert!(cfg.api_key.is_none());
        assert_eq!(
            cfg.request_timeout,
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn common_config_strips_trailing_slash_and_honours_timeout() {
        let cfg = LlamaCppCommon::from_options(&json!({
            "base_url": "http://h:1/v1/",
            "request_timeout_secs": 7
        }))
        .unwrap();
        assert_eq!(cfg.base_url, "http://h:1/v1");
        assert_eq!(cfg.request_timeout, Duration::from_secs(7));
    }

    #[test]
    fn common_config_requires_base_url() {
        let err = LlamaCppCommon::from_options(&json!({})).expect_err("must fail");
        match err {
            RuntimeError::Config(msg) => assert!(msg.contains("base_url"), "{msg}"),
            other => panic!("expected Config, got {other:?}"),
        }
        assert!(LlamaCppCommon::from_options(&Value::Null).is_err());

        // A blank or whitespace-only value is as absent as a missing key.
        for blank in ["", "   "] {
            assert!(
                LlamaCppCommon::from_options(&json!({ "base_url": blank })).is_err(),
                "{blank:?} should be rejected"
            );
        }
    }

    #[test]
    fn common_config_ignores_a_zero_timeout_and_falls_back_to_the_default() {
        let cfg = LlamaCppCommon::from_options(&json!({
            "base_url": "http://h:1/v1",
            "request_timeout_secs": 0
        }))
        .unwrap();
        assert_eq!(
            cfg.request_timeout,
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
    }

    #[tokio::test]
    async fn common_config_api_key_env_present_and_missing() {
        let _lock = ENV_LOCK.lock().await;
        let mut o = options();
        o["api_key_env"] = json!("UNI_XERVO_LLAMACPP_TEST_KEY");

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var("UNI_XERVO_LLAMACPP_TEST_KEY") };
        let err = LlamaCppCommon::from_options(&o).err().unwrap();
        assert!(
            err.to_string().contains("UNI_XERVO_LLAMACPP_TEST_KEY"),
            "{err}"
        );

        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var("UNI_XERVO_LLAMACPP_TEST_KEY", "sekrit") };
        let cfg = LlamaCppCommon::from_options(&o).unwrap();
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
        let e = map_llamacpp_status(500, too_large, Some(512));
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");
        assert!(!e.is_retryable());
        assert!(e.to_string().contains("max_input_tokens=512"), "{e}");

        let e = map_llamacpp_status(
            400,
            r#"{"error":{"type":"exceed_context_size","message":"x"}}"#,
            Some(512),
        );
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");

        assert!(matches!(
            map_llamacpp_status(500, "boom", Some(512)),
            RuntimeError::Unavailable
        ));
        assert!(matches!(
            map_llamacpp_status(503, "", Some(512)),
            RuntimeError::Unavailable
        ));
        assert!(matches!(
            map_llamacpp_status(429, "", Some(512)),
            RuntimeError::RateLimited
        ));
        assert!(matches!(
            map_llamacpp_status(401, "", Some(512)),
            RuntimeError::Unauthorized
        ));
        assert!(matches!(
            map_llamacpp_status(403, "", Some(512)),
            RuntimeError::Unauthorized
        ));
        match map_llamacpp_status(404, r#"{"error":{"message":"no route"}}"#, Some(512)) {
            RuntimeError::ApiError(m) => {
                assert!(m.contains("404") && m.contains("no route"), "{m}")
            }
            other => panic!("{other:?}"),
        }
    }

    /// A 501 from an endpoint the server was not started for is permanent:
    /// `Unavailable` would have it retried and would trip the breaker.
    #[test]
    fn an_endpoint_that_is_not_enabled_is_not_retryable() {
        let body = r#"{"error":{"code":501,"message":"This server does not support reranking. Start it with `--reranking`","type":"not_supported_error"}}"#;
        assert_eq!(classify_error_body(body).0, LlamaCppErrorKind::NotSupported);

        let e = map_llamacpp_status(501, body, None);
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");
        assert!(!e.is_retryable(), "{e:?}");
        // The server's own advice is the useful part, so keep it.
        assert!(e.to_string().contains("--reranking"), "{e}");
    }

    #[test]
    fn status_mapping_without_a_token_budget_advises_server_flags_instead() {
        let too_large = r#"{"error":{"message":"input is too large to process. increase the physical batch size"}}"#;
        let e = map_llamacpp_status(500, too_large, None);
        assert!(matches!(e, RuntimeError::InferenceError(_)), "{e:?}");
        assert!(!e.is_retryable());

        let msg = e.to_string();
        assert!(!msg.contains("max_input_tokens"), "{msg}");
        assert!(msg.contains("--ctx-size"), "{msg}");

        // Classification of everything else is unaffected by the budget.
        assert!(matches!(
            map_llamacpp_status(429, "", None),
            RuntimeError::RateLimited
        ));
    }
}
