//! [`ModelTask::Embed`](crate::api::ModelTask::Embed) on `POST /v1/embeddings`.
//!
//! `llama-server` exposes an OpenAI-compatible embeddings endpoint, but it
//! differs from hosted embedding APIs in one way that matters for production
//! ingestion: **it never truncates embedding input**. Encoder models such as
//! BGE (BERT) are non-causal, so the whole prompt has to fit in a single
//! physical batch (`--ubatch-size`) and inside the context window
//! (`--ctx-size`); anything longer is rejected with an HTTP error instead of
//! being cut to size.
//!
//! This module therefore runs a deterministic, tokenizer-aware pre-pass before
//! every embedding call:
//!
//! 1. Texts that provably fit are sent as plain strings. Every WordPiece token
//!    consumes at least one character, so a text with at most
//!    `max_input_tokens - 2` characters cannot exceed the budget once the two
//!    special tokens (`[CLS]`/`[SEP]`) are added.
//! 2. Longer texts are sent to the server's native `POST /tokenize` endpoint
//!    with `add_special: true`, which returns the exact id sequence the server
//!    would embed, special tokens included.
//! 3. If that sequence exceeds `max_input_tokens`, the pre-pass keeps the
//!    first `max_input_tokens - 1` ids and re-appends the trailing special
//!    token, then sends the **token array** (not text) to `/v1/embeddings`.
//!    llama.cpp passes integer arrays through verbatim, so the server embeds
//!    exactly the bounded sequence — no lossy detokenize round trip.
//!
//! Batches preserve input order and produce exactly one vector per input.
//!
//! # Options
//!
//! Beyond the shared keys documented on the [parent module](super):
//!
//! | key | type | required | meaning |
//! |-----|------|----------|---------|
//! | `tokenizer_base_url` | string | no | server root for `/tokenize`; defaults to `base_url` with a trailing `/v1` removed |
//! | `max_input_tokens` | integer ≥ 3 | yes | total token budget **including** special tokens (512 for BGE) |
//! | `embedding_dimensions` | integer > 0 | yes | expected vector width (384 for BGE Small) |
//!
//! `max_input_tokens` must not exceed the server's `--ubatch-size` or
//! `--ctx-size`; if it does, the server's rejection surfaces as
//! [`RuntimeError::InferenceError`] naming the option.

use super::{
    LlamaCppCommon, MIN_MAX_INPUT_TOKENS, PROVIDER_ID, post_json, required_option,
    resolve_tokenizer_base_url,
};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::parse_openai_embeddings_response;
use crate::traits::{EmbedResult, EmbeddingModel};
use async_trait::async_trait;
use futures::future::try_join_all;
use reqwest::Client;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved, validated embedding configuration for one alias.
#[derive(Debug, Clone)]
pub(crate) struct EmbedConfig {
    /// Options shared with every other task.
    pub common: LlamaCppCommon,
    /// Server root for `/tokenize`, no trailing slash.
    pub tokenizer_base_url: String,
    /// Total token budget including special tokens.
    pub max_input_tokens: usize,
    /// Expected embedding width.
    pub dimensions: u32,
}

impl EmbedConfig {
    /// Parse the alias `options`. The catalog validator already enforces the
    /// same rules, but `ModelProvider::load` is public, so this is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        let common = LlamaCppCommon::from_options(options)?;

        let tokenizer_base_url = match options.get("tokenizer_base_url").and_then(|v| v.as_str()) {
            Some(raw) if !raw.trim().is_empty() => raw.trim().trim_end_matches('/').to_string(),
            _ => resolve_tokenizer_base_url(&common.base_url),
        };

        let max_input_tokens = options
            .get("max_input_tokens")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| required_option("max_input_tokens"))?
            as usize;
        if max_input_tokens < MIN_MAX_INPUT_TOKENS {
            return Err(RuntimeError::Config(format!(
                "Option 'max_input_tokens' for provider '{}' must be at least {} \
                 (two special tokens plus one content token)",
                PROVIDER_ID, MIN_MAX_INPUT_TOKENS
            )));
        }

        let dimensions = options
            .get("embedding_dimensions")
            .and_then(|v| v.as_u64())
            .filter(|d| *d > 0 && *d <= u32::MAX as u64)
            .ok_or_else(|| required_option("embedding_dimensions"))?
            as u32;

        Ok(Self {
            common,
            tokenizer_base_url,
            max_input_tokens,
            dimensions,
        })
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
    cfg: EmbedConfig,
}

impl LlamaCppEmbeddingModel {
    pub(crate) fn new(
        client: Client,
        cb: crate::reliability::CircuitBreakerWrapper,
        model_id: String,
        cfg: EmbedConfig,
    ) -> Self {
        Self {
            client,
            cb,
            model_id,
            cfg,
        }
    }

    /// POST JSON, naming `max_input_tokens` in any input-too-long error.
    async fn post_json(&self, url: String, payload: &Value) -> Result<Value> {
        post_json(
            &self.client,
            &self.cfg.common,
            url,
            payload,
            Some(self.cfg.max_input_tokens),
        )
        .await
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
            format!("{}/embeddings", self.cfg.common.base_url),
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
        // rejections (`InferenceError`, produced only by `map_llamacpp_status`)
        // are a configuration problem, not a server outage, so they are
        // returned as `Ok(Err(..))` from the closure — the breaker sees a
        // success and stays closed — and flattened here.
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
    use crate::provider::llamacpp::DEFAULT_REQUEST_TIMEOUT_SECS;
    use std::time::Duration;

    fn options() -> Value {
        json!({
            "base_url": "http://127.0.0.1:8080/v1",
            "max_input_tokens": 512,
            "embedding_dimensions": 384
        })
    }

    // --- config -------------------------------------------------------------

    #[test]
    fn config_happy_path_and_defaults() {
        let cfg = EmbedConfig::from_options(&options()).unwrap();
        assert_eq!(cfg.common.base_url, "http://127.0.0.1:8080/v1");
        assert_eq!(cfg.tokenizer_base_url, "http://127.0.0.1:8080");
        assert_eq!(cfg.max_input_tokens, 512);
        assert_eq!(cfg.dimensions, 384);
        assert!(cfg.common.api_key.is_none());
        assert_eq!(
            cfg.common.request_timeout,
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn config_strips_trailing_slash_and_honours_explicit_tokenizer_url_and_timeout() {
        let cfg = EmbedConfig::from_options(&json!({
            "base_url": "http://h:1/v1/",
            "tokenizer_base_url": "http://tok:2/",
            "max_input_tokens": 3,
            "embedding_dimensions": 4,
            "request_timeout_secs": 7
        }))
        .unwrap();
        assert_eq!(cfg.common.base_url, "http://h:1/v1");
        assert_eq!(cfg.tokenizer_base_url, "http://tok:2");
        assert_eq!(cfg.common.request_timeout, Duration::from_secs(7));
    }

    #[test]
    fn config_requires_base_url_max_tokens_and_dimensions() {
        for missing in ["base_url", "max_input_tokens", "embedding_dimensions"] {
            let mut o = options();
            o.as_object_mut().unwrap().remove(missing);
            let err = EmbedConfig::from_options(&o).expect_err("must fail");
            match err {
                RuntimeError::Config(msg) => assert!(msg.contains(missing), "{msg}"),
                other => panic!("expected Config, got {other:?}"),
            }
        }
        assert!(EmbedConfig::from_options(&Value::Null).is_err());
    }

    #[test]
    fn config_rejects_tiny_max_input_tokens() {
        let mut o = options();
        o["max_input_tokens"] = json!(2);
        let err = EmbedConfig::from_options(&o).err().unwrap();
        assert!(err.to_string().contains("at least 3"), "{err}");
    }

    #[test]
    fn config_rejects_zero_dimensions() {
        let mut o = options();
        o["embedding_dimensions"] = json!(0);
        assert!(EmbedConfig::from_options(&o).is_err());
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

    // --- dimension validation -----------------------------------------------

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
