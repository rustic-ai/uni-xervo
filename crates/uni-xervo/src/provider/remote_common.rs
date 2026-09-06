//! Shared utilities for all remote (HTTP API) providers: HTTP status mapping,
//! API key resolution, circuit breaker management, and Google-style payload
//! construction.

use crate::api::{ModelAliasSpec, ModelRuntimeKey};
use crate::error::{Result, RuntimeError};
use crate::reliability::{CircuitBreakerConfig, CircuitBreakerWrapper};
use crate::traits::{EmbedResult, TokenUsage};
use reqwest::Client;
#[cfg(any(feature = "provider-gemini", feature = "provider-vertexai"))]
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Map an HTTP response status to a `RuntimeError` for non-success codes.
/// Returns `Ok(response)` when the status is 2xx.
#[cfg(any(
    feature = "provider-openai",
    feature = "provider-gemini",
    feature = "provider-vertexai",
    feature = "provider-mistral",
    feature = "provider-anthropic",
    feature = "provider-voyageai",
    feature = "provider-cohere",
    feature = "provider-azure-openai",
))]
pub(crate) fn check_http_status(
    provider_name: &str,
    response: reqwest::Response,
) -> std::result::Result<reqwest::Response, RuntimeError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    Err(match status.as_u16() {
        429 => RuntimeError::RateLimited,
        401 | 403 => RuntimeError::Unauthorized,
        500..=599 => RuntimeError::Unavailable,
        _ => RuntimeError::ApiError(format!("{} API error: {}", provider_name, status)),
    })
}

/// Resolve an API key from the spec's options JSON.
///
/// Looks for `options[option_key]` to get a custom env var name; falls back to
/// `default_env` if unset. Then reads the value from the environment.
#[cfg(any(
    feature = "provider-openai",
    feature = "provider-gemini",
    feature = "provider-vertexai",
    feature = "provider-mistral",
    feature = "provider-anthropic",
    feature = "provider-voyageai",
    feature = "provider-cohere",
    feature = "provider-azure-openai",
))]
pub(crate) fn resolve_api_key(
    options: &serde_json::Value,
    option_key: &str,
    default_env: &str,
) -> Result<String> {
    let env_var_name = options
        .get(option_key)
        .and_then(|v| v.as_str())
        .unwrap_or(default_env);

    std::env::var(env_var_name)
        .map_err(|_| RuntimeError::Config(format!("{} env var not set", env_var_name)))
}

/// Decode an OpenAI-format `/embeddings` response body into an [`EmbedResult`].
///
/// Two modes:
///
/// * `expected_len == None` — **lenient**, the historical `remote/openai`
///   behaviour: items without an `embedding` array are skipped, non-numeric
///   elements are dropped, and vectors come back in `data` order.
/// * `expected_len == Some(n)` — **strict**: `data` must be an array of exactly
///   `n` items, every item must carry a numeric `embedding` array, and when
///   every item has an integer `index` the vectors are placed by that index
///   (duplicate or out-of-range indexes are an error). Otherwise placement is
///   positional.
///
/// Malformed responses surface as [`RuntimeError::ApiError`] tagged with
/// `provider_name`. The `usage` object (`prompt_tokens`, `total_tokens`) is
/// mapped the same way in both modes; `completion_tokens` is always `0`.
pub(crate) fn parse_openai_embeddings_response(
    provider_name: &str,
    body: &serde_json::Value,
    expected_len: Option<usize>,
) -> Result<EmbedResult> {
    let usage = body.get("usage").map(|u| {
        let prompt = u["prompt_tokens"].as_u64().unwrap_or(0) as usize;
        let total = u["total_tokens"].as_u64().unwrap_or(prompt as u64) as usize;
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: total,
        }
    });

    let Some(expected) = expected_len else {
        let mut vectors = Vec::new();
        if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(embedding) = item.get("embedding").and_then(|e| e.as_array()) {
                    let vec: Vec<f32> = embedding
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    vectors.push(vec);
                }
            }
        }
        return Ok(EmbedResult { vectors, usage });
    };

    let malformed = |detail: String| {
        RuntimeError::ApiError(format!(
            "{} embeddings response malformed: {}",
            provider_name, detail
        ))
    };

    let data = body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| malformed("missing 'data' array".to_string()))?;
    if data.len() != expected {
        return Err(malformed(format!(
            "expected {} embeddings, got {}",
            expected,
            data.len()
        )));
    }

    let mut decoded: Vec<(Option<usize>, Vec<f32>)> = Vec::with_capacity(expected);
    for (pos, item) in data.iter().enumerate() {
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| malformed(format!("item {} has no 'embedding' array", pos)))?;
        let mut vec = Vec::with_capacity(embedding.len());
        for (j, v) in embedding.iter().enumerate() {
            let f = v
                .as_f64()
                .ok_or_else(|| malformed(format!("item {} element {} is not a number", pos, j)))?;
            vec.push(f as f32);
        }
        let index = match item.get("index") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .map(|i| i as usize)
                    .ok_or_else(|| malformed(format!("item {} has a non-integer 'index'", pos)))?,
            ),
        };
        decoded.push((index, vec));
    }

    let all_indexed = decoded.iter().all(|(i, _)| i.is_some());
    let vectors = if all_indexed {
        let mut slots: Vec<Option<Vec<f32>>> = (0..expected).map(|_| None).collect();
        for (index, vec) in decoded {
            let i = index.expect("checked all_indexed");
            if i >= expected {
                return Err(malformed(format!(
                    "index {} out of range for {} inputs",
                    i, expected
                )));
            }
            if slots[i].is_some() {
                return Err(malformed(format!("duplicate index {}", i)));
            }
            slots[i] = Some(vec);
        }
        slots
            .into_iter()
            .map(|s| s.expect("every slot filled: len == expected and no duplicates"))
            .collect()
    } else {
        decoded.into_iter().map(|(_, v)| v).collect()
    };

    Ok(EmbedResult { vectors, usage })
}

struct BreakerEntry {
    breaker: CircuitBreakerWrapper,
    last_access: Instant,
}

/// Shared circuit-breaker management for all remote providers.
pub(crate) struct RemoteProviderBase {
    pub(crate) client: Client,
    breakers: Mutex<HashMap<ModelRuntimeKey, BreakerEntry>>,
    last_cleanup: Mutex<Instant>,
}

impl RemoteProviderBase {
    pub(crate) const BREAKER_TTL: Duration = Duration::from_secs(30 * 60);
    const CLEANUP_INTERVAL: Duration = Duration::from_secs(5 * 60);

    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            client: Client::new(),
            breakers: Mutex::new(HashMap::new()),
            last_cleanup: Mutex::new(now),
        }
    }

    pub(crate) fn circuit_breaker_for(&self, spec: &ModelAliasSpec) -> CircuitBreakerWrapper {
        let key = ModelRuntimeKey::new(spec);
        let now = Instant::now();
        self.maybe_cleanup(now);

        let mut breakers = self.breakers.lock().unwrap();
        let entry = breakers.entry(key).or_insert_with(|| BreakerEntry {
            breaker: CircuitBreakerWrapper::new(CircuitBreakerConfig::default()),
            last_access: now,
        });
        entry.last_access = now;
        entry.breaker.clone()
    }

    fn maybe_cleanup(&self, now: Instant) {
        let should_cleanup = {
            let mut last = self.last_cleanup.lock().unwrap();
            if now.duration_since(*last) >= Self::CLEANUP_INTERVAL {
                *last = now;
                true
            } else {
                false
            }
        };
        if !should_cleanup {
            return;
        }

        let mut breakers = self.breakers.lock().unwrap();
        breakers.retain(|_, entry| now.duration_since(entry.last_access) < Self::BREAKER_TTL);
    }

    #[cfg(test)]
    pub(crate) fn insert_test_breaker(&self, key: ModelRuntimeKey, age: Duration) {
        let now = Instant::now();
        let mut breakers = self.breakers.lock().unwrap();
        breakers.insert(
            key,
            BreakerEntry {
                breaker: CircuitBreakerWrapper::new(CircuitBreakerConfig::default()),
                last_access: now.checked_sub(age).unwrap_or(now),
            },
        );
    }

    #[cfg(test)]
    pub(crate) fn breaker_count(&self) -> usize {
        let breakers = self.breakers.lock().unwrap();
        breakers.len()
    }

    #[cfg(test)]
    pub(crate) fn force_cleanup_now_for_test(&self) {
        let mut last = self.last_cleanup.lock().unwrap();
        *last = Instant::now()
            .checked_sub(Self::CLEANUP_INTERVAL + Duration::from_secs(1))
            .unwrap_or(Instant::now());
    }
}

/// Build a Google-style generateContent payload used by Gemini and Vertex AI.
#[cfg(any(feature = "provider-gemini", feature = "provider-vertexai"))]
pub(crate) fn build_google_generate_payload(
    messages: &[crate::traits::Message],
    options: &crate::traits::GenerationOptions,
) -> serde_json::Value {
    use crate::traits::MessageRole;

    // Collect system messages into a separate system_instruction field
    let system_parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .map(|m| m.text())
        .collect();

    let contents: Vec<_> = messages
        .iter()
        .filter(|m| m.role != MessageRole::System)
        .map(|message| {
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "model",
                MessageRole::System => unreachable!("system messages filtered above"),
            };
            json!({
                "role": role,
                "parts": [{ "text": message.text() }]
            })
        })
        .collect();

    let mut payload = serde_json::Map::new();
    payload.insert("contents".to_string(), json!(contents));

    if !system_parts.is_empty() {
        let combined = system_parts.join("\n");
        payload.insert(
            "system_instruction".to_string(),
            json!({ "parts": [{ "text": combined }] }),
        );
    }

    let mut generation_config = serde_json::Map::new();
    if let Some(temperature) = options.temperature {
        generation_config.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = options.top_p {
        generation_config.insert("topP".to_string(), json!(top_p));
    }
    if let Some(max_tokens) = options.max_tokens {
        generation_config.insert("maxOutputTokens".to_string(), json!(max_tokens));
    }
    if !generation_config.is_empty() {
        payload.insert(
            "generationConfig".to_string(),
            serde_json::Value::Object(generation_config),
        );
    }

    serde_json::Value::Object(payload)
}

#[cfg(test)]
mod embeddings_decoder_tests {
    use super::parse_openai_embeddings_response;
    use crate::error::RuntimeError;
    use serde_json::json;

    fn body(items: serde_json::Value) -> serde_json::Value {
        json!({ "object": "list", "data": items, "usage": { "prompt_tokens": 7, "total_tokens": 7 } })
    }

    #[test]
    fn lenient_skips_items_without_embedding_and_non_numeric_elements() {
        let b = body(json!([
            { "index": 0, "embedding": [1.0, "x", 2.0] },
            { "index": 1 },
        ]));
        let r = parse_openai_embeddings_response("T", &b, None).unwrap();
        assert_eq!(r.vectors, vec![vec![1.0, 2.0]]);
        let usage = r.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.total_tokens, 7);
        assert_eq!(usage.completion_tokens, 0);
    }

    #[test]
    fn lenient_missing_data_yields_empty() {
        let r = parse_openai_embeddings_response("T", &json!({}), None).unwrap();
        assert!(r.vectors.is_empty());
        assert!(r.usage.is_none());
    }

    #[test]
    fn strict_places_by_index_when_out_of_order() {
        let b = body(json!([
            { "index": 1, "embedding": [1.0] },
            { "index": 0, "embedding": [0.0] },
        ]));
        let r = parse_openai_embeddings_response("T", &b, Some(2)).unwrap();
        assert_eq!(r.vectors, vec![vec![0.0], vec![1.0]]);
    }

    #[test]
    fn strict_positional_when_index_missing() {
        let b = body(json!([{ "embedding": [1.0] }, { "embedding": [2.0] }]));
        let r = parse_openai_embeddings_response("T", &b, Some(2)).unwrap();
        assert_eq!(r.vectors, vec![vec![1.0], vec![2.0]]);
    }

    fn assert_api_error(r: crate::error::Result<crate::traits::EmbedResult>, needle: &str) {
        match r {
            Err(RuntimeError::ApiError(msg)) => {
                assert!(msg.contains(needle), "message {msg:?} lacks {needle:?}")
            }
            other => panic!("expected ApiError containing {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn strict_rejects_missing_data() {
        assert_api_error(
            parse_openai_embeddings_response("T", &json!({}), Some(1)),
            "missing 'data'",
        );
    }

    #[test]
    fn strict_rejects_count_mismatch() {
        let b = body(json!([{ "index": 0, "embedding": [1.0] }]));
        assert_api_error(
            parse_openai_embeddings_response("T", &b, Some(2)),
            "expected 2 embeddings, got 1",
        );
    }

    #[test]
    fn strict_rejects_non_numeric_element() {
        let b = body(json!([{ "index": 0, "embedding": [1.0, "x"] }]));
        assert_api_error(
            parse_openai_embeddings_response("T", &b, Some(1)),
            "is not a number",
        );
    }

    #[test]
    fn strict_rejects_item_without_embedding() {
        let b = body(json!([{ "index": 0 }]));
        assert_api_error(
            parse_openai_embeddings_response("T", &b, Some(1)),
            "no 'embedding' array",
        );
    }

    #[test]
    fn strict_rejects_duplicate_and_out_of_range_index() {
        let dup = body(json!([
            { "index": 0, "embedding": [1.0] },
            { "index": 0, "embedding": [2.0] },
        ]));
        assert_api_error(
            parse_openai_embeddings_response("T", &dup, Some(2)),
            "duplicate index 0",
        );
        let oob = body(json!([
            { "index": 0, "embedding": [1.0] },
            { "index": 5, "embedding": [2.0] },
        ]));
        assert_api_error(
            parse_openai_embeddings_response("T", &oob, Some(2)),
            "index 5 out of range",
        );
    }
}
