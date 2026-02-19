//! Shared utilities for all remote (HTTP API) providers: HTTP status mapping,
//! API key resolution, circuit breaker management, and Google-style payload
//! construction.

use crate::api::{ModelAliasSpec, ModelRuntimeKey};
use crate::error::{Result, RuntimeError};
use crate::reliability::{CircuitBreakerConfig, CircuitBreakerWrapper};
use reqwest::Client;
#[cfg(any(feature = "provider-gemini", feature = "provider-vertexai"))]
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Map an HTTP response status to a `RuntimeError` for non-success codes.
/// Returns `Ok(response)` when the status is 2xx.
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
///
/// Messages alternate roles: even indices are `"user"`, odd are `"model"`.
#[cfg(any(feature = "provider-gemini", feature = "provider-vertexai"))]
pub(crate) fn build_google_generate_payload(
    messages: &[String],
    options: &crate::traits::GenerationOptions,
) -> serde_json::Value {
    let contents: Vec<_> = messages
        .iter()
        .enumerate()
        .map(|(i, message)| {
            let role = if i % 2 == 0 { "user" } else { "model" };
            json!({
                "role": role,
                "parts": [{ "text": message }]
            })
        })
        .collect();

    let mut payload = serde_json::Map::new();
    payload.insert("contents".to_string(), json!(contents));

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
