//! Public API types for configuring models, catalogs, and runtime behavior.

use crate::error::{Result, RuntimeError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The kind of inference task a model performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// Produce dense vector embeddings from text.
    Embed,
    /// Re-score a set of documents against a query.
    Rerank,
    /// Generate text (chat completions, summarization, etc.).
    Generate,
}

/// Controls when a model or provider is initialized during runtime startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WarmupPolicy {
    /// Load immediately during [`ModelRuntime::builder().build()`](crate::runtime::ModelRuntimeBuilder::build).
    /// Startup blocks until the load completes (or fails).
    Eager,
    /// Defer loading until the first inference request. This is the default.
    #[default]
    Lazy,
    /// Spawn loading in a background task at startup. Inference calls that arrive
    /// before loading finishes will trigger a blocking wait.
    Background,
}

impl std::fmt::Display for WarmupPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eager => write!(f, "eager"),
            Self::Lazy => write!(f, "lazy"),
            Self::Background => write!(f, "background"),
        }
    }
}

/// Declarative specification that maps a human-readable alias to a concrete
/// provider and model.
///
/// A model catalog is a `Vec<ModelAliasSpec>` — either built programmatically or
/// parsed from JSON with [`catalog_from_str`] / [`catalog_from_file`].
///
/// # Example JSON
///
/// ```json
/// {
///   "alias": "embed/default",
///   "task": "embed",
///   "provider_id": "local/candle",
///   "model_id": "sentence-transformers/all-MiniLM-L6-v2",
///   "warmup": "lazy"
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAliasSpec {
    /// Human-readable name used to request this model (e.g. `"embed/default"`).
    /// Must contain a `/` separator.
    pub alias: String,
    /// The inference task this alias targets.
    pub task: ModelTask,
    /// Identifier of the provider that will load this model (e.g. `"local/candle"`,
    /// `"remote/openai"`).
    pub provider_id: String,
    /// Model identifier understood by the provider — typically a HuggingFace repo ID
    /// for local providers or an API model name for remote providers.
    pub model_id: String,
    /// Optional HuggingFace revision (branch, tag, or commit hash).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// When this model should be initialized. Defaults to [`WarmupPolicy::Lazy`].
    #[serde(default)]
    pub warmup: WarmupPolicy,
    /// If `true`, a failed eager warmup aborts runtime startup. Defaults to `false`.
    #[serde(default)]
    pub required: bool,
    /// Per-inference timeout in seconds. `None` means no timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    /// Model load timeout in seconds. Defaults to 600 s if unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_timeout: Option<u64>,
    /// Retry configuration for transient inference failures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,
    /// Provider-specific options (e.g. `{"isq": "Q4K"}` for mistral.rs,
    /// `{"api_key_env": "MY_KEY"}` for remote providers). Defaults to `{}`.
    #[serde(default)]
    pub options: serde_json::Value,
}

/// Configuration for exponential-backoff retries on transient inference errors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the initial call).
    pub max_attempts: u32,
    /// Base delay in milliseconds; doubled on each subsequent attempt.
    pub initial_backoff_ms: u64,
}

impl RetryConfig {
    /// Compute the backoff duration for the given 1-based `attempt` number.
    ///
    /// Uses `initial_backoff_ms * 2^(attempt - 1)` with saturating arithmetic.
    pub fn get_backoff(&self, attempt: u32) -> std::time::Duration {
        std::time::Duration::from_millis(
            self.initial_backoff_ms * 2u64.pow(attempt.saturating_sub(1)),
        )
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 100,
        }
    }
}

/// Deduplication key used by the runtime to share a single loaded model instance
/// across multiple aliases that point to the same provider, model, revision, and
/// options.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelRuntimeKey {
    /// The task type (embed, rerank, generate).
    pub task: ModelTask,
    /// Provider that owns this model instance.
    pub provider_id: String,
    /// Model identifier within the provider.
    pub model_id: String,
    /// Optional HuggingFace revision.
    pub revision: Option<String>,
    /// Hash of the provider-specific options JSON. Two specs with semantically
    /// equivalent options (same keys/values, any object-key order) produce the
    /// same hash.
    pub variant_hash: u64,
}

impl ModelRuntimeKey {
    /// Derive a runtime key from a [`ModelAliasSpec`], hashing the options JSON
    /// in a key-order-independent manner.
    pub fn new(spec: &ModelAliasSpec) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::Hasher;

        // Hash all JSON option shapes with deterministic key ordering.
        // This avoids collisions for non-object values while preserving
        // object-order independence for semantically equivalent JSON.
        hash_json_value(&spec.options, &mut hasher);

        Self {
            task: spec.task,
            provider_id: spec.provider_id.clone(),
            model_id: spec.model_id.clone(),
            revision: spec.revision.clone(),
            variant_hash: hasher.finish(),
        }
    }
}

/// Recursively hash a JSON value in a deterministic, key-order-independent way.
///
/// Each JSON variant is prefixed with a unique discriminant byte to avoid
/// collisions between structurally different values (e.g. `null` vs `false`).
/// Object keys are sorted before hashing so that `{"a":1,"b":2}` and
/// `{"b":2,"a":1}` produce the same hash.
fn hash_json_value<H: std::hash::Hasher>(value: &serde_json::Value, hasher: &mut H) {
    use std::hash::Hash;

    match value {
        serde_json::Value::Null => {
            0u8.hash(hasher);
        }
        serde_json::Value::Bool(v) => {
            1u8.hash(hasher);
            v.hash(hasher);
        }
        serde_json::Value::Number(v) => {
            2u8.hash(hasher);
            v.to_string().hash(hasher);
        }
        serde_json::Value::String(v) => {
            3u8.hash(hasher);
            v.hash(hasher);
        }
        serde_json::Value::Array(values) => {
            4u8.hash(hasher);
            values.len().hash(hasher);
            for v in values {
                hash_json_value(v, hasher);
            }
        }
        serde_json::Value::Object(map) => {
            5u8.hash(hasher);
            map.len().hash(hasher);

            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            for (k, v) in entries {
                k.hash(hasher);
                hash_json_value(v, hasher);
            }
        }
    }
}

impl ModelAliasSpec {
    /// Validate invariants: alias must be non-empty and contain a `'/'`, timeouts
    /// must be non-zero when set.
    pub fn validate(&self) -> Result<()> {
        if self.alias.is_empty() {
            return Err(RuntimeError::Config("Alias cannot be empty".to_string()));
        }
        if !self.alias.contains('/') {
            return Err(RuntimeError::Config(format!(
                "Alias '{}' must be in 'task/name' format",
                self.alias
            )));
        }
        if self.timeout == Some(0) {
            return Err(RuntimeError::Config(
                "Inference timeout must be greater than 0".to_string(),
            ));
        }
        if self.load_timeout == Some(0) {
            return Err(RuntimeError::Config(
                "Load timeout must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Parse a single `ModelAliasSpec` from a JSON value.
    pub fn from_json(value: serde_json::Value) -> Result<Self> {
        let spec: Self = serde_json::from_value(value)
            .map_err(|e| RuntimeError::Config(format!("Invalid ModelAliasSpec JSON: {}", e)))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Parse a single `ModelAliasSpec` from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let spec: Self = serde_json::from_str(s)
            .map_err(|e| RuntimeError::Config(format!("Invalid ModelAliasSpec JSON: {}", e)))?;
        spec.validate()?;
        Ok(spec)
    }
}

/// Parse a catalog (array) of `ModelAliasSpec` from a JSON string.
pub fn catalog_from_str(s: &str) -> Result<Vec<ModelAliasSpec>> {
    let specs: Vec<ModelAliasSpec> = serde_json::from_str(s)
        .map_err(|e| RuntimeError::Config(format!("Invalid catalog JSON: {}", e)))?;
    for spec in &specs {
        spec.validate()?;
    }
    Ok(specs)
}

/// Read and parse a catalog from a JSON file.
///
/// The file must contain a JSON array of model alias specs.
pub fn catalog_from_file(path: impl AsRef<Path>) -> Result<Vec<ModelAliasSpec>> {
    let path = path.as_ref();
    let contents = std::fs::read_to_string(path).map_err(|e| {
        RuntimeError::Config(format!(
            "Failed to read catalog file '{}': {}",
            path.display(),
            e
        ))
    })?;
    catalog_from_str(&contents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const VALID_JSON: &str = r#"{
        "alias": "embed/default",
        "task": "embed",
        "provider_id": "local/candle",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2"
    }"#;

    const VALID_CATALOG_JSON: &str = r#"[
        {
            "alias": "embed/default",
            "task": "embed",
            "provider_id": "local/candle",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2"
        },
        {
            "alias": "chat/fast",
            "task": "generate",
            "provider_id": "local/mistralrs",
            "model_id": "mistralai/Mistral-7B-v0.1",
            "warmup": "background",
            "required": false,
            "options": { "isq": "Q4K" }
        }
    ]"#;

    #[test]
    fn from_json_str_parses_valid_spec() {
        let spec = ModelAliasSpec::from_json_str(VALID_JSON).unwrap();
        assert_eq!(spec.alias, "embed/default");
        assert_eq!(spec.task, ModelTask::Embed);
        assert_eq!(spec.provider_id, "local/candle");
        assert_eq!(spec.warmup, WarmupPolicy::Lazy); // default
        assert!(!spec.required); // default
    }

    #[test]
    fn from_json_value_parses_valid_spec() {
        let value = json!({
            "alias": "embed/fast",
            "task": "embed",
            "provider_id": "local/fastembed",
            "model_id": "BAAI/bge-small-en-v1.5",
            "required": true,
            "warmup": "eager"
        });
        let spec = ModelAliasSpec::from_json(value).unwrap();
        assert_eq!(spec.alias, "embed/fast");
        assert_eq!(spec.warmup, WarmupPolicy::Eager);
        assert!(spec.required);
    }

    #[test]
    fn from_json_str_rejects_missing_slash_in_alias() {
        let json = r#"{"alias":"noSlash","task":"embed","provider_id":"x","model_id":"y"}"#;
        assert!(ModelAliasSpec::from_json_str(json).is_err());
    }

    #[test]
    fn from_json_str_rejects_invalid_json() {
        assert!(ModelAliasSpec::from_json_str("{not valid}").is_err());
    }

    #[test]
    fn catalog_from_str_parses_array() {
        let specs = catalog_from_str(VALID_CATALOG_JSON).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].alias, "embed/default");
        assert_eq!(specs[1].alias, "chat/fast");
        assert_eq!(specs[1].options["isq"], "Q4K");
    }

    #[test]
    fn catalog_from_str_rejects_invalid_spec() {
        let json = r#"[{"alias":"bad","task":"embed","provider_id":"x","model_id":"y"}]"#;
        assert!(catalog_from_str(json).is_err()); // alias has no '/'
    }

    #[test]
    fn catalog_from_file_reads_and_parses() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_catalog.json");
        std::fs::write(&path, VALID_CATALOG_JSON).unwrap();
        let specs = catalog_from_file(&path).unwrap();
        assert_eq!(specs.len(), 2);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn catalog_from_file_errors_on_missing_file() {
        assert!(catalog_from_file("/nonexistent/path/catalog.json").is_err());
    }

    #[test]
    fn runtime_key_distinguishes_non_object_options() {
        let mut spec_null = ModelAliasSpec::from_json_str(VALID_JSON).unwrap();
        spec_null.options = serde_json::Value::Null;

        let mut spec_bool = spec_null.clone();
        spec_bool.options = json!(true);

        let mut spec_array = spec_null.clone();
        spec_array.options = json!(["a", 1]);

        let key_null = ModelRuntimeKey::new(&spec_null);
        let key_bool = ModelRuntimeKey::new(&spec_bool);
        let key_array = ModelRuntimeKey::new(&spec_array);

        assert_ne!(key_null, key_bool);
        assert_ne!(key_null, key_array);
        assert_ne!(key_bool, key_array);
    }

    #[test]
    fn runtime_key_nested_option_order_independence() {
        let mut spec1 = ModelAliasSpec::from_json_str(VALID_JSON).unwrap();
        spec1.options = json!({
            "outer": {
                "b": [3, 2, 1],
                "a": {"y": 2, "x": 1}
            }
        });

        let mut spec2 = ModelAliasSpec::from_json_str(VALID_JSON).unwrap();
        spec2.options = json!({
            "outer": {
                "a": {"x": 1, "y": 2},
                "b": [3, 2, 1]
            }
        });

        let key1 = ModelRuntimeKey::new(&spec1);
        let key2 = ModelRuntimeKey::new(&spec2);
        assert_eq!(key1, key2);
    }
}
