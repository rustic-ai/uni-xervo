//! Compile-time validation of provider-specific options JSON.
//!
//! Called during [`ModelRuntimeBuilder::build`](crate::runtime::ModelRuntimeBuilder::build) and
//! [`ModelRuntime::register`](crate::runtime::ModelRuntime::register) to reject
//! unknown or malformed options before any model loading occurs.

use crate::api::ModelTask;
use crate::error::{Result, RuntimeError};
use serde_json::Value;

/// Validate provider-specific options for the given `provider_id` and `task`.
///
/// Returns `Ok(())` if the options are valid or the provider is unknown (unknown
/// providers are silently accepted to allow third-party extensions).
pub fn validate_provider_options(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
) -> Result<()> {
    match provider_id {
        "remote/openai" | "remote/gemini" | "remote/mistral" | "remote/voyageai" => {
            validate_string_keys_only(provider_id, options, &["api_key_env"])
        }
        "remote/anthropic" => {
            validate_string_keys_only(provider_id, options, &["api_key_env", "anthropic_version"])
        }
        "remote/cohere" => {
            validate_string_keys_only(provider_id, options, &["api_key_env", "input_type"])
        }
        "remote/azure-openai" => validate_string_keys_only(
            provider_id,
            options,
            &["api_key_env", "resource_name", "api_version"],
        ),
        "remote/vertexai" => validate_vertexai_options(provider_id, task, options),
        "local/candle" | "local/fastembed" => {
            validate_string_keys_only(provider_id, options, &["cache_dir"])
        }
        "local/mistralrs" => validate_mistralrs_options(provider_id, task, options),
        _ => Ok(()),
    }
}

/// Parse `options` as a JSON object map, returning `None` for null and an
/// error for non-object types.
fn as_object<'a>(
    provider_id: &str,
    options: &'a Value,
) -> Result<Option<&'a serde_json::Map<String, Value>>> {
    match options {
        Value::Null => Ok(None),
        Value::Object(map) => Ok(Some(map)),
        _ => Err(RuntimeError::Config(format!(
            "Options for provider '{}' must be a JSON object or null",
            provider_id
        ))),
    }
}

/// Return an error if `map` contains any key not in `allowed`.
fn reject_unknown_keys(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(RuntimeError::Config(format!(
                "Unknown option '{}' for provider '{}'",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Require that all specified keys, if present, are strings.
fn require_string_keys(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<()> {
    for key in keys {
        if let Some(value) = map.get(*key)
            && !value.is_string()
        {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a string",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Require that the named key, if present, is a positive (> 0) integer.
fn require_positive_u64(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key) {
        let Some(v) = value.as_u64() else {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a positive integer",
                key, provider_id
            )));
        };
        if v == 0 {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be greater than 0",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Validate that the embedding_dimensions option is a positive integer and only
/// used for embed tasks.
fn require_embedding_dimensions(
    provider_id: &str,
    task: ModelTask,
    map: &serde_json::Map<String, Value>,
) -> Result<()> {
    if map.contains_key("embedding_dimensions") {
        require_positive_u64(provider_id, map, "embedding_dimensions")?;
        if task != ModelTask::Embed {
            return Err(RuntimeError::Config(
                "Option 'embedding_dimensions' is only valid for embed tasks".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate providers whose options are all optional string keys.
fn validate_string_keys_only(
    provider_id: &str,
    options: &Value,
    allowed_keys: &[&str],
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(provider_id, map, allowed_keys)?;
    require_string_keys(provider_id, map, allowed_keys)
}

/// Validate Vertex AI-specific options: string keys plus optional
/// `embedding_dimensions`.
fn validate_vertexai_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(
        provider_id,
        map,
        &[
            "api_token_env",
            "project_id",
            "location",
            "publisher",
            "embedding_dimensions",
        ],
    )?;
    require_string_keys(
        provider_id,
        map,
        &["api_token_env", "project_id", "location", "publisher"],
    )?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate mistral.rs-specific options: ISQ type, boolean flags, GGUF files,
/// and embedding dimensions.
fn validate_mistralrs_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };

    reject_unknown_keys(
        provider_id,
        map,
        &[
            "isq",
            "force_cpu",
            "paged_attention",
            "max_num_seqs",
            "chat_template",
            "tokenizer_json",
            "embedding_dimensions",
            "gguf_files",
        ],
    )?;

    require_string_keys(
        provider_id,
        map,
        &["isq", "chat_template", "tokenizer_json"],
    )?;

    for key in ["force_cpu", "paged_attention"] {
        if let Some(value) = map.get(key)
            && !value.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a boolean",
                key, provider_id
            )));
        }
    }

    require_positive_u64(provider_id, map, "max_num_seqs")?;
    require_embedding_dimensions(provider_id, task, map)?;

    if let Some(value) = map.get("gguf_files") {
        let Some(items) = value.as_array() else {
            return Err(RuntimeError::Config(format!(
                "Option 'gguf_files' for provider '{}' must be an array of strings",
                provider_id
            )));
        };
        if items.iter().any(|item| !item.is_string()) {
            return Err(RuntimeError::Config(format!(
                "Option 'gguf_files' for provider '{}' must be an array of strings",
                provider_id
            )));
        }
    }

    Ok(())
}
