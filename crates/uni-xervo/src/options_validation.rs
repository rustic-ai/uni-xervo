//! Compile-time validation of provider-specific options JSON.
//!
//! Called during [`ModelRuntimeBuilder::build`](crate::runtime::ModelRuntimeBuilder::build) and
//! [`ModelRuntime::register`](crate::runtime::ModelRuntime::register) to reject
//! unknown or malformed options before any model loading occurs.

use crate::api::ModelTask;
use crate::error::{Result, RuntimeError};
use serde_json::Value;

/// Wire-format identifier for a [`ModelTask`] variant, matching the
/// `#[serde(rename_all = "snake_case")]` representation. Used in error
/// messages.
fn task_wire_name(task: ModelTask) -> &'static str {
    match task {
        ModelTask::Embed => "embed",
        ModelTask::Rerank => "rerank",
        ModelTask::Generate => "generate",
        ModelTask::Raw => "raw",
        ModelTask::EmbedImage => "embed_image",
        ModelTask::EmbedAudio => "embed_audio",
        ModelTask::EmbedMultimodal => "embed_multimodal",
        ModelTask::EmbedSparse => "embed_sparse",
        ModelTask::EmbedMultiVector => "embed_multi_vector",
        ModelTask::EmbedHybrid => "embed_hybrid",
        ModelTask::Nlp => "nlp",
        ModelTask::DocumentExtract => "document_extract",
        ModelTask::Transcribe => "transcribe",
        ModelTask::Ocr => "ocr",
    }
}

/// The option keys one task of a bundled provider accepts, on top of the
/// provider's [`ProviderOptions::common`] keys.
pub(crate) struct TaskOptions {
    pub(crate) task: ModelTask,
    pub(crate) keys: &'static [&'static str],
}

/// A bundled provider's supported tasks and the option keys each accepts.
///
/// This table is the one place the accepted `(provider, task)` pairs and
/// option keys are declared: [`validate_provider_options`] rejects anything
/// outside it, [`crate::schemas`] publishes it, and a test checks that every
/// file under `schemas/provider-options/` declares exactly these keys per
/// task and that each provider's `capabilities()` lists exactly these tasks.
/// Value rules (types, ranges, enums) live in the per-provider validators
/// below and are checked against the schemas by fixture cases.
pub(crate) struct ProviderOptions {
    pub(crate) id: &'static str,
    /// Keys every task of the provider accepts.
    pub(crate) common: &'static [&'static str],
    pub(crate) tasks: &'static [TaskOptions],
}

impl ProviderOptions {
    pub(crate) fn task(&self, task: ModelTask) -> Option<&TaskOptions> {
        self.tasks.iter().find(|t| t.task == task)
    }

    /// Tasks whose task-specific keys include `key`.
    fn tasks_with_key(&self, key: &str) -> Vec<ModelTask> {
        self.tasks
            .iter()
            .filter(|t| t.keys.contains(&key))
            .map(|t| t.task)
            .collect()
    }
}

const fn task(task: ModelTask, keys: &'static [&'static str]) -> TaskOptions {
    TaskOptions { task, keys }
}

const ONNX_COMMON: &[&str] = &[
    "artifact",
    "max_batch_size",
    "execution_providers",
    "graph_optimization_level",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "cache_dir",
];

/// Every bundled provider, in the order the schemas list them.
pub(crate) const BUNDLED_PROVIDERS: &[ProviderOptions] = &[
    ProviderOptions {
        id: "remote/openai",
        common: &["api_key_env", "base_url"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::Generate, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/mistral",
        common: &["api_key_env"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::Generate, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/voyageai",
        common: &["api_key_env"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::Rerank, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/gemini",
        common: &["api_key_env", "api_version"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::EmbedMultimodal, &[]),
            task(ModelTask::Generate, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/anthropic",
        common: &["api_key_env", "anthropic_version"],
        tasks: &[task(ModelTask::Generate, &[])],
    },
    ProviderOptions {
        id: "remote/cohere",
        common: &["api_key_env", "input_type"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::EmbedMultimodal, &[]),
            task(ModelTask::Generate, &[]),
            task(ModelTask::Rerank, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/azure-openai",
        common: &["api_key_env", "resource_name", "api_version"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::Generate, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/vertexai",
        common: &["api_token_env", "project_id", "location", "publisher"],
        tasks: &[
            task(ModelTask::Embed, &["embedding_dimensions"]),
            task(ModelTask::Generate, &[]),
        ],
    },
    ProviderOptions {
        id: "remote/llamacpp",
        common: &["base_url", "api_key_env", "request_timeout_secs"],
        tasks: &[
            task(
                ModelTask::Embed,
                &[
                    "tokenizer_base_url",
                    "max_input_tokens",
                    "embedding_dimensions",
                ],
            ),
            task(ModelTask::Generate, &[]),
            task(ModelTask::Rerank, &[]),
            task(ModelTask::DocumentExtract, &["style"]),
        ],
    },
    ProviderOptions {
        id: "local/candle",
        common: &["cache_dir"],
        tasks: &[task(ModelTask::Embed, &[])],
    },
    ProviderOptions {
        id: "local/onnx",
        common: ONNX_COMMON,
        tasks: &[
            task(
                ModelTask::Embed,
                &[
                    "tokenizer_path",
                    "pooling",
                    "normalize",
                    "dimensions",
                    "max_seq_len",
                    "token_type_ids",
                    "output_name",
                ],
            ),
            task(ModelTask::Rerank, &["max_seq_len", "style", "instruction"]),
            task(ModelTask::Raw, &[]),
            task(
                ModelTask::Nlp,
                &[
                    // Shared.
                    "tokenizer_path",
                    "generation",
                    // v1 cascade only.
                    "onnx_path",
                    "label_maps_path",
                    "max_seq_len",
                    // kniv v7 only.
                    "variant",
                    "calibration_path",
                    "label_bank_path",
                ],
            ),
            task(
                ModelTask::EmbedImage,
                &[
                    "onnx_path",
                    "image_size",
                    "dimensions",
                    "normalization",
                    "pool",
                    "normalize",
                    "output_name",
                ],
            ),
            task(
                ModelTask::Ocr,
                &[
                    "onnx_path",
                    "char_dict_path",
                    "image_height",
                    "image_width",
                    "normalization",
                    "blank_class",
                    "use_space_char",
                    "max_image_width",
                    "output_name",
                    // Optional DBNet detection stage.
                    "det_onnx_path",
                    "det_model_id",
                    "det_limit_side",
                    "det_bin_threshold",
                    "det_box_score_threshold",
                    "det_unclip_ratio",
                    "det_min_box_size",
                    "det_input_name",
                    "det_output_name",
                ],
            ),
            task(
                ModelTask::DocumentExtract,
                &["style", "onnx_path", "tokenizer_path", "max_seq_len"],
            ),
            task(
                ModelTask::EmbedSparse,
                &[
                    "tokenizer_path",
                    "sparse_method",
                    "output_name",
                    "output_index",
                    "max_seq_len",
                    "token_type_ids",
                    "top_k",
                ],
            ),
            task(
                ModelTask::EmbedMultiVector,
                &[
                    "tokenizer_path",
                    "dimensions",
                    "normalize",
                    "drop_special_tokens",
                    "output_name",
                    "output_index",
                    "max_seq_len",
                    "token_type_ids",
                ],
            ),
            // Hybrid is preset-driven: the preset declares each head's output
            // and recipe. Options only override pass-wide globals.
            task(
                ModelTask::EmbedHybrid,
                &["tokenizer_path", "max_seq_len", "token_type_ids", "top_k"],
            ),
        ],
    },
    ProviderOptions {
        id: "local/whisper-cpp",
        common: &["model_path", "default_language", "cache_dir"],
        tasks: &[task(ModelTask::Transcribe, &[])],
    },
];

/// The table entry for a bundled provider, or `None` for a third-party one.
pub(crate) fn bundled_provider(provider_id: &str) -> Option<&'static ProviderOptions> {
    BUNDLED_PROVIDERS.iter().find(|p| p.id == provider_id)
}

/// Quote and join task names for an error message: `'embed', 'rerank'`.
fn task_list(tasks: impl IntoIterator<Item = ModelTask>) -> String {
    tasks
        .into_iter()
        .map(|t| format!("'{}'", task_wire_name(t)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rewrite whole-number floats in `options` (`128.0`) as integers (`128`).
///
/// JSON does not distinguish the two and JSON Schema's `integer` accepts both,
/// but `serde_json` keeps `128.0` a float, so every integer option read with
/// `as_u64` would reject it. Tools that write numbers as floats (Python's
/// `json`, many YAML loaders) would pass the published schema and then fail at
/// startup. The runtime applies this once, where a spec enters it, so
/// validation, providers and the model cache key all see the same value.
/// Fractional floats are left alone.
pub(crate) fn normalize_whole_floats(options: &mut Value) {
    match options {
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if f.is_finite() && f.fract() == 0.0 {
                // `as` saturates, so check the round trip before converting.
                if f >= 0.0 && f <= u64::MAX as f64 && (f as u64) as f64 == f {
                    *options = Value::from(f as u64);
                } else if f < 0.0 && f >= i64::MIN as f64 && (f as i64) as f64 == f {
                    *options = Value::from(f as i64);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(normalize_whole_floats),
        Value::Object(map) => map.values_mut().for_each(normalize_whole_floats),
        _ => {}
    }
}

/// Validate provider-specific options for the given `provider_id` and `task`.
///
/// Returns `Ok(())` if the options are valid or the provider is unknown (unknown
/// providers are silently accepted to allow third-party extensions).
///
/// For a bundled provider, the task must be one it supports and every option
/// key one that task accepts (see [`BUNDLED_PROVIDERS`]); the provider's own
/// validator then checks the values.
pub fn validate_provider_options(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
) -> Result<()> {
    let Some(provider) = bundled_provider(provider_id) else {
        return Ok(());
    };
    let Some(task_options) = provider.task(task) else {
        return Err(RuntimeError::Config(format!(
            "Provider '{}' does not support task '{}'; it supports {}",
            provider_id,
            task_wire_name(task),
            task_list(provider.tasks.iter().map(|t| t.task)),
        )));
    };

    if let Value::Object(map) = options {
        for key in map.keys() {
            let key = key.as_str();
            if provider.common.contains(&key) || task_options.keys.contains(&key) {
                continue;
            }
            // Name the task a misplaced key belongs to, so the error says why
            // it does not fit rather than just "unknown".
            let owners = provider.tasks_with_key(key);
            if owners.is_empty() {
                return Err(RuntimeError::Config(format!(
                    "Unknown option '{}' for provider '{}'",
                    key, provider_id
                )));
            }
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' applies only to task{} {} (got '{}')",
                key,
                provider_id,
                if owners.len() == 1 { "" } else { "s" },
                task_list(owners),
                task_wire_name(task),
            )));
        }
    }

    match provider_id {
        "remote/openai" => validate_openai_options(provider_id, task, options),
        "remote/mistral" | "remote/voyageai" => {
            validate_with_embedding_dimensions(provider_id, task, options, &["api_key_env"])
        }
        "remote/gemini" => validate_with_embedding_dimensions(
            provider_id,
            task,
            options,
            &["api_key_env", "api_version"],
        ),
        "remote/anthropic" => {
            validate_string_keys_only(provider_id, options, &["api_key_env", "anthropic_version"])
        }
        "remote/cohere" => validate_with_embedding_dimensions(
            provider_id,
            task,
            options,
            &["api_key_env", "input_type"],
        ),
        "remote/azure-openai" => validate_azure_openai_options(provider_id, task, options),
        "remote/vertexai" => validate_vertexai_options(provider_id, task, options),
        "remote/llamacpp" => validate_llamacpp_options(provider_id, task, options),
        "local/candle" => validate_string_keys_only(provider_id, options, &["cache_dir"]),
        "local/onnx" => validate_local_onnx_options(provider_id, task, options),
        "local/whisper-cpp" => validate_whisper_cpp_options(provider_id, task, options),
        _ => Ok(()),
    }
}

/// Validate option values for `local/whisper-cpp`: all are strings.
fn validate_whisper_cpp_options(
    provider_id: &str,
    _task: ModelTask,
    options: &Value,
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    require_string_keys(
        provider_id,
        map,
        &["model_path", "default_language", "cache_dir"],
    )?;
    Ok(())
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

/// Validate the kniv generation and reject options meant for the other one.
///
/// The two generations of the kniv NLP family take different options
/// entirely: v1 names one graph and a 128-token cap, v7 reads its graph set
/// and window size from `onnx/onnx_meta.json` and takes a precision variant
/// instead. Mixing them is always a mistake, and catching it here means the
/// error arrives before a multi-gigabyte download rather than after.
fn validate_nlp_generation(provider_id: &str, map: &serde_json::Map<String, Value>) -> Result<()> {
    const V1_ONLY: [&str; 3] = ["onnx_path", "label_maps_path", "max_seq_len"];
    const V7_ONLY: [&str; 3] = ["variant", "calibration_path", "label_bank_path"];

    let generation = map.get("generation").and_then(Value::as_str);
    if let Some(g) = generation
        && !matches!(g, "v1" | "v7")
    {
        return Err(RuntimeError::Config(format!(
            "Option 'generation' for provider '{provider_id}' must be one of: v1, v7"
        )));
    }

    let declared_v7 = generation == Some("v7");
    let declared_v1 = generation == Some("v1");
    let has_v7_key = V7_ONLY.iter().find(|k| map.contains_key(**k));
    let has_v1_key = V1_ONLY.iter().find(|k| map.contains_key(**k));

    if let (Some(v7), Some(v1)) = (has_v7_key, has_v1_key) {
        return Err(RuntimeError::Config(format!(
            "Options '{v7}' (kniv v7) and '{v1}' (kniv v1 cascade) cannot both be set \
             for provider '{provider_id}'"
        )));
    }
    if declared_v7 && let Some(v1) = has_v1_key {
        return Err(RuntimeError::Config(format!(
            "Option '{v1}' does not apply to a kniv v7 model for provider \
             '{provider_id}'; v7 reads its graph set and window size from \
             onnx/onnx_meta.json"
        )));
    }
    if declared_v1 && let Some(v7) = has_v7_key {
        return Err(RuntimeError::Config(format!(
            "Option '{v7}' does not apply to the kniv v1 cascade for provider \
             '{provider_id}'"
        )));
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

/// Require that all specified keys, if present, are booleans.
fn require_bool_keys(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<()> {
    for key in keys {
        if let Some(value) = map.get(*key)
            && !value.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a boolean",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Require that the named key, if present, is a positive (> 0) integer that
/// also fits in `usize`. Several call sites downcast the validated value into
/// a `usize` field; without the upper bound, oversized values pass validation
/// here only to fail later in `serde_json::from_value` with a less actionable
/// error. On 64-bit targets `usize::MAX as u64 == u64::MAX`, so the upper
/// bound is a no-op there; on 32-bit it catches values above `u32::MAX`.
/// Validate that an optional option is a non-negative integer when present.
///
/// Unlike [`require_positive_u64`], this permits `0` — used for index-like
/// options (e.g. `output_index`) where the first element is a valid choice.
fn require_non_negative_u64(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key)
        && value.as_u64().is_none()
    {
        return Err(RuntimeError::Config(format!(
            "Option '{}' for provider '{}' must be a non-negative integer",
            key, provider_id
        )));
    }
    Ok(())
}

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
        if v > usize::MAX as u64 {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must not exceed {}",
                key,
                provider_id,
                usize::MAX
            )));
        }
    }
    Ok(())
}

/// If present, require `key` to be a number strictly inside `(0.0, 1.0)`.
fn require_unit_interval(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key) {
        let in_range = value.as_f64().is_some_and(|v| v > 0.0 && v < 1.0);
        if !in_range {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a number in (0.0, 1.0)",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// If present, require `key` to be a number strictly greater than `0.0`.
fn require_positive_f64(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key) {
        let positive = value.as_f64().is_some_and(|v| v > 0.0);
        if !positive {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a number greater than 0.0",
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
    require_string_keys(provider_id, map, allowed_keys)
}

/// Validate providers that accept string keys plus an optional
/// `embedding_dimensions` integer.
fn validate_with_embedding_dimensions(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
    string_keys: &[&str],
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    require_string_keys(provider_id, map, string_keys)?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate OpenAI options: `api_key_env`, optional `base_url`, and optional
/// `embedding_dimensions`. The `base_url` option lets users target OpenAI-
/// compatible servers (OpenRouter, vLLM, LM Studio, Ollama, internal proxies).
fn validate_openai_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    require_string_keys(provider_id, map, &["api_key_env", "base_url"])?;
    require_absolute_http_url(provider_id, map, "base_url")?;
    require_embedding_dimensions(provider_id, task, map)
}

/// If present, require `key` to be a non-empty absolute `http://` or
/// `https://` URL. Callers must have already checked the key is a string.
fn require_absolute_http_url(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key) {
        let trimmed = value.as_str().unwrap_or("").trim();
        if trimmed.is_empty() {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a non-empty URL",
                key, provider_id
            )));
        }
        if !(trimmed.starts_with("http://") || trimmed.starts_with("https://")) {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be an absolute http(s) URL",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Validate `remote/llamacpp` options.
///
/// `base_url` is required for every task: llama.cpp has no hosted default
/// endpoint, so there is nothing to fall back to. The rest is task-keyed.
///
/// `embed` additionally requires `max_input_tokens` and
/// `embedding_dimensions` — there is no model-id → dimensions table, and the
/// token budget drives the client-side truncation pre-pass. Those three keys
/// are embed-only and rejected elsewhere, rather than silently ignored.
///
/// `document_extract` takes one optional key, `style`, naming the output
/// dialect of the loaded document VLM so the right parser is applied. Its
/// value is checked here rather than at load, so a typo fails when the catalog
/// is read.
///
/// `generate` and `rerank` take no keys of their own. In particular there is
/// no option for vision, and none for the rerank response shape: the first
/// depends on whether `llama-server` was started with `--mmproj` and the
/// second follows from the request key this crate chooses. Both are properties
/// of the server process or the wire call, not things the catalog can know.
/// Accepted `style` values for `remote/llamacpp` `document_extract`.
///
/// Kept in step with `doc_parse::style_from_str` by hand: that module is
/// feature-gated on the providers that run a document VLM, and this validator
/// is not.
const LLAMACPP_DOC_STYLES: [&str; 3] = ["granite-docling", "mineru", "olmocr"];

fn validate_llamacpp_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let is_embed = matches!(task, ModelTask::Embed);

    let mut required: Vec<&str> = vec!["base_url"];
    if is_embed {
        required.extend(["max_input_tokens", "embedding_dimensions"]);
    }

    let Some(map) = as_object(provider_id, options)? else {
        return Err(RuntimeError::Config(format!(
            "Provider '{}' requires options: {}",
            provider_id,
            required.join(", ")
        )));
    };

    for key in required {
        if !map.contains_key(key) {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' is required",
                key, provider_id
            )));
        }
    }

    require_string_keys(
        provider_id,
        map,
        &["base_url", "tokenizer_base_url", "api_key_env", "style"],
    )?;
    require_absolute_http_url(provider_id, map, "base_url")?;
    require_absolute_http_url(provider_id, map, "tokenizer_base_url")?;
    require_positive_u64(provider_id, map, "request_timeout_secs")?;

    if let Some(style) = map.get("style").and_then(|v| v.as_str())
        && !LLAMACPP_DOC_STYLES.contains(&style)
    {
        return Err(RuntimeError::Config(format!(
            "Option 'style' for provider '{}' must be one of {} (got '{}')",
            provider_id,
            LLAMACPP_DOC_STYLES.join(", "),
            style
        )));
    }

    if is_embed {
        require_positive_u64(provider_id, map, "max_input_tokens")?;
        if map["max_input_tokens"].as_u64().unwrap_or(0) < 3 {
            return Err(RuntimeError::Config(format!(
                "Option 'max_input_tokens' for provider '{}' must be at least 3 \
                 (two special tokens plus one content token)",
                provider_id
            )));
        }
    }
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate Azure OpenAI options: string keys, `api_version`, and optional
/// `embedding_dimensions`.
fn validate_azure_openai_options(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    require_string_keys(
        provider_id,
        map,
        &["api_key_env", "resource_name", "api_version"],
    )?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate Vertex AI-specific options: string keys plus optional
/// `embedding_dimensions`.
fn validate_vertexai_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    require_string_keys(
        provider_id,
        map,
        &["api_token_env", "project_id", "location", "publisher"],
    )?;
    require_embedding_dimensions(provider_id, task, map)
}

fn validate_local_onnx_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };

    require_positive_u64(provider_id, map, "max_batch_size")?;
    require_positive_u64(provider_id, map, "inter_op_num_threads")?;
    require_positive_u64(provider_id, map, "intra_op_num_threads")?;
    require_string_keys(provider_id, map, &["artifact", "cache_dir"])?;

    if matches!(task, ModelTask::Nlp) {
        require_string_keys(
            provider_id,
            map,
            &[
                "onnx_path",
                "tokenizer_path",
                "label_maps_path",
                "variant",
                "calibration_path",
                "label_bank_path",
                "generation",
            ],
        )?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        validate_nlp_generation(provider_id, map)?;
    }

    if matches!(task, ModelTask::DocumentExtract) {
        require_string_keys(provider_id, map, &["style", "onnx_path", "tokenizer_path"])?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        if let Some(value) = map.get("style") {
            let s = value.as_str().unwrap_or("");
            if !matches!(s, "granite-docling" | "mineru" | "olmocr") {
                return Err(RuntimeError::Config(format!(
                    "Option 'style' for provider '{}' must be one of: granite-docling, mineru, olmocr",
                    provider_id
                )));
            }
        }
    }

    if matches!(task, ModelTask::Ocr) {
        require_string_keys(
            provider_id,
            map,
            &[
                "onnx_path",
                "char_dict_path",
                "normalization",
                "output_name",
            ],
        )?;
        require_positive_u64(provider_id, map, "image_height")?;
        require_positive_u64(provider_id, map, "image_width")?;
        require_positive_u64(provider_id, map, "max_image_width")?;
        require_non_negative_u64(provider_id, map, "blank_class")?;
        if let Some(value) = map.get("use_space_char")
            && !value.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'use_space_char' for provider '{}' must be a boolean",
                provider_id
            )));
        }
        if let Some(value) = map.get("normalization") {
            let s = value.as_str().unwrap_or("");
            if !matches!(s, "siglip" | "imagenet") {
                return Err(RuntimeError::Config(format!(
                    "Option 'normalization' for provider '{}' must be one of: siglip, imagenet",
                    provider_id
                )));
            }
        }
        // Detection-stage keys are checked even when `det_onnx_path` is
        // absent: a bad value is a mistake whether or not the stage is on.
        require_string_keys(
            provider_id,
            map,
            &[
                "det_onnx_path",
                "det_model_id",
                "det_input_name",
                "det_output_name",
            ],
        )?;
        require_positive_u64(provider_id, map, "det_limit_side")?;
        require_positive_u64(provider_id, map, "det_min_box_size")?;
        require_unit_interval(provider_id, map, "det_bin_threshold")?;
        require_unit_interval(provider_id, map, "det_box_score_threshold")?;
        require_positive_f64(provider_id, map, "det_unclip_ratio")?;
    }

    if matches!(task, ModelTask::EmbedImage) {
        require_string_keys(
            provider_id,
            map,
            &["onnx_path", "normalization", "pool", "output_name"],
        )?;
        require_positive_u64(provider_id, map, "image_size")?;
        require_positive_u64(provider_id, map, "dimensions")?;
        if let Some(value) = map.get("normalize")
            && !value.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'normalize' for provider '{}' must be a boolean",
                provider_id
            )));
        }
        if let Some(value) = map.get("normalization") {
            let s = value.as_str().unwrap_or("");
            if !matches!(s, "siglip" | "imagenet") {
                return Err(RuntimeError::Config(format!(
                    "Option 'normalization' for provider '{}' must be one of: siglip, imagenet",
                    provider_id
                )));
            }
        }
        if let Some(value) = map.get("pool") {
            let s = value.as_str().unwrap_or("");
            if !matches!(s, "none" | "mean") {
                return Err(RuntimeError::Config(format!(
                    "Option 'pool' for provider '{}' must be one of: none, mean",
                    provider_id
                )));
            }
        }
    }

    if matches!(task, ModelTask::Embed) {
        require_string_keys(
            provider_id,
            map,
            &["tokenizer_path", "pooling", "output_name"],
        )?;
        require_positive_u64(provider_id, map, "dimensions")?;
        require_positive_u64(provider_id, map, "max_seq_len")?;

        if let Some(pooling) = map.get("pooling") {
            let Some(pooling) = pooling.as_str() else {
                return Err(RuntimeError::Config(format!(
                    "Option 'pooling' for provider '{}' must be a string",
                    provider_id
                )));
            };
            match pooling {
                "cls" | "mean" | "max" | "last-token" => {}
                _ => {
                    return Err(RuntimeError::Config(format!(
                        "Option 'pooling' for provider '{}' must be one of cls, mean, max, last-token",
                        provider_id
                    )));
                }
            }
        }

        if let Some(v) = map.get("normalize")
            && !v.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'normalize' for provider '{}' must be a boolean",
                provider_id
            )));
        }
        if let Some(v) = map.get("token_type_ids")
            && !v.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'token_type_ids' for provider '{}' must be a boolean",
                provider_id
            )));
        }
    }

    if matches!(task, ModelTask::EmbedSparse) {
        require_string_keys(provider_id, map, &["tokenizer_path", "output_name"])?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        require_positive_u64(provider_id, map, "top_k")?;
        require_non_negative_u64(provider_id, map, "output_index")?;

        // `sparse_method` selects the post-processing recipe: SPLADE-style MLM
        // logits (term expansion over the full vocab) vs BGE-M3 lexical weights
        // (re-weighting input token ids). The two are not interchangeable.
        if let Some(method) = map.get("sparse_method") {
            let Some(method) = method.as_str() else {
                return Err(RuntimeError::Config(format!(
                    "Option 'sparse_method' for provider '{}' must be a string",
                    provider_id
                )));
            };
            if !matches!(method, "mlm" | "lexical") {
                return Err(RuntimeError::Config(format!(
                    "Option 'sparse_method' for provider '{}' must be one of: mlm, lexical",
                    provider_id
                )));
            }
        }

        if let Some(v) = map.get("token_type_ids")
            && !v.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'token_type_ids' for provider '{}' must be a boolean",
                provider_id
            )));
        }
    }

    if matches!(task, ModelTask::EmbedMultiVector) {
        require_string_keys(provider_id, map, &["tokenizer_path", "output_name"])?;
        require_positive_u64(provider_id, map, "dimensions")?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        require_non_negative_u64(provider_id, map, "output_index")?;

        for key in ["normalize", "drop_special_tokens", "token_type_ids"] {
            if let Some(v) = map.get(key)
                && !v.is_boolean()
            {
                return Err(RuntimeError::Config(format!(
                    "Option '{}' for provider '{}' must be a boolean",
                    key, provider_id
                )));
            }
        }
    }

    if matches!(task, ModelTask::EmbedHybrid) {
        require_string_keys(provider_id, map, &["tokenizer_path"])?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        require_positive_u64(provider_id, map, "top_k")?;
        require_bool_keys(provider_id, map, &["token_type_ids"])?;
    }

    if matches!(task, ModelTask::Rerank) {
        require_string_keys(provider_id, map, &["style", "instruction"])?;
        require_positive_u64(provider_id, map, "max_seq_len")?;
        if let Some(style) = map.get("style").and_then(|v| v.as_str()) {
            match style {
                "cross-encoder" | "generative" => {}
                _ => {
                    return Err(RuntimeError::Config(format!(
                        "Option 'style' for provider '{}' must be one of cross-encoder, generative",
                        provider_id
                    )));
                }
            }
        }
    }

    if let Some(level) = map.get("graph_optimization_level") {
        let Some(level) = level.as_str() else {
            return Err(RuntimeError::Config(format!(
                "Option 'graph_optimization_level' for provider '{}' must be a string",
                provider_id
            )));
        };
        match level {
            "disable" | "basic" | "extended" | "all" => {}
            _ => {
                return Err(RuntimeError::Config(format!(
                    "Option 'graph_optimization_level' for provider '{}' must be one of disable, basic, extended, all",
                    provider_id
                )));
            }
        }
    }

    if let Some(execution_providers) = map.get("execution_providers") {
        validate_execution_provider_array(
            provider_id,
            execution_providers,
            &["cpu", "cuda", "coreml", "directml"],
        )?;
    }

    Ok(())
}

fn validate_execution_provider_array(
    provider_id: &str,
    value: &Value,
    allowed: &[&str],
) -> Result<()> {
    let Some(values) = value.as_array() else {
        return Err(RuntimeError::Config(format!(
            "Option 'execution_providers' for provider '{}' must be an array of strings",
            provider_id
        )));
    };
    for item in values {
        match item.as_str() {
            Some(name) if allowed.contains(&name) => {}
            Some(other) => {
                return Err(RuntimeError::Config(format!(
                    "Execution provider '{}' is not supported for provider '{}'",
                    other, provider_id
                )));
            }
            None => {
                return Err(RuntimeError::Config(format!(
                    "Option 'execution_providers' for provider '{}' must be an array of strings",
                    provider_id
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn whole_floats_become_integers() {
        let mut v = json!({
            "max_seq_len": 128.0,
            "neg": -2.0,
            "ratio": 1.5,
            "big": 1e300,
            "nested": [3.0, {"k": 4.0}],
            "already": 7,
            "s": "8.0"
        });
        normalize_whole_floats(&mut v);
        assert_eq!(
            v,
            json!({
                "max_seq_len": 128,
                "neg": -2,
                "ratio": 1.5,
                "big": 1e300,
                "nested": [3, {"k": 4}],
                "already": 7,
                "s": "8.0"
            })
        );
        assert!(v["max_seq_len"].is_u64());
    }

    #[test]
    fn validation_accepts_whole_floats_once_normalized() {
        let mut options = json!({"embedding_dimensions": 256.0});
        assert!(validate_provider_options("remote/openai", ModelTask::Embed, &options).is_err());
        normalize_whole_floats(&mut options);
        assert!(validate_provider_options("remote/openai", ModelTask::Embed, &options).is_ok());
    }
}
