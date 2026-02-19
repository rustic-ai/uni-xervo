# Configuration

Catalogs are JSON arrays of `ModelAliasSpec` entries.

## Canonical schema shape

```json
[
  {
    "alias": "embed/default",
    "task": "embed",
    "provider_id": "local/candle",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": null,
    "warmup": "lazy",
    "required": false,
    "timeout": 10,
    "load_timeout": 600,
    "retry": {
      "max_attempts": 3,
      "initial_backoff_ms": 100
    },
    "options": null
  }
]
```

## Field constraints

- `alias`: string matching `.+/.+`.
- `task`: one of `embed`, `rerank`, `generate`.
- `warmup`: one of `eager`, `lazy`, `background`.
- `timeout`, `load_timeout`: integer >= 1.
- `retry.max_attempts`, `retry.initial_backoff_ms`: integer >= 1.
- `options`: object or null, strict provider-specific keys only.

## Provider options reference

| Provider ID | Allowed option keys | Notes |
| --- | --- | --- |
| `local/candle` | `cache_dir` | Per-model local cache path |
| `local/fastembed` | `cache_dir` | Per-model local cache path |
| `local/mistralrs` | `isq`, `force_cpu`, `paged_attention`, `max_num_seqs`, `chat_template`, `tokenizer_json`, `embedding_dimensions`, `gguf_files` | Quantization and local runtime tuning |
| `remote/openai` | `api_key_env` | Override env var name for API key |
| `remote/gemini` | `api_key_env` | Override env var name for API key |
| `remote/vertexai` | `api_token_env`, `project_id`, `location`, `publisher`, `embedding_dimensions` | OAuth token + project/location metadata |
| `remote/mistral` | `api_key_env` | Override env var name for API key |
| `remote/anthropic` | `api_key_env`, `anthropic_version` | `anthropic_version` defaults to `2023-06-01` |
| `remote/voyageai` | `api_key_env` | Override env var name for API key |
| `remote/cohere` | `api_key_env`, `input_type` | `input_type` used for embedding mode |
| `remote/azure-openai` | `api_key_env`, `resource_name`, `api_version` | `resource_name` required; `api_version` default `2024-10-21` |

Provider-specific model/config links:

- [Provider Reference Pages](providers/index.md)

## Runtime builder paths

- Programmatic catalog: `.catalog(Vec<ModelAliasSpec>)`
- JSON string catalog: `.catalog_from_str(&str)`
- JSON file catalog: `.catalog_from_file(path)`

## Helpful APIs

- `runtime.contains_alias(alias)`
- `runtime.prefetch_all()`
- `runtime.prefetch(&[aliases])`
- `runtime.embedding(alias)`
- `runtime.reranker(alias)`
- `runtime.generator(alias)`
