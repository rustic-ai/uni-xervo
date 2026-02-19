# Uni-Xervo

Uni-Xervo is a unified Rust runtime for model serving across local and remote providers. It gives you one catalog-driven API for embeddings, reranking, and text generation.

## What you get

- Alias-based model resolution (`task/name`) instead of hardcoded provider model IDs.
- A single runtime for local and hosted providers.
- Typed task APIs:
  - `EmbeddingModel`
  - `RerankerModel`
  - `GeneratorModel`
- Reliability controls per alias:
  - inference timeout (`timeout`)
  - load timeout (`load_timeout`)
  - retry policy (`retry`)
- Strict provider option validation with JSON Schema support.

## Capability matrix

| Provider ID | Type | Embed | Rerank | Generate | Default auth env | Key options |
| --- | --- | --- | --- | --- | --- | --- |
| `local/candle` | local | Yes | No | No | N/A | `cache_dir` |
| `local/fastembed` | local | Yes | No | No | N/A | `cache_dir` |
| `local/mistralrs` | local | Yes | No | Yes | N/A | `isq`, `force_cpu`, `paged_attention`, `max_num_seqs`, `chat_template`, `tokenizer_json`, `embedding_dimensions`, `gguf_files` |
| `remote/openai` | remote | Yes | No | Yes | `OPENAI_API_KEY` | `api_key_env` |
| `remote/gemini` | remote | Yes | No | Yes | `GEMINI_API_KEY` | `api_key_env` |
| `remote/vertexai` | remote | Yes | No | Yes | `VERTEX_AI_TOKEN` | `api_token_env`, `project_id`, `location`, `publisher`, `embedding_dimensions` |
| `remote/mistral` | remote | Yes | No | Yes | `MISTRAL_API_KEY` | `api_key_env` |
| `remote/anthropic` | remote | No | No | Yes | `ANTHROPIC_API_KEY` | `api_key_env`, `anthropic_version` |
| `remote/voyageai` | remote | Yes | Yes | No | `VOYAGE_API_KEY` | `api_key_env` |
| `remote/cohere` | remote | Yes | Yes | Yes | `CO_API_KEY` | `api_key_env`, `input_type` |
| `remote/azure-openai` | remote | Yes | No | Yes | `AZURE_OPENAI_API_KEY` | `api_key_env`, `resource_name`, `api_version` |

## User developer view

For application developers, the main contract is:

1. Build a catalog of `ModelAliasSpec` entries.
2. Register providers with `ModelRuntime::builder()`.
3. Resolve typed handles by alias.
4. Call `embed`, `rerank`, or `generate` without provider-specific branching in your app logic.

## Framework developer view

For platform and library contributors, important implementation concepts are:

- Runtime key deduplication for shared model instances.
- Per-key load mutexes to prevent duplicate concurrent loads.
- Provider and model warmup policies (`eager`, `lazy`, `background`).
- Instrumented wrappers that enforce timeout/retry and emit metrics.
- Remote provider circuit breakers and HTTP status mapping.

## Start here

- [Getting Started](getting-started/index.md)
- [Core Concepts](concepts/index.md)
- [Guides](guides/index.md)
- [Reference](reference/index.md)
- [API Reference (rustdoc)](api/uni_xervo/index.html)
- [Internals](internals/index.md)
