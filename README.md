# Uni-Xervo

Unified Rust runtime for embedding, reranking, and generation across local and remote model providers.

`uni-xervo` gives you one runtime and one API surface for mixed model stacks, so application code stays stable while you swap providers, models, and execution modes.

## Overview

Uni-Xervo is built around three core ideas:

- Model aliases: your app requests models by stable names like `embed/default` or `generate/llm`.
- Provider abstraction: local and remote providers implement the same task traits.
- Runtime deduplication: equivalent model specs share one loaded instance.

Core tasks:

- `embed` for vector embeddings
- `rerank` for relevance scoring
- `generate` for LLM text generation

## Why Uni-Xervo?

- Keep product code provider-agnostic.
- Mix local and remote models in one runtime.
- Enforce config correctness with schema-backed option validation.
- Control startup behavior with lazy, eager, or background warmup.
- Add retries/timeouts per model alias instead of hard-coding behavior.

## Provider Support

| Provider ID | Tasks | Cargo Feature |
| --- | --- | --- |
| `local/candle` | `embed` | `provider-candle` |
| `local/fastembed` | `embed` | `provider-fastembed` |
| `local/mistralrs` | `embed`, `generate` | `provider-mistralrs` |
| `remote/openai` | `embed`, `generate` | `provider-openai` |
| `remote/gemini` | `embed`, `generate` | `provider-gemini` |
| `remote/vertexai` | `embed`, `generate` | `provider-vertexai` |
| `remote/mistral` | `embed`, `generate` | `provider-mistral` |
| `remote/anthropic` | `generate` | `provider-anthropic` |
| `remote/voyageai` | `embed`, `rerank` | `provider-voyageai` |
| `remote/cohere` | `embed`, `rerank`, `generate` | `provider-cohere` |
| `remote/azure-openai` | `embed`, `generate` | `provider-azure-openai` |

## Installation

Use only the features you need.

```toml
[dependencies]
uni-xervo = { version = "0.1.0", default-features = false, features = ["provider-candle"] }
tokio = { version = "1", features = ["full"] }
```

Default feature set:

- `provider-candle`

If you want local embeddings + OpenAI generation:

```toml
[dependencies]
uni-xervo = { version = "0.1.0", default-features = false, features = ["provider-candle", "provider-openai"] }
tokio = { version = "1", features = ["full"] }
```

GPU acceleration flag:

- `gpu-cuda` for CUDA-enabled builds.

## Quick Start (Rust)

```rust
use uni_xervo::api::{ModelAliasSpec, ModelTask};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = ModelAliasSpec {
        alias: "embed/local".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: Default::default(),
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    let embedder = runtime.embedding("embed/local").await?;
    let vectors = embedder.embed(vec!["hello world"]).await?;
    println!("vector dims = {}", vectors[0].len());

    Ok(())
}
```

## JSON Config Example (`generate/llm`)

Model catalogs are JSON arrays of `ModelAliasSpec`.

`model-catalog.json`:

```json
[
  {
    "alias": "embed/default",
    "task": "embed",
    "provider_id": "local/candle",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "warmup": "lazy",
    "required": true,
    "options": null
  },
  {
    "alias": "generate/llm",
    "task": "generate",
    "provider_id": "remote/openai",
    "model_id": "gpt-4o-mini",
    "warmup": "lazy",
    "timeout": 30,
    "retry": {
      "max_attempts": 3,
      "initial_backoff_ms": 200
    },
    "options": {
      "api_key_env": "OPENAI_API_KEY"
    }
  }
]
```

## Load JSON Config and Run Generation

```rust
use uni_xervo::provider::{LocalCandleProvider, RemoteOpenAIProvider};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::GenerationOptions;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .register_provider(RemoteOpenAIProvider::new())
        .catalog_from_file("model-catalog.json")?
        .build()
        .await?;

    let llm = runtime.generator("generate/llm").await?;
    let result = llm
        .generate(
            &[
                "You are a concise assistant.".to_string(),
                "Understood.".to_string(),
                "Explain what embeddings are in one paragraph.".to_string(),
            ],
            GenerationOptions {
                max_tokens: Some(200),
                temperature: Some(0.3),
                top_p: Some(0.9),
            },
        )
        .await?;

    println!("{}", result.text);
    Ok(())
}
```

## Configuration and Validation

- Catalog schema: `schemas/model-catalog.schema.json`
- Provider option schemas: `schemas/provider-options/*.schema.json`
- Unknown keys or wrong value types fail fast during runtime build/register.

Default remote credential env vars:

| Provider ID | Default credential env var | Extra required options |
| --- | --- | --- |
| `remote/openai` | `OPENAI_API_KEY` | None |
| `remote/gemini` | `GEMINI_API_KEY` | None |
| `remote/vertexai` | `VERTEX_AI_TOKEN` | `project_id` option or `VERTEX_AI_PROJECT` |
| `remote/mistral` | `MISTRAL_API_KEY` | None |
| `remote/anthropic` | `ANTHROPIC_API_KEY` | None |
| `remote/voyageai` | `VOYAGE_API_KEY` | None |
| `remote/cohere` | `CO_API_KEY` | None |
| `remote/azure-openai` | `AZURE_OPENAI_API_KEY` | `resource_name` option |

## CLI Prefetch Utility

The repository includes a prefetch CLI target (`src/bin/prefetch.rs`) to pre-download local model artifacts:

```bash
cargo run --bin prefetch -- model-catalog.json --dry-run
cargo run --bin prefetch -- model-catalog.json
```

Remote providers are skipped by design because they do not cache local weights.

## Development

```bash
# Build
./scripts/build.sh

# Format + check + test
./scripts/test.sh

# Ignored integration tests (real providers)
./scripts/test-integration.sh
```

Integration tests for real providers are gated by `EXPENSIVE_TESTS=1` and relevant API credentials.

## Docs

- User guide: `docs/USER_GUIDE.md`
- Testing guide: `TESTING.md`
- Website docs: `website/`

## License

Apache-2.0 (`LICENSE`).
