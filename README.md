# Uni-Xervo: Unified Model Runtime

`uni-xervo` is a standalone Rust crate that provides a unified runtime for managing local and remote machine learning models. It abstracts away the complexity of loading, caching, and inferencing with different backends (Candle, OpenAI, etc.) behind a clean, task-based API.

## Features

- **Task-Based API**: Strongly typed interfaces for `Embedding`, `Reranking`, and `Generation`.
- **Provider Abstraction**: Pluggable providers for local (Candle) and remote (OpenAI) backends.
- **Model Registry**: Global, async-aware registry that ensures only one instance of a model is loaded per configuration.
- **Alias Resolution**: Decouples application logic from specific model IDs using a configurable catalog.

## Usage

```rust
use uni_xervo::api::{ModelAliasSpec, ModelTask};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. configuration
    let catalog = vec![ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: Default::default(),
        required: true,
        timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    }];

    // 2. Build runtime
    let runtime = ModelRuntime::builder()
        .catalog(catalog)
        .register_provider(LocalCandleProvider::new())
        .build()
        .await?;

    // 3. Resolve and use
    let model = runtime.embedding("embed/default").await?;
    let vectors = model.embed(vec!["hello world"]).await?;
    
    Ok(())
}
```

## Providers

| Provider   | ID              | Description                                                |
| ---------- | --------------- | ---------------------------------------------------------- |
| **Candle** | `local/candle`  | Local execution using HuggingFace Candle (CPU/CUDA/Metal). |
| **OpenAI** | `remote/openai` | Remote execution via OpenAI API.                           |
| **Gemini** | `remote/gemini` | Remote execution via Gemini API.                           |
| **Vertex AI** | `remote/vertexai` | Remote execution via Google Vertex AI API.             |

## CI and Distribution

- Continuous integration runs on every PR and push to `main`:
  - `./scripts/test.sh`
- Crate publishing is handled by GitHub Actions on tags matching `v*`
  (for example `v0.1.0`) via `cargo publish --locked`.
- Set repository secret `CARGO_REGISTRY_TOKEN` to enable publish.
- The pushed tag must match `Cargo.toml` version exactly (without the `v` prefix).

## Scripts

- `./scripts/build.sh` - Build the crate in debug mode.
- `./scripts/build.sh --release` - Build the crate in release mode.
- `./scripts/test.sh` - Run formatting, compile checks, and tests.
- `./scripts/test-integration.sh` - Run ignored integration tests with all features.

## License

Apache 2.0
