# Quick Start

This quickstart shows:

1. building a runtime,
2. registering providers,
3. loading a catalog,
4. calling embedding and generation APIs.

```rust
use uni_xervo::api::catalog_from_file;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::GenerationOptions;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse catalog JSON (array of ModelAliasSpec).
    let catalog = catalog_from_file("./catalog.json")?;

    // Build runtime and register compiled providers.
    let mut builder = ModelRuntime::builder().catalog(catalog);

    #[cfg(feature = "provider-candle")]
    {
        builder = builder.register_provider(uni_xervo::provider::candle::LocalCandleProvider::new());
    }
    #[cfg(feature = "provider-openai")]
    {
        builder = builder.register_provider(uni_xervo::provider::openai::RemoteOpenAIProvider::new());
    }

    let runtime = builder.build().await?;

    // Embedding by alias.
    let embedder = runtime.embedding("embed/default").await?;
    let vectors = embedder.embed(vec!["hello", "world"]).await?;
    println!("embedded {} vectors with dim {}", vectors.len(), embedder.dimensions());

    // Generation by alias.
    let generator = runtime.generator("generate/default").await?;
    let out = generator
        .generate(
            &["You are a concise assistant.".into(), "Summarize Rust ownership in one sentence.".into()],
            GenerationOptions {
                max_tokens: Some(120),
                temperature: Some(0.2),
                top_p: Some(0.95),
            },
        )
        .await?;
    println!("{}", out.text);

    Ok(())
}
```

## Example catalog

```json
[
  {
    "alias": "embed/default",
    "task": "embed",
    "provider_id": "local/candle",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "warmup": "eager",
    "timeout": 10,
    "load_timeout": 300,
    "options": {
      "cache_dir": ".uni_cache/candle"
    }
  },
  {
    "alias": "generate/default",
    "task": "generate",
    "provider_id": "remote/openai",
    "model_id": "gpt-4o-mini",
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

## Common pitfalls

- `CapabilityMismatch`: alias task does not match resolved model interface.
- `ProviderNotFound`: provider feature disabled or provider not registered.
- `Config`: invalid alias format (`task/name`) or invalid provider options.

## Next

- [Model Catalog](../concepts/model-catalog.md)
- [Configuration](../reference/configuration.md)
