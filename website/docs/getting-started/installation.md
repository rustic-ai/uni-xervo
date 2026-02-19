# Installation

Add Uni-Xervo to your Rust project.

## Minimal install (default local Candle provider)

```toml
[dependencies]
uni-xervo = "0.1.0"
```

## Explicit feature selection (recommended)

```toml
[dependencies]
uni-xervo = { version = "0.1.0", default-features = false, features = [
  "provider-candle",
  "provider-fastembed",
  "provider-mistralrs",
  "provider-openai",
  "provider-gemini",
  "provider-vertexai",
  "provider-mistral",
  "provider-anthropic",
  "provider-voyageai",
  "provider-cohere",
  "provider-azure-openai"
] }
```

Enable only what you need to keep build and binary size smaller.

## GPU acceleration

```toml
[dependencies]
uni-xervo = { version = "0.1.0", default-features = false, features = [
  "provider-candle",
  "gpu-cuda"
] }
```

`gpu-cuda` must be paired with one or more providers and requires a valid CUDA toolchain.

## Remote auth environment variables

Set the variables for providers you use:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `VERTEX_AI_TOKEN`
- `VERTEX_AI_PROJECT` (optional fallback for Vertex project)
- `MISTRAL_API_KEY`
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`
- `CO_API_KEY`
- `AZURE_OPENAI_API_KEY`

You can override key variable names per alias with provider options such as `api_key_env` or `api_token_env`.

## Next

- [Quick Start](quickstart.md)
- [Feature Flags](../reference/feature-flags.md)
