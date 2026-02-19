# Feature Flags

Uni-Xervo providers are feature-gated.

## Provider features

- `provider-candle`
- `provider-fastembed`
- `provider-mistralrs`
- `provider-openai`
- `provider-gemini`
- `provider-vertexai`
- `provider-mistral`
- `provider-anthropic`
- `provider-voyageai`
- `provider-cohere`
- `provider-azure-openai`

## Acceleration features

- `gpu-cuda`

## Cargo examples

```toml
# Local-only footprint
uni-xervo = { version = "0.1.0", default-features = false, features = [
  "provider-candle",
  "provider-fastembed"
] }

# Remote-only footprint
uni-xervo = { version = "0.1.0", default-features = false, features = [
  "provider-openai",
  "provider-cohere",
  "provider-vertexai"
] }
```

## Runtime registration reminder

Enabling features compiles provider code; it does not auto-register providers.

Register each provider in `ModelRuntime::builder()` before `build()`.
