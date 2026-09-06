# Uni-Xervo

Unified Rust runtime for embedding (dense, learned-sparse, multi-vector / ColBERT, and single-pass hybrid), reranking, generation, OCR, NLP, document extraction, transcription, and raw ONNX execution across local and remote model providers.

`uni-xervo` gives you one runtime and one API surface for mixed model stacks, so application code stays stable while you swap providers, models, and execution modes.

## Overview

Uni-Xervo is built around three core ideas:

- Model aliases: your app requests models by stable names like `embed/default` or `generate/llm`.
- Provider abstraction: local and remote providers implement the same task traits.
- Runtime deduplication: equivalent model specs share one loaded instance.

Core tasks (the 14 `ModelTask` variants):

- `embed` for dense vector embeddings
- `embed_sparse` for learned-sparse term-weight vectors (SPLADE / BGE-M3 sparse)
- `embed_multi_vector` for per-token / late-interaction (ColBERT) embeddings
- `embed_hybrid` for single-pass dense + sparse + multi-vector heads from one multi-output graph (BGE-M3)
- `embed_image`, `embed_audio`, `embed_multimodal` for non-text and heterogeneous embeddings
- `rerank` for relevance scoring
- `generate` for text generation, vision, image generation, and speech synthesis
- `nlp` for structured analysis (POS / NER / DEP / SRL / dialog-act)
- `document_extract` for structured blocks from document page images
- `ocr` for optical character recognition
- `transcribe` for speech-to-text
- `raw` for task-agnostic ONNX tensor execution

Each task is backed by a trait (`EmbeddingModel`, `SparseEmbeddingModel`,
`MultiVectorEmbeddingModel`, `HybridEmbeddingModel`, `RerankerModel`,
`GeneratorModel`, `NlpModel`, `DocumentExtractionModel`, `OcrModel`,
`TranscriptionModel`, `RawTensorModel`, `ImageEmbeddingModel`,
`AudioEmbeddingModel`, `MultimodalEmbeddingModel`), all
sharing a common [`ModelInfo`] supertrait (`model_id()` +
`active_execution_providers()`). The [`prelude`] module re-exports the runtime,
every task trait, the common input/result types, and the host-side scoring
helpers (`max_sim`, `colbert_rerank`, `sparse_dot`) in one `use`.

## Why Uni-Xervo?

- Keep product code provider-agnostic.
- Mix local and remote models in one runtime.
- Retrieval beyond dense vectors: learned-sparse (SPLADE / BGE-M3) and multi-vector / late-interaction (ColBERT) embeddings, with host-side scoring helpers (`max_sim`, `colbert_rerank`, `sparse_dot`).
- Multimodal generation: text, vision, diffusion (image gen), and speech pipelines.
- Enforce config correctness with schema-backed option validation.
- Control startup behavior with lazy, eager, or background warmup.
- Add retries/timeouts per model alias instead of hard-coding behavior.

## Provider Support

| Provider ID | Tasks | Cargo Feature |
| --- | --- | --- |
| `local/candle` | `embed` | `provider-candle` |
| `local/onnx` | `raw`, `rerank`, `embed`, `embed_sparse`, `embed_multi_vector`, `embed_hybrid`, `embed_image`, `nlp`, `ocr`, `document_extract` | `provider-onnx` (or `provider-onnx-dynamic`) |
| `local/mistralrs` | `embed`, `generate` (text, vision, diffusion, speech), `document_extract` | `provider-mistralrs` |
| `local/whisper-cpp` | `transcribe` | `provider-whisper-cpp` (opt-in, not default) |
| `remote/openai` | `embed`, `generate` | `provider-openai` |
| `remote/gemini` | `embed`, `generate`, `embed_multimodal` | `provider-gemini` |
| `remote/vertexai` | `embed`, `generate` | `provider-vertexai` |
| `remote/mistral` | `embed`, `generate` | `provider-mistral` |
| `remote/anthropic` | `generate` | `provider-anthropic` |
| `remote/voyageai` | `embed`, `rerank` | `provider-voyageai` |
| `remote/cohere` | `embed`, `rerank`, `generate`, `embed_multimodal` | `provider-cohere` |
| `remote/azure-openai` | `embed`, `generate` | `provider-azure-openai` |
| `remote/llamacpp` | `embed` (llama.cpp `llama-server`, tokenizer-aware truncation) | `provider-llamacpp` |

## Installation

Defaults give you all three local backends and all eight remote API providers on CPU. Two follow-up questions:

1. **Want a leaner build?** Pass `default-features = false` and pick what you need.
2. **Want GPU?** Add `gpu-cuda` (Linux/Windows + NVIDIA) or `gpu-metal` (macOS/iOS).

```toml
[dependencies]
# Defaults: candle + mistralrs + onnx + all 8 remote providers, CPU only.
uni-xervo = "0.18"
tokio = { version = "1", features = ["full"] }
```

### The feature surface

The features split into three independent axes:

- **Providers**: `provider-candle`, `provider-mistralrs`, `provider-onnx` (local) and `provider-{openai, gemini, vertexai, mistral, anthropic, voyageai, cohere, azure-openai, llamacpp}` (remote). All on by default. `provider-whisper-cpp` (local speech-to-text) is opt-in — it builds whisper.cpp's C/C++ source via CMake, so it stays out of the default `cargo build` toolchain requirement.
- **ORT linking** (matters only if you use ONNX): `provider-onnx` (default, statically linked CPU bundle, self-contained) or `provider-onnx-dynamic` (load-dynamic, BYO ORT via `ORT_DYLIB_PATH`). Mutually exclusive.
- **GPU acceleration**: `gpu-cuda` or `gpu-metal`. Off by default. Purely additive — at runtime, ORT registers the GPU EP first and silently falls back to CPU.

To bring your own ORT build (ROCm, OpenVINO, custom CPU/GPU builds, sandboxed CI, …), use `provider-onnx-dynamic` and point `ORT_DYLIB_PATH` at the vendor-supplied ORT library at deploy. The per-alias `execution_providers` option accepts `"cpu"`, `"cuda"`, `"coreml"`, and `"directml"` (chained, e.g. `["cuda", "cpu"]`, for fallback); other identifiers are rejected at catalog load. See `docs/migrations/0.9.0-feature-surface.md`.

### Common build recipes

```toml
# Default — everything except GPU.
uni-xervo = "0.18"

# Add NVIDIA GPU (Linux / Windows).
uni-xervo = { version = "0.18", features = ["gpu-cuda"] }

# Add Apple GPU + Neural Engine (macOS / iOS).
uni-xervo = { version = "0.18", features = ["gpu-metal"] }

# Lean — only candle (no ORT, no remote providers).
uni-xervo = { version = "0.18", default-features = false, features = ["provider-candle"] }

# Remote-only — no native deps.
uni-xervo = { version = "0.18", default-features = false, features = [
    "provider-openai",
    "provider-anthropic",
] }

# BYO ONNX Runtime (ROCm, OpenVINO, custom builds, sandboxed CI).
uni-xervo = { version = "0.18", default-features = false, features = [
    "provider-candle",
    "provider-mistralrs",
    "provider-onnx-dynamic",
] }
# Then at runtime: ORT_DYLIB_PATH=/path/to/libonnxruntime.so ./your-binary
```

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

    let embedder = runtime.embedder("embed/local").await?;
    let result = embedder.embed(&["hello world"]).await?;
    println!("vector dims = {}", result.vectors[0].len());

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
use uni_xervo::traits::{GenerationOptions, Message};

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
                Message::user("You are a concise assistant."),
                Message::assistant("Understood."),
                Message::user("Explain what embeddings are in one paragraph."),
            ],
            GenerationOptions {
                max_tokens: Some(200),
                temperature: Some(0.3),
                top_p: Some(0.9),
                ..Default::default()
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
| `remote/llamacpp` | none (optional `api_key_env`) | `base_url`, `max_input_tokens`, `embedding_dimensions` options |

## CLI Prefetch Utility

The repository includes a prefetch CLI target (`src/bin/prefetch.rs`) to pre-download local model artifacts:

```bash
cargo run --bin prefetch -- model-catalog.json --dry-run
cargo run --bin prefetch -- model-catalog.json
```

Remote providers are skipped by design because they do not cache local weights.

HF-backed `local/onnx` aliases are treated as local artifacts for prefetch purposes because Uni-Xervo snapshots the full repository into cache before loading the ONNX model.

## ONNX Raw Runtime

`local/onnx` is a raw tensor runtime, not a high-level task adapter. Uni-Xervo handles model resolution, snapshot download, ONNX Runtime loading, signature introspection, batching, and timeout/retry wrappers. Your application still owns preprocessing and output interpretation.

Full developer documentation:

- website ONNX section: `website/docs/onnx/`

Use:

- `provider_id: "local/onnx"`
- `task: "raw"`

`model_id` can be either:

- a local path to a `.onnx` file, or
- a Hugging Face repo ID

If a repo contains multiple `.onnx` files, set `options.artifact`.

Example catalog entry:

```json
{
  "alias": "raw/classifier",
  "task": "raw",
  "provider_id": "local/onnx",
  "model_id": "smokxy/sequence_classification_onnx",
  "options": {
    "artifact": "model.onnx",
    "execution_providers": ["cpu"]
  }
}
```

Example runtime usage:

```rust
use ndarray::arr2;
use uni_xervo::traits::{TensorBatch, TensorValue};

let runner = runtime.raw_tensor_model("raw/classifier").await?;

let mut batch = TensorBatch::new();
batch.insert(
    "input_ids".to_string(),
    TensorValue::I64(arr2(&[[101_i64, 1045, 2293, 3185, 102]]).into_dyn()),
);
batch.insert(
    "attention_mask".to_string(),
    TensorValue::I64(arr2(&[[1_i64, 1, 1, 1, 1]]).into_dyn()),
);

let outputs = runner.run(&batch).await?;
println!("{:?}", outputs.keys().collect::<Vec<_>>());
```

Key ONNX options:

- `artifact`
- `max_batch_size`
- `execution_providers`
- `graph_optimization_level`
- `inter_op_num_threads`
- `intra_op_num_threads`

## Measuring Consumer Binary Size

To measure how much `uni-xervo` adds to a real app, compare a tiny baseline binary against tiny consumer binaries that reference specific provider types:

```bash
./scripts/measure-size.sh
```

The script builds stripped release binaries for:

- a baseline Tokio app with no `uni-xervo`
- `uni-xervo` core with no providers
- `provider-candle`
- `provider-onnx`
- `provider-openai`
- `provider-mistralrs`

This gives a more useful incremental footprint than looking at debug binaries or the repository's `prefetch` CLI alone.

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

- Contributing guide: `CONTRIBUTING.md`
- Development guide: `DEVELOPMENT.md`
- Community guidelines: `COMMUNITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Support guide: `SUPPORT.md`
- Security policy: `SECURITY.md`
- User guide: `docs/USER_GUIDE.md`
- Testing guide: `TESTING.md`
- Website docs: `website/`

## License

Apache-2.0 (`LICENSE`).
