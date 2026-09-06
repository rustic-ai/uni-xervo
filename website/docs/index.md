# Uni-Xervo

Uni-Xervo is a unified Rust runtime for model serving across local and remote providers. It gives you one catalog-driven API for embeddings, reranking, generation, and raw ONNX execution.

## What you get

- Alias-based model resolution (`task/name`) instead of hardcoded provider model IDs.
- A single runtime for local and hosted providers.
- Typed task APIs (all share the `ModelInfo` supertrait, which provides
  `model_id()` and `active_execution_providers()`):
  - **Core**: `EmbeddingModel`, `RerankerModel`, `GeneratorModel`, `RawTensorModel`.
  - **Sparse & multi-vector** (introduced in 0.16.0): `SparseEmbeddingModel`
    (learned-sparse / SPLADE / BGE-M3 sparse) and `MultiVectorEmbeddingModel`
    (per-token / ColBERT late-interaction).
  - **Multimodal extension** (introduced in 0.13.0): `ImageEmbeddingModel`,
    `AudioEmbeddingModel`, `MultimodalEmbeddingModel`, `NlpModel`,
    `DocumentExtractionModel`, `TranscriptionModel`, `OcrModel`.
- Reliability controls per alias:
  - inference timeout (`timeout`)
  - load timeout (`load_timeout`)
  - retry policy (`retry`)
- Multimodal generation: vision (image understanding), diffusion (image generation), and speech synthesis via `local/mistralrs`.
- Strict provider option validation with JSON Schema support.

## Capability matrix

| Provider ID | Type | Embed | Rerank | Generate | Raw | Image embed | Multimodal embed | NLP | OCR | ASR | Doc extract | Default auth env |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `local/candle` | local | ✓ | | | | | | | | | | N/A |
| `local/onnx` | local | ✓ | ✓ | | ✓ | ✓ | | ✓ | ✓ | | scaffold | N/A |
| `local/mistralrs` | local | ✓ | | ✓ | | | | | | | ✓ | N/A |
| `local/whisper-cpp` | local | | | | | | | | | ✓ | | N/A |
| `remote/openai` | remote | ✓ | | ✓ | | | | | | | | `OPENAI_API_KEY` |
| `remote/gemini` | remote | ✓ | | ✓ | | | ✓ | | | | | `GEMINI_API_KEY` |
| `remote/vertexai` | remote | ✓ | | ✓ | | | | | | | | `VERTEX_AI_TOKEN` |
| `remote/mistral` | remote | ✓ | | ✓ | | | | | | | | `MISTRAL_API_KEY` |
| `remote/anthropic` | remote | | | ✓ | | | | | | | | `ANTHROPIC_API_KEY` |
| `remote/voyageai` | remote | ✓ | ✓ | | | | | | | | | `VOYAGE_API_KEY` |
| `remote/cohere` | remote | ✓ | ✓ | ✓ | | | ✓ | | | | | `CO_API_KEY` |
| `remote/azure-openai` | remote | ✓ | | ✓ | | | | | | | | `AZURE_OPENAI_API_KEY` |
| `remote/llamacpp` | remote | ✓ | | | | | | | | | | none (optional `api_key_env`) |

**Reading the matrix**:

- `✓` — task is wired with a real model implementation today.
- `scaffold` — catalog wiring, capability advertising, and options validation
  are production-ready; the inference path returns
  `RuntimeError::Unavailable` until an upstream ONNX export ships.
  Reusable building blocks (`autoreg::greedy_decode`, the DocTags / MinerU /
  olmOCR output parsers) are tested and available for the wiring PR. See
  the [provider page](reference/providers/onnx.md) for details.
- Empty — task is not supported on this provider.

For per-provider option details and example catalog entries, see the
[provider reference](reference/providers/index.md). For the trait surface
and resolver methods, see the [API reference](api/uni_xervo/index.html).

## User developer view

For application developers, the main contract is:

1. Build a catalog of `ModelAliasSpec` entries.
2. Register providers with `ModelRuntime::builder()`.
3. Resolve typed handles by alias.
4. Call `embed`, `rerank`, `generate`, or `raw_tensor_model` without provider-specific branching in your app logic.

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
