# Task trait surface

uni-xervo's task traits are the typed interfaces the runtime hands back when you
resolve an alias. There are 13 `ModelTask` variants and a matching trait per
task. Every task trait shares the same one-business-method-per-trait,
`Send + Sync`, `async_trait` shape, sits on the common
[`ModelInfo`](#the-modelinfo-supertrait) supertrait, and gains a matching
`ModelRuntime` resolver, instrumented wrapper (timeout + retry + metrics), and
`ModelTask` variant.

## The `ModelInfo` supertrait

Every task trait is a subtrait of `ModelInfo`, which carries the metadata common
to all loaded handles:

```rust
pub trait ModelInfo: Send + Sync {
    /// HuggingFace repo ID or API model name.
    fn model_id(&self) -> &str;

    /// ONNX Runtime execution providers requested for the session, in priority
    /// order. Empty for remote and non-ONNX models. Defaults to `Vec::new()`.
    fn active_execution_providers(&self) -> Vec<String> { Vec::new() }
}
```

`model_id()` and `active_execution_providers()` now live on `ModelInfo`, **not**
on the individual task traits — so a caller holding any typed handle
(`Arc<dyn EmbeddingModel>`, `Arc<dyn RerankerModel>`, …) can identify the model
and inspect its requested execution backends uniformly. See
[Verifying the GPU EP actually loaded](gpu-setup.md#verifying-the-gpu-ep-actually-loaded)
for `active_execution_providers()` in practice.

## The 14 task traits

The original quartet — `EmbeddingModel`, `RerankerModel`, `GeneratorModel`,
`RawTensorModel` — plus ten retrieval / multimodal / structured-output traits:

| Trait | Method | Resolver | `ModelTask` |
| --- | --- | --- | --- |
| `EmbeddingModel` | `embed(&[&str]) -> EmbedResult` | `runtime.embedding(alias)` | `Embed` |
| `RerankerModel` | `rerank(&str, &[&str]) -> Vec<ScoredDoc>` | `runtime.reranker(alias)` | `Rerank` |
| `GeneratorModel` | `generate(&[Message], GenerationOptions) -> GenerationResult` | `runtime.generator(alias)` | `Generate` |
| `RawTensorModel` | `run(&TensorBatch) -> TensorBatch` | `runtime.raw_tensor_model(alias)` | `Raw` |
| `ImageEmbeddingModel` | `embed(Vec<ImageInput>) -> EmbedResult` | `runtime.image_embedder(alias)` | `EmbedImage` |
| `AudioEmbeddingModel` | `embed(Vec<AudioInput>) -> EmbedResult` | `runtime.audio_embedder(alias)` | `EmbedAudio` |
| `MultimodalEmbeddingModel` | `embed(Vec<MultimodalInput>) -> EmbedResult` | `runtime.multimodal_embedder(alias)` | `EmbedMultimodal` |
| [`SparseEmbeddingModel`](sparse-embeddings.md) | `embed(&[&str]) -> SparseEmbedResult` | `runtime.sparse_embedder(alias)` | `EmbedSparse` |
| [`MultiVectorEmbeddingModel`](multi-vector-embeddings.md) | `embed(&[&str]) -> MultiVectorEmbedResult` | `runtime.multi_vector_embedder(alias)` | `EmbedMultiVector` |
| [`HybridEmbeddingModel`](multi-vector-embeddings.md#bge-m3-three-heads-one-export) | `embed(&[&str], HeadSet) -> HybridEmbedResult` | `runtime.hybrid_embedder(alias)` | `EmbedHybrid` |
| [`NlpModel`](nlp.md) | `analyze(Vec<NlpRequest>) -> Vec<NlpResult>` | `runtime.nlp_model(alias)` | `Nlp` |
| `DocumentExtractionModel` | `extract(Vec<ImageInput>, DocExtractOptions) -> Vec<DocExtractResult>` | `runtime.document_extractor(alias)` | `DocumentExtract` |
| [`TranscriptionModel`](#transcriptionmodel-is-batch-primary) | `transcribe(Vec<AudioInput>, TranscribeOptions) -> Vec<TranscribeResult>` + `transcribe_one(...)` | `runtime.transcriber(alias)` | `Transcribe` |
| [`OcrModel`](ocr.md) | `recognize(Vec<ImageInput>) -> Vec<OcrResult>` | `runtime.ocr_model(alias)` | `Ocr` |

`ModelTask` is `#[non_exhaustive]` — downstream pattern matches must add a
wildcard arm.

## The `prelude`

`use uni_xervo::prelude::*;` brings the runtime, catalog/config types, **every**
task trait (so their methods are in scope on `Arc<dyn …>` handles), the common
input/result types, and the host-side scoring helpers
(`max_sim` / `colbert_rerank` / `sparse_dot`) into scope in one import. Concrete
provider types stay in `uni_xervo::provider` — import the ones you register
explicitly.

## Result types

### `EmbedResult`

Every dense embedder — `EmbeddingModel`, `ImageEmbeddingModel`,
`AudioEmbeddingModel`, and `MultimodalEmbeddingModel` — returns
`EmbedResult { vectors: Vec<Vec<f32>>, usage: Option<TokenUsage> }`. Remote
providers (Cohere, Gemini) populate `usage` when their APIs report it; local
providers leave it `None`.

`EmbeddingModel::embed` now takes a `&[&str]` slice and returns this
`EmbedResult` — read the vectors off `result.vectors`:

```rust
// 0.16.0: &[&str] in, EmbedResult out (was Vec<&str> -> Vec<Vec<f32>>).
async fn embed(&self, texts: &[&str]) -> Result<EmbedResult>;

let result = model.embed(&["hello", "world"]).await?;
let first: &Vec<f32> = &result.vectors[0];
```

### `SparseEmbedResult` and `MultiVectorEmbedResult`

The two retrieval-oriented text embedders carry their own result shapes, both
following the same `usage` convention:

```rust
// SparseEmbeddingModel::embed(&[&str]) -> SparseEmbedResult
pub struct SparseEmbedResult {
    pub vectors: Vec<Vec<(u32, f32)>>,  // one (term_id, weight) list per input
    pub usage: Option<TokenUsage>,
}

// MultiVectorEmbeddingModel::embed(&[&str]) -> MultiVectorEmbedResult
pub struct MultiVectorEmbedResult {
    pub vectors: Vec<Vec<Vec<f32>>>,    // per input → per token → vector (ragged)
    pub usage: Option<TokenUsage>,
}
```

Score them host-side with `uni_xervo::score::{sparse_dot, max_sim, colbert_rerank}`
— no index required. See the [Sparse embeddings](sparse-embeddings.md) and
[Multi-vector embeddings](multi-vector-embeddings.md) guides for the full
treatment.

### `NlpResult`

`NlpModel::analyze` returns one `NlpResult` per request, carrying per-token
POS / NER / DEP, sentence boundaries, SRL frames, merged named-entity spans,
and dialog-act classifications. See the dedicated
[Structured NLP guide](nlp.md) for the full result-type reference, a worked
example, and the 0.14.0 migration notes.

### `DocExtractResult`

Structured document blocks with reading order, optional bounding boxes,
and a concatenated `plain_markdown` field:

```rust
pub struct DocBlock {
    pub kind: DocBlockKind,  // Text | Heading | List | Table | Figure | Formula | …
    pub content: String,
    pub bbox: Option<[f32; 4]>,
    pub reading_order: u32,
}
```

### `TranscribeResult`

```rust
pub struct TranscribeResult {
    pub language: String,
    pub segments: Vec<TranscribeSegment>,
}
```

Each segment has `start_ms` / `end_ms` / `text` / optional `speaker` /
optional `words`. Word-level timestamps populate when
`TranscribeOptions::word_timestamps = true`.

#### `TranscriptionModel` is batch-primary

The canonical method is the batch one — `transcribe(Vec<AudioInput>, …) ->
Vec<TranscribeResult>` — matching the batch-in / batch-out convention of the
other tasks. `transcribe_one(AudioInput, …) -> TranscribeResult` is a defaulted
convenience that wraps a single input and unwraps the single result:

```rust
async fn transcribe(&self, audios: Vec<AudioInput>, options: TranscribeOptions)
    -> Result<Vec<TranscribeResult>>;

// Defaulted on the trait — implementors only write `transcribe`.
async fn transcribe_one(&self, audio: AudioInput, options: TranscribeOptions)
    -> Result<TranscribeResult>;
```

### `OcrResult`

```rust
pub struct OcrResult {
    pub blocks: Vec<OcrBlock>,  // text + bbox + confidence
    pub plain_text: String,
}
```

## Inputs

### `AudioInput`

```rust
pub enum AudioInput {
    Bytes { data: Vec<u8>, media_type: String },
    Pcm { sample_rate: u32, channels: u16, samples: Vec<f32> },
}
```

No `Path` variant — providers stay I/O-free; callers fetch files
themselves and pass bytes.

### `MultimodalBlock` + `MultimodalInput`

```rust
pub enum MultimodalBlock {
    Text(String),
    Image(ImageInput),
    Audio(AudioInput),
}
pub struct MultimodalInput {
    pub blocks: Vec<MultimodalBlock>,
}
```

Distinct from `ContentBlock` (used by `GeneratorModel`) so generators
stay unchanged.

### `NlpRequest`

`NlpModel::analyze` takes a batch of `NlpRequest { text, tasks }`, where `tasks`
is an `NlpTasks` bitflag selecting which heads to populate. See the
[Structured NLP guide](nlp.md) for details.

## Built-in provider coverage

Today's provider matrix for the task-specific traits beyond the original
quartet:

| Provider | Image embed | Sparse | Multi-vector | Hybrid | NLP | Doc extract | Transcribe | OCR | Multimodal embed |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `local/onnx` | ✓ | ✓ | ✓ | ✓ | ✓ | scaffold | | ✓ | |
| `remote/llamacpp` | | | | | | ✓ | | | |
| `remote/cohere` | | | | | | | | | ✓ |
| `remote/gemini` | | | | | | | | | ✓ |
| `local/whisper-cpp` | | | | | | | ✓ | | |

`local/onnx` **Sparse** and **Multi-vector** are the learned-sparse (SPLADE /
BGE-M3) and ColBERT late-interaction heads — see the
[Sparse embeddings](sparse-embeddings.md) and
[Multi-vector embeddings](multi-vector-embeddings.md) guides. **Hybrid**
(`BGEM3Hybrid`) fuses the dense, sparse, and multi-vector heads of a multi-output
graph into a single forward pass — see
[BGE-M3: three heads, one export](multi-vector-embeddings.md#bge-m3-three-heads-one-export).

`embed_audio` (`AudioEmbeddingModel`) is a defined task and trait but has no
bundled provider implementation yet, so it appears in neither matrix.

`remote/llamacpp` **Doc extract** is the live path: a GGUF document VLM on a
`llama-server` started with `--mmproj` — see
[`remote/llamacpp` → document_extract](../reference/providers/llamacpp.md#document_extract).

`local/onnx` **OCR** supports an optional DBNet detection stage for full-page
detect→recognize (set `det_onnx_path`); without it, OCR is single-stage
recognition — see [`local/onnx` → OCR-only keys](../reference/providers/onnx.md#ocr-only-keys-task-ocr).

`scaffold` (here, `local/onnx` **Doc extract**) means catalog wiring + options
validation are production-ready, but the inference path returns
`RuntimeError::Unavailable` until an upstream prerequisite ships (a canonical
ONNX export of Granite-Docling / MinerU / olmOCR). The reusable building blocks
(`provider::local_onnx::autoreg::greedy_decode`, the shared `doc_parse` DocTags
/ MinerU / olmOCR output parsers) are tested and available — and are exactly
what the live `remote/llamacpp` path reuses.

## Instrumentation

Each new resolver wraps the loaded handle in an `Instrumented*Model`
adapter — same shape as the existing wrappers around `EmbeddingModel`,
`RerankerModel`, etc. The wrapper applies:

- **Timeout** — per-call deadline from `ModelAliasSpec::timeout`. A
  hit surfaces as `RuntimeError::Timeout`.
- **Retry** — exponential backoff on retryable errors
  (`RateLimited` / `Timeout` / `Unavailable`) up to
  `RetryConfig::max_attempts`.
- **Metrics** — `model_inference.duration_seconds` (histogram) and
  `model_inference.total` (counter), labeled with `alias` / `task` /
  `provider` / `status`.

For `TranscriptionModel`, the batched `transcribe` is the instrumented entry
point; the timeout applies batch-wide, not per-item. `transcribe_one` routes
through it.

## Migration: `ModelTask` is `#[non_exhaustive]`

`ModelTask` carries 13 variants and is `#[non_exhaustive]`, so exhaustive
matches against it do not compile downstream. Add a wildcard arm:

```rust
match spec.task {
    ModelTask::Embed => { /* ... */ }
    ModelTask::Rerank => { /* ... */ }
    ModelTask::Generate => { /* ... */ }
    ModelTask::Raw => { /* ... */ }
    _ => return Err(/* unknown task for this provider */),
}
```

## See also

- [Sparse embeddings](sparse-embeddings.md) — `SparseEmbeddingModel` and the
  SPLADE / BGE-M3 sparse presets.
- [Multi-vector embeddings](multi-vector-embeddings.md) —
  `MultiVectorEmbeddingModel` and the ColBERT presets, scored with MaxSim.
- [Structured NLP](nlp.md) — the `NlpModel` result-type reference.
- [OCR](ocr.md) — `OcrModel` and the `local/onnx` PP-OCR pipeline.
- [Provider reference](../reference/providers/index.md) — per-provider
  capability matrix and option keys.
- [Feature flags](../reference/feature-flags.md) — which providers
  ship which tasks.
- [API reference (rustdoc)](../api/uni_xervo/index.html) — full trait
  signatures.
