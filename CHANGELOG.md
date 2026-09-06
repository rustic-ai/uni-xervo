# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.18.0] - 2026-09-05

### Added

- **`remote/llamacpp` embedding provider** (`provider-llamacpp`, default-on) —
  targets a llama.cpp `llama-server`. `llama-server` never truncates embedding
  input and rejects prompts longer than its physical batch / context, so the
  provider bounds every input deterministically before calling
  `/v1/embeddings`: texts that provably fit (≤ `max_input_tokens - 2` chars)
  go straight through as strings; longer texts are tokenized via the server's
  native `/tokenize` (`add_special: true`) and, when over budget, truncated to
  the first `max_input_tokens - 1` ids plus the trailing special token and sent
  as a token array (no lossy detokenize round trip). Batches keep input order;
  responses are strictly decoded (count, `index`, width). Required options:
  `base_url`, `max_input_tokens`, `embedding_dimensions`; optional
  `tokenizer_base_url`, `api_key_env` (no key by default), and
  `request_timeout_secs`. Server input-size rejections map to a non-retryable
  `InferenceError` that does not trip the circuit breaker. Ships with an
  in-process mock `llama-server` test harness (`tests/common/mock_http.rs`) and
  an `EXPENSIVE_TESTS` smoke test against a real server.

### Changed

- **Internal:** the OpenAI-format embeddings response decoder is shared via
  `remote_common::parse_openai_embeddings_response`; `remote/openai` behaviour
  is unchanged (lenient mode), `remote/llamacpp` uses strict mode.

## [0.17.0] - 2026-06-23

### Added

- **Single-pass hybrid embeddings** — `HybridEmbeddingModel` /
  `ModelTask::EmbedHybrid`, resolved via `runtime.hybrid_embedder(..)`. One ONNX
  forward pass on a multi-output graph yields the dense, learned-sparse, and
  multi-vector (ColBERT) heads together; a `HeadSet` bitflag selects which to
  materialize and `HybridEmbedResult` carries each as an `Option`. The
  `BGEM3Hybrid` preset wires all three BGE-M3 heads of `aapot/bge-m3-onnx`, so a
  hybrid retrieval pipeline pays one weight load and one pass instead of three.
  Validated end-to-end against the live repo for parity with the per-task
  `embed` / `embed_sparse` / `embed_multi_vector` paths. See
  `examples/embed_hybrid.rs`.
- **`BGEM3Dense` dense embedding preset** — BGE-M3 dense vectors from the
  multilingual community multi-output export `aapot/bge-m3-onnx` (the same graph
  that already serves the sparse and ColBERT tasks), as an alternative to the
  official `BAAI/bge-m3` dense preset. The bare `aapot/bge-m3-onnx` id now
  resolves to the dense head for the `embed` task (the task disambiguates one
  repo serving all three heads). `EmbeddingPreset` gains an `output_name` field
  so a dense preset can pin a specific head on a multi-output graph.

## [0.16.0] - 2026-06-23

### Added

- **Learned-sparse text embeddings** — `SparseEmbeddingModel` / `ModelTask::EmbedSparse`,
  resolved via `runtime.sparse_embedder(..)`. Two recipes (`sparse_method`):
  SPLADE-style MLM logits (`mlm`, term expansion) and the BGE-M3 lexical head
  (`lexical`). Output is a `Vec<(term_id, weight)>` per input.
- **Multi-vector / ColBERT late-interaction embeddings** — `MultiVectorEmbeddingModel`
  / `ModelTask::EmbedMultiVector`, via `runtime.multi_vector_embedder(..)`. Emits
  per-token vectors (ragged, padding stripped, optionally L2-normalized) for MaxSim.
- **Host-side scoring helpers** — `score::{max_sim, colbert_rerank, sparse_dot}` so
  the sparse/multi-vector outputs are usable for in-process reranking without an index.
- **`local/onnx` presets** (all validated end-to-end against their live HF repos):
  SPLADE++ v1/v2 and the BGE-M3 sparse head; `answerai-colbert-small-v1`,
  `GTE-ModernColBERT-v1`, `mxbai-edge-colbert-v0` (17M/32M), and the BGE-M3 ColBERT head.
- **`ModelInfo` base trait** — `model_id()` + `active_execution_providers()`, the
  supertrait of every task trait. `RerankerModel`, `GeneratorModel`, and
  `RawTensorModel` gain `model_id()`.
- **`uni_xervo::prelude`** — `use uni_xervo::prelude::*` re-exports the runtime,
  catalog/config types, every task trait, the common I/O & result types, and the
  scoring helpers.
- **`ModelRuntime::embedder(..)`** — agent-noun alias for `embedding(..)`.
- **`examples/encode_corpus_multivec.rs`** — a ColBERT multi-vector producer that
  encodes a JSONL corpus into an `MVEC` binary for a downstream recall benchmark.

### Changed

- **BREAKING — `EmbeddingModel::embed` now returns `EmbedResult { vectors, usage }`**
  (was `Vec<Vec<f32>>`), matching the other embedders. The seven remote embedders
  (OpenAI / Cohere / Gemini / Mistral / Voyage / Azure / Vertex) now report the
  per-call token usage they previously discarded.
- **BREAKING — text embedders take `&[&str]`** (was `Vec<&str>`) on
  `EmbeddingModel`, `SparseEmbeddingModel`, and `MultiVectorEmbeddingModel`,
  matching `RerankerModel::rerank`.
- **BREAKING — `model_id()` moved to the `ModelInfo` supertrait** (no longer
  declared per task trait); `active_execution_providers()` likewise unified onto
  `ModelInfo`.
- **BREAKING — `TranscriptionModel` is now batch-primary**: `transcribe(Vec<AudioInput>)
  -> Vec<TranscribeResult>` with a defaulted `transcribe_one` convenience (the old
  single-`transcribe` / `transcribe_many` pair is gone).
- **BREAKING — dropped the unused `: Any` supertrait** from the embedding / NLP /
  document / OCR / ASR traits (downcasting was always on the outer `Arc<dyn Any>`
  handle, never the trait object).

## [0.15.0] - 2026-06-16

### Added

- **`local/onnx` × `OcrModel` — optional DBNet detection stage.** OCR can now
  run full-page detect → crop → recognize → reading-order when `det_onnx_path`
  is set; without it, behavior is unchanged (single-stage recognition). New
  `det_*` options (`det_onnx_path`, `det_model_id`, `det_limit_side`,
  `det_bin_threshold`, `det_box_score_threshold`, `det_unclip_ratio`,
  `det_min_box_size`, `det_input_name`, `det_output_name`). The accuracy-
  sensitive postprocessing (binarize → iterative 4-connectivity flood fill →
  mean-prob score → axis-aligned unclip → TB-YX reading order) is pure-Rust
  with no new dependency, unit-tested on synthetic probability maps, and
  verified end-to-end against PP-OCRv5 (`monkt/paddleocr-onnx`).
- **`local/mistralrs` × `document_extract` — olmOCR-2.** Revived
  `DocumentExtractionModel` on the mistral.rs vision pipeline: olmOCR-2
  (`allenai/olmOCR-2-7B-1025`, a Qwen2.5-VL fine-tune), one page per request,
  the image-only no-anchor prompt, low-temperature first pass, parsed via the
  shared `doc_parse` module. New `style` option.
- **Shared `doc_parse` module** — the DocTags / MinerU / olmOCR output parsers
  were hoisted out of `local_onnx` so both `local/onnx` and `local/mistralrs`
  reuse them.
- **`image::preprocess_batch_hw`** (non-square `(h, w)` preprocessing) and
  `preprocess_det` (resize to a multiple of 32 + scale factors).
- **New companion crate `uni-xervo-pdf`** (optional workspace member) — tiered
  PDF document extraction that escalates per page across `Native` → `Ocr` →
  `Vlm`, composing the core OCR/VLM models by alias. Family-aligned via the
  `PdfExt` extension trait (`runtime.pdf_extractor(..)`), pure-Rust `hayro`
  rasterization (no FFI) and `lopdf` native text behind features, with
  provenance-bearing results and cross-tier numeric verification.

### Fixed

- **OCR recognition confidence** — recognizers whose ONNX output is an
  already-softmaxed distribution (e.g. PP-OCR) were double-softmaxed, collapsing
  confidence toward `1/n_classes`. `class_confidence` now detects a normalized
  row and reads the probability directly; logit outputs still get a softmax.
  (Text/argmax was always correct; this fixes the trust signal.)

### Documentation

- New `guides/pdf-extraction.md` for the `uni-xervo-pdf` companion; the
  `local/onnx` and `local/mistralrs` provider references, the multimodal
  trait-surface capability matrix, and the feature-flags table were updated for
  the OCR detection stage and the olmOCR-2 document-extraction path.

## [0.14.0] - 2026-06-14

### Changed (breaking)

- **`local/onnx` × `NlpModel`** — all structured-NLP result structs
  (`NlpResult`, `NlpToken`, `NlpSentence`, `DepLink`, `SrlFrame`, `SrlRole`,
  `SpeechAct`) are now `#[non_exhaustive]` and derive `Default`. Construct them
  via `Default` (e.g. `NlpResult::default()` plus field assignment) rather than
  struct literals; future heads can then be added without breaking callers.
- **`DepLink::head`** is now `Option<usize>` (was `usize`). `Some(i)` is a
  0-based **global** index into `NlpResult::tokens`; `None` marks the sentence
  root. This also fixes a latent correctness bug: the old `usize` head was a
  chunk-local index into the model's tokenized sequence (special tokens
  included), which did not align with the special-token-free `tokens` array,
  and roots — encoded by the cascade as a self-loop — were not represented at
  all.

### Added

- **`SpeechAct::scores`** — the full per-class dialog-act softmax, parallel to
  `NlpLabelMaps::cls`; `confidence` equals `scores[argmax]`. Lets consumers do
  thresholded multi-label gating instead of relying only on the top-1 label.
- **`NlpResult::entities`** (`Vec<NerEntity>`) — merged, BIO-collapsed
  named-entity spans (inclusive token span + half-open byte char span),
  alongside the retained per-token `NlpToken::ner` BIO tags.
- **`NlpToken::word_index`** — dense, monotonic subword→word grouping sourced
  from the tokenizer's word ids (with a byte-gap fallback), so consumers
  regroup subwords into words without parsing metaspace markers.
- **`NlpModel::label_maps()`** → `Option<&NlpLabelMaps>` (default `None`) —
  exposes the model's per-head label vocabularies (`pos` / `ner` / `deprel` /
  `srl` / `cls`). The ONNX cascade returns them; remote/mock models return
  `None`. The instrumentation wrapper forwards the method.
- SRL span/label contract is now documented on `SrlFrame` / `SrlRole`
  (inclusive `[first, last]` global token spans; predicate excluded from roles).
- Decode-path unit tests for the new helpers (`softmax_full`, `remap_dep_head`
  including self-loop / special-token roots and cross-chunk globality,
  `merge_ner_spans`, `is_word_boundary`, `label_maps_from_value` failure paths)
  and wrapper-forwarding tests; the end-to-end cascade test gained subset-task
  and richer-output assertions.

### Documentation

- New `guides/nlp.md` consolidating the `NlpModel` surface with a worked
  `analyze()` example and a 0.14.0 migration section; the provider reference
  and multimodal trait-surface pages cross-link it, and doc dependency pins
  were bumped to 0.14.0.

## [0.13.1] - 2026-06-08

### Fixed

- **`local/onnx` × `NlpModel`** — `OnnxNlpModel::analyze` failed
  unconditionally with `argmax_axis only supports axis=1, got 2`. The
  dependency-parse head decode called `argmax_axis(&arc_scores, 2)` on a
  2-D `[seq, seq]` score matrix, where the correct per-token decode is a
  per-row argmax (`axis=1`). Because the heads were decoded eagerly,
  upstream of the per-task `NlpTasks::DEP` gate, every `analyze()` call
  errored regardless of which tasks were requested. The head decode now
  calls `argmax_last_axis` directly, and the dead-`axis` `argmax_axis`
  wrapper (only `axis=1` was ever valid) is removed.

### Added

- No-model decode-path unit tests for the `OnnxNlpModel` tensor-shape
  helpers (per-row head argmax, `[token, head, class]` relation indexing,
  the seq-mismatch guard, 1-D CLS argmax, softmax confidence, SRL BIO span
  collapsing). These run on every `provider-onnx` build, unlike the
  existing `#[ignore]`d, model-gated end-to-end cascade test.

## [0.13.0] - 2026-05-26

### Added (deferred follow-ups)

- **`local/whisper-cpp` × `TranscriptionModel`** — promoted from scaffold
  to real `whisper-rs` integration. `transcribe()` now downloads the
  ggml `.bin` weights from HuggingFace, decodes the input audio to
  16 kHz mono f32 PCM, and runs `whisper_rs::WhisperContext::full(...)`
  on a blocking thread via `tokio::task::spawn_blocking`. Word
  timestamps populate `TranscribeSegment::words` when
  `TranscribeOptions::word_timestamps = true`. Sampling is greedy
  (`SamplingStrategy::Greedy { best_of: 1 }`); beam search is a
  follow-up.
- Audio decode (v1): `AudioInput::Pcm` at 16 kHz, mono or stereo
  (stereo collapsed to mono). `AudioInput::Bytes` decoded only when
  `media_type == "audio/wav"` (16-bit PCM mono/stereo, in-tree decoder,
  no `symphonia` dep). Other formats / sample rates rejected with a
  clear error pointing at the supported shape.
- `provider-whisper-cpp` feature now activates `whisper-rs` 0.13. Builds
  whisper.cpp's C/C++ source via CMake; toolchain requirement (cmake +
  cc) documented in the Cargo.toml feature comment. Still opt-in, NOT
  in default features.
- Whisper diarization is not surfaced (whisper.cpp doesn't natively
  diarize); `TranscribeSegment::speaker` is always `None`. A
  pyannote-style segmenter would land as a separate concern.
- **Reusable greedy autoregressive decoder helper**
  (`src/provider/local_onnx/autoreg.rs`) — `greedy_decode(&AutoregConfig, |running_tokens| logits)`.
  The callback owns all model state (ONNX session, KV cache, image
  embeddings, attention masks, …); the helper only handles the
  decoder-loop logic (argmax → append → stop on EOS / max-new-tokens).
  9 unit tests cover the loop semantics with scripted-logits callbacks.
  Reserved for downstream VLM / decoder-LM provider impls; not wired
  into any task yet.
- **`local/onnx` × `DocumentExtractionModel`** — `extract()` still
  returns `Unavailable` (no canonical ONNX export of Granite-Docling /
  MinerU / olmOCR-2 exists yet — see the function's doc comment for the
  expected schema). The error message and tracing event are improved to
  point at the autoregressive decoder helper, the shipping output
  parsers, and the precise blockers per model family.

### Added

- **`local/onnx` × `NlpModel`** (PR-3) — first concrete provider implementation
  on top of the PR-1 trait surface. Targets the kniv-deberta cascade family
  (canonical default `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`),
  producing POS / NER / DEP / SRL / dialog-act CLS from one DeBERTa-v3
  encoder pass. SRL multi-pass orchestration (one forward per detected verb
  via `predicate_idx`), chunking for inputs longer than `max_seq_len`, and
  label decoding from the repo's `label_maps.json` are all handled
  provider-side; callers just see populated `NlpResult` heads. New option
  keys on `local/onnx` when `task = nlp`: `onnx_path` (default
  `onnx/cascade.onnx`), `tokenizer_path` (default `tokenizer.json`),
  `label_maps_path` (default `label_maps.json`), `max_seq_len` (default
  128).
- `LocalOnnxProvider::capabilities` now advertises `ModelTask::Nlp`.
- `validate_provider_options` accepts `(local/onnx, nlp)`; the multimodal-
  task-rejection gate now carries a small allow-list of pairs that have
  shipped impls.
- Expensive integration test `kniv_deberta_nlp_cascade_end_to_end` in
  `tests/onnx_models_expensive_test.rs`, gated on `EXPENSIVE_TESTS=1`.
- **`remote/cohere` × `MultimodalEmbeddingModel`** (PR-6) — wires Cohere
  Embed v4 via the existing `v2/embed` endpoint with multimodal `inputs`
  (text + image; audio rejected with a clear error per Cohere v4's spec).
  Images can be supplied as URL or raw bytes — the latter are base64-encoded
  into a data URL.
- **`remote/gemini` × `MultimodalEmbeddingModel`** (PR-6) — wires Gemini
  Embedding 2 via `batchEmbedContents` with multimodal `parts` (text +
  image + audio + video). PCM audio is encoded to 16-bit mono WAV inline
  before upload; container-form `AudioInput::Bytes` is forwarded as-is.
- Both `remote/cohere` and `remote/gemini` add `base64` as a feature dep.
- **`local/onnx` × `ImageEmbeddingModel`** (PR-2a) — wires ViT-style image
  embedders (SigLIP / SigLIP-2 / CLIP / OpenCLIP) on the `local/onnx`
  provider. New option keys when `task = embed_image`: `onnx_path`,
  `image_size`, `dimensions`, `normalization` (`siglip` or `imagenet`),
  `pool` (`none` for pre-pooled `[batch, dim]` outputs, `mean` for
  `[batch, tokens, dim]` patch sequences), `normalize` (L2-norm),
  `output_name`.
- New shared image-preprocessing helper at
  `src/provider/local_onnx/image.rs` (decode → resize-square → normalize →
  NCHW float tensor). Will be reused by PR-2b (OCR) and PR-5 (document
  extraction).
- `image` crate is now a feature dep of `provider-onnx` and
  `provider-onnx-dynamic` (previously only pulled in by `provider-mistralrs`).
- **`local/onnx` × `OcrModel`** (PR-2b) — CRNN + CTC-style recognition
  targeting the PaddleOCR-rec / EasyOCR family of ONNX-exported OCR
  models. The impl:
  - Loads a `[batch, time, n_classes]` model output.
  - Greedy-decodes via CTC: per-step argmax → collapse consecutive
    duplicates → drop the blank class.
  - Maps remaining indices through a character dictionary file
    (one entry per line).
  - Returns one `OcrBlock` per image with normalized full-image bbox and
    average softmax confidence over emitted characters.
  - New option keys when `task = ocr`: `onnx_path` (required),
    `char_dict_path` (required), `image_height` (default 48),
    `image_width` (default 320), `normalization` (`siglip` | `imagenet`,
    default `imagenet`), `blank_class` (default 0), `output_name`.
  - Two-stage detection + recognition is deferred; callers pre-crop their
    regions and pass one image per region.
- **`local/whisper-cpp` × `TranscriptionModel`** (PR-4) — new provider
  driver scaffold for speech-to-text. v1 ships the catalog wiring,
  capability advertising, options validation, and the registered
  resolver. The actual ggml-model load + `whisper-rs` inference path
  lands in a follow-up to keep the default `cargo build` toolchain
  requirement at "just Rust" (whisper.cpp compiles C/C++ source).
  Until then `transcribe()` returns `RuntimeError::Unavailable` with a
  clear message. New `provider-whisper-cpp` feature flag (opt-in, not
  default). Options: `model_path` (default `ggml-base.bin`),
  `default_language`, `cache_dir`.
- **`local/onnx` × `DocumentExtractionModel`** (PR-5) — provider scaffold
  for VLM-based document parsing. Three output styles selected via the
  `style` option:
  - `style: "granite-docling"` (default, reference) — Granite-Docling's
    DocTags output. Typed tags (`<heading>`, `<table>`, `<formula>`, …)
    with optional `loc="x0,y0,x1,y1"` bboxes.
  - `style: "mineru"` — MinerU 2.5's structured Markdown with
    `$$..$$`-delimited LaTeX. Heuristic block detection (heading / list /
    table / figure / formula / text).
  - `style: "olmocr"` — olmOCR-2's Markdown with `$..$` inline LaTeX
    (reuses the MinerU block-level heuristics; inline math stays in text
    blocks verbatim).
  - The three style-specific output parsers (`parse_doctags`,
    `parse_mineru_markdown`, `parse_olmocr_markdown`) are
    production-ready and fully unit-tested. The VLM inference loop
    (vision encoder + LLM decoder generation) is deferred to a follow-up
    that picks concrete ONNX-exported variants; until then `extract()`
    returns `RuntimeError::Unavailable`. Options: `style`, `onnx_path`,
    `tokenizer_path`, `max_seq_len`.

### Added

- **Multimodal trait surface.** Seven new model traits for image / audio /
  mixed-modality embedding, structured NLP, document extraction, speech
  transcription, and OCR. Each sits alongside the existing
  `EmbeddingModel` / `RerankerModel` / `GeneratorModel` traits and follows
  the same one-business-method-per-trait, `Send + Sync`, `async_trait`
  shape:
  - `ImageEmbeddingModel::embed(Vec<ImageInput>) -> EmbedResult`
  - `AudioEmbeddingModel::embed(Vec<AudioInput>) -> EmbedResult`
  - `MultimodalEmbeddingModel::embed(Vec<MultimodalInput>) -> EmbedResult`
  - `NlpModel::analyze(Vec<NlpRequest>) -> Vec<NlpResult>`
  - `DocumentExtractionModel::extract(Vec<ImageInput>, DocExtractOptions) -> Vec<DocExtractResult>`
  - `TranscriptionModel::transcribe(AudioInput, TranscribeOptions) -> TranscribeResult`
    plus a default-fan-out `transcribe_many` for ingest batching that
    providers can override for genuine internal batching.
  - `OcrModel::recognize(Vec<ImageInput>) -> Vec<OcrResult>`
- **`ModelTask` extension.** Seven new variants on `ModelTask`:
  `EmbedImage`, `EmbedAudio`, `EmbedMultimodal`, `Nlp`, `DocumentExtract`,
  `Transcribe`, `Ocr`. The enum is now `#[non_exhaustive]` to make future
  variants non-breaking. Downstream pattern matches without a wildcard
  arm will not compile against 0.13.0 — add `_ => { ... }` to fix.
- **`ModelRuntime` resolvers.** Seven new methods returning instrumented
  trait handles: `image_embedder`, `audio_embedder`, `multimodal_embedder`,
  `nlp_model`, `document_extractor`, `transcriber`, `ocr_model`. Each
  cache-hits per alias, downcast-checks the underlying provider handle, and
  returns `RuntimeError::ProviderCapabilityMissing` on mismatch.
- **`EmbedResult { vectors, usage }`.** The new embed traits return a
  wrapped result that can carry per-call `TokenUsage`. The existing
  `EmbeddingModel::embed` signature is unchanged.
- **Shared types.** `AudioInput` (Bytes / Pcm), `MultimodalBlock`,
  `MultimodalInput`, `Modality`, `NlpTasks` (bitflag — POS/NER/DEP/SRL/CLS),
  `NlpRequest<'a>`, `NlpResult`, `NlpToken`, `NlpSentence`, `DepLink`,
  `SrlFrame`, `SrlRole`, `SpeechAct`, `DocExtractOptions`, `DocOutputFormat`,
  `DocExtractResult`, `DocBlock`, `DocBlockKind`, `OcrResult`, `OcrBlock`,
  `TranscribeOptions`, `TranscribeResult`, `TranscribeSegment`,
  `TranscribeWord`. All re-exported from `uni_xervo::traits::*`.
- **Instrumentation.** Each new trait gains an `Instrumented*Model`
  wrapper applying timeout (`spec.timeout`) + retry-with-backoff
  (`spec.retry`) + metrics (`model_inference.duration_seconds` and
  `model_inference.total` keyed on `alias` / `task` / `provider`). The
  `transcribe_many` batched method shares the same timeout envelope —
  operators tuning `timeout` for batched ingest should budget for the
  full batch, not a per-item slice.

### Changed

- **`ModelTask` is now `#[non_exhaustive]`** (see Added). Downstream
  match sites without a wildcard arm must add one.
- **`validate_provider_options`** rejects every (bundled-provider × new-task)
  pair with a clear `Config` error. No bundled provider implements any of
  the new traits in this release; impls land in follow-up PRs and will
  flip their entries to the "supported" branch as they ship.

### Preserved

- **`RawTensorModel` and `ModelTask::Raw` are unchanged.** They remain
  first-class public API for customers running custom ONNX graphs through
  the escape hatch. The new managed traits are siblings, not replacements.
  Existing `runtime.raw_tensor_model(alias)` call sites require no code
  changes.

### Reference models

The canonical default for `NlpModel` is
[`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall),
a DeBERTa-v3-xsmall multi-head cascade producing POS / NER / DEP / SRL / CLS
in a single forward pass. It will be wired up as the `nlp/default` catalog
alias once the `local/onnx` provider gains an `NlpModel` implementation
in a follow-up release. No catalog entry ships in this release.

### Dependencies

- New direct dependencies: `bitflags = "2"` (for `NlpTasks`), `futures = "0.3"`
  (for `TranscriptionModel::transcribe_many`'s default fan-out via
  `try_join_all`).

## [0.12.0] - 2026-05-15

### Added

- `remote/vertexai`: `location: "global"` now routes through Vertex AI's [global endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint) (`aiplatform.googleapis.com`) instead of the regional `{location}-aiplatform.googleapis.com` host. The URL path keeps `locations/global/...` as Vertex AI expects. Any other `location` value continues to use the regional host (unchanged default: `us-central1`).

### Docs

- `reference/providers/vertexai.md` and `reference/configuration.md`: document the new `location: "global"` value and link to Google's global-endpoint docs.

## [0.11.0] - 2026-05-01

### Added

- `local/onnx`: `pooling: "last-token"` for decoder-style LLM embedders (Qwen3-Embedding, NV-Embed) — takes the rightmost non-pad hidden state as the sentence representation. The existing `cls`, `mean`, and `max` modes remain available.
- `local/onnx`: built-in preset `Qwen3Embedding06B` → `onnx-community/Qwen3-Embedding-0.6B-ONNX` (1024-dim, last-token pool, external-data sidecar handled by the existing `additional_files` download path).
- `local/onnx`: generative-style reranker behind `style: "generative"` for Qwen3-Reranker-style decoder-LM rerankers that score by emitting `yes`/`no` token logits rather than a single relevance logit. Default artifact `onnx/model_q4.onnx` (Qwen3-Reranker-0.6B-ONNX publishes only quantized variants); customize the prompt's task description via the new `instruction` option. Score is `softmax([yes, no])[0]` — a probability in `[0, 1]`. The existing `style: "cross-encoder"` (default) path is unchanged.
- `local/onnx`: decoder-LM session inputs (`position_ids`, `past_key_values.*`) are now auto-detected from `session.inputs()` and fed transparently for both embedders and the generative reranker. Encoder-style models (BGE, MiniLM, MPNet, ModernBERT) are unaffected. The new internal `decoder_inputs` module centralises input classification, dim resolution via dim symbols + positional fallback, and zero-tensor KV allocation.
- `tests/onnx_models_expensive_test.rs`: `EXPENSIVE_TESTS=1`-gated end-to-end suite covering BGE-small-en-v1.5, BGE-large-en-v1.5, BGE-reranker-base, Qwen3-Embedding-0.6B, and Qwen3-Reranker-0.6B against real HuggingFace weights through the public `EmbeddingModel` and `RerankerModel` APIs. Run with `EXPENSIVE_TESTS=1 cargo nextest run --features provider-onnx --test onnx_models_expensive_test --run-ignored all`.
- `scripts/test-integration.sh`: now also runs `onnx_models_expensive_test` alongside `real_providers_test`, so a single integration run exercises both remote APIs and local ONNX models.

### Fixed

- `local/onnx` cross-encoder reranker: `token_type_ids` is now auto-detected from `session.inputs()` instead of hardcoded. Models that don't declare it (e.g. `BAAI/bge-reranker-base`, XLM-R-based cross-encoders) no longer fail with `"Invalid input name: token_type_ids"` at the first rerank call. Models that do declare it (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`) continue to receive it.
- `remote/openai`: `base_url` validation now rejects empty, whitespace-only, and relative URLs at `ModelRuntimeBuilder::build` time instead of failing later inside `reqwest` with a less actionable error. Requires the value to start with `http://` or `https://`.
- `require_positive_u64` validator: also rejects values exceeding `usize::MAX`. Options that downcast into `usize` fields (`max_seq_len`, `max_batch_size`, `max_num_images`, `embedding_dimensions`, `max_num_seqs`, ONNX thread/batch knobs, etc.) now fail validation with a clear message on 32-bit targets rather than panicking later in `serde_json::from_value`. No-op on 64-bit (`usize::MAX as u64 == u64::MAX`).

### Docs

- New "Mixing GPU and CPU placement in one catalog" section in `guides/onnx.md` with a Rust catalog example showing one ONNX embedder pinned to CPU and one reranker on `["cuda", "cpu"]` in the same `ModelRuntime`.
- New "Device placement" section in `reference/providers/mistralrs.md` covering `force_cpu` defaults and the (build feature × `force_cpu`) result matrix. Includes a per-spec mixing example and a cross-provider note that `local/mistralrs` does not accept `execution_providers` (it's candle-backed, not ORT-backed).
- `reference/providers/onnx.md` option lists updated for the new `pooling: "last-token"`, rerank `style`, and rerank `instruction` keys, plus a cross-reference to the new mixing-placement guide.

## [0.10.0] - 2026-05-01

### Added

- `remote/openai` now accepts a `base_url` option to target OpenAI-compatible servers (OpenRouter, vLLM, LM Studio, Ollama, internal proxies). Default remains `https://api.openai.com/v1`.
- `local/mistralrs` exposes `max_seq_len`, `max_batch_size`, `max_image_shape`, and `max_num_images` to override the auto-device-mapper's planning reservation. Lowering these lets layers fit on small GPUs where the bf16-sized default reservation otherwise places zero layers on-device.
- `local/mistralrs` accepts a `uqff_files` option for loading pre-quantized UQFF models (mistralrs's native format). Loads quantized weights directly, bypassing the bf16-then-quantize flow that forces the full unquantized footprint into VRAM at load time. Required for fitting larger multimodal models (e.g. Gemma 4 E2B) on small (8 GB) GPUs. Mutually exclusive with `gguf_files` and `isq`.

## [0.9.0] - 2026-04-27

Feature-flag surface collapsed from a per-EP matrix to a small capability surface: CPU is always on, `gpu-cuda` and `gpu-metal` are the two opt-in GPU knobs, and ROCm / DirectML / OpenVINO / QNN / TensorRT / WebGPU support is reachable via `provider-onnx-dynamic` + a vendor-supplied ORT build. The Microsoft tarball-fetcher pipeline is gone, build.rs is now a ~30-line validator, and `gpu-metal` finally activates the CoreML execution provider so ORT actually reaches Apple GPU/ANE.

### Breaking

- **Default features expanded** to include all local backends *and* all remote providers: `provider-candle`, `provider-mistralrs`, `provider-onnx`, `provider-openai`, `provider-gemini`, `provider-vertexai`, `provider-mistral`, `provider-anthropic`, `provider-voyageai`, `provider-cohere`, `provider-azure-openai`. Build time and binary size grow accordingly; users who want a leaner build should use `default-features = false` and select explicitly.
- **Removed Cargo features**: `gpu-tensorrt`, `gpu-coreml`, `gpu-wgpu`, `gpu-directml`, `gpu-qnn`, `gpu-rocm`, `gpu-openvino`, and the internal `_ort-fetched-base`.
- **`gpu-metal` redefined**: now also activates `ort?/coreml` so the CoreML EP is registered when ORT is in the build. Previously `gpu-metal` only flipped Metal kernels for candle/mistralrs and ORT silently stayed on CPU.
- **`OnnxExecutionProvider` enum**: `Cpu`, `Cuda`, `CoreMl` are always available; `Rocm`, `DirectMl`, `OpenVino`, `Qnn`, `TensorRt`, `WebGpu` are accepted by the parser unconditionally and dispatched only under `provider-onnx-dynamic` (a `RuntimeError::Config` is returned otherwise pointing the user at the right feature). The legacy aliases `"nvrtx"` and `"wgpu"` are no longer accepted — use `"tensorrt"` and `"webgpu"`.
- **`execution_providers` strings**: vendor strings (`"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"`) require `provider-onnx-dynamic` plus `ORT_DYLIB_PATH` pointing at a vendor-supplied ORT library that contains the requested EP. Under the bundled `provider-onnx`, requesting them fails fast with a clear `Config` error.
- **build.rs simplification**: the Microsoft tarball fetcher (`build/ort_vendor.rs`) is deleted. Build-deps `ureq`, `flate2`, `tar`, `zip`, `sha2`, `hex` are dropped. The `UNI_XERVO_ORT_DIST_DIR` build-time override is gone.

### Migration

| Old feature | New path |
|---|---|
| `gpu-coreml` | `gpu-metal` (now bundles CoreML EP automatically) |
| `gpu-tensorrt` | use `gpu-cuda` (covers NVIDIA via the CUDA EP) |
| `gpu-directml` | `provider-onnx-dynamic` + Microsoft Windows GPU tarball, `ORT_DYLIB_PATH` |
| `gpu-rocm` | `provider-onnx-dynamic` + AMD `onnxruntime-rocm` package |
| `gpu-openvino` | `provider-onnx-dynamic` + Intel `openvino-onnxruntime` package |
| `gpu-qnn` | `provider-onnx-dynamic` + Microsoft Windows ARM64x tarball |
| `gpu-wgpu` | no replacement — file an issue if you need it |

If you were combining `provider-onnx` with `gpu-coreml` on macOS, just switch to `gpu-metal` — it now covers both candle/mistralrs Metal kernels *and* the ORT CoreML EP in one feature.

### Removed

- `build/ort_vendor.rs` and the entire Microsoft tarball-fetching pipeline.
- `_ort-fetched-base` internal scaffolding feature.
- Six `gpu-*` features and their accompanying `OnnxExecutionProvider` variants.

## [0.8.0] - 2026-04-27

Native ONNX embedding lane: `local/onnx` now serves `ModelTask::Embed` directly, and `provider-fastembed` is removed. The new path is a thin tokenize → ORT → pool → L2-normalize adapter on top of `RawTensorModel`, with a built-in preset table for the 25 text-embedding models that `fastembed-rs` previously surfaced.

Parity vs. `fastembed-rs` v5.13 was verified end-to-end across all 25 presets: 22 hit cosine ≈ 1.000 (numerical-noise match), 3 dynamic-quantized variants land at ≈ 0.99 (quant scales are input-dependent and amplify minor tokenizer-setup differences).

### Breaking

- **Removed `provider-fastembed` feature, `LocalFastEmbedProvider` type, and `local/fastembed` provider id.** Catalog migration: change `provider_id: "local/fastembed"` to `provider_id: "local/onnx"` and keep `task: "Embed"`. **All 35 FastEmbed alias strings** (`"BGESmallENV15"`, `"all-MiniLM-L6-v2"`, `"NomicEmbedTextV15"`, etc.) are preserved by the new preset table — no `model_id` changes required for catalog entries that used FastEmbed-style names.
- **Removed `fastembed/cuda` and `fastembed/metal` activations** from `gpu-cuda` / `gpu-tensorrt` / `gpu-metal`. ORT EPs alone now drive embedding-model GPU execution.
- **Removed `fastembed?/ort-…` conditional activations** from `provider-onnx`, `provider-onnx-dynamic`, and the internal `_ort-fetched-base`. The ORT linking-mode story is now single-source-of-truth — no more "FastEmbed needs an ORT mode" build.rs check.
- **Schema:** `schemas/provider-options/fastembed.schema.json` deleted. `schemas/model-catalog.schema.json` no longer routes `local/fastembed` to a sibling schema.

### Added

- **Preset registry for embedding models** (`src/provider/local_onnx/presets.rs`): aliases → `(hf_repo, onnx_path, tokenizer_path, additional_files, dimensions, pooling, normalize, max_seq_len, token_type_ids)`. Pass-through is supported for any model_id not in the table — supply the same fields via `options`.
- **Per-task ONNX option validation** (`local/onnx` + `Embed`): `pooling` (`"cls" | "mean" | "max"`), `normalize`, `dimensions`, `max_seq_len`, `token_type_ids`, `output_name`, `tokenizer_path` join the existing `artifact` / `max_batch_size` / `execution_providers` / `graph_optimization_level` / thread-count keys.
- **External-data ONNX support**: presets carry `additional_files` for models that ship their weights in a `.onnx_data` sidecar (BGE-M3, multilingual-E5-large), so ORT can resolve the relative external-data references at session creation.
- **Two new expensive-tests**: `test_local_onnx_bge_small_embedding` (native) and `test_local_onnx_vs_fastembed_parity_full` (parameterized parity sweep across all 25 presets, gated on both `provider-onnx` and `provider-fastembed`). The parity test is gone now that fastembed is gone, but it lived long enough to validate the migration.

### Migration

- **Catalog YAML/JSON**: `provider_id: local/fastembed` → `provider_id: local/onnx`. Alias strings unchanged.
- **Code**: `LocalFastEmbedProvider::new()` → `LocalOnnxProvider::new()`. Same `EmbeddingModel` trait, same return type.
- **Cargo.toml**: drop `provider-fastembed` from your feature list. If you only had it for embedding, `provider-onnx` (or `provider-onnx-dynamic`) is now the single source.
- **Cache directory**: embeddings now cache under `.uni_cache/onnx-embed/<sanitized-repo>/` instead of `.uni_cache/fastembed/<alias>/`. Old cache entries are ignored (won't be reused, but won't break anything either).

## [0.7.0] - 2026-04-26

Provider unification: `local/onnx-reranker` is folded into `local/onnx`. Aligns ONNX with the rest of the provider matrix (one provider per backend, multiple tasks per provider — same shape as `cohere`, `mistralrs`).

### Breaking

- **Removed `LocalOnnxRerankerProvider` (the type) and the `local/onnx-reranker` provider id.** The cross-encoder rerank task is now served by the existing `LocalOnnxProvider`, which declares both `ModelTask::Raw` and `ModelTask::Rerank` in its capabilities and dispatches in `load()`.
- **Migration:** wherever you registered `LocalOnnxRerankerProvider`, register `LocalOnnxProvider::new()` instead (or omit the second registration entirely if `LocalOnnxProvider` is already registered for `Raw`). In catalog specs, change `provider_id: "local/onnx-reranker"` to `provider_id: "local/onnx"`; the `task: "Rerank"` field already routes to the correct backend.
- **File layout:** `src/provider/local_onnx.rs` is now the unified provider entry; raw and rerank task implementations moved to private `src/provider/local_onnx/raw.rs` and `src/provider/local_onnx/rerank.rs` submodules. No effect on the public API beyond the removal above.
- **Renamed `OnnxRunner` → `RawTensorModel`.** The other three task traits (`EmbeddingModel`, `RerankerModel`, `GeneratorModel`) name semantic capabilities; `OnnxRunner` named the transport mechanism (ONNX) instead. The new name aligns with the `*Model` suffix and describes what the trait is — raw tensor I/O. The trait shape (`run`, `run_batch`, `input_signature`, `output_signature`, `max_batch_size`, `active_execution_providers`) is unchanged. Rename also applies to:
    - `runtime.onnx_runner(alias)` → `runtime.raw_tensor_model(alias)`
    - `reliability::InstrumentedOnnxRunner` → `InstrumentedRawTensorModel`
    - `mock::MockOnnxRunner` → `MockRawTensorModel`
    - `src/traits/onnx_runner.rs` → `src/traits/raw_tensor_model.rs`
    - `tests/onnx_runner_test.rs` → `tests/raw_tensor_model_test.rs`
  `ModelTask::Raw` (the task variant in catalog specs) is unchanged.

See [`docs/migrations/0.7.0-onnx-provider-unification.md`](docs/migrations/0.7.0-onnx-provider-unification.md) for the full migration guide.

## [0.6.1] - 2026-04-26

Follow-up review pass on `0.6.0` (PR #22 review feedback):

### Fixed
- `provider-fastembed` now requires an ORT linking mode (`provider-onnx` / `provider-onnx-dynamic` / a `gpu-*`). Previously it pulled `dep:ort` without selecting a linking strategy, producing a confusing ort-internal build error. `build.rs` now emits a clear configuration error.
- `OnnxExecutionProvider::from_str` returns `Option<Self>` and surfaces unknown EP names (e.g. typos) as `RuntimeError::Config` instead of silently selecting CPU. Added enum variants for `Rocm`, `OpenVino`, `Qnn`, `TensorRt`, and `WebGpu` so users on those targets actually get the requested provider.
- `LocalOnnxProvider::load` now validates the requested EP list **before** the load-dynamic preflight. Misconfigurations (e.g. requesting `cuda` without `gpu-cuda`) report the precise feature-mismatch error, matching `LocalOnnxRerankerProvider`.
- `parse_execution_providers_option` treats an empty array as `None` (fall back to feature-aware default) and returns `RuntimeError::Config` for entries with the wrong shape.
- Aligned `libloading` direct dep to `0.9` to match `ort 2.0.0-rc.12`'s transitive version (was `0.8`, would have pulled two majors into the graph).
- `preflight_dylib_test` second case bumped from a 1s to a 60s timeout. The legitimate slow path is a real HuggingFace model download; the failure mode we're guarding against (the ort `OnceLock` deadlock) is *forever*, so any finite bound detects it.
- Updated stale `docs/migrations/0.5.4-load-dynamic.md` reference in `preflight_ort_dylib`'s error message and test assertion to `0.6.0-final-feature-surface.md`.
- Packaging fix: added `/build/**` to `[package].include` so `cargo publish`'s verify step finds `build/ort_vendor.rs`.

## [0.6.0] - 2026-04-26

This is the **final, settled** ONNX/GPU feature surface. The 0.5.x series went through three iterations (0.5.4 / 0.5.5 / 0.5.6) trying to get the deployment story right; 0.6.0 closes it. After this release, the canonical migration doc is `docs/migrations/0.6.0-final-feature-surface.md` — the per-version migration trail (`0.5.4-load-dynamic.md`, `0.5.6-bundled-cpu-default.md`) has been deleted.

### Breaking

- **Every `gpu-*` feature now bundles its ONNX Runtime automatically at build time.** No `ORT_DYLIB_PATH` setup at runtime for any vendor that has a prebuilt distribution. Users who previously had to manage tarballs at deploy time should remove that setup from their deploy scripts. Specifically:
    - `gpu-cuda`, `gpu-tensorrt`, `gpu-coreml`: pyke fetches the right artifact at build via `ort/download-binaries`. EP sidecars (`libonnxruntime_providers_cuda.so` etc.) are staged into `target/<profile>/` automatically by `ort/copy-dylibs`.
    - `gpu-directml`, `gpu-qnn`: a new `build/ort_vendor.rs` fetcher downloads Microsoft's prebuilt at build time, stages the dylibs into `OUT_DIR`, and embeds an absolute rpath into the binary.
    - `gpu-rocm`, `gpu-openvino`: there's no central prebuilt for these (AMD and Intel ship their own packages). build.rs emits a `cargo:warning=` explaining the deployment requirement; users supply the vendor's runtime via `ORT_DYLIB_PATH`. Internally these activate the same load-dynamic base as `provider-onnx-dynamic`.
- **Mutual exclusion**: only one of `provider-onnx` (Mode A), Mode B (`gpu-cuda` / `gpu-tensorrt` / `gpu-coreml` / `gpu-wgpu`), Mode C (`gpu-directml` / `gpu-qnn` / `gpu-rocm` / `gpu-openvino`), or `provider-onnx-dynamic` (Mode E) may be active at a time. `build.rs` panics with a clear message naming the conflict.

### Added

- **`gpu-tensorrt`** — pyke's NVIDIA TensorRT-RTX bundle. NVIDIA only.
- **`gpu-wgpu`** — vendor-neutral GPU via WebGPU. Single binary works across NVIDIA/AMD/Intel on Linux + Windows. Lower per-vendor performance than native EPs but covers any GPU.
- **`build/ort_vendor.rs`** module — fetches Microsoft's prebuilt ONNX Runtime tarballs at build time for vendors pyke doesn't ship. Includes SHA256-pinned vendor specs, idempotent caching at `~/.cache/uni-xervo/ort/<ORT_VERSION>/`, and a `UNI_XERVO_ORT_DIST_DIR` bypass for sandboxed builds.
- **`docs/migrations/0.6.0-final-feature-surface.md`** — single canonical migration guide covering all paths from 0.5.x.
- **Cross-platform support matrix** in `docs/proposals/gpu-runtime-architecture.md`.

### Changed

- ORT pinned version updated to **1.25.0** (was 1.24.x via api-24 in earlier 0.5.x). pyke ships `cu13` by default; override with `ORT_CUDA_VERSION=12` if needed.
- `provider-onnx-dynamic` semantics unchanged from 0.5.6: power-user escape, BYO tarball.
- `preflight_ort_dylib` and `libloading` are now compiled in only under load-dynamic modes (which now includes `_ort-fetched-base` for the Mode-C and Mode-D features).

### Removed

- `docs/migrations/0.5.4-load-dynamic.md` — described an obsolete intermediate state.
- `docs/migrations/0.5.6-bundled-cpu-default.md` — same.

### Migration cheatsheet

| Were on `0.5.x` with… | Switch to `0.6.0`'s | Action |
|---|---|---|
| `provider-onnx` (had to set `ORT_DYLIB_PATH`) | `provider-onnx` | Now bundles. Remove `ORT_DYLIB_PATH` from deploy. |
| `provider-onnx` (already bundled in 0.5.6) | `provider-onnx` | No change. |
| `gpu-cuda` | `gpu-cuda` | Now bundles. Remove `ORT_DYLIB_PATH`. cuDNN still a host requirement. |
| `gpu-coreml` | `gpu-coreml` | Now bundles via pyke macOS. |
| `gpu-directml` | `gpu-directml` | Now bundles via build.rs MS fetch. |
| `gpu-rocm` / `gpu-openvino` | same + `provider-onnx-dynamic` (or activate alone — implies dynamic internally) | These remain BYO; no change in deployment, just bundled cargo features. |
| `provider-onnx-dynamic` | `provider-onnx-dynamic` | No change — power-user escape preserved. |

## [0.5.6] - 2026-04-26

### Breaking

- **`provider-onnx` semantics flip from load-dynamic back to bundled CPU.** Restores the pre-`0.5.4` zero-config user experience: `cargo build --features provider-onnx` produces a self-contained binary with `libonnxruntime.a` statically linked in (~70 MB extra after dead-code elimination). No `ORT_DYLIB_PATH`, no separate runtime tarball, no env-var setup.
- Users who relied on `ORT_DYLIB_PATH` and the load-dynamic deployment model in `0.5.4` / `0.5.5` must switch their feature flag from `provider-onnx` to `provider-onnx-dynamic`. No source-code changes needed; the API surface is identical. See `docs/migrations/0.5.6-bundled-cpu-default.md` for step-by-step guidance.

### Added

- **`provider-onnx-dynamic` feature** for the load-dynamic deployment model. Tiny binary, requires `ORT_DYLIB_PATH` at runtime, supports any vendor EP (CUDA / ROCm / CoreML / DirectML / OpenVINO / QNN) that's in the runtime tarball.
- **GPU features auto-imply `provider-onnx-dynamic`** — `gpu-cuda`, `gpu-rocm`, `gpu-coreml`, `gpu-directml`, `gpu-openvino`, `gpu-qnn` all activate the load-dynamic path automatically. (pyke's bundled CPU lib has no GPU EPs, so combining `gpu-*` with bundled `provider-onnx` is a configuration error.)
- **Build-time mutual-exclusion guard**: `build.rs` panics with a clear message if both `provider-onnx` and `provider-onnx-dynamic` are activated simultaneously. Catches a common feature-unification mistake before the inevitable ort link error.
- New migration guide at `docs/migrations/0.5.6-bundled-cpu-default.md`.

### Changed

- `dep:libloading` is now activated only by `provider-onnx-dynamic` (it's used by the deadlock preflight, which is meaningless in the bundled-CPU mode where there's no dlopen).
- `preflight_ort_dylib` and `default_dylib_name` in `provider::onnx_ep` are now `#[cfg(feature = "provider-onnx-dynamic")]`. Compiled out of bundled-CPU builds entirely.
- The order of validation in `OnnxCrossEncoder::load` and `LocalOnnxProvider::load` is now: EP-list validation → preflight → HF download → ort session. EP-feature mismatches (e.g. requesting `cuda` without `gpu-cuda`) now surface a precise "feature not enabled" error rather than being masked by a missing-dylib error.
- `fastembed`'s `ort` linking-mode is now selected through `fastembed?/ort-download-binaries` (under `provider-onnx`) and `fastembed?/ort-load-dynamic` (under `provider-onnx-dynamic`) feature-conditional activation. Its `Cargo.toml` dep no longer hard-codes a linking mode.

### Migration cheatsheet

| Were on `0.5.5`'s | Switch to `0.5.6`'s |
|---|---|
| `provider-onnx` (had to set `ORT_DYLIB_PATH`) | `provider-onnx-dynamic` (env var still set) |
| `provider-onnx` + want bundled CPU | `provider-onnx` (now bundles; remove `ORT_DYLIB_PATH` from deploy) |
| `gpu-cuda` (had to set `ORT_DYLIB_PATH`) | `gpu-cuda` (no source change; `provider-onnx-dynamic` implied) |
| `gpu-rocm`, `gpu-coreml`, `gpu-directml`, `gpu-openvino`, `gpu-qnn` | same — each now implies `provider-onnx-dynamic` |

## [0.5.5] - 2026-04-25

### Fixed

- **`provider-onnx` no longer hangs indefinitely when the ONNX Runtime dynamic library is missing.** Previously, a misconfigured `ORT_DYLIB_PATH` (or its absence on a host without system ORT) would cause any ORT-backed session creation to block forever in `futex_wait`. Root cause: an upstream re-entrant `OnceLock` deadlock in `ort` 2.0.0-rc.12's load-dynamic error path ([pykeio/ort#560](https://github.com/pykeio/ort/issues/560), fixed upstream in [`17ed7277`](https://github.com/pykeio/ort/commit/17ed7277) but **not yet released** as of `=2.0.0-rc.12`).
- New pre-flight check `provider::onnx_ep::preflight_ort_dylib` runs before any ORT API call in `LocalOnnxProvider::load` and `OnnxCrossEncoder::load`. It attempts `libloading::Library::new` against `ORT_DYLIB_PATH` (or the platform default) and converts a load failure into a clear `RuntimeError::Config` in milliseconds, with a pointer to `docs/migrations/0.5.4-load-dynamic.md`.
- New regression test `tests/preflight_dylib_test.rs` — guards against re-introducing the hang. Both tests complete in <10ms.

### Added

- `libloading = "0.8"` as an optional dep gated by `provider-onnx`. Already present transitively via ort's `load-dynamic`; we now expose it directly so the preflight can use it.

## [0.5.4] - 2026-04-25

### Breaking

- The `ort` crate is now compiled with `load-dynamic` instead of `download-binaries`. The CPU-only ONNX Runtime binary that previously shipped with the Rust crate is no longer bundled. Consumers must download Microsoft's official ONNX Runtime release tarball matching their target hardware and set `ORT_DYLIB_PATH` (or rely on system library paths) before any ORT-backed provider session is created. See `docs/migrations/0.5.4-load-dynamic.md` for step-by-step instructions and `docs/proposals/gpu-runtime-architecture.md` for the rationale.
- `fastembed` similarly switched from `ort-download-binaries` to `ort-load-dynamic`. Same deployment requirement.
- The previous "build with `gpu-cuda`, get CUDA at runtime" promise was already broken (the bundled ORT had no CUDA EP); load-dynamic makes the contract honest. Existing `LocalOnnxProvider` / `LocalOnnxRerankerProvider` users on CPU-only hosts must now point `ORT_DYLIB_PATH` at the CPU tarball before tests/binaries run.

### Added

- **Runtime-selectable execution providers via ORT load-dynamic.** A single uni-xervo binary can now run on NVIDIA / AMD / Apple / DirectML / Intel hardware by swapping the ORT tarball at deploy time — no rebuild required. Per-spec EP selection through `options.execution_providers` (already supported) is now the canonical knob.
- New cargo features as **default-EP-list hints** (additive over the universal load-dynamic base):
  - `gpu-rocm = ["ort/rocm"]` — Linux only.
  - `gpu-openvino = ["ort/openvino"]` — Linux + Windows.
  - `gpu-qnn = ["ort/qnn"]` — Windows ARM64 (Snapdragon Hexagon NPU).
- `gpu-metal` is no longer commented out — declaring the feature is safe on all platforms; the `build.rs` guard fails fast if you try to *activate* it on a non-Apple target.
- `RawTensorModel::active_execution_providers()` and `RerankerModel::active_execution_providers()` — return the requested EP list as stable string ids (e.g. `["cuda", "cpu"]`). Default impls return empty for non-ORT runners. Documents the "requested vs. attached" distinction (the ORT 2.0 Rust binding doesn't yet expose `Session::providers()`).
- Build script (`build.rs`) now guards `gpu-metal`, `gpu-coreml`, `gpu-directml`, `gpu-rocm`, `gpu-qnn` to their respective target operating systems with copy-pasteable error messages.
- New tests: `tests/onnx_ep_resolution_test.rs` (3 tests, no I/O) verifies feature-gated EP validation runs *before* any HF download — wrong EP requests fail fast with a clear `Config` error.
- Inline `active_execution_providers()` assertion added to the existing reranker e2e test.

### Changed

- `gpu-cuda` is now an opt-in *hint* rather than a load-bearing build flag for the ORT path. The CUDA EP is available at runtime to any load-dynamic binary whose ORT tarball includes it. The candle/mistralrs CUDA paths still require `gpu-cuda` (and `nvcc` at build time) — that's a structural property of those upstream crates.
- `OnnxCrossEncoder::load` now validates the requested execution-provider list **before** downloading any HF assets. A misconfigured spec (e.g. CUDA requested without `gpu-cuda` enabled) errors out instantly instead of after a multi-megabyte download.
- The `Cargo.toml` GPU feature block is reorganized into "Tier 1" (runtime-selectable through ORT) and "Tier 2" (build-locked through candle/mistralrs) groups with inline rationale.

## [0.5.3] - 2026-04-25

### Added
- Execution-provider selection for `LocalOnnxRerankerProvider`. Specify `"execution_providers": ["cuda"]` (or `["cuda", "cpu"]`, etc.) in the catalog spec options to run reranking on CUDA / CoreML / DirectML, mirroring the knob already on `LocalOnnxProvider`. Without the option, defaults to `[Cuda, Cpu]` when `gpu-cuda` is enabled, `[Cpu]` otherwise.
- New `gpu_cuda_inference_test` module with `EXPENSIVE_TESTS=1`-gated tests that exercise reranker, embed, and mistralrs generate on CUDA. Includes inline doc on host-side requirements (ORT-CUDA shared libs, candle-cuda PTX-toolchain match).

### Changed
- Extracted shared ONNX execution-provider selection (`OnnxExecutionProvider` enum, `build_execution_providers`, `default_execution_providers`, `parse_execution_providers_option`) into a new `provider::onnx_ep` module. Both `LocalOnnxProvider` and `LocalOnnxRerankerProvider` import from it; the duplicated copy that previously lived in `local_onnx.rs` is gone.
- `LocalOnnxOptions` (used by `LocalOnnxProvider`) now accepts `execution_providers` as either a JSON array (`["cuda", "cpu"]`) or a single string (`"cuda"`); previously only the array form parsed.

## [0.5.2] - 2026-04-25

### Added
- `LocalOnnxRerankerProvider` (`local/onnx-reranker`) — local ONNX cross-encoder reranker (BERT-style models such as `cross-encoder/ms-marco-MiniLM-L6-v2`). Handles HF download, WordPiece tokenization, and batched ORT inference. Returns raw logits as `ScoredDoc`s sorted descending; the caller is responsible for any normalization (sigmoid, etc.). Hoisted from uni-db so all uni-xervo consumers get local rerank without re-implementing it. Gated by the existing `provider-onnx` feature.
- Deferred-followups proposal at `docs/proposals/onnx-reranker-followups.md` capturing four enhancements considered during the hoist (score-normalization API, shared HF download helper, BERT-pair tokenizer extraction, ORT session pool).

### Changed
- `provider-onnx` feature now also pulls `tokenizers` (previously only gated by `provider-candle`), required by the new reranker provider.

## [0.4.0] - 2026-04-15

### Added
- Configurable `embedding_dimensions` option for all remote embedding providers (OpenAI, Gemini, Cohere, Mistral, Azure OpenAI, Voyage AI). Overrides the default dimensions reported by the model handle, enabling immediate adoption of new upstream models without a crate release.
- Configurable `api_version` option for the Gemini provider (default: `v1beta`), allowing users to switch API versions as Google promotes endpoints to stable.

### Fixed
- Corrected default embedding dimensions for newer upstream models: `gemini-embedding-001` (3072), Cohere `embed-v4.0` (1536), Mistral `codestral-embed` (1536), Voyage AI `voyage-3-lite` (512).

## [0.3.0] - 2026-04-13

### Breaking Changes
- Updated all dependencies to latest versions and migrated to mistralrs 0.8 API.

### Changed
- Upgraded `mistralrs` from 0.4 to 0.8.
- Upgraded `candle-core`, `candle-nn`, `candle-transformers` to 0.10.
- Upgraded `tokenizers` to 0.22.
- Upgraded `fastembed` to 5.9.0.
- Upgraded `reqwest` to 0.13.
- Upgraded `thiserror` to 2.
- Upgraded `criterion` (dev) to 0.8.

## [0.2.0] - 2026-03-12

### Breaking Changes
- `GeneratorModel::generate()` signature changed: `messages: &[String]` → `messages: &[Message]`.
- `GenerationResult` now has two additional required fields: `images: Vec<GeneratedImage>`, `audio: Option<AudioOutput>`.
- All provider implementations updated accordingly.
- Migration: replace `&["text".to_string()]` with `&[Message::user("text")]`.

### Added
- **Multimodal message types**: `Message`, `MessageRole`, `ContentBlock`, `ImageInput` for structured conversation input.
- **Vision generation**: Process images + text via mistralrs vision pipeline (`"pipeline": "vision"`).
- **Image generation**: Diffusion pipeline (FLUX) via mistralrs (`"pipeline": "diffusion"`).
- **Speech synthesis**: Audio generation (Dia) via mistralrs (`"pipeline": "speech"`).
- **GGUF model support**: Load quantized GGUF models in mistralrs text pipeline.
- **dtype control**: Configure model precision (`f32`, `f16`, `bf16`, `auto`) for mistralrs pipelines.
- **ISQ quantization**: In-situ quantization support for text and vision pipelines.
- **Embedding validation**: NaN/Inf detection in embedding outputs.
- **Explicit message roles**: `System`, `User`, `Assistant` roles replace index-based role inference.
- `GenerationOptions` gains `width` and `height` fields for diffusion image sizing.

## [0.1.1] - 2026-02-23

### Changed
- Upgraded `thiserror` from 1.0 to 2, aligning with the transitive dependency ecosystem.
- Upgraded `reqwest` from 0.11 to 0.12, replacing the legacy `hyper` 0.14 / `http` 0.2 HTTP stack with the modern `hyper` 1.x stack and eliminating several duplicate transitive dependencies.

## [0.1.0] - 2026-02-19

### Added
- Provider options schema files and runtime validation for provider-specific options.
- Dedicated public API and error taxonomy improvements with clearer error variants.
- Minimal GitHub Actions workflows for CI and crates.io publishing.
- Expanded website documentation content aligned with Uni-Xervo branding.

### Changed
- Improved runtime load timeout handling and reliability behavior.
- Refined packaging metadata and include list for cleaner crates.io distribution.

### Fixed
- Correct provider capability declarations for Gemini embedding support.
- Gated mock module export for test/testing builds only.
