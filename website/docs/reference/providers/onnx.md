# local/onnx

## Uni-Xervo support

- Provider ID: `local/onnx`
- Feature flag: `provider-onnx` (or `provider-onnx-dynamic`)
- Capabilities: `raw`, `rerank`, `embed`, `embed_sparse`,
  `embed_multi_vector`, `embed_hybrid`, `embed_image`, `nlp`, `ocr`,
  `document_extract` (scaffold)

A single ONNX-Runtime-backed provider that dispatches by `task`:

- **`raw`** — arbitrary tensor execution (`RawTensorModel` trait).
- **`rerank`** — cross-encoder rerankers via `RerankerModel`.
- **`embed`** — dense text embeddings via `EmbeddingModel` (replaces the retired `local/fastembed` provider as of 0.8.0; the same alias strings still resolve).
- **`embed_image`** — ViT-style image embedders (SigLIP-2, CLIP, OpenCLIP)
  via `ImageEmbeddingModel`. New in 0.13.0.
- **`nlp`** — structured NLP via `NlpModel`, serving two generations of the
  kniv family. `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` is the v1
  cascade (POS / NER / DEP / SRL / dialog-act CLS in one forward pass);
  `dragonscale-ai/kniv-v7-xlarge-u-en` and `-d-en` are kniv v7 (eight graphs
  over word-level windows, adding morphology, lemmas, sentiment,
  coreference, caller-supplied entity types and relations with polarity,
  modality, holder and time). New in 0.13.0; result surface extended in
  0.14.0 (merged entities, full dialog-act distribution, word alignment,
  exposed label vocabularies) and in 0.19.0 (kniv v7). See the
  [Structured NLP guide](../../guides/nlp.md).
- **`ocr`** — text recognition (CRNN + CTC) via `OcrModel`, with an
  **optional DBNet detection stage** for full-page OCR. Recognition is new
  in 0.13.0; the two-stage detect→recognize path is new in 0.15.0 (see
  [OCR-only keys](#ocr-only-keys-task-ocr)).
- **`document_extract`** — VLM-based document parsing via
  `DocumentExtractionModel`. On `local/onnx` this remains a scaffold:
  catalog wiring, options validation, the reusable `autoreg::greedy_decode`
  decoder helper, and three style-specific output parsers (Granite-Docling
  DocTags / MinerU Markdown / olmOCR Markdown — now in the shared
  `doc_parse` module) all ship and are unit tested, but `extract()` returns
  `RuntimeError::Unavailable` until an upstream canonical ONNX export of one
  of the three target VLMs ships. The olmOCR-2 path is **live today on
  [`remote/llamacpp`](llamacpp.md#document_extract)** — a GGUF document VLM
  on a `llama-server` started with `--mmproj`, which reuses the same
  `doc_parse` parsers.

## Provider options

Options are validated per task (unknown keys are rejected with a precise `RuntimeError::Config`).

### Common keys (all tasks)

- `artifact` (string) — explicit `.onnx` filename within an HF repo (auto-detected if a single match exists).
- `max_batch_size` (integer)
- `execution_providers` — array of EP identifiers. Accepted by option validation: `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`. Other identifiers (e.g. `"rocm"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"`) are rejected at catalog load with `RuntimeError::Config`. Defaults to a feature-aware list: `["cuda", "cpu"]` under `gpu-cuda`, `["coreml", "cpu"]` under `gpu-metal`, `["cpu"]` otherwise. Vendor EPs are never default — opt in explicitly. Selecting `"cuda"`/`"coreml"` still requires the matching GPU feature (or a `provider-onnx-dynamic` build pointed at a vendor ORT library via `ORT_DYLIB_PATH`). **Per-spec**, so a single catalog can mix GPU and CPU placement on a `gpu-cuda` build (e.g. embedder on `["cpu"]`, reranker on `["cuda", "cpu"]`) — see the [mixed-catalog recipe](../../guides/onnx.md#mixing-gpu-and-cpu-placement-in-one-catalog). No other provider takes a per-spec device option: `local/candle` places from the build features alone, and `remote/llamacpp` runs inference in the server process, where the device is a `llama-server` flag. If a requested EP fails to initialize (missing cuDNN, wrong CUDA major version, etc.), ORT silently falls through to the next EP in the list — see [GPU setup → verifying the GPU EP actually loaded](../../guides/gpu-setup.md#verifying-the-gpu-ep-actually-loaded) for how to detect this. See `docs/migrations/0.9.0-feature-surface.md` for vendor-build setup.
- `graph_optimization_level` — `"disable" | "basic" | "extended" | "all"`.
- `inter_op_num_threads`, `intra_op_num_threads` (integers)
- `cache_dir` (string) — overrides `UNI_CACHE_DIR` and the default `.uni_cache/onnx-…/` location.

### Embed-only keys

- `pooling` — `"cls" | "mean" | "max" | "last-token"`. Required when no preset matches. Use `"last-token"` for decoder-style LLM embedders (Qwen3-Embedding, NV-Embed) that take the rightmost non-pad hidden state as the sentence representation; classic BERT-family embedders use `"cls"`, sentence-transformers use `"mean"`.
- `normalize` (bool, default `true`) — apply L2 normalization after pooling.
- `dimensions` (integer) — required when no preset matches; validated against the model's actual output.
- `max_seq_len` (integer, default `512`) — truncation cap for tokenizer input.
- `token_type_ids` (bool) — whether the model accepts a `token_type_ids` tensor (BERT-family yes; MPNet, ModernBERT, BAAI's M3 export, Qdrant's E5-large export, Qwen3-Embedding — no). Required for pass-through models.
- `tokenizer_path` (string, default `"tokenizer.json"`) — relative path within the HF repo.
- `output_name` (string) — explicit ONNX output to read; defaults to the first session output (typically `last_hidden_state`).

Decoder-LM embedders that declare `position_ids` and `past_key_values.*` placeholder inputs (e.g. `onnx-community/Qwen3-Embedding-0.6B-ONNX`) are auto-detected from the session metadata and fed correctly — no extra option is needed beyond `pooling: "last-token"`.

### Sparse-embed-only keys (`task = embed_sparse`)

- `sparse_method` — `"mlm" | "lexical"`. Required when no preset matches.
  `"mlm"` is the SPLADE recipe (MLM logits over the full vocab →
  `log(1 + ReLU(x))` + masked max-pool; term ids are vocabulary indices with
  expansion). `"lexical"` is the BGE-M3 sparse head (one ReLU weight per token,
  keyed by input token id, max over duplicates; no expansion).
- `output_name` (string, optional) — explicit output tensor to read.
- `output_index` (integer, default from preset / `0`) — positional output
  selection for multi-output graphs (BGE-M3 emits dense / sparse / ColBERT;
  sparse is output `1`). Used when `output_name` is absent.
- `top_k` (integer, optional) — keep only the highest-weight terms per vector.
- `max_seq_len` (integer, default `512`), `tokenizer_path` (string),
  `token_type_ids` (bool) — as for dense embed.

The `embed_sparse` output is one `Vec<(term_id, weight)>` per input. Score two
vectors with [`uni_xervo::score::sparse_dot`](../../guides/sparse-embeddings.md).
Presets: `prithivida/Splade_PP_en_v1` (`mlm`) and the BGE-M3 sparse head
(`lexical`). See the [sparse embeddings guide](../../guides/sparse-embeddings.md).

### Multi-vector-only keys (`task = embed_multi_vector`)

- `dimensions` (integer) — per-token vector dim; required when no preset matches.
- `normalize` (bool, default `true`) — L2-normalize each per-token vector
  (cosine MaxSim).
- `drop_special_tokens` (bool, default `false`) — drop common special-token
  vectors (`[CLS]`/`[SEP]`/`<s>`/`</s>` etc.) in addition to padding.
- `output_name` (string, optional) / `output_index` (integer, default from
  preset / `0`) — output selection; ColBERT is output `2` for BGE-M3's
  multi-output graph.
- `max_seq_len` (integer, default `512`), `tokenizer_path` (string),
  `token_type_ids` (bool) — as for dense embed.

The `embed_multi_vector` output is a ragged `Vec<Vec<f32>>` per input (one vector
per retained token). Score with
[`uni_xervo::score::max_sim`](../../guides/multi-vector-embeddings.md). Presets:
`answerdotai/answerai-colbert-small-v1` and the BGE-M3 ColBERT head. See the
[multi-vector embeddings guide](../../guides/multi-vector-embeddings.md).

### Hybrid-only keys (`task = embed_hybrid`)

Single-pass multi-head embedding: one forward pass on a multi-output graph yields
the dense, sparse, and ColBERT heads together. This task is **preset-driven** —
the preset declares each head's output and recipe, so options carry only
pass-wide globals (per-head settings live in the preset, not the catalog):

- `max_seq_len` (integer, default from preset / `512`)
- `tokenizer_path` (string), `token_type_ids` (bool) — as for dense embed.
- `top_k` (integer, optional) — cap on retained sparse terms per vector.

Resolve with `runtime.hybrid_embedder(alias)`; call
`model.embed(texts, heads)` with a
[`HeadSet`](../index.md) (`HeadSet::ALL`, or a subset like
`HeadSet::DENSE | HeadSet::SPARSE`). The `HybridEmbedResult` carries `dense`
(`Vec<Vec<f32>>`), `sparse` (`Vec<SparseVector>`), and `multi_vector`
(ragged `Vec<Vec<Vec<f32>>>`), each `Some` iff its head was requested and
exposed (see `model.available_heads()`). Models without a hybrid preset return a
`RuntimeError::Config` — use the per-task resolvers instead. Preset:
`BGEM3Hybrid` (all three BGE-M3 heads of `aapot/bge-m3-onnx`). See the
[multi-vector embeddings guide](../../guides/multi-vector-embeddings.md#bge-m3-three-heads-one-export).

### Rerank-only keys

- `max_seq_len` (integer, default `512`)
- `style` — `"cross-encoder"` (default) or `"generative"`. Cross-encoder is for single-logit BERT-family models like `cross-encoder/ms-marco-MiniLM-L6-v2` and `BAAI/bge-reranker-base` (auto-detects whether the model takes a `token_type_ids` input). Generative is for decoder-LM rerankers like `Qwen3-Reranker-0.6B` that score by emitting `yes`/`no` token logits.
- `instruction` (string, generative only) — task description shown to the reranker. Default: `"Given a web search query, retrieve relevant passages that answer the query"`. Customize for domain-specific tasks (code search, medical literature, etc.).

### NLP-only keys (`task = nlp`)

Shared by both kniv generations:

- `tokenizer_path` (string, default `"tokenizer.json"`).
- `generation` (string, `"v1"` or `"v7"`) — pins the generation instead of
  inferring it. Inference: a repo carrying `onnx/onnx_meta.json` is v7, a repo
  carrying `onnx/cascade.onnx` is v1.

kniv v1 cascade only:

- `onnx_path` (string, default `"onnx/cascade.onnx"`) — path within the HF
  repo to the cascade ONNX file.
- `label_maps_path` (string, default `"label_maps.json"`) — JSON file
  containing the model's per-head id-to-label arrays (`pos`, `ner`,
  `srl`, `cls`, `deprel`). Each entry's position is the model's class
  index.
- `max_seq_len` (integer, default `128`) — chunking cap. Inputs longer
  than this are split into non-overlapping token windows; per-token byte
  offsets in `NlpToken::{start, end}` stay correct because tokenization
  runs once over the full text.

kniv v7 only:

- `variant` (string, default `""` = fp32) — the precision **fallback chain**
  for the three encoder graphs. `"int8,fp16"` loads, per graph, the first
  suffix published for it, else the fp32 file; that resolves to `plain-int8`,
  `marker-int8` and `label-fp16`, because the label graph publishes no INT8
  file (it runs once per schema and is cached). Only the files the chain needs
  are downloaded.
- `calibration_path` (string, default `"calibration.json"`) — the release's
  dev-tuned decoding thresholds, readable via `NlpModel::calibration()` and
  overridable per request.
- `label_bank_path` (string, default `"label_bank.json"`) — marks the
  relations whose thresholds were tuned individually; the rest fall back to
  `unseen_offset`.

The two sets do not overlap, and mixing them is rejected at catalog
registration — before any download. v7 takes no `max_seq_len`: its window is
512 words and its subword budget comes from `onnx_meta.json` (1,024 for
UltraBERT, 512 for DeBERTa).

`graph_optimization_level`, `intra_op_num_threads` and `inter_op_num_threads`
are honoured on the v7 path. Set `intra_op_num_threads` when the model runs
alongside an embedder, or ONNX Runtime takes every core.

On GPUs use `"variant": "fp16"`: the INT8 graphs use dynamic per-channel
quantisation, which CUDA serves through its own CPU fallback.

SRL multi-pass orchestration on the v1 path (one forward per detected verb via
`predicate_idx`) is handled provider-side and invisible to callers; v7 scores
every predicate in one batched call.

### Image-embed-only keys (`task = embed_image`)

- `onnx_path` (string, required) — `.onnx` path within the HF repo.
- `image_size` (integer, required) — square input edge in pixels (e.g.
  384 for SigLIP-2-So400m-patch16-384).
- `dimensions` (integer, required) — expected output embedding dim.
- `normalization` (`"siglip"` | `"imagenet"`, default `"siglip"`) —
  SigLIP/SigLIP-2 mean=std=(0.5, 0.5, 0.5); ImageNet stats otherwise.
- `pool` (`"none"` | `"mean"`, default `"none"`) — `"none"` for
  pre-pooled `[batch, dim]` outputs; `"mean"` to mean-pool a
  `[batch, tokens, dim]` patch sequence.
- `normalize` (bool, default `true`) — L2-normalize each row.
- `output_name` (string, optional) — explicit output tensor name override.

Images may be supplied as `ImageInput::Bytes` (PNG/JPEG/WebP);
`ImageInput::Url` is rejected — callers fetch URLs upstream.

### OCR-only keys (`task = ocr`)

Recognition (always required):

- `onnx_path` (string, required) — recognizer (CRNN/CTC) `.onnx` within the repo.
- `char_dict_path` (string, required) — character dictionary file within
  the HF repo, one entry per line. Class 0 is the CTC blank by default.
- `image_height` (integer, default 48), `image_width` (integer, default 320).
- `normalization` (`"siglip"` | `"imagenet"`, default `"imagenet"`). PP-OCR
  recognizers expect `"siglip"` (i.e. `(x/255 − 0.5)/0.5`).
- `blank_class` (integer, default 0).
- `output_name` (string, optional).

Optional detection stage (DBNet) — present `det_onnx_path` to enable two-stage
detect→recognize:

- `det_onnx_path` (string) — DBNet detector `.onnx` within the repo. Its
  presence is what turns on detection.
- `det_model_id` (string, default = the spec's `model_id`) — separate HF repo
  for the detector when it is not co-located with the recognizer.
- `det_limit_side` (integer > 0, default 960) — longest side the page is resized
  to (rounded to a multiple of 32) before detection.
- `det_bin_threshold` (number in `(0, 1)`, default 0.3) — probability-map
  binarization threshold.
- `det_box_score_threshold` (number in `(0, 1)`, default 0.6) — minimum mean
  box probability to keep a region.
- `det_unclip_ratio` (number > 0, default 1.5) — box dilation (unclip) ratio.
- `det_min_box_size` (integer > 0, default 3) — minimum box side in detector px.
- `det_input_name` (string, default `"x"`), `det_output_name` (string, optional).

**Without a detector**, OCR is single-stage recognition: callers pre-crop their
text regions and pass one image per region, and each call returns one
`OcrBlock` per image with a normalized whole-image bbox.

**With a detector**, `recognize()` runs the whole page — detect text regions →
crop → recognize each → order top-to-bottom then left-to-right — returning one
`OcrBlock` per detected region with its pixel-coordinate bbox. A blank page
yields an empty result (no error). Confidence is the average per-character
probability, robust to recognizers that emit raw logits *or* an
already-softmaxed distribution. Detection geometry is axis-aligned in v1
(horizontal text); rotated/multi-column handling is a future refinement.

### Document-extract-only keys (`task = document_extract`)

- `style` (`"granite-docling"` | `"mineru"` | `"olmocr"`, default
  `"granite-docling"`) — selects which output parser to apply.
- `onnx_path` (string, optional once the upstream model ships).
- `tokenizer_path` (string, optional).
- `max_seq_len` (integer, optional).

The shipped output parsers convert the VLM's decoded text into typed
`DocBlock` entries with reading order, optional bboxes, and a
concatenated `plain_markdown` field. End-to-end inference is gated on
an upstream canonical ONNX export — see the module doc comment in the
source for the expected ONNX input schema.

Authoritative option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/crates/uni-xervo/schemas/provider-options/onnx.schema.json>

## Model IDs

`model_id` can be:

- a Hugging Face repo ID (e.g. `"BAAI/bge-small-en-v1.5"`), or
- a built-in alias for the embed task (e.g. `"BGESmallENV15"`, `"all-MiniLM-L6-v2"` — see [Embedding presets](#embedding-presets) below), or
- a local path to a `.onnx` file (raw / rerank tasks).

HF-backed aliases download into the per-task cache directory (`onnx-raw`, `onnx-reranker`, or `onnx-embed`) before ORT session creation. If the repo contains multiple `.onnx` files, set `options.artifact`.

## Embedding presets

The embed task ships with a preset table covering 25 popular text-embedding models. When `model_id` matches a preset alias, the HF repo, ONNX path, pooling kind, dimensions, and `token_type_ids` flag are filled in automatically; per-spec `options` may still override any field.

| Alias (canonical) | HF repo | Dim | Pooling | `token_type_ids` |
|---|---|---:|---|:---:|
| `AllMiniLML6V2` (`all-MiniLM-L6-v2`) | Qdrant/all-MiniLM-L6-v2-onnx | 384 | mean | ✓ |
| `AllMiniLML6V2Q` | Xenova/all-MiniLM-L6-v2 (quantized) | 384 | mean | ✓ |
| `AllMiniLML12V2` | Xenova/all-MiniLM-L12-v2 | 384 | mean | ✓ |
| `AllMiniLML12V2Q` | Xenova/all-MiniLM-L12-v2 (quantized) | 384 | mean | ✓ |
| `AllMpnetBaseV2` (`all-mpnet-base-v2`) | Xenova/all-mpnet-base-v2 | 768 | mean | ✗ |
| `BGESmallENV15` (`bge-small-en-v1.5`) | Xenova/bge-small-en-v1.5 | 384 | cls | ✓ |
| `BGESmallENV15Q` | Qdrant/bge-small-en-v1.5-onnx-Q | 384 | cls | ✓ |
| `BGEBaseENV15` (`bge-base-en-v1.5`) | Xenova/bge-base-en-v1.5 | 768 | cls | ✓ |
| `BGEBaseENV15Q` | Qdrant/bge-base-en-v1.5-onnx-Q | 768 | cls | ✓ |
| `BGELargeENV15` (`bge-large-en-v1.5`) | Xenova/bge-large-en-v1.5 | 1024 | cls | ✓ |
| `BGELargeENV15Q` | Qdrant/bge-large-en-v1.5-onnx-Q | 1024 | cls | ✓ |
| `BGESmallZHV15` | Xenova/bge-small-zh-v1.5 | 512 | cls | ✓ |
| `BGELargeZHV15` | Xenova/bge-large-zh-v1.5 | 1024 | cls | ✓ |
| `BGEM3` | BAAI/bge-m3 (external data) | 1024 | cls | ✗ |
| `NomicEmbedTextV1` | nomic-ai/nomic-embed-text-v1 | 768 | mean | ✓ |
| `NomicEmbedTextV15` (`nomic-embed-text-v1.5`) | nomic-ai/nomic-embed-text-v1.5 | 768 | mean | ✓ |
| `NomicEmbedTextV15Q` | nomic-ai/nomic-embed-text-v1.5 (quantized) | 768 | mean | ✓ |
| `ParaphraseMLMiniLML12V2` | Xenova/paraphrase-multilingual-MiniLM-L12-v2 | 384 | mean | ✓ |
| `ParaphraseMLMiniLML12V2Q` | Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q | 384 | mean | ✓ |
| `ParaphraseMLMpnetBaseV2` | Xenova/paraphrase-multilingual-mpnet-base-v2 | 768 | mean | ✗ |
| `MultilingualE5Small` (`multilingual-e5-small`) | intfloat/multilingual-e5-small | 384 | mean | ✓ |
| `MultilingualE5Base` (`multilingual-e5-base`) | intfloat/multilingual-e5-base | 768 | mean | ✗ |
| `MultilingualE5Large` (`multilingual-e5-large`) | Qdrant/multilingual-e5-large-onnx (external data) | 1024 | mean | ✗ |
| `MxbaiEmbedLargeV1` (`mxbai-embed-large-v1`) | mixedbread-ai/mxbai-embed-large-v1 | 1024 | cls | ✓ |
| `ModernBertEmbedLarge` | lightonai/modernbert-embed-large | 1024 | mean | ✗ |

All 25 presets normalize output to L2 norm ≈ 1.0 and use `max_seq_len: 512` to match `fastembed-rs` defaults.

## Runtime contract

By task:

- **`raw`** → `runtime.raw_tensor_model(alias)` returns `Arc<dyn RawTensorModel>`. Methods: `run`, `run_batch`, `input_signature`, `output_signature`, `max_batch_size`, `active_execution_providers`.
- **`rerank`** → `runtime.reranker(alias)` returns `Arc<dyn RerankerModel>`. Method: `rerank(query, docs)` → `Vec<ScoredDoc>`.
- **`embed`** → `runtime.embedding(alias)` returns `Arc<dyn EmbeddingModel>`. Method: `embed(&[&str])` → `EmbedResult` whose `vectors` field holds one row per input (each of length `dimensions()`, L2-normalized when `normalize: true`); `usage` is `None` for `local/onnx`.
- **`embed_image`** → `runtime.image_embedder(alias)` returns
  `Arc<dyn ImageEmbeddingModel>`. Method: `embed(images)` → `EmbedResult`
  (vectors + optional `TokenUsage`).
- **`nlp`** → `runtime.nlp_model(alias)` returns `Arc<dyn NlpModel>`.
  Method: `analyze(requests)` → `Vec<NlpResult>` with populated `tokens`,
  `sentences`, `frames`, `speech_acts`, and merged `entities` per the requested
  `NlpTasks` bitflag. A v7 model also fills `windows` with the per-window
  analysis — typed entities, coreference clusters and facts — and accepts
  conversation `turns` and an `ExtractionSchema` on the request.
  `NlpModel::label_maps()` exposes the model's per-head label vocabularies and
  `NlpModel::calibration()` its decoding thresholds. See the
  [Structured NLP guide](../../guides/nlp.md) for the full result-type
  reference and a worked example.
- **`ocr`** → `runtime.ocr_model(alias)` returns `Arc<dyn OcrModel>`.
  Method: `recognize(images)` → `Vec<OcrResult>`. With `det_onnx_path`
  set, each result holds one block per detected region (pixel bboxes, reading
  order); otherwise one whole-image block per input.
- **`document_extract`** → `runtime.document_extractor(alias)` returns
  `Arc<dyn DocumentExtractionModel>`. Method: `extract(pages, options)`
  → `Vec<DocExtractResult>` (scaffold on `local/onnx` — returns `Unavailable`;
  use [`remote/llamacpp`](llamacpp.md#document_extract) for a GGUF document VLM).

## Example catalog entries

### Embed (preset alias)

```json
{
  "alias": "embed/local",
  "task": "embed",
  "provider_id": "local/onnx",
  "model_id": "BGESmallENV15"
}
```

### Embed (pass-through, custom HF model)

```json
{
  "alias": "embed/custom",
  "task": "embed",
  "provider_id": "local/onnx",
  "model_id": "Snowflake/snowflake-arctic-embed-m",
  "options": {
    "artifact": "onnx/model.onnx",
    "pooling": "cls",
    "dimensions": 768,
    "token_type_ids": true,
    "execution_providers": ["cuda", "cpu"]
  }
}
```

### Rerank

```json
{
  "alias": "rerank/cross",
  "task": "rerank",
  "provider_id": "local/onnx",
  "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2"
}
```

### Raw tensor

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

### NLP (kniv v7)

```json
{
  "alias": "nlp/default",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-v7-xlarge-u-en",
  "warmup": "background",
  "options": {
    "variant": "int8,fp16",
    "execution_providers": ["cpu"],
    "intra_op_num_threads": 8
  }
}
```

### NLP (kniv v1 cascade)

```json
{
  "alias": "nlp/cascade",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
  "warmup": "background",
  "options": {
    "onnx_path": "onnx/cascade.onnx",
    "max_seq_len": 128
  }
}
```

### Image embed (SigLIP-2-style ViT)

```json
{
  "alias": "embed/siglip2",
  "task": "embed_image",
  "provider_id": "local/onnx",
  "model_id": "google/siglip2-so400m-patch16-384",
  "options": {
    "onnx_path": "onnx/model.onnx",
    "image_size": 384,
    "dimensions": 1152,
    "normalization": "siglip",
    "pool": "none"
  }
}
```

### OCR (two-stage: PP-OCRv5 detection + recognition)

Verified end-to-end against `monkt/paddleocr-onnx` (Apache-2.0): PP-OCRv5
DBNet detection + English recognizer. Drop `det_onnx_path` for single-stage
(pre-cropped) recognition.

```json
{
  "alias": "ocr/ppocr-en",
  "task": "ocr",
  "provider_id": "local/onnx",
  "model_id": "monkt/paddleocr-onnx",
  "options": {
    "onnx_path": "languages/english/rec.onnx",
    "char_dict_path": "languages/english/dict.txt",
    "image_height": 48,
    "image_width": 320,
    "normalization": "siglip",
    "det_onnx_path": "detection/v5/det.onnx"
  }
}
```

### Document extract (scaffold; returns Unavailable until wired)

`local/onnx` document extraction is gated on an upstream ONNX export. For a
working VLM document extractor today, use the olmOCR-2 path on
[`remote/llamacpp`](llamacpp.md#document_extract).

```json
{
  "alias": "doc/granite",
  "task": "document_extract",
  "provider_id": "local/onnx",
  "model_id": "ibm-granite/granite-docling-258M",
  "options": {
    "style": "granite-docling"
  }
}
```

## External references

- ONNX Runtime docs: <https://onnxruntime.ai/docs/>
- Hugging Face ONNX docs: <https://huggingface.co/docs/optimum/exporters/onnx/overview>
