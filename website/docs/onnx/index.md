# ONNX

`local/onnx` is Uni-Xervo's ONNX Runtime integration. It serves a range of tasks: `Raw` (arbitrary tensor execution), `Rerank` (cross-encoder and generative rerankers), `Embed` (dense text embeddings — replaces the retired `local/fastembed` provider as of 0.8.0; the same alias strings still resolve via the embedding preset table), `EmbedSparse` (learned-sparse SPLADE / BGE-M3 term-weight vectors), `EmbedMultiVector` (ColBERT-style per-token late-interaction embeddings), `EmbedImage` (image embeddings), `Nlp` (structured NLP), `Ocr` (text recognition, with an optional DBNet detection stage), and `DocumentExtract` (scaffolded — see below).

This section is the developer-facing guide for using ONNX models with Uni-Xervo. It explains the mental model, the configuration surface, and the end-to-end application flow.

## What Uni-Xervo handles

With `local/onnx`, Uni-Xervo handles:

- alias-based model resolution,
- model catalog validation,
- local-path and Hugging Face repo resolution,
- full HF snapshot download and caching,
- ONNX Runtime session creation,
- input/output signature introspection,
- batch validation and `run_batch()` orchestration,
- timeout, retry, and warmup wrappers.

## What your application handles

Your application still owns:

- tokenization and preprocessing,
- image/audio/tabular feature preparation,
- building `TensorBatch` inputs,
- interpreting output tensors,
- task-specific postprocessing such as argmax, softmax, span decoding, pooling, or label mapping.

That is the intended boundary. `local/onnx` is a runtime primitive, not a high-level transformer framework.

## When to use `local/onnx`

Good fits:

- custom numeric or scientific models,
- tabular regression/classification,
- HF transformer exports where you already control tokenization,
- sequence classification and NER pipelines,
- any ONNX graph where raw tensor I/O is the right abstraction.

If you want a provider that already knows what "embed" means, use `local/onnx` with `task: "embed"` (preset-driven; replaces `local/fastembed`), `local/candle`, or `remote/llamacpp`. If you want direct tensor control, use `local/onnx` with `task: "raw"`.

## Start here

- [Developer Flow](developer-flow.md)
- [Configuration](configuration.md)
- [Usage Patterns](usage-patterns.md)
