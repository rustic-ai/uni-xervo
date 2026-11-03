# Guides

Practical guides for operating Uni-Xervo in production environments.

- [Provider Selection](provider-selection.md): choose providers by capability, latency, and control.
- [ONNX Runtime](onnx.md): use `local/onnx` for raw tensor execution, HF snapshots, and ONNX Runtime session management.
- [Config Validation](config-validation.md): enforce schema correctness in CI and startup.
- [Multimodal Generation](multimodal-generation.md): vision generation with `remote/llamacpp`.
- [Structured NLP](nlp.md): POS / NER / dependency / SRL / dialog-act analysis via `NlpModel` and the kniv-deberta cascade.
- [Multimodal Trait Surface](multimodal-traits.md): the full task-trait surface — the `ModelInfo` supertrait plus the image / audio / multimodal / sparse / multi-vector embed, NLP, document extract, transcription, and OCR traits.
- [Sparse Embeddings](sparse-embeddings.md): learned-sparse term-weight vectors (SPLADE / BGE-M3 sparse) via `SparseEmbeddingModel`, for the lexical half of hybrid retrieval.
- [Multi-Vector Embeddings](multi-vector-embeddings.md): per-token (ColBERT / late-interaction) vectors via `MultiVectorEmbeddingModel`, scored with host-side MaxSim.
- [OCR](ocr.md): optical character recognition with bounding boxes and confidences via `OcrModel` and the `local/onnx` PP-OCR pipeline.
- [Tiered PDF Extraction](pdf-extraction.md): the `uni-xervo-pdf` companion crate — escalate per page across native text → OCR → doc-VLM, with provenance and cross-tier verification.

For full ONNX developer documentation, use the dedicated [ONNX](../onnx/index.md) section.
