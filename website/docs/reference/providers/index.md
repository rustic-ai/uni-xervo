# Providers

This section contains one page per Uni-Xervo provider.

Each provider page documents:

- Uni-Xervo provider ID and feature flag.
- Supported capabilities — any of the snake_case `ModelTask` wire names:
  `embed`, `rerank`, `generate`, `raw`, `embed_image`, `embed_audio`,
  `embed_multimodal`, `embed_sparse`, `embed_multi_vector`, `nlp`,
  `document_extract`, `transcribe`, `ocr`.
- Provider-specific `options` keys accepted by Uni-Xervo.
- Authoritative external links for model availability and model request/configuration docs.

## Provider pages

### Local providers

- [local/candle](candle.md)
- [local/onnx](onnx.md)
- [local/mistralrs](mistralrs.md)
- [local/whisper-cpp](whisper-cpp.md)

### Remote providers

- [remote/openai](openai.md)
- [remote/gemini](gemini.md)
- [remote/vertexai](vertexai.md)
- [remote/mistral](mistral.md)
- [remote/anthropic](anthropic.md)
- [remote/voyageai](voyageai.md)
- [remote/cohere](cohere.md)
- [remote/azure-openai](azure-openai.md)
- [remote/llamacpp](llamacpp.md)
