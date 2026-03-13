# local/mistralrs

## Uni-Xervo support

- Provider ID: `local/mistralrs`
- Feature flag: `provider-mistralrs`
- Capabilities: `embed`, `generate`
- Pipeline types: `text` (default), `vision`, `diffusion`, `speech`

## Pipeline types

| Pipeline | Description | Output |
| --- | --- | --- |
| `text` | Standard LLM text generation (default) | `result.text` |
| `vision` | Image + text understanding | `result.text` |
| `diffusion` | Text-to-image generation | `result.images` |
| `speech` | Text-to-audio synthesis | `result.audio` |

## Uni-Xervo provider options

### Common options (all pipelines)

| Option | Type | Description |
| --- | --- | --- |
| `pipeline` | string | Pipeline type: `text`, `vision`, `diffusion`, `speech`. Default: `text` |
| `dtype` | string | Model precision: `auto`, `f16`, `bf16`, `f32`. See [dtype](#dtype) |
| `force_cpu` | boolean | Force CPU inference |

### Text pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `isq` | string | In-situ quantization type (e.g. `Q4K`, `Q8_0`) |
| `paged_attention` | boolean | Enable paged attention |
| `max_num_seqs` | integer > 0 | Maximum concurrent sequences |
| `chat_template` | string | Custom chat template |
| `tokenizer_json` | string | Path to tokenizer.json |
| `embedding_dimensions` | integer > 0 | Override output dimensions for embeddings (embed task only) |
| `gguf_files` | array of strings | GGUF filenames to load in GGUF mode |

### Diffusion pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `diffusion_loader_type` | string | Required. One of: `flux`, `flux_offloaded` |

### Speech pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `speech_loader_type` | string | Required. One of: `dia` |

### Pipeline-specific option validity

| Option | text | vision | diffusion | speech |
| --- | --- | --- | --- | --- |
| `pipeline` | Yes | Yes | Yes | Yes |
| `dtype` | Yes | Yes | Yes | Yes |
| `force_cpu` | Yes | Yes | Yes | Yes |
| `isq` | Yes | No | No | No |
| `paged_attention` | Yes | No | No | No |
| `max_num_seqs` | Yes | No | No | No |
| `chat_template` | Yes | No | No | No |
| `tokenizer_json` | Yes | No | No | No |
| `embedding_dimensions` | Yes | No | No | No |
| `gguf_files` | Yes | No | No | No |
| `diffusion_loader_type` | No | No | Yes | No |
| `speech_loader_type` | No | No | No | Yes |

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/mistralrs.schema.json>

## Dtype

Model precision control. Available on all four pipeline types.

| Value | Description |
| --- | --- |
| `auto` | Automatic selection (BF16 on GPU, F32 on CPU) |
| `f16` | 16-bit floating point |
| `bf16` | Brain floating point 16 |
| `f32` | 32-bit floating point |

**Default resolution logic:**

1. Explicit `dtype` value in catalog options, if set
2. `f32` when running on CPU or without GPU support
3. `auto` otherwise

## Available models

`local/mistralrs` delegates model support to the upstream `mistral.rs` engine.

Authoritative model/support references:

- mistral.rs docs: <https://ericlbuehler.github.io/mistral.rs/>
- mistral.rs repository: <https://github.com/EricLBuehler/mistral.rs>

## Generation API

Uni-Xervo generation API exposes:

- `max_tokens`
- `temperature`
- `top_p`
- `width` (diffusion only)
- `height` (diffusion only)

`GenerationResult` output fields:

- `text` — generated text (text and vision pipelines)
- `usage` — optional token usage stats
- `images` — generated images (diffusion pipeline)
- `audio` — generated audio (speech pipeline)

## Example catalog entries

### Text generation (basic)

```json
{
  "alias": "generate/local",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "options": {
    "isq": "Q4K",
    "paged_attention": true,
    "max_num_seqs": 8
  }
}
```

### Text generation with GGUF

```json
{
  "alias": "generate/gguf",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
  "options": {
    "gguf_files": ["mistral-7b-instruct-v0.2.Q4_K_M.gguf"]
  }
}
```

### Text generation with ISQ + dtype

```json
{
  "alias": "generate/isq",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "options": {
    "isq": "Q8_0",
    "dtype": "bf16"
  }
}
```

### Vision

```json
{
  "alias": "vision/qwen",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "Qwen/Qwen2-VL-2B-Instruct",
  "options": {
    "pipeline": "vision",
    "dtype": "bf16"
  }
}
```

### Diffusion (image generation)

```json
{
  "alias": "image/flux",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "options": {
    "pipeline": "diffusion",
    "diffusion_loader_type": "flux"
  }
}
```

### Speech synthesis

```json
{
  "alias": "tts/dia",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "nari-labs/Dia-1.6B",
  "options": {
    "pipeline": "speech",
    "speech_loader_type": "dia"
  }
}
```
