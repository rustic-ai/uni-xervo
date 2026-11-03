# Multimodal Generation

Generation takes structured `Message` values rather than plain strings, so a
single turn can mix text and images. This guide covers the message API, the
vision workflow, and what the `GenerationResult` fields mean in practice.

## What is wired today

| Modality | Task | Provider | Output |
| --- | --- | --- | --- |
| Text in → text out | `generate` | every generation provider | `result.text` |
| Text + images in → text out | `generate` | `remote/llamacpp` (server started with `--mmproj`), `remote/openai`, `remote/gemini`, `remote/cohere` | `result.text` |
| Page images in → structured document out | `document_extract` | `remote/llamacpp` | `DocExtractResult` |
| Audio in → text out | `transcribe` | `local/whisper-cpp` | `TranscriptResult` |

!!! warning "Image generation and speech synthesis have no bundled provider"
    `GenerationResult.images`, `GenerationResult.audio` and
    `GenerationOptions.width` / `height` are part of the public API but no
    bundled provider populates or reads them. `local/mistralrs`, which served
    the diffusion and speech pipelines, was removed in 0.19.0: the published
    crate was frozen at a release that could not run its own vision pipeline,
    while pulling in nearly half of the default dependency tree.
    `llama-server` has no image-generation endpoint. Use a dedicated service
    for these until a provider lands; the trait surface is unchanged, so one
    can be added without an API break.

## Message types

### Core types

```rust
use uni_xervo::traits::{ContentBlock, ImageInput, Message, MessageRole};

// Simple text message (convenience constructor)
let msg = Message::user("Hello, world!");

// Full message with role
let msg = Message {
    role: MessageRole::User,
    content: vec![ContentBlock::Text("Hello".to_string())],
};
```

### Message roles

- `MessageRole::System` — system instructions
- `MessageRole::User` — user input
- `MessageRole::Assistant` — model responses (for multi-turn conversations)

### Content blocks

- `ContentBlock::Text(String)` — text content
- `ContentBlock::Image(ImageInput)` — image content

Blocks keep their order on the wire, which matters: document VLMs are trained
with the instruction ahead of the page, and some chat models attend
differently depending on whether the image precedes or follows the question.

### Image input

```rust
// From raw bytes
let input = ImageInput::Bytes {
    data: image_bytes,
    media_type: "image/jpeg".to_string(),
};

// From URL — passed to the provider verbatim, so the server fetches it
let input = ImageInput::Url("https://example.com/image.jpg".to_string());
```

On OpenAI-compatible endpoints (`remote/llamacpp`, `remote/openai`,
`remote/azure-openai`) `ImageInput::Bytes` is sent as a
`data:{media_type};base64,…` URI inside an `image_url` content part.

## Vision workflow

### Catalog configuration

Vision is a property of the server process, not of the alias: a
`llama-server` started with a multimodal projector accepts image parts, one
started without it does not. So there is no `pipeline` option to set — point
the alias at a vision-capable server.

```sh
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -c 4096 --port 8081
```

(`-hf` fetches the projector alongside the model; with local files, pass
`--mmproj` explicitly. A projector costs context — image tokens are added to
the prompt — so a vision server wants noticeably more `--ctx-size` than the
same model serving text.)

```json
{
  "alias": "vision/smolvlm",
  "task": "generate",
  "provider_id": "remote/llamacpp",
  "model_id": "local",
  "options": {
    "base_url": "http://127.0.0.1:8081/v1",
    "request_timeout_secs": 180
  }
}
```

### Rust code

```rust
let image_bytes = std::fs::read("scene.jpg")?;

let message = Message {
    role: MessageRole::User,
    content: vec![
        ContentBlock::Image(ImageInput::Bytes {
            data: image_bytes,
            media_type: "image/jpeg".to_string(),
        }),
        ContentBlock::Text("What objects are in this image?".to_string()),
    ],
};

let vision = runtime.generator("vision/smolvlm").await?;
let result = vision.generate(&[message], GenerationOptions::default()).await?;
println!("{}", result.text);
```

A runnable version is at `crates/uni-xervo/examples/vision_describe.rs`
(`cargo run --example vision_describe --features provider-llamacpp`).

### Choosing a model

`ggml-org/SmolVLM-500M-Instruct-GGUF` is small enough to run comfortably on
CPU and is the easiest starting point. Caption quality scales with size: step
up to `SmolVLM-Instruct` (~2.2B) or a Qwen2.5-VL build for richer output.
Because the model lives in the server process, swapping it is a server restart
rather than a catalog change.

Sending an image to a text-only server fails with a non-retryable error naming
the missing multimodal support — not a silently dropped image.

!!! note "Reasoning models return the answer separately from the thinking"
    On a model with a thinking phase, `llama-server` puts the chain of thought
    in `reasoning_content` and the answer in `content`; only `content` becomes
    `result.text`. If `max_tokens` runs out *during* thinking, the call
    succeeds with non-zero token usage but empty text. Raise `max_tokens`.

## Document extraction

Page images in, structured blocks out — a different task from vision
generation, with its own prompt and parser. See
[`remote/llamacpp` → document_extract](../reference/providers/llamacpp.md#document_extract)
and the [tiered PDF extraction guide](pdf-extraction.md).

## Quantized models

Quantization is a server-side concern: `llama-server` loads a GGUF file, so
the quantization is chosen when the server starts and the catalog neither
knows nor cares.

```sh
llama-server -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 8192 --port 8081
```

The same applies to precision and device placement — `--n-gpu-layers` and
friends are server flags. Nothing in the alias options changes them.

## Sampling options

`GenerationOptions` fields are forwarded only when set, so the server's own
defaults apply otherwise:

| Field | Forwarded as | Notes |
| --- | --- | --- |
| `max_tokens` | `max_tokens` | Per-call completion cap |
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `width` / `height` | — | Ignored; no bundled provider generates images |
