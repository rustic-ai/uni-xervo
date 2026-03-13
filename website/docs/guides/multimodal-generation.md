# Multimodal Generation

Uni-Xervo 0.2.0 extends `local/mistralrs` with four pipeline types: text, vision, diffusion, and speech. This guide covers when to use each pipeline, how to configure them, and how to work with the structured message API.

## When to use each pipeline

| Pipeline | Use case | Input | Output |
| --- | --- | --- | --- |
| `text` | Standard LLM chat and completion (default) | Text messages | `result.text` |
| `vision` | Image understanding with text prompts | Images + text messages | `result.text` |
| `diffusion` | Text-to-image generation | Text prompt | `result.images` |
| `speech` | Text-to-audio synthesis | Text prompt | `result.audio` |

## Message types

Generation in 0.2.0 uses structured `Message` types instead of plain strings.

### Core types

```rust
use uni_xervo::{Message, MessageRole, ContentBlock, ImageInput};

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
- `ContentBlock::Image(ImageInput)` — image content (vision pipeline)

### Image input

```rust
// From raw bytes
let input = ImageInput::Bytes {
    data: image_bytes,
    media_type: "image/jpeg".to_string(),
};

// From URL
let input = ImageInput::Url("https://example.com/image.jpg".to_string());
```

## Vision workflow

Process images with text prompts using a vision model.

### Catalog configuration

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

let vision = runtime.generator("vision/qwen").await?;
let result = vision.generate(&[message], GenerationOptions::default()).await?;
println!("{}", result.text);
```

## Diffusion workflow

Generate images from text prompts using FLUX models.

### Catalog configuration

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

### Rust code

```rust
let gen = runtime.generator("image/flux").await?;
let result = gen.generate(
    &[Message::user("A serene mountain landscape at sunset")],
    GenerationOptions {
        width: Some(1024),
        height: Some(1024),
        ..Default::default()
    },
).await?;

for image in &result.images {
    std::fs::write("output.png", &image.data)?;
}
```

## Speech workflow

Synthesize audio from text using Dia models.

### Catalog configuration

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

### Rust code

```rust
let tts = runtime.generator("tts/dia").await?;
let result = tts.generate(
    &[Message::user("[S1] Hello, welcome!")],
    GenerationOptions::default(),
).await?;

if let Some(audio) = &result.audio {
    // audio.pcm_data — raw PCM samples
    // audio.sample_rate — e.g. 24000
    // audio.channels — e.g. 1
}
```

## GGUF models

Load quantized text models in GGUF format by specifying the filenames.

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

!!! note
    `gguf_files` is only valid with the `text` pipeline. Vision, diffusion, and speech pipelines do not support GGUF.

## Model precision (dtype)

Control model precision with the `dtype` option. Available on all four pipeline types.

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

### Example

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

## Migration from 0.1.x

The `generate()` API changed from `&[String]` to `&[Message]`.

**Before (0.1.x):**

```rust
let result = generator.generate(
    &["Hello".to_string()],
    GenerationOptions::default(),
).await?;
```

**After (0.2.0):**

```rust
let result = generator.generate(
    &[Message::user("Hello")],
    GenerationOptions::default(),
).await?;
```

The `Message::user()` convenience constructor creates a single text message with `MessageRole::User`. For multimodal content, build `Message` structs directly with multiple `ContentBlock` entries.
