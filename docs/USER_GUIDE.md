# Uni-Xervo User & Developer Guide

Welcome to Uni-Xervo! This guide is designed to help you integrate `uni-xervo` into your Rust applications and extend it with custom capabilities.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Hybrid Local Setup (Multiple Providers)](#hybrid-local-setup-multiple-providers)
4. [Configuration](#configuration)
5. [Messages & Multimodal Content](#messages--multimodal-content)
6. [Usage Patterns](#usage-patterns)
    - [Text Embeddings](#text-embeddings)
    - [Reranking](#reranking)
    - [Text Generation (LLM)](#text-generation-llm)
    - [Vision Generation](#vision-generation)
    - [Image Generation (Diffusion)](#image-generation-diffusion)
    - [Speech Synthesis](#speech-synthesis)
7. [Advanced Topics](#advanced-topics)
    - [Warm-up Policies](#warm-up-policies)
    - [GGUF Models](#gguf-models)
    - [Model Precision (dtype)](#model-precision-dtype)
    - [Mistralrs Pipeline Options](#mistralrs-pipeline-options)
    - [Error Handling](#error-handling)
8. [Developer Guide: Adding Providers](#developer-guide-adding-providers)

---

## Installation

Add `uni-xervo` to your `Cargo.toml`. Select the features corresponding to the providers you intend to use to keep your build size optimized.

```toml
[dependencies]
uni-xervo = { version = "0.2.0", default-features = false, features = ["provider-candle"] }
tokio = { version = "1", features = ["full"] }
```

**Available Features:**
- `provider-candle`: Local inference using Hugging Face Candle (Default).
- `provider-mistralrs`: High-performance local inference via mistral.rs (text, vision, diffusion, speech).
- `provider-fastembed`: Optimized local embeddings via FastEmbed.
- `provider-openai`: Remote API support for OpenAI.
- `provider-gemini`: Remote API support for Google Gemini.
- `provider-vertexai`: Remote API support for Google Vertex AI.
- `provider-mistral`: Remote API support for Mistral AI.
- `provider-anthropic`: Remote API support for Anthropic.
- `provider-voyageai`: Remote API support for Voyage AI.
- `provider-cohere`: Remote API support for Cohere.
- `provider-azure-openai`: Remote API support for Azure OpenAI.

---

## Quick Start

This example demonstrates how to set up a runtime with a local provider and generate an embedding.

```rust
use std::sync::Arc;
use uni_xervo::api::{ModelTask, ModelAliasSpec};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define your model catalog
    let spec = ModelAliasSpec {
        alias: "embed/local".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: Default::default(), // Lazy by default
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    // 2. Build the runtime
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Get a typed handle to the model
    let model = runtime.embedding("embed/local").await?;

    // 4. Run inference
    let embeddings = model.embed(vec!["Hello, world!"]).await?;
    println!("Embedding vector length: {}", embeddings[0].len());

    Ok(())
}
```

## Hybrid Local Setup (Multiple Providers)

This example shows how to combine `candle` for embeddings and `mistral.rs` for generation in a single runtime.

**Dependencies:**
```toml
uni-xervo = { version = "0.2.0", features = ["provider-candle", "provider-mistralrs"] }
```

**Code:**
```rust
use uni_xervo::api::{ModelTask, ModelAliasSpec, WarmupPolicy};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{GenerationOptions, Message};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure Candle for Embedding (BGE Small)
    let embed_spec = ModelAliasSpec {
        alias: "embed/bge".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    // 2. Configure Mistral.rs for Generation (Gemma 3)
    let gen_spec = ModelAliasSpec {
        alias: "chat/gemma3".to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: "google/gemma-3-270m".to_string(), // Gemma 3
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({
            "isq": "Q4K",
            "max_num_seqs": 4
        }),
    };

    // 3. Build Runtime with both providers
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![embed_spec, gen_spec])
        .build()
        .await?;

    // 4. Use Embedding
    let embedder = runtime.embedding("embed/bge").await?;
    let _vec = embedder.embed(vec!["Search query"]).await?;
    println!("BGE embedding generated.");

    // 5. Use Generation
    let generator = runtime.generator("chat/gemma3").await?;
    let res = generator.generate(
        &[Message::user("Explain the importance of Rust safety.")],
        GenerationOptions::default()
    ).await?;
    println!("Gemma 3 says: {}", res.text);

    Ok(())
}
```

---

## Configuration

Uni-Xervo uses a **Model Catalog** to map application-specific aliases (e.g., `search/query`) to concrete provider implementations. This allows you to swap backends without changing application code.

### The `ModelAliasSpec`
```rust
pub struct ModelAliasSpec {
    pub alias: String,       // e.g., "chat/support-bot"
    pub task: ModelTask,     // Embed, Rerank, or Generate
    pub provider_id: String, // e.g., "remote/openai", "local/candle"
    pub model_id: String,    // e.g., "gpt-4o", "sentence-transformers/all-MiniLM-L6-v2"
    pub timeout: Option<u64>,      // Per-inference timeout (seconds)
    pub load_timeout: Option<u64>, // Model load timeout (seconds)
    pub options: Value,      // Provider-specific JSON options
    // ...
}
```

`options` are validated per provider at build/register time. Unknown keys and wrong value types return a configuration error.
Schema files are available under `schemas/` (for example, `schemas/model-catalog.schema.json`).

---

## Messages & Multimodal Content

Generation in Uni-Xervo uses structured `Message` objects instead of plain strings. This enables multi-turn conversations, role-based prompting, and multimodal inputs (text + images).

### Message Types

```rust
use uni_xervo::traits::{Message, MessageRole, ContentBlock, ImageInput};
```

**`MessageRole`** defines who is speaking:
- `MessageRole::System` — system instructions
- `MessageRole::User` — user input
- `MessageRole::Assistant` — model responses (for multi-turn context)

**`ContentBlock`** represents a piece of content:
- `ContentBlock::Text(String)` — text content
- `ContentBlock::Image(ImageInput)` — image content (for vision models)

**`ImageInput`** specifies an image source:
- `ImageInput::Bytes { data: Vec<u8>, media_type: String }` — raw image bytes
- `ImageInput::Url(String)` — image URL

### Convenience Constructors

For text-only messages, use the shorthand constructors:

```rust
// Single-role messages with text content
let user_msg = Message::user("What is Rust?");
let system_msg = Message::system("You are a helpful assistant.");
let assistant_msg = Message::assistant("Rust is a systems language.");
```

### Multi-turn Conversations

```rust
let messages = &[
    Message::system("You are a concise assistant."),
    Message::user("What is ownership in Rust?"),
    Message::assistant("Ownership is Rust's memory management model."),
    Message::user("How does borrowing relate to it?"),
];

let result = generator.generate(messages, GenerationOptions::default()).await?;
```

### Multimodal Messages (Image + Text)

For vision models, construct messages with mixed content blocks:

```rust
let image_data = std::fs::read("photo.jpg")?;

let message = Message {
    role: MessageRole::User,
    content: vec![
        ContentBlock::Image(ImageInput::Bytes {
            data: image_data,
            media_type: "image/jpeg".to_string(),
        }),
        ContentBlock::Text("Describe this image.".to_string()),
    ],
};
```

### Migration from 0.1.x

The `GeneratorModel::generate()` signature changed from `&[String]` to `&[Message]`:

```rust
// Before (0.1.x)
generator.generate(&["Hello".to_string()], opts).await?;

// After (0.2.0)
generator.generate(&[Message::user("Hello")], opts).await?;
```

---

## Usage Patterns

### Text Embeddings
Used for semantic search, clustering, and classification.

```rust
let embedder = runtime.embedding("embed/local").await?;
let vectors = embedder.embed(vec![
    "Rust is a systems programming language.",
    "Machine learning is fascinating."
]).await?;
```

### Reranking
Re-scores a list of documents based on their relevance to a query.

```rust
let reranker = runtime.reranker("rerank/fast").await?;
let docs = vec!["Doc A content...", "Doc B content..."];
let scores = reranker.rerank("query string", &docs).await?;

for doc in scores {
    println!("Index: {}, Score: {}", doc.index, doc.score);
}
```

### Text Generation (LLM)
Used for chat bots, summarization, and content creation.

```rust
use uni_xervo::traits::{GenerationOptions, Message};

let generator = runtime.generator("chat/gpt4").await?;
let result = generator.generate(
    &[Message::user("Explain quantum computing in one sentence.")],
    GenerationOptions {
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    }
).await?;

println!("Response: {}", result.text);
```

The `GenerationResult` contains:
```rust
pub struct GenerationResult {
    pub text: String,                  // Generated text
    pub usage: Option<TokenUsage>,     // Token counts
    pub images: Vec<GeneratedImage>,   // Generated images (diffusion)
    pub audio: Option<AudioOutput>,    // Generated audio (speech)
}
```

### Vision Generation

Vision models process images alongside text prompts. Use the mistralrs vision pipeline to run models like Qwen2-VL or Gemma-3n locally.

**Catalog config:**
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

**Code:**
```rust
use uni_xervo::traits::{Message, MessageRole, ContentBlock, ImageInput, GenerationOptions};

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

### Image Generation (Diffusion)

Generate images from text prompts using the mistralrs diffusion pipeline (FLUX models).

**Catalog config:**
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

**Code:**
```rust
use uni_xervo::traits::{Message, GenerationOptions};

let gen = runtime.generator("image/flux").await?;
let result = gen.generate(
    &[Message::user("A serene mountain landscape at sunset")],
    GenerationOptions {
        width: Some(1024),
        height: Some(1024),
        ..Default::default()
    },
).await?;

// Access generated images
for image in &result.images {
    std::fs::write("output.png", &image.data)?;
    println!("Generated image: {} ({} bytes)", image.media_type, image.data.len());
}
```

### Speech Synthesis

Generate audio from text using the mistralrs speech pipeline (Dia models).

**Catalog config:**
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

**Code:**
```rust
use uni_xervo::traits::{Message, GenerationOptions};

let tts = runtime.generator("tts/dia").await?;
let result = tts.generate(
    &[Message::user("[S1] Hello, welcome to Uni-Xervo!")],
    GenerationOptions::default(),
).await?;

if let Some(audio) = &result.audio {
    println!(
        "Audio: {} samples, {}Hz, {} channels",
        audio.pcm_data.len(),
        audio.sample_rate,
        audio.channels
    );
    // Write PCM data to a WAV file using your preferred audio library
}
```

---

## Advanced Topics

### Warm-up Policies
Control when models are loaded into memory to optimize startup time vs. first-request latency.

- **`Lazy` (Default):** Model loads when `embed()` or `generate()` is first called.
    - *Pro:* Fast application startup.
    - *Con:* First request is slow.
- **`Eager`:** Model loads during `ModelRuntime::build()`.
    - *Pro:* Fast first request.
    - *Con:* Application startup blocks until all models are loaded.
- **`Background`:** `ModelRuntime::build()` returns immediately; models load in a detached thread.
    - *Pro:* Fast startup + Fast eventual request.
    - *Con:* Request fails or blocks if called before background load finishes (implementation dependent).

**Global Configuration:**
```rust
let runtime = ModelRuntime::builder()
    .warmup_policy(uni_xervo::api::WarmupPolicy::Eager) // Global default
    .build()
    .await?;
```

**Per-Model Configuration:**
You can override the policy in the `ModelAliasSpec`.

### GGUF Models

GGUF is a quantized model format optimized for CPU inference with reduced memory usage. The mistralrs text pipeline supports loading GGUF models directly.

**Catalog config:**
```json
{
    "alias": "chat/phi-gguf",
    "task": "generate",
    "provider_id": "local/mistralrs",
    "model_id": "microsoft/Phi-3.5-mini-instruct",
    "options": {
        "gguf_files": ["Phi-3.5-mini-instruct-Q4_K_M.gguf"]
    }
}
```

GGUF support is only available for the text pipeline. Vision, diffusion, and speech pipelines do not support GGUF files.

### Model Precision (dtype)

Control the floating-point precision for mistralrs model loading:

| Value | Description |
|-------|-------------|
| `auto` | Automatic selection (typically BF16 on GPU, F32 on CPU) |
| `f16` | 16-bit floating point |
| `bf16` | Brain floating point 16 |
| `f32` | 32-bit floating point (default on CPU) |

**Catalog config:**
```json
{
    "alias": "chat/model",
    "task": "generate",
    "provider_id": "local/mistralrs",
    "model_id": "google/gemma-3-270m",
    "options": {
        "dtype": "bf16"
    }
}
```

The `dtype` option is available for all four mistralrs pipeline types (text, vision, diffusion, speech).

### Mistralrs Pipeline Options

The `local/mistralrs` provider supports four pipeline types. Each pipeline accepts a different set of options:

| Option | text | vision | diffusion | speech |
|--------|:----:|:------:|:---------:|:------:|
| `pipeline` | Y | Y | Y | Y |
| `isq` | Y | Y | - | - |
| `dtype` | Y | Y | Y | Y |
| `force_cpu` | Y | Y | Y | Y |
| `gguf_files` | Y | - | - | - |
| `paged_attention` | Y | Y | - | - |
| `max_num_seqs` | Y | Y | - | - |
| `chat_template` | Y | Y | - | - |
| `tokenizer_json` | Y | Y | - | - |
| `embedding_dimensions` | Y* | - | - | - |
| `diffusion_loader_type` | - | - | Y | - |
| `speech_loader_type` | - | - | - | Y |

\* `embedding_dimensions` is only valid for the `Embed` task.

**Pipeline selection:** Set `"pipeline": "text"` (default), `"vision"`, `"diffusion"`, or `"speech"` in the model options.

**ISQ values:** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`

**Diffusion loader types:** `flux`, `flux_offloaded`

**Speech loader types:** `dia`

---

## Developer Guide: Adding Providers

To add support for a new backend (e.g., Anthropic, Cohere, or a custom in-house model server), implement the `ModelProvider` trait.

### 1. Implement `ModelProvider`

```rust
use async_trait::async_trait;
use uni_xervo::traits::{ModelProvider, ProviderCapabilities, ProviderHealth, LoadedModelHandle};
use uni_xervo::api::ModelAliasSpec;

pub struct MyCustomProvider;

#[async_trait]
impl ModelProvider for MyCustomProvider {
    fn provider_id(&self) -> &'static str {
        "my-custom-provider"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Generate],
        }
    }

    async fn health(&self) -> ProviderHealth {
        // Ping your service
        ProviderHealth::Healthy
    }

    async fn load(&self, spec: &ModelAliasSpec) -> uni_xervo::error::Result<LoadedModelHandle> {
        // 1. Parse spec.options
        // 2. Initialize your client/model
        // 3. Return it wrapped in Arc
        let model = MyCustomModel::new(&spec.model_id);
        Ok(std::sync::Arc::new(std::sync::Arc::new(model)))
    }
}
```

### 2. Implement Capability Traits
Your model struct must implement the trait corresponding to the task (e.g., `GeneratorModel`).

```rust
use uni_xervo::traits::{GeneratorModel, GenerationResult, GenerationOptions, Message};

struct MyCustomModel { name: String }

#[async_trait]
impl GeneratorModel for MyCustomModel {
    async fn generate(&self, messages: &[Message], opts: GenerationOptions) -> uni_xervo::error::Result<GenerationResult> {
        // Call your API here
        Ok(GenerationResult {
            text: "Hello from custom provider!".to_string(),
            usage: None,
            images: vec![],
            audio: None,
        })
    }
}
```

### 3. Register the Provider
```rust
let runtime = ModelRuntime::builder()
    .register_provider(MyCustomProvider)
    // ...
```
