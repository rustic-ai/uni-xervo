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
    - [Raw ONNX Runtime](#raw-onnx-runtime)
    - [Vision Generation](#vision-generation)
    - [Document Extraction](#document-extraction)
7. [Advanced Topics](#advanced-topics)
    - [Warm-up Policies](#warm-up-policies)
    - [GGUF and Quantized Models](#gguf-and-quantized-models)
    - [Error Handling](#error-handling)
8. [Developer Guide: Adding Providers](#developer-guide-adding-providers)

---

## Installation

The default build gives you all three local backends and all eight remote API providers on CPU. Most users just need:

```toml
[dependencies]
uni-xervo = "0.9"
tokio = { version = "1", features = ["full"] }
```

Pass `default-features = false` and select explicitly when you want a leaner build.

### The feature surface

Features split into three independent axes:

**1. Providers** (which model backends to compile in — all on by default)

| Feature | Default? | Backend |
| --- | --- | --- |
| `provider-candle` | ✓ | Hugging Face Candle (local, embeddings). |
| `provider-onnx` | ✓ | ONNX Runtime (local; raw, rerank, embed). Subsumes the retired `provider-fastembed` as of 0.8.0. |
| `provider-onnx-dynamic` |   | Same surface as `provider-onnx` but `dlopen`s ORT at runtime via `ORT_DYLIB_PATH`. Mutually exclusive with `provider-onnx`. |
| `provider-openai` | ✓ | Remote OpenAI. |
| `provider-gemini` | ✓ | Remote Google Gemini. |
| `provider-vertexai` | ✓ | Remote Google Vertex AI. |
| `provider-mistral` | ✓ | Remote Mistral AI. |
| `provider-anthropic` | ✓ | Remote Anthropic. |
| `provider-voyageai` | ✓ | Remote Voyage AI. |
| `provider-cohere` | ✓ | Remote Cohere. |
| `provider-azure-openai` | ✓ | Remote Azure OpenAI. |
| `provider-llamacpp` | ✓ | Remote llama.cpp `llama-server`: embeddings with tokenizer-aware truncation, text and vision generation, reranking, document extraction. |

**2. ORT linking** (only matters if you use ONNX)

- `provider-onnx` — bundled, statically linked, self-contained. Default.
- `provider-onnx-dynamic` — BYO via `ORT_DYLIB_PATH`. Use this for ROCm, DirectML, OpenVINO, QNN, TensorRT, or WebGPU vendor builds; request the EP per alias by setting `execution_providers` to one of `"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"` (chained with `"cpu"` as fallback).

`build.rs` panics if both are enabled.

**3. GPU acceleration** (purely additive; off by default)

- `gpu-cuda` — Adds the CUDA EP to ORT (when bundled) plus CUDA kernels for candle. Linux + Windows + NVIDIA.
- `gpu-metal` — Adds the CoreML EP to ORT (when bundled) plus Metal kernels for candle. macOS / iOS, covers Apple GPU + Neural Engine.

CPU is always available — there is no "CPU feature." At runtime, ORT registers the GPU EP first and silently falls back to CPU per-op if the GPU path isn't available.

For other GPU vendors (AMD ROCm, Intel OpenVINO, Microsoft DirectML, Qualcomm QNN, TensorRT, WebGPU), use `provider-onnx-dynamic` plus a vendor-supplied ORT build via `ORT_DYLIB_PATH` at deploy, and select the EP per alias via `execution_providers`. See [`docs/migrations/0.9.0-feature-surface.md`](migrations/0.9.0-feature-surface.md).

### Common build recipes

```toml
# Defaults — everything except GPU.
uni-xervo = "0.9"

# Add NVIDIA GPU.
uni-xervo = { version = "0.9", features = ["gpu-cuda"] }

# Add Apple GPU + Neural Engine.
uni-xervo = { version = "0.9", features = ["gpu-metal"] }

# Lean local-only build.
uni-xervo = { version = "0.9", default-features = false, features = ["provider-candle"] }

# Remote-only — no native deps.
uni-xervo = { version = "0.9", default-features = false, features = [
    "provider-openai",
    "provider-anthropic",
] }
```

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

This example shows how to combine `candle` for in-process embeddings and a
llama.cpp `llama-server` for generation in a single runtime.

**Dependencies:**
```toml
uni-xervo = { version = "0.19", features = ["provider-candle", "provider-llamacpp"] }
```

Start the generation server first:

```sh
llama-server -hf ggml-org/gemma-3-270m-it-GGUF -c 4096 --port 8081
```

**Code:**
```rust
use uni_xervo::api::{ModelTask, ModelAliasSpec, WarmupPolicy};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::provider::RemoteLlamaCppProvider;
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

    // 2. Configure a llama-server for Generation (Gemma 3)
    let gen_spec = ModelAliasSpec {
        alias: "chat/gemma3".to_string(),
        task: ModelTask::Generate,
        provider_id: "remote/llamacpp".to_string(),
        // A single-model server accepts any value here.
        model_id: "local".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({
            "base_url": "http://127.0.0.1:8081/v1",
            "request_timeout_secs": 180
        }),
    };

    // 3. Build Runtime with both providers
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .register_provider(RemoteLlamaCppProvider::new())
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
    pub task: ModelTask,     // Embed, Rerank, Generate, or Raw
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

### ONNX alias shape

For `local/onnx`, use:

- `task: ModelTask::Raw`
- `provider_id: "local/onnx"`

`model_id` may be:

- a local path to a `.onnx` file, or
- a Hugging Face repo ID

If the repo contains multiple `.onnx` files, set `options.artifact`.

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

### Raw ONNX Runtime

Use `local/onnx` when you want Uni-Xervo to manage ONNX Runtime sessions, batching, caching, and Hugging Face snapshots while your application keeps control of preprocessing and output decoding.

**Catalog config:**
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

`model_id` can be:

- a local `.onnx` path
- a Hugging Face repo ID

If the HF repo contains multiple `.onnx` files, set `options.artifact`.

**Code:**
```rust
use ndarray::arr2;
use uni_xervo::traits::{TensorBatch, TensorValue};

let runner = runtime.raw_tensor_model("raw/classifier").await?;

println!("Inputs: {:#?}", runner.input_signature());
println!("Outputs: {:#?}", runner.output_signature());

let mut batch = TensorBatch::new();
batch.insert(
    "input_ids".to_string(),
    TensorValue::I64(arr2(&[[101_i64, 1045, 2293, 3185, 102]]).into_dyn()),
);
batch.insert(
    "attention_mask".to_string(),
    TensorValue::I64(arr2(&[[1_i64, 1, 1, 1, 1]]).into_dyn()),
);

let outputs = runner.run(&batch).await?;
println!("Returned tensors: {:?}", outputs.keys().collect::<Vec<_>>());
```

Uni-Xervo validates tensor names, dtypes, and shapes against the graph signature, but it does not interpret logits, labels, embeddings, or spans for you.

### Vision Generation

Vision models process images alongside text prompts. Point a `generate` alias
at a `llama-server` started with a multimodal projector; whether images are
accepted is a property of that server process, so there is no option for it.

```sh
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -c 4096 --port 8081
```

**Catalog config:**
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

let vision = runtime.generator("vision/smolvlm").await?;
let result = vision.generate(&[message], GenerationOptions::default()).await?;
println!("{}", result.text);
```

Sending an image to a text-only server fails with a non-retryable error naming
the missing multimodal support, rather than silently dropping the image.

> **Image generation and speech synthesis have no bundled provider.**
> `GenerationResult.images`, `.audio` and `GenerationOptions.width` / `height`
> remain public API, but the provider that served them (`local/mistralrs`) was
> removed in 0.19.0 and `llama-server` has no image-generation endpoint. Use a
> dedicated service for these; the trait surface is unchanged, so a provider
> can be added later without an API break.

### Document Extraction

A document VLM turns page images into structured blocks — tables, formulas,
headings and reading order, not just the text an OCR model would return. The
server needs a projector and a document model; `style` tells the provider
which output dialect to parse.

**Catalog config:**
```json
{
    "alias": "docext/olmocr",
    "task": "document_extract",
    "provider_id": "remote/llamacpp",
    "model_id": "local",
    "options": {
        "base_url": "http://127.0.0.1:8099/v1",
        "style": "olmocr",
        "request_timeout_secs": 600
    }
}
```

**Code:**
```rust
use uni_xervo::traits::{DocExtractOptions, DocOutputFormat, ImageInput};

let extractor = runtime.document_extractor("docext/olmocr").await?;
let pages = extractor
    .extract(
        vec![ImageInput::Bytes {
            data: std::fs::read("page1.png")?,
            media_type: "image/png".to_string(),
        }],
        DocExtractOptions {
            output: DocOutputFormat::Markdown,
            include_tables: true,
            include_formulas: true,
            include_bboxes: false,
        },
    )
    .await?;

println!("{}", pages[0].plain_markdown);
println!("{} block(s) in reading order", pages[0].blocks.len());
```

Pages are sent one request each, in order, so a multi-page document is a
single call regardless of the server's context size. `style` accepts
`olmocr` (default), `mineru` and `granite-docling`.

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

### GGUF and Quantized Models

GGUF is a quantized model format optimized for CPU inference with reduced
memory usage. With `remote/llamacpp` the quantization, the precision and the
device are all server-side concerns: `llama-server` loads the GGUF file, so
they are chosen when the server starts and the catalog neither knows nor
cares.

```sh
llama-server -m Phi-3.5-mini-instruct-Q4_K_M.gguf -c 8192 --port 8081
```

```sh
# Offload as much as fits onto the GPU
llama-server -m Phi-3.5-mini-instruct-Q4_K_M.gguf -c 8192 --n-gpu-layers 99 --port 8081
```

The alias options carry only the endpoint and the HTTP timeout, so swapping
quantization is a server restart rather than a catalog change.

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
