# Uni-Xervo User & Developer Guide

Welcome to Uni-Xervo! This guide is designed to help you integrate `uni-xervo` into your Rust applications and extend it with custom capabilities.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Hybrid Local Setup (Multiple Providers)](#hybrid-local-setup-multiple-providers)
4. [Configuration](#configuration)
5. [Usage Patterns](#usage-patterns)
    - [Text Embeddings](#text-embeddings)
    - [Reranking](#reranking)
    - [Text Generation (LLM)](#text-generation-llm)
5. [Advanced Topics](#advanced-topics)
    - [Warm-up Policies](#warm-up-policies)
    - [Error Handling](#error-handling)
6. [Developer Guide: Adding Providers](#developer-guide-adding-providers)

---

## Installation

Add `uni-xervo` to your `Cargo.toml`. Select the features corresponding to the providers you intend to use to keep your build size optimized.

```toml
[dependencies]
uni-xervo = { version = "0.1.0", default-features = false, features = ["provider-candle"] }
tokio = { version = "1", features = ["full"] }
```

**Available Features:**
- `provider-candle`: Local inference using Hugging Face Candle (Default).
- `provider-mistralrs`: High-performance local inference via mistral.rs.
- `provider-fastembed`: Optimized local embeddings via FastEmbed.
- `provider-openai`: Remote API support for OpenAI.
- `provider-gemini`: Remote API support for Google Gemini.
- `provider-vertexai`: Remote API support for Google Vertex AI.

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
uni-xervo = { version = "0.1.0", features = ["provider-candle", "provider-mistralrs"] }
```

**Code:**
```rust
use uni_xervo::api::{ModelTask, ModelAliasSpec, WarmupPolicy};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::GenerationOptions;
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
        &["Explain the importance of Rust safety.".to_string()],
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
use uni_xervo::traits::GenerationOptions;

let generator = runtime.generator("chat/gpt4").await?;
let result = generator.generate(
    &["Explain quantum computing in one sentence.".to_string()],
    GenerationOptions {
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    }
).await?;

println!("Response: {}", result.text);
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
use uni_xervo::traits::{GeneratorModel, GenerationResult, GenerationOptions};

struct MyCustomModel { name: String }

#[async_trait]
impl GeneratorModel for MyCustomModel {
    async fn generate(&self, messages: &[String], opts: GenerationOptions) -> uni_xervo::error::Result<GenerationResult> {
        // Call your API here
        Ok(GenerationResult {
            text: "Hello from custom provider!".to_string(),
            usage: None,
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
