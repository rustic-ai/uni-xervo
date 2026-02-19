# System Architecture

## Overview
Uni-Xervo is a unified, task-based runtime for machine learning models. It abstracts the differences between local (Candle, Mistral.rs) and remote (OpenAI, Gemini) providers behind a consistent, type-safe API. The system is designed for high performance, ease of use, and stability in production environments.

## Core Concepts

### 1. Model Runtime (`ModelRuntime`)
The central entry point of the library. It manages the lifecycle of providers and models.
- **Responsibility:**
    - Resolves application-level aliases (e.g., `embed/local-fast`) to specific provider configurations.
    - Manages the `ModelRegistry` to cache and deduplicate loaded models.
    - Exposes task-specific methods (`embedding()`, `reranker()`, `generator()`) to retrieve strongly-typed model handles.

### 2. Model Registry (`ModelRegistry`)
A thread-safe cache for loaded model instances.
- **Mechanism:** Uses a `RwLock<HashMap<ModelRuntimeKey, LoadedInstance>>` to store active models.
- **Deduplication:** A `ModelRuntimeKey` uniquely identifies a model configuration (Provider + Model ID + Revision + Options). Concurrent requests for the same key join a shared loader lock (`Mutex`) to ensure a model is loaded only once ("thundering herd" protection).

### 3. Providers (`ModelProvider`)
The bridge between the runtime and the underlying ML framework or API.
- **Interface:** The `ModelProvider` trait defines the contract:
    - `capabilities()`: What tasks does this provider support?
    - `load()`: Instantiate a model based on a `ModelAliasSpec`.
    - `health()`: Check provider status.
    - `warmup()`: Optional pre-initialization (e.g., connecting to HF API).

### 4. Tasks & Models
The system is strictly task-oriented. Models are typed by their capability, not their implementation.
- **`EmbeddingModel`**: Returns vector embeddings (`Vec<f32>`).
- **`RerankerModel`**: Re-scores a list of documents against a query.
- **`GeneratorModel`**: Generates text completions (LLM).

### 5. Alias Specification (`ModelAliasSpec`)
Configuration object defining *what* to load.
- **Fields:**
    - `alias`: Application-facing name (e.g., `search/vector`).
    - `task`: The capability required (`Embed`, `Rerank`, `Generate`).
    - `provider_id`: ID of the registered provider to use.
    - `model_id`: Provider-specific model identifier (e.g., `bert-base-uncased`, `gpt-4o`).
    - `warmup`: `Lazy` (default), `Eager`, or `Background`.

## Key Workflows

### Model Resolution & Loading
1.  **Request:** User calls `runtime.embedding("my-alias")`.
2.  **Lookup:** Runtime resolves `my-alias` to a `ModelAliasSpec`.
3.  **Cache Check:**
    - Checks `ModelRegistry` (Read Lock). If found, returns `Arc<Model>`.
4.  **Load Coordination:**
    - If missing, acquires a granular `Loader Lock` for the specific `ModelRuntimeKey`.
    - Double-checks cache (in case another thread finished loading).
5.  **Instantiation:**
    - Calls `provider.load(spec)`.
    - Provider returns a type-erased `LoadedModelHandle` (`Arc<dyn Any + Send + Sync>`).
6.  **Type Check:** Runtime verifies the handle implements the requested trait (e.g., `EmbeddingModel`).
7.  **Cache Update:** Stores handle in registry and returns it.

### Warm-up Strategies
Controlled via `WarmupPolicy`:
- **Lazy (Default):** Model loads on the first inference request. Zero startup overhead.
- **Eager:** Model loads during `ModelRuntime::build()`. `build()` blocks until completion.
- **Background:** `build()` returns immediately, and a detached Tokio task loads the model. Failures are logged but do not crash the runtime.

## Concurrency & Safety
- **State Management:** Uses `Arc` for shared ownership and `RwLock`/`Mutex` for synchronization.
- **Thread Safety:** All core traits (`ModelProvider`, `EmbeddingModel`, etc.) require `Send + Sync`.
- **Async/Await:** Built on `Tokio`. Heavy compute (local inference) should ideally be offloaded to blocking threads within the provider implementation if it doesn't support native async.

## Error Handling
- **`thiserror`:** Used for internal error definitions.
- **`anyhow`:** (Internal usage only, if any). Public API returns strongly typed `uni_xervo::error::Result`.
- **Recoverability:** Errors are propagated (e.g., `ProviderNotFound`, `CapabilityMismatch`) allowing the application to implement fallbacks.

## Extension Points
- **New Providers:** Implement the `ModelProvider` trait and register it via `ModelRuntime::builder().register_provider()`.
- **New Tasks:** Add a new variant to `ModelTask` enum and a corresponding trait (e.g., `AudioTranscriberModel`).
