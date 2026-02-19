# uni-xervo: Product Spec, Design, and Architecture

**Version:** 0.1.0
**Status:** Active Development
**Last Updated:** 2026-02-14

---

## 1. Executive Summary

uni-xervo is a standalone Rust crate that provides a unified, pluggable runtime for
managing machine learning model backends. It serves as the single model abstraction
layer across Dragonscale products, eliminating repeated one-off model integrations
while preserving performance and operational control.

The runtime provides:

- A **task-based API** with strongly typed interfaces for embeddings, reranking, and text generation.
- A **provider plugin system** for local (Candle, FastEmbed, mistral.rs) and remote (OpenAI, Gemini) backends.
- A **global in-process model registry** that guarantees one loaded instance per unique model configuration.
- **Alias-based routing** that decouples application code from concrete model identifiers.
- **Configurable lifecycle policies** (eager, lazy, background warmup) and **reliability controls** (circuit breakers, retry, backoff).

### Why a Standalone Crate

uni-xervo was extracted from an internal monorepo into its own repository to:

1. Enable reuse across multiple Dragonscale products without pulling in the full database stack.
2. Allow independent versioning, release cadence, and CI/CD.
3. Clarify the dependency boundary: consumers depend on uni-xervo, not the other way around.
4. Make the provider plugin ecosystem extensible by third parties without touching core product code.

---

## 2. Problem Statement

Model integrations across Dragonscale products are fragmented:

- Different stacks use different backends with inconsistent error handling, retry, and observability.
- Read paths and write paths may instantiate model services independently, wasting memory and producing unpredictable latency.
- Adding a new provider (e.g., Anthropic, Cohere, Ollama) requires duplicated work in every consuming project.
- Caching and warmup strategies are ad hoc and not standardized.
- Provider credentials, timeouts, and failure policies are scattered across codebases.

This produces higher tail latency, duplicate memory usage, integration bugs, and slower delivery of new model capabilities.

---

## 3. Product Goals and Non-Goals

### Goals

- **G1: One runtime, many products.** Any Dragonscale Rust project can integrate uni-xervo as a library dependency and get access to all supported model backends through a single API.
- **G2: Task-typed safety.** Callers request models by task (embed, rerank, generate). The type system prevents using an embedding model as a generator.
- **G3: Provider agnosticism.** Application code references aliases (`embed/default`), not provider-specific model identifiers. Swapping from local Candle to remote OpenAI is a config change, not a code change.
- **G4: Singleton instances.** The registry guarantees one loaded model per unique configuration key within a process. No duplicate weight loading, no duplicate connections.
- **G5: Lifecycle control.** Operators configure warmup policy per alias. Critical models block startup. Optional models load in the background or on first request.
- **G6: Reliability.** Remote providers are wrapped in circuit breakers with configurable retry and backoff. Failures produce structured, actionable errors.
- **G7: Observability.** Load latency, inference latency, cache hit/miss, and failure counters are emitted via the `metrics` crate and `tracing` spans.
- **G8: Minimal integration surface.** Host applications wire up providers and a catalog at startup, then resolve aliases throughout their codebase. No provider-specific code leaks into business logic.

### Non-Goals (Current Phase)

- **NG1:** No cross-process model sharing or distributed cache coordination. uni-xervo is in-process only.
- **NG2:** No multi-tenant billing, rate partitioning, or usage metering. Those are host-layer concerns.
- **NG3:** No model training, fine-tuning, or RLHF workflows.
- **NG4:** No UI, dashboard, or console. Library-first.
- **NG5:** No streaming generation API (planned for a future phase).
- **NG6:** No automatic fallback chains between providers (planned for phase 2).

---

## 4. Users and Use Cases

### Primary Users

| User | Needs |
|------|-------|
| **Backend engineers** integrating model capabilities into products | Simple API, type safety, no provider lock-in |
| **Product teams** building search, retrieval, ranking, and generation features | Alias-based config, swap models without code changes |
| **Ops/SRE teams** managing runtime behavior | Warmup policies, health checks, circuit breakers, observability |
| **Platform engineers** adding new model backends | Clean provider trait, feature-gated compilation, conformance tests |

### Core Use Cases

1. **Embedding at write time.** A database writer resolves `embed/default` and generates vectors for incoming documents. The model is warmed eagerly at startup; the same instance is shared across all writer threads.

2. **Embedding at query time.** A query executor resolves the same `embed/default` alias to generate a query vector for vector similarity search. It reuses the already-loaded model instance from the write path.

3. **Multi-model catalog.** An application configures multiple embedding aliases for different domains:
   - `embed/default` -> local Candle, all-MiniLM-L6-v2
   - `embed/legal` -> remote OpenAI, text-embedding-3-large
   - `embed/multilingual` -> local FastEmbed, MultilingualE5Small

4. **LLM generation.** A product feature resolves `llm/chat` to a locally-running mistral.rs model for summarization or question answering, with temperature and token limits configured via `GenerationOptions`.

5. **Provider swap via config.** An operator switches `embed/default` from `local/candle` to `remote/openai` by changing one config field. No application code changes. No recompilation (if both provider features are enabled).

6. **Graceful degradation.** A remote provider goes down. The circuit breaker opens, returning `Unavailable` errors immediately instead of blocking on timeouts. When the provider recovers, the half-open probe succeeds and traffic resumes.

---

## 5. Functional Requirements

### FR-1: Model Catalog

The system MUST support a catalog of named aliases with task-aware specifications.

- **Alias format:** `<task>/<name>` (e.g., `embed/default`, `llm/chat`, `rerank/legal`).
- **Alias uniqueness:** No two catalog entries may share the same alias.
- **Spec fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | `String` | Yes | Unique alias in `task/name` format |
| `task` | `ModelTask` | Yes | `Embed`, `Rerank`, or `Generate` |
| `provider_id` | `String` | Yes | Provider identifier (e.g., `local/candle`) |
| `model_id` | `String` | Yes | Provider-specific model reference |
| `revision` | `Option<String>` | No | Model version/revision (e.g., HF commit hash) |
| `warmup` | `WarmupPolicy` | No | `Eager`, `Lazy` (default), or `Background` |
| `required` | `bool` | No | If true, warmup failure fails startup |
| `timeout` | `Option<u64>` | No | Per-inference timeout in seconds |
| `load_timeout` | `Option<u64>` | No | Model load timeout in seconds (default: 600s when omitted) |
| `retry` | `Option<RetryConfig>` | No | Retry policy for inference calls |
| `options` | `serde_json::Value` | No | Provider-specific configuration (API key env, quantization, etc.) |

- **Validation:** The builder MUST reject invalid aliases (empty, missing `/`), duplicate aliases, unknown provider IDs, and invalid provider options (unknown keys/wrong types) at build time with actionable error messages.

### FR-2: Provider Plugin Architecture

The system MUST support a pluggable provider model where each provider:

1. Self-identifies via a unique `provider_id` string.
2. Declares its supported tasks via `ProviderCapabilities`.
3. Implements a `load(spec) -> LoadedModelHandle` method that returns a type-erased handle.
4. Implements a `health() -> ProviderHealth` method for operational monitoring.

Providers MUST be feature-gated at compile time. Enabling `provider-candle` pulls in Candle dependencies; disabling it removes them entirely. The core runtime compiles and functions with zero providers enabled (useful for testing with mocks).

**Current providers:**

| Provider | ID | Feature Flag | Supported Tasks | Backend |
|----------|----|-------------|-----------------|---------|
| Candle | `local/candle` | `provider-candle` | Embed | HuggingFace Candle (BERT) |
| FastEmbed | `local/fastembed` | `provider-fastembed` | Embed | ONNX Runtime via fastembed |
| mistral.rs | `local/mistralrs` | `provider-mistralrs` | Embed, Generate | mistral.rs engine |
| OpenAI | `remote/openai` | `provider-openai` | Embed, Generate | OpenAI REST API |
| Gemini | `remote/gemini` | `provider-gemini` | Embed, Generate | Google Gemini REST API |

### FR-3: Task-Specific APIs

The system MUST expose strongly typed, async, `Send + Sync` interfaces per task:

**Embedding:**
```rust
#[async_trait]
pub trait EmbeddingModel: Send + Sync + Any {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> u32;
    fn model_id(&self) -> &str;
}
```

**Reranking:**
```rust
#[async_trait]
pub trait RerankerModel: Send + Sync {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>>;
}
```

**Generation:**
```rust
#[async_trait]
pub trait GeneratorModel: Send + Sync {
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult>;
}
```

The runtime MUST provide typed accessor methods (`embedding()`, `reranker()`, `generator()`) that downcast the stored handle and return a compile-time-safe reference to the correct trait object. Requesting the wrong task type for an alias MUST produce a `CapabilityMismatch` error.

### FR-4: Global Model Registry

The system MUST provide a shared, process-level registry of loaded model instances.

- **Key:** `ModelRuntimeKey` = `(task, provider_id, model_id, revision, variant_hash)`.
  - `variant_hash` is computed from the sorted, deterministic hash of provider-specific `options`.
  - Two aliases pointing to the same `(task, provider, model, revision, options)` MUST share one loaded instance.
- **Deduplication:** Concurrent requests for the same key MUST NOT trigger multiple loads. A per-key loader lock ensures exactly-once initialization with all waiters receiving the same handle.
- **Concurrency model:** Read-heavy access is optimized via `RwLock`. The common path (cache hit) acquires only a read lock. The slow path (first load) acquires a per-key `Mutex` to serialize initialization, then a write lock to insert.

### FR-5: Warmup Policies

Each catalog entry declares a warmup policy:

| Policy | Behavior |
|--------|----------|
| `Eager` | Model is loaded during `builder.build().await`. If `required: true` and loading fails, `build()` returns an error. If `required: false`, the failure is logged but build succeeds. |
| `Background` | `build()` spawns a detached tokio task to load the model. Build returns immediately. The model becomes available when loading completes. |
| `Lazy` | No loading at build time. The model is loaded on the first call to `embedding()`, `reranker()`, or `generator()`. |

### FR-6: Lifecycle and Health

- Each provider exposes a `health() -> ProviderHealth` method returning `Healthy`, `Degraded(reason)`, or `Unhealthy(reason)`.
- The runtime tracks which aliases have been resolved and which are still pending.
- Future: eviction/unload hooks for local models under memory pressure.

### FR-7: Observability

The system MUST emit:

| Signal | Type | Description |
|--------|------|-------------|
| `model_load.duration_seconds` | Histogram | Time to load a model by provider and task |
| `model_load.total` | Counter | Load attempts with `result` label (success/failure) |
| Cache hit/miss | Counter | Registry lookup results (planned) |
| Inference latency | Histogram | Per-call inference time (planned) |
| Loaded instances | Gauge | Number of currently loaded model instances (planned) |

All metrics use the `metrics` crate. Tracing spans wrap load and inference operations.

### FR-8: Reliability

**Circuit breaker** (implemented for remote providers):
- Three states: `Closed` -> `Open` -> `HalfOpen` -> `Closed`.
- Configurable `failure_threshold` (default 5) and `open_wait_seconds` (default 10).
- When open, calls fail immediately with `Unavailable` instead of hitting the network.

**Retry with backoff** (implemented for OpenAI):
- 3 attempts with exponential backoff (100ms, 200ms, 400ms).
- Retries on HTTP 429 (rate limited) and 5xx (server error).
- Non-retryable errors (401, 403, 4xx) fail immediately.

**Planned:**
- Per-alias fallback chains (e.g., `embed/default` -> try local Candle, fall back to remote OpenAI).
- Configurable timeout per provider.

### FR-9: Security

- Provider credentials (API keys) are read from environment variables, never hardcoded.
- The `api_key_env` option allows configuring which env var holds the key (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).
- Credentials MUST NOT appear in log output, error messages, or metric labels.
- Remote providers use HTTPS (TLS) by default.

### FR-10: Host Integration

Host applications integrate uni-xervo by:

1. Constructing a `ModelRuntimeBuilder`.
2. Registering provider instances.
3. Supplying a catalog of alias specs.
4. Calling `.build().await` to create the runtime.
5. Passing `Arc<ModelRuntime>` to subsystems that need model access.
6. Calling `runtime.embedding("embed/default").await` (or `reranker`, `generator`) to resolve and use models.

```rust
let runtime = ModelRuntime::builder()
    .register_provider(LocalCandleProvider::new())
    .register_provider(RemoteOpenAIProvider::new())
    .catalog(catalog_specs)
    .build()
    .await?;

let embedder = runtime.embedding("embed/default").await?;
let vectors = embedder.embed(vec!["hello world"]).await?;
```

---

## 6. Non-Functional Requirements

### Performance

| Metric | Target |
|--------|--------|
| Model handle lookup (cache hit) | < 1ms P50 |
| Concurrent same-key first load | Exactly 1 actual load, all waiters share result |
| Registry lock contention | Minimized via read-heavy `RwLock` + per-key loader `Mutex` |

### Availability

- Degraded operation if optional aliases fail to warm.
- Strict mode: fail startup if a `required: true` alias fails.
- Circuit breakers prevent cascade failures from remote provider outages.

### Scalability

- Dozens of aliases per process.
- High request concurrency with minimal lock contention.
- Memory usage proportional to number of loaded models (no hidden duplication).

### Compatibility

- Rust stable toolchain (edition 2024).
- Async runtime: tokio.
- No unsafe code outside of `VarBuilder::from_mmaped_safetensors` (Candle provider, for memory-mapped weight loading).

---

## 7. Architecture

### 7.1 Crate Structure

```
uni-xervo/
  Cargo.toml              # Feature-gated dependencies
  src/
    lib.rs                 # Module declarations
    api.rs                 # Domain types: ModelTask, WarmupPolicy, ModelAliasSpec, ModelRuntimeKey
    error.rs               # RuntimeError enum (10 variants)
    traits.rs              # Core traits: ModelProvider, EmbeddingModel, RerankerModel, GeneratorModel
    provider.rs            # Feature-gated module router + re-exports
    provider/
      candle.rs            # local/candle  (Embed)
      fastembed.rs          # local/fastembed (Embed)
      openai.rs            # remote/openai (Embed, Generate)
      gemini.rs            # remote/gemini (Embed, Generate)
      mistralrs.rs         # local/mistralrs (Embed, Generate)
    runtime.rs             # ModelRuntime, ModelRuntimeBuilder, ModelRegistry
    reliability.rs         # CircuitBreakerWrapper
    mock.rs                # Mock implementations (crate tests only)
  tests/
    api_validation_test.rs
    deduplication_test.rs
    embedding_model_test.rs
    error_handling_test.rs
    generator_model_test.rs
    provider_capability_test.rs
    real_providers_test.rs
    reranker_model_test.rs
    runtime_lifecycle_test.rs
    warmup_policy_test.rs
    integration_test.rs
  docs/
    first-spec.md          # This document
```

### 7.2 High-Level Component Diagram

```
                   Host Application
                         |
                         | Arc<ModelRuntime>
                         v
              +---------------------+
              |    ModelRuntime      |
              |---------------------|
              | .embedding(alias)   |    Typed accessors return
              | .reranker(alias)    | <- Arc<dyn EmbeddingModel>,
              | .generator(alias)   |    Arc<dyn GeneratorModel>, etc.
              +---------------------+
                    |           |
          +---------+           +---------+
          v                               v
  +----------------+            +------------------+
  | Catalog        |            | ModelRegistry     |
  | (alias -> spec)|            | (key -> instance) |
  +----------------+            +------------------+
          |                               |
          v                               v
  +------------------+          +------------------+
  | Provider Registry|          | Loader Locks     |
  | (id -> provider) |          | (key -> Mutex)   |
  +------------------+          | dedup init       |
          |                     +------------------+
          v
  +-------+-------+-------+-------+-------+
  | Candle | FEmbed| OpenAI| Gemini|Mistral|
  | local  | local | remote| remote| local |
  +--------+-------+-------+-------+-------+
```

### 7.3 Data Flow: Alias Resolution

```
1. Caller: runtime.embedding("embed/default").await
2. Runtime: catalog lookup -> ModelAliasSpec
3. Runtime: compute ModelRuntimeKey from spec
4. Registry: read lock -> check cache
   4a. Cache HIT -> return Arc<dyn EmbeddingModel>
   4b. Cache MISS -> continue to step 5
5. Registry: acquire per-key loader Mutex (blocks concurrent loaders for same key)
6. Registry: double-check cache (another thread may have loaded while we waited)
   6a. Now present -> release lock, return handle
   6b. Still absent -> continue to step 7
7. Runtime: look up provider by spec.provider_id
8. Provider: load(spec) -> LoadedModelHandle (type-erased Arc<dyn Any>)
9. Registry: write lock -> insert instance
10. Runtime: downcast handle to Arc<dyn EmbeddingModel>
11. Return to caller
```

### 7.4 Key Data Structures

```rust
// --- Domain Types (api.rs) ---

pub enum ModelTask { Embed, Rerank, Generate }

pub enum WarmupPolicy { Eager, Lazy, Background }

pub struct ModelAliasSpec {
    pub alias: String,            // "embed/default"
    pub task: ModelTask,          // Embed
    pub provider_id: String,      // "local/candle"
    pub model_id: String,         // "all-MiniLM-L6-v2"
    pub revision: Option<String>, // HF commit hash
    pub warmup: WarmupPolicy,     // Eager
    pub required: bool,           // true
    pub options: Value,           // {"cache_dir": "~/.cache/..."}
}

pub struct ModelRuntimeKey {
    pub task: ModelTask,
    pub provider_id: String,
    pub model_id: String,
    pub revision: Option<String>,
    pub variant_hash: u64,        // deterministic hash of sorted options
}

// --- Core Traits (traits.rs) ---

pub trait ModelProvider: Send + Sync {
    fn provider_id(&self) -> &'static str;
    fn capabilities(&self) -> ProviderCapabilities;
    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle>;
    async fn health(&self) -> ProviderHealth;
}

pub type LoadedModelHandle = Arc<dyn Any + Send + Sync>;

// --- Runtime (runtime.rs) ---

pub struct ModelRuntime {
    providers: HashMap<String, Box<dyn ModelProvider>>,
    registry: ModelRegistry,
    catalog: HashMap<String, ModelAliasSpec>,
}

struct ModelRegistry {
    instances: RwLock<HashMap<ModelRuntimeKey, LoadedInstance>>,
    loader_locks: Mutex<HashMap<ModelRuntimeKey, Arc<Mutex<()>>>>,
}
```

### 7.5 Concurrency Model

| Component | Synchronization | Rationale |
|-----------|----------------|-----------|
| Registry instance cache | `tokio::sync::RwLock` | Read-heavy. Most calls are cache hits (read lock only). |
| Per-key loader | `tokio::sync::Mutex` (one per key) | Prevents thundering herd. Only one task loads a given model; others await the same result. |
| Loader lock map | `tokio::sync::Mutex` | Protects the map of per-key mutexes. Held briefly to get-or-insert a key mutex. |
| Provider state (e.g., Candle model) | `tokio::sync::Mutex<Option<LoadedModel>>` | Lazy loading within a provider. Once loaded, the inner Option is Some and subsequent callers skip init. |
| FastEmbed ONNX model | `std::sync::Mutex<TextEmbedding>` + dedicated thread | ONNX Runtime is not async. A dedicated 8MB-stack thread runs inference; async callers use oneshot channels. |
| Circuit breaker | `std::sync::Mutex<Inner>` | Simple state machine. Held briefly per call to check/update state. |

### 7.6 Type Safety via Double-Arc Pattern

Providers return `LoadedModelHandle = Arc<dyn Any + Send + Sync>`. The concrete value stored inside is `Arc<dyn EmbeddingModel>` (or `RerankerModel`, `GeneratorModel`). The runtime downcasts in two steps:

```
LoadedModelHandle                         Arc<dyn Any + Send + Sync>
  -> downcast_ref::<Arc<dyn EmbeddingModel>>()
  -> Arc<dyn EmbeddingModel>              Typed, safe to call .embed()
```

This double-Arc pattern allows:
- Type erasure at the registry level (all handles stored uniformly).
- Type recovery at the accessor level (compile-time safe task API).
- Shared ownership (multiple callers hold `Arc` clones to the same model).

---

## 8. Provider System Design

### 8.1 Provider Trait Contract

Every provider MUST implement:

```rust
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Unique identifier (e.g., "local/candle", "remote/openai").
    fn provider_id(&self) -> &'static str;

    /// Tasks this provider can handle.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Load a model instance for the given spec.
    /// Returns a type-erased handle wrapping Arc<dyn TaskTrait>.
    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle>;

    /// Operational health check.
    async fn health(&self) -> ProviderHealth;
}
```

**Conventions:**
- `provider_id` follows `<locality>/<name>` format: `local/candle`, `remote/openai`.
- `load()` MUST validate that `spec.task` is in `capabilities().supported_tasks`.
- `load()` MUST return `CapabilityMismatch` for unsupported tasks.
- The returned handle MUST wrap `Arc<dyn EmbeddingModel>` for embed tasks, `Arc<dyn RerankerModel>` for rerank, `Arc<dyn GeneratorModel>` for generate.

### 8.2 Provider Implementations

#### local/candle

- **Backend:** HuggingFace Candle (pure Rust, no FFI).
- **Tasks:** Embed.
- **Models:** BERT-family sentence transformers:
  - `all-MiniLM-L6-v2` (384 dims, default)
  - `bge-small-en-v1.5` (384 dims)
  - `bge-base-en-v1.5` (768 dims)
- **Loading:** Downloads config, tokenizer, and safetensors weights from HuggingFace Hub. Memory-maps weights via `VarBuilder::from_mmaped_safetensors`.
- **Inference pipeline:** Tokenize -> BERT forward pass -> mean pooling over non-padding tokens -> L2 normalization.
- **Lazy internal loading:** The model downloads and loads on first `embed()` call (within the provider's own `Mutex`), not at `provider.load()` time. The runtime's registry dedup still applies at the outer level.
- **Feature flag:** `provider-candle` (default).

#### local/fastembed

- **Backend:** ONNX Runtime via the `fastembed` crate.
- **Tasks:** Embed.
- **Models:** AllMiniLML6V2, BGEBaseENV15, BGESmallENV15, NomicEmbedTextV15, MultilingualE5Small.
- **Thread safety:** ONNX Runtime requires an 8MB stack. Inference runs on a dedicated `std::thread` with `EMBEDDING_THREAD_STACK_SIZE = 8 * 1024 * 1024`. Results are communicated back to the async runtime via `tokio::sync::oneshot`.
- **Initialization:** Model creation is offloaded to `tokio::task::spawn_blocking`.
- **Feature flag:** `provider-fastembed`.

#### local/mistralrs

- **Backend:** mistral.rs engine.
- **Tasks:** Embed, Generate.
- **Loading:** Uses mistral.rs `Model` with configurable ISQ (In-Situ Quantization) types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K through Q8K, HQQ4, HQQ8, F8E4M3, AFQ2 through AFQ8.
- **Configuration (`options` keys):**
  - `isq`: ISQ quantization type string.
  - `force_cpu`: Force CPU execution even when GPU is available.
  - `paged_attention`: Enable paged attention.
  - `max_num_seqs`: Maximum number of sequences.
  - `chat_template`: Custom chat template.
  - `tokenizer_json`: Custom tokenizer path.
  - `embedding_dimensions`: Override output dimensions for embedding models.
- **Embedding:** Probes actual dimensions at load time if `embedding_dimensions` is not specified.
- **Generation:** Maps input messages to alternating User/Assistant roles. Supports `temperature`, `top_p`, `max_tokens` via `GenerationOptions`. Returns `GenerationResult` with `TokenUsage`.
- **Feature flag:** `provider-mistralrs`.

#### remote/openai

- **Backend:** OpenAI REST API (`/v1/embeddings`, `/v1/chat/completions`).
- **Tasks:** Embed, Generate.
- **Authentication:** API key read from env var (configurable via `options.api_key_env`, default `OPENAI_API_KEY`).
- **Reliability:**
  - Circuit breaker (failure_threshold=5, open_wait=10s).
  - Retry: 3 attempts with exponential backoff (100ms base). Retries on HTTP 429 and 5xx.
  - Non-retryable: 401/403 -> `Unauthorized`, other 4xx -> `ApiError`.
- **Dimension mapping:**
  - `text-embedding-3-large` -> 3072
  - `text-embedding-3-small`, `text-embedding-ada-002` -> 1536
- **Feature flag:** `provider-openai`.

#### remote/gemini

- **Backend:** Google Gemini REST API.
- **Tasks:** Embed, Generate.
- **Authentication:** API key read from env var (configurable via `options.api_key_env`, default `GEMINI_API_KEY`).
- **Embedding:** Uses `batchEmbedContents` endpoint. Default dimensions: 768.
- **Generation:** Uses `generateContent` endpoint. Returns `GenerationResult` (token usage not currently reported by Gemini API response parsing).
- **Reliability:** Circuit breaker (same config as OpenAI).
- **Feature flag:** `provider-gemini`.

### 8.3 Adding a New Provider

To add a new provider (e.g., `remote/anthropic`):

1. **Add feature flag** in `Cargo.toml`:
   ```toml
   provider-anthropic = ["dep:reqwest"]
   ```

2. **Create module** `src/provider/anthropic.rs` implementing `ModelProvider`.

3. **Register in router** `src/provider.rs`:
   ```rust
   #[cfg(feature = "provider-anthropic")]
   pub mod anthropic;
   ```

4. **Write tests:** Add mock-based unit tests in `tests/` and `#[ignore]`-gated integration tests in `tests/real_providers_test.rs`.

5. **No core runtime changes.** The runtime, registry, and builder are provider-agnostic.

---

## 9. Error Taxonomy

```rust
pub enum RuntimeError {
    Config(String),           // Invalid alias format, missing fields, unknown provider
    ProviderNotFound(String), // No registered provider matches spec.provider_id
    CapabilityMismatch(String), // Provider doesn't support the requested task
    Load(String),             // Model download, weight loading, or initialization failure
    ApiError(String),         // HTTP/API transport or response-shape failure
    InferenceError(String),   // Local/remote model inference pipeline failure
    RateLimited,              // HTTP 429 from remote provider
    Unauthorized,             // HTTP 401/403 from remote provider
    Timeout,                  // Request exceeded configured timeout
    Unavailable,              // Circuit breaker open, provider down
}
```

**Design principles:**
- Every variant carries enough context to diagnose the problem (alias, provider, model, stage).
- Variants map cleanly to operational responses: `RateLimited` -> back off, `Unauthorized` -> check credentials, `Unavailable` -> check provider health.
- The `Result<T>` type alias uses `RuntimeError` throughout the crate.

---

## 10. Reliability Design

### 10.1 Circuit Breaker

```
                    success
            +-------------------+
            |                   |
            v                   |
  +--------+--------+    +-----------+
  |     Closed      |--->| Half-Open |
  | (normal traffic)|    | (1 probe) |
  +--------+--------+    +-----+-----+
            |                   |
            | failure_threshold |  failure
            v                   v
  +--------+--------+    +-----+-----+
  |      Open       |<---| Half-Open |
  | (reject all)    |    | (re-open) |
  +--------+--------+    +-----------+
            |
            | open_wait_seconds elapsed
            v
        Half-Open
```

**Configuration:**
- `failure_threshold`: Number of consecutive failures before opening (default: 5).
- `open_wait_seconds`: Seconds to wait before allowing a probe (default: 10).

**Scope:** One circuit breaker per remote model runtime key (`task + provider_id + model_id + revision + options`). Local providers do not use circuit breakers.

### 10.2 Retry Policy (OpenAI)

- Max attempts: 3.
- Backoff: Exponential (100ms * 2^attempt).
- Retryable conditions: HTTP 429, HTTP 5xx.
- Non-retryable: HTTP 401, 403, other 4xx.

### 10.3 Planned: Fallback Chains

Future support for per-alias fallback:
```yaml
- alias: embed/default
  primary:
    provider: local/candle
    model: all-MiniLM-L6-v2
  fallback:
    provider: remote/openai
    model: text-embedding-3-small
```

When the primary fails (after exhausting retries), the runtime transparently falls back to the secondary provider.

---

## 11. Configuration Model

### 11.1 Programmatic (Rust API)

```rust
let catalog = vec![
    ModelAliasSpec {
        alias: "embed/default".into(),
        task: ModelTask::Embed,
        provider_id: "local/candle".into(),
        model_id: "all-MiniLM-L6-v2".into(),
        revision: None,
        warmup: WarmupPolicy::Eager,
        required: true,
        options: json!({}),
    },
    ModelAliasSpec {
        alias: "embed/openai".into(),
        task: ModelTask::Embed,
        provider_id: "remote/openai".into(),
        model_id: "text-embedding-3-small".into(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        options: json!({ "api_key_env": "OPENAI_API_KEY" }),
    },
    ModelAliasSpec {
        alias: "llm/local".into(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".into(),
        model_id: "mistralai/Mistral-7B-Instruct-v0.3".into(),
        revision: None,
        warmup: WarmupPolicy::Background,
        required: false,
        options: json!({ "isq": "Q4K", "force_cpu": true }),
    },
];
```

### 11.2 Declarative (YAML/JSON, host-layer responsibility)

uni-xervo does not parse config files itself. Host applications deserialize their config format into `Vec<ModelAliasSpec>` and pass it to the builder. Example YAML that a host might support:

```yaml
model_runtime:
  aliases:
    - alias: embed/default
      task: embed
      provider: local/candle
      model: all-MiniLM-L6-v2
      warmup: eager
      required: true

    - alias: embed/legal
      task: embed
      provider: remote/openai
      model: text-embedding-3-large
      warmup: lazy
      options:
        api_key_env: OPENAI_API_KEY

    - alias: llm/chat
      task: generate
      provider: local/mistralrs
      model: mistralai/Mistral-7B-Instruct-v0.3
      warmup: background
      options:
        isq: Q4K
        force_cpu: true
```

### 11.3 Provider-Specific Options

Options are provider-interpreted JSON and validated against provider schemas. Unknown keys and wrong types are rejected at build/register time.

| Provider | Option | Type | Description |
|----------|--------|------|-------------|
| remote/openai | `api_key_env` | string | Env var name for API key |
| remote/gemini | `api_key_env` | string | Env var name for API key |
| local/fastembed | `cache_dir` | string | Model cache directory |
| local/mistralrs | `isq` | string | ISQ quantization type |
| local/mistralrs | `force_cpu` | bool | Force CPU execution |
| local/mistralrs | `paged_attention` | bool | Enable paged attention |
| local/mistralrs | `max_num_seqs` | number | Max concurrent sequences |
| local/mistralrs | `chat_template` | string | Custom chat template |
| local/mistralrs | `embedding_dimensions` | number | Override embedding output dimensions |

Schemas:
- `schemas/model-catalog.schema.json`
- `schemas/provider-options/openai.schema.json`
- `schemas/provider-options/gemini.schema.json`
- `schemas/provider-options/candle.schema.json`
- `schemas/provider-options/fastembed.schema.json`
- `schemas/provider-options/mistralrs.schema.json`

---

## 12. Integration with Uni (Primary Consumer)

### 12.1 Current Integration Points

uni-xervo is consumed by three crates in the Uni database:

**`uni` (main database API):**
- `UniBuilder` constructs a `ModelRuntimeBuilder`, registers providers, supplies catalog from database config.
- The built `Arc<ModelRuntime>` is stored on the `Uni` struct and passed to writer and query contexts.

**`uni-store` (storage layer):**
- `Writer` holds `model_runtime: Option<Arc<ModelRuntime>>`.
- `get_embedding_service()` resolves embedding models via the runtime, with legacy fallback for backward compatibility.
- `XervoEmbeddingAdapter` bridges `uni_xervo::traits::EmbeddingModel` to `uni_store::EmbeddingService`.

**Query engine integration:**
- Query execution contexts propagate `model_runtime` through request execution.
- Vector search operations resolve embedding aliases from schema config to generate query vectors at runtime.

### 12.2 Dependency Direction

```
uni (database API)
  -> uni-store (storage, embedding at write time)
       -> uni-xervo (model runtime)
  -> uni-query (query engine, embedding at query time)
       -> uni-xervo (model runtime)
```

uni-xervo has **zero** dependencies on Uni. It is a pure model runtime library. All integration logic lives in the consuming crates via adapter patterns.

### 12.3 Adapter Pattern

`XervoEmbeddingAdapter` in `uni-store` implements uni-store's `EmbeddingService` trait by delegating to uni-xervo's `EmbeddingModel` trait:

```rust
pub struct XervoEmbeddingAdapter {
    model: Arc<dyn EmbeddingModel>,
}

impl EmbeddingService for XervoEmbeddingAdapter {
    async fn embed(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.model.embed(texts.to_vec()).await.unwrap()
    }
    fn dimensions(&self) -> u32 { self.model.dimensions() }
    fn model_name(&self) -> &str { self.model.model_id() }
}
```

This pattern keeps uni-xervo independent while allowing seamless integration.

---

## 13. Testing Strategy

### 13.1 Test Architecture

Tests are split into two tiers:

**Tier 1: Fast mock tests (~87+ tests, ~2 seconds)**
- Use `MockProvider`, `MockEmbeddingModel`, `MockRerankerModel`, `MockGeneratorModel` in crate tests and `tests/common/mock_support.rs` for integration tests.
- Mock helpers are test-only and not exported as public runtime API.
- Cover all core runtime logic without network access, model downloads, or API keys.
- Run on every CI push.

**Tier 2: Real provider integration tests (~10 tests, `#[ignore]`)**
- Gated behind `EXPENSIVE_TESTS=1` environment variable.
- Require model downloads (100MB+), network access, and API keys for remote providers.
- Cover end-to-end flows with real backends.
- Run nightly or before releases.

### 13.2 Test Coverage Matrix

| Test File | Count | Covers |
|-----------|-------|--------|
| `api_validation_test.rs` | 12 | Alias validation, key determinism, serde roundtrip |
| `deduplication_test.rs` | 5 | Registry dedup, thundering herd, shared instances |
| `embedding_model_test.rs` | 8 | Embed API, batching, dimensions, failure propagation |
| `generator_model_test.rs` | 8 | Generate API, options, usage, failure propagation |
| `reranker_model_test.rs` | 8 | Rerank API, scoring, indices, failure propagation |
| `error_handling_test.rs` | 14 | All error variants, propagation, display strings |
| `provider_capability_test.rs` | 8 | Capability checks, health status, task mismatch |
| `runtime_lifecycle_test.rs` | 12 | Builder, registration, resolution, downcast, multi-provider |
| `warmup_policy_test.rs` | 8 | Eager/lazy/background warmup, failure handling, mixed policies |
| `real_providers_test.rs` | 8+ | Real Candle, FastEmbed, OpenAI, Gemini, mistral.rs, RAG workflow |
| `integration_test.rs` | 2 | End-to-end Candle embedding, warmup policies |

### 13.3 Mock System Design

The mock module provides configurable test doubles:

- **`MockProvider`**: Configurable `provider_id`, `supported_tasks`, `health`, `load_delay_ms`, `fail_on_load`. Factory methods: `embed_only()`, `generate_only()`, `rerank_only()`, `failing()`.
- **`MockEmbeddingModel`**: Returns zero vectors of configurable dimensions. Tracks `call_count`. Supports `fail_on_embed`.
- **`MockRerankerModel`**: Returns descending scores (`1/(i+1)`). Tracks `call_count`. Supports `fail_on_rerank`.
- **`MockGeneratorModel`**: Returns configurable `response_text`. Reports `TokenUsage`. Tracks `call_count`. Supports `fail_on_generate`.
- **Helper functions**: `make_spec()`, `runtime_with_embed()`, `runtime_with_generator()`, `runtime_with_reranker()` for concise test setup.

### 13.4 Running Tests

```bash
# Fast tests (CI, every push)
cargo test
cargo nextest run              # parallel execution

# Integration tests (nightly, pre-release)
EXPENSIVE_TESTS=1 cargo test --test real_providers_test -- --ignored

# Specific provider
EXPENSIVE_TESTS=1 cargo test test_candle_local_embedding -- --ignored

# All features
cargo test --all-features
```

---

## 14. Feature Flag Design

```toml
[features]
default = ["provider-candle"]

provider-candle = [
    "dep:candle-core", "dep:candle-nn", "dep:candle-transformers",
    "dep:tokenizers", "dep:hf-hub",
]
provider-fastembed = ["dep:fastembed"]
provider-openai = ["dep:reqwest"]
provider-gemini = ["dep:reqwest"]
provider-mistralrs = ["dep:mistralrs"]
```

**Design rationale:**
- Each provider is independently toggleable. A consumer that only needs OpenAI embeddings can compile with `--no-default-features --features provider-openai` and avoid pulling in Candle, ONNX, or mistral.rs.
- `provider-openai` and `provider-gemini` share `reqwest` but are independent features because a user may want one without the other.
- The core runtime (`api`, `error`, `traits`, `runtime`, `reliability`, `mock`) compiles with zero provider features. This enables pure-mock testing with minimal compile time.
- Default is `provider-candle` because it's the most common local embedding backend and has no system-level dependencies (pure Rust).

---

## 15. Deployment Modes

### In-Process (Current, Default)

The runtime lives in the same process as the host application. This is the lowest-latency option and the recommended mode for database engines, CLI tools, and backend services.

```
[Host Process]
  +-- Application Logic
  +-- Arc<ModelRuntime>
       +-- loaded models (in-memory weights, connections)
```

### Service Mode (Future, Planned)

Wrap the runtime behind a gRPC or HTTP service for:
- GPU consolidation across multiple processes.
- Multi-language client support (Python, Go, Node via gRPC).
- Centralized model management and monitoring.

The same catalog, provider, and trait abstractions would be reused. The service would add a network transport layer and client SDK.

---

## 16. Roadmap

### Phase 1 (Complete): MVP

- Core runtime, catalog, registry with dedup.
- Providers: Candle, FastEmbed, OpenAI, Gemini, mistral.rs.
- Tasks: Embed, Generate (Rerank trait defined, no production provider yet).
- Circuit breaker and retry for remote providers.
- Mock-based test suite (87+ tests).
- Integration test framework with `EXPENSIVE_TESTS` gating.
- Extraction from monorepo into standalone crate.

### Phase 2: Hardening and Operational Maturity

- Per-alias fallback chains.
- Configurable per-provider timeouts.
- Enhanced observability: inference latency histograms, cache hit/miss counters, loaded instance gauge.
- Tracing span coverage for full request lifecycle.
- Lifecycle states (`Loading`, `Ready`, `Failed`) visible via API.
- Graceful shutdown hooks.
- Reranker provider implementation (e.g., Cohere Rerank, Gemini reranker).
- Streaming generation support.

### Phase 3: Scale and Ecosystem

- Memory policies and LRU eviction for local models.
- Optional service-mode deployment (gRPC).
- Provider conformance test suite (automated validation for new providers).
- Published crate on crates.io.
- Additional providers: Anthropic, Cohere, Ollama, vLLM.
- Multimodal embedding support (CLIP, BLIP).
- Batch/queue inference mode for high-throughput pipelines.

---

## 17. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Provider dependency churn (Candle, mistral.rs breaking changes) | Build failures, API drift | Feature gates isolate blast radius. Pin versions. CI per-feature matrix. |
| Memory pressure from many loaded local models | OOM, degraded performance | Warmup policy limits eager loads. Phase 3 adds LRU eviction. Monitor with metrics. |
| Remote provider outages | Request failures, user-facing errors | Circuit breakers, retry with backoff, planned fallback chains. |
| Configuration sprawl | Hard to reason about behavior | Validated catalog schema. `task/name` alias convention. Build-time rejection of invalid configs. |
| Mock/production behavior divergence | Tests pass but production breaks | Real-provider integration tests in CI (nightly). Mock behavior matches trait contracts. |
| ONNX Runtime stack overflow (FastEmbed) | Crashes in high-concurrency scenarios | Dedicated 8MB-stack thread for ONNX inference. Document requirement. |

---

## 18. Security Considerations

- **Credentials:** API keys read from environment variables only. Never stored in config files shipped with the binary, never logged, never included in error messages or metrics labels.
- **Network:** Remote providers use HTTPS. `reqwest` configured with `rustls-tls`.
- **Model weights:** Downloaded from HuggingFace Hub over HTTPS. Cached locally in user's home directory (`~/.cache/huggingface/`). No credential material stored in weight files.
- **Unsafe code:** Limited to `VarBuilder::from_mmaped_safetensors` in the Candle provider for memory-mapped weight loading. This is a well-audited pattern from the Candle ecosystem.
- **Supply chain:** Feature flags limit the dependency surface. A consumer using only `provider-openai` does not pull in ML framework dependencies.

---

## 19. Acceptance Criteria

1. A host application can configure 3+ aliases across 2+ providers and resolve all of them through a single `ModelRuntime` instance.
2. Two aliases pointing to the same `(task, provider, model, revision, options)` share exactly one loaded model instance.
3. Concurrent first requests for the same alias result in exactly one model load (thundering herd prevention).
4. `Eager` warmup blocks `build()` until loading completes. Failure of a `required: true` eager alias fails the build.
5. `Background` warmup returns from `build()` immediately. The model loads asynchronously.
6. `Lazy` warmup does not load until the first `embedding()`/`generator()`/`reranker()` call.
7. Circuit breaker opens after configured failures and rejects calls immediately. Half-open probe succeeds and closes the breaker.
8. All 87+ mock-based tests pass. All ignored integration tests pass when run with appropriate env vars.
9. No provider-specific code leaks into host application business logic.
10. Adding a new provider requires only: one new file, one feature flag, one `provider.rs` entry. Zero changes to core runtime.

---

## 20. Open Questions

1. **Streaming generation:** Should `GeneratorModel` support `Stream<Item = String>` for token-by-token output? If so, what's the trait signature?
2. **Alias versioning:** Should aliases support explicit versions (`embed/default@v2`) or is catalog-level versioning sufficient?
3. **Fallback scope:** Should fallback chains be configured per-alias, per-task, or globally?
4. **Memory caps:** When should memory limits and LRU eviction be enforced -- phase 2 or phase 3?
5. **Reranker providers:** Which reranker backend should be the first real implementation? Cohere Rerank, Gemini, or a local cross-encoder via Candle?
6. **Batch inference API:** Should there be a separate batched inference path for high-throughput pipelines, or is the current `Vec<&str>` input sufficient?
7. **Model hot-reloading:** Should the runtime support swapping a model behind an alias without restarting the process?
