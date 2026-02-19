# Architecture

```mermaid
flowchart LR
    A[App code] --> B[ModelRuntime]
    B --> C[Alias lookup]
    C --> D[ModelAliasSpec]
    D --> E[ModelRuntimeKey]
    E --> F{Registry hit?}
    F -->|Yes| G[Reuse loaded handle]
    F -->|No| H[Per-key load mutex]
    H --> I[Provider.load(spec)]
    I --> J[Model warmup]
    J --> K[Cache handle in registry]
    K --> L[Typed resolver]
    G --> L
    L --> M[Instrumented wrapper timeout/retry/metrics]
    M --> N[embed/rerank/generate call]
```

## Build-time flow

1. Register providers in builder.
2. Ingest catalog and validate each `ModelAliasSpec`.
3. Enforce provider existence and options validation.
4. Apply provider warmup policy.
5. Apply per-alias model warmup policy.

## Runtime flow

1. Resolve alias via catalog.
2. Compute runtime key (includes normalized options hash).
3. Return cached instance if already loaded.
4. Otherwise load under key-level mutex and cache.
5. Downcast to task trait and wrap in instrumented model.

## Design notes for contributors

- `options_validation` should be updated whenever new provider options are added.
- Provider option schemas under `schemas/provider-options/` should mirror runtime validation.
- Remote providers should use shared helpers in `provider/remote_common.rs` for consistent auth resolution, HTTP status mapping, and circuit breaker behavior.
- New providers must declare accurate `capabilities()` so runtime task resolution remains correct.
