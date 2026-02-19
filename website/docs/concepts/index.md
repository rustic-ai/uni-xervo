# Concepts

Uni-Xervo is organized around four core concepts:

- Model catalog: declarative alias-to-provider mapping.
- Runtime loading: safe lazy/eager/background model initialization.
- Reliability: timeout, retry, and remote circuit breaker controls.
- Capability-driven APIs: embed/rerank/generate with typed trait handles.

## Request lifecycle

1. App asks for alias handle (`runtime.embedding("embed/default")`).
2. Runtime resolves alias to `ModelAliasSpec`.
3. Runtime computes a `ModelRuntimeKey` (task + provider + model + revision + options hash).
4. Existing loaded instance is reused, or provider load is coordinated under a per-key mutex.
5. Returned handle is wrapped with instrumentation (metrics + timeout/retry).

## Why this matters

- Application code stays provider-agnostic.
- Equivalent aliases share loaded instances.
- Cold start and failure handling are explicit and configurable.
