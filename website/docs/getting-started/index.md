# Getting Started

This section gets you from dependency setup to a production-ready runtime configuration.

## Prerequisites

- Rust `1.85+` (crate `edition = "2024"`).
- Tokio async runtime.
- Provider feature flags enabled for the providers you plan to register.
- Required API keys/tokens in environment variables for remote providers.

## Recommended path

1. [Installation](installation.md)
2. [Quick Start](quickstart.md)
3. [CLI Reference](cli-reference.md)

## Build strategy

- Start small: one local embedding alias and one remote generation alias.
- Keep app code alias-based (`embed/default`, `generate/chat`) so provider swaps are config-only changes.
- Turn on strict validation in CI by checking catalogs against `schemas/model-catalog.schema.json`.

## After first boot

- Deep dive catalog semantics: [Model Catalog](../concepts/model-catalog.md)
- Understand load behavior: [Runtime Loading](../concepts/runtime-loading.md)
- Add resilience controls: [Reliability](../concepts/reliability.md)
