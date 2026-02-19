# Development Guide

This document covers local development workflows for `uni-xervo`.

## Prerequisites

- Rust toolchain `1.85+`
- `cargo fmt`
- Optional: `cargo-nextest`
- Optional (website docs): Python + Poetry

## Repository Layout

- `src/`: runtime core, provider traits, provider implementations
- `tests/`: integration and behavior tests
- `schemas/`: model catalog and provider option schemas
- `scripts/`: build/test/docs helper scripts
- `website/`: MkDocs documentation site
- `docs/`: architecture and long-form project docs

## Common Commands

```bash
# Build
./scripts/build.sh

# Release build
./scripts/build.sh --release

# Quality gate: format + check + test
./scripts/test.sh

# Integration tests (real providers, expensive)
./scripts/test-integration.sh

# Build rustdoc + website docs
./scripts/doc.sh
```

## Feature-Oriented Development

Providers are feature-gated. During provider work, run targeted checks:

```bash
cargo check --locked --no-default-features --features provider-openai
cargo check --locked --no-default-features --features provider-candle,provider-openai
```

## Integration Tests and Credentials

Real-provider tests are ignored unless explicitly enabled.

```bash
EXPENSIVE_TESTS=1 cargo test --all-features --test real_providers_test -- --ignored
```

Common env vars:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `VERTEX_AI_TOKEN`
- `VERTEX_AI_PROJECT`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `VOYAGE_API_KEY`
- `CO_API_KEY`
- `AZURE_OPENAI_API_KEY`

## Local Prefetch CLI

Use the prefetch binary to pre-warm local model caches from a catalog JSON file:

```bash
cargo run --bin prefetch -- model-catalog.json --dry-run
cargo run --bin prefetch -- model-catalog.json
```

## Release Process

See `RELEASE.md` for the tag-driven crates.io release flow.
