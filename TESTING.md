# uni-xervo Testing Guide

## Script Shortcuts

```bash
# Build the crate
./scripts/build.sh

# Run standard quality gates
./scripts/test.sh

# Run the remote-provider integration tests (expensive, needs API keys)
./scripts/test-integration.sh

# Run the local-model acceptance matrix for one platform (downloads models)
./scripts/acceptance.sh cpu
```

## Deployment Acceptance Matrix

`scripts/acceptance.sh` runs every opt-in real-model suite for one platform
target and writes a report of what passed, what each suite cost, and which
model revisions the run used:

```bash
scripts/acceptance.sh cpu       # default features; any machine
scripts/acceptance.sh coreml    # local/onnx with gpu-metal, plus the CoreML tests (macOS)
scripts/acceptance.sh cuda      # local/onnx with gpu-cuda, plus the CUDA tests (needs nvcc)
scripts/acceptance.sh cpu --only ocr,hybrid-bge-m3   # a subset
```

- **Suites:**
  - ONNX embedding, reranking, sparse, multi-vector and BGE-M3 hybrid (with
    the widths pinned: 1024-wide dense and per-token outputs, a 250002-term
    sparse vocabulary, parity with the per-task models);
  - kniv NLP v1 and v7;
  - PP-OCR detection and recognition, and PDF to OCR;
  - execution-provider reporting, including a real registration skip when
    `ORT_DYLIB_PATH` names an ONNX Runtime library;
  - whisper.cpp;
  - and, for `coreml` or `cuda`, the accelerator inference tests.
- **Output:** the report is `target/acceptance/<target>/report.md`, with one
  log per suite beside it. Each suite records passed, failed and skipped
  counts, wall time, peak memory, and the exact command, so any row can be
  rerun alone.
- **Model manifest:** the commit each Hugging Face repository in the cache
  resolved to. Set those as an alias's `revision` to reproduce a run.
- **Requirements:** network on the first run (models download into the cache),
  no API keys. A suite this machine cannot build, such as whisper.cpp without
  cmake, is reported as skipped rather than failed.
- **Exit status:** non-zero if any suite failed.
- **CI:** `.github/workflows/acceptance.yml` runs the `cpu` target on demand
  (Actions → Acceptance), caching models between runs, and uploads the report.

The latest recorded results are in [`docs/TEST_MATRIX.md`](docs/TEST_MATRIX.md).
The matrix covers the local backends; remote providers need keys and are
exercised by `scripts/test-integration.sh`.

## Recommended Runner: cargo-nextest

We use [cargo-nextest](https://nexte.st/) as our primary test runner for its speed and enhanced reporting.

### Installation

```bash
cargo install cargo-nextest
```

### Running Tests

```bash
# Run all tests (fast mock tests)
cargo nextest run

# Run integration tests (requires EXPENSIVE_TESTS=1)
EXPENSIVE_TESTS=1 cargo nextest run --test real_providers_test

# Run tests with CI profile (retries enabled)
cargo nextest run --profile ci
```

## Test Categories

### 1. **Unit & Mock Tests** (Fast, No Dependencies)
These tests use mock providers and run without network access or model downloads.

```bash
# Run all fast tests (default)
cargo nextest run -p uni-xervo

# Run with standard cargo test
cargo test -p uni-xervo
```

**Coverage**: 87+ tests covering all core functionality with mocks.

---

### 2. **Integration Tests** (Expensive, Real Providers)
These tests use real ML models and API providers. They require:
- Large model downloads (100MB+)
- Network access
- API keys for remote providers
- Significant compute time

#### Running Integration Tests

Set the `EXPENSIVE_TESTS` environment variable to enable:

```bash
# Run ALL integration tests (downloads models + uses APIs)
EXPENSIVE_TESTS=1 cargo nextest run -p uni-xervo --test real_providers_test

# Run specific test
EXPENSIVE_TESTS=1 cargo nextest run -p uni-xervo --test real_providers_test -- -E 'test(test_local_onnx_bge_small_embedding)'
```

#### Environment Variables

For remote provider tests, set API keys:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export VERTEX_AI_TOKEN="..."
export VERTEX_AI_PROJECT="your-gcp-project-id"
```

---

## Test Matrix

### Local Embedding Providers

| Provider | Test | Model | Size | Features Required |
|----------|------|-------|------|-------------------|
| **ONNX (Embed)** | `test_local_onnx_bge_small_embedding` | BGESmallENV15 | ~130MB | `provider-onnx` |
| **Candle** | `test_candle_local_embedding` | all-MiniLM-L6-v2 | ~90MB | `provider-candle` |
### Remote Embedding Providers

| Provider | Test | Model | API Key | Features Required |
|----------|------|-------|---------|-------------------|
| **OpenAI** | `test_openai_remote_embedding` | text-embedding-3-small | `OPENAI_API_KEY` | `provider-openai` |
| **Gemini** | `test_gemini_remote_embedding` | embedding-001 | `GEMINI_API_KEY` | `provider-gemini` |
| **Vertex AI** | `test_vertexai_remote_embedding` | text-embedding-005 | `VERTEX_AI_TOKEN` + `VERTEX_AI_PROJECT` | `provider-vertexai` |
| **llama.cpp** | `test_llamacpp_remote_embedding` | `LLAMACPP_EMBED_MODEL` (default bge-small-en-v1.5) | none; needs `LLAMACPP_BASE_URL` of a running `llama-server` (`LLAMACPP_EMBED_DIMENSIONS`, `LLAMACPP_MAX_INPUT_TOKENS` optional) | `provider-llamacpp` |

### Remote Generation Providers

| Provider | Test | Model | API Key | Features Required |
|----------|------|-------|---------|-------------------|
| **Gemini** | `test_gemini_remote_generation` | gemini-pro | `GEMINI_API_KEY` | `provider-gemini` |
| **Vertex AI** | `test_vertexai_remote_generation` | gemini-1.5-flash | `VERTEX_AI_TOKEN` + `VERTEX_AI_PROJECT` | `provider-vertexai` |

### Multi-Provider Tests

| Test | Description | Requirements |
|------|-------------|--------------|
| `test_multi_provider_integration` | Tests multiple providers in single runtime | At least one provider + API keys |
| `test_rag_workflow` | Full RAG workflow: embed → retrieve → generate | `local/onnx` (Embed) + Gemini API |

---

## Running Specific Test Suites

### Test Only Local Providers (No API Keys Needed)

```bash
EXPENSIVE_TESTS=1 cargo nextest run -p uni-xervo \
  --test real_providers_test \
  -E 'test(test_local_onnx_bge_small_embedding) or test(test_candle_local_embedding)'
```

### Test Only Remote Providers (Requires API Keys)

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export VERTEX_AI_TOKEN="..."
export VERTEX_AI_PROJECT="your-gcp-project-id"

EXPENSIVE_TESTS=1 cargo nextest run -p uni-xervo \
  --test real_providers_test \
  -E 'test(remote)'
```

### Test RAG Workflow

```bash
export GEMINI_API_KEY="..."

EXPENSIVE_TESTS=1 cargo nextest run -p uni-xervo --test real_providers_test -E 'test(test_rag_workflow)'
```

---

## Feature Flags

Enable specific providers:

```bash
# Enable all providers (default)
cargo nextest run -p uni-xervo --all-features

# Enable specific providers only
cargo nextest run -p uni-xervo --no-default-features \
  --features provider-onnx,provider-openai
```

Available features:
- `provider-candle` - Local Candle-based models
- `provider-onnx` - Local ONNX Runtime models (raw, rerank, embed)
- `provider-openai` - Remote OpenAI API
- `provider-gemini` - Remote Gemini API
- `provider-vertexai` - Remote Google Vertex AI API
- `provider-llamacpp` - Remote llama.cpp `llama-server` embeddings (mock-server tests in `tests/llamacpp_provider_test.rs` run without a server)

---

## CI/CD Recommendations

### GitHub Actions Example

```yaml
# Fast tests (run on every PR)
- name: Install cargo-nextest
  uses: taiki-e/install-action@nextest

- name: Run unit tests
  run: cargo nextest run -p uni-xervo --profile ci

# Expensive tests (run nightly or on release)
- name: Run integration tests
  env:
    EXPENSIVE_TESTS: "1"
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    VERTEX_AI_TOKEN: ${{ secrets.VERTEX_AI_TOKEN }}
    VERTEX_AI_PROJECT: ${{ secrets.VERTEX_AI_PROJECT }}
  run: cargo nextest run -p uni-xervo --test real_providers_test --profile ci
```

### Local Development Workflow

1. **During development**: Run fast mock tests
   ```bash
   cargo nextest run
   ```

2. **Before committing**: Run integration tests for changed providers
   ```bash
   EXPENSIVE_TESTS=1 cargo nextest run --test real_providers_test -E 'test(test_local_onnx_bge_small_embedding)'
   ```

3. **Before release**: Run full integration test suite
   ```bash
   export OPENAI_API_KEY="..."
   export GEMINI_API_KEY="..."
   export VERTEX_AI_TOKEN="..."
   export VERTEX_AI_PROJECT="your-gcp-project-id"
   EXPENSIVE_TESTS=1 cargo nextest run --test real_providers_test
   ```

---

## Troubleshooting

### "Skipping test - set EXPENSIVE_TESTS=1 to run"
The integration test is gated behind the `EXPENSIVE_TESTS` environment variable. Set it to run:
```bash
EXPENSIVE_TESTS=1 cargo nextest run ...
```

### "Skipping - provider-X feature not enabled"
Enable the required feature:
```bash
cargo nextest run -p uni-xervo --features provider-onnx
```

### "Skipping - API_KEY not set"
Export the required API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Model download failures
- Check your internet connection
- Verify you have disk space (models can be 100MB+)
- Models are cached in `~/.cache/huggingface/` (Linux) or `~/Library/Caches/` (macOS)

---

## Performance Expectations

| Test Type | Duration | Disk Usage | Network | RAM |
|-----------|----------|------------|---------|-----|
| Mock tests | ~0.1 seconds | 0 MB | None | <100 MB |
| ONNX Embed (first run) | ~30 seconds | ~130 MB | Yes | ~500 MB |
| Candle (first run) | ~30 seconds | ~90 MB | Yes | ~500 MB |
| OpenAI API | ~1 second | 0 MB | Yes | <100 MB |
| Gemini API | ~1 second | 0 MB | Yes | <100 MB |

*First run includes model download time. Subsequent runs are much faster due to caching.*
