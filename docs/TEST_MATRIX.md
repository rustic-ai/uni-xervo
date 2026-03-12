# Test Matrix

Comprehensive test results for uni-xervo local backends. Remote provider tests (OpenAI, Anthropic, etc.) are excluded — they require API keys.

**Last run**: 2026-03-12 | **Platform**: Linux x86_64 (CPU-only, no GPU) | **Rust**: stable

---

## Summary

| Category | Passed | Failed | Skipped | Total |
|----------|--------|--------|---------|-------|
| Regular (non-ignored) | 174 | 0 | 39 | 213 |
| Expensive — fastembed | 2 | 0 | 0 | 2 |
| Expensive — candle | 4 | 0 | 0 | 4 |
| Expensive — mistralrs (small) | 11 | 1 | 0 | 12 |
| Expensive — mistralrs (heavy) | 0 | 0 | 3 | 3 |
| **Total** | **191** | **1** | **42** | **234** |

---

## Local Provider Matrix — Expensive Tests (Real Models)

These tests download real models from HuggingFace and run actual inference. Requires `EXPENSIVE_TESTS=1`.

### fastembed

| Interface | Model | Model Size | Test Name | Result | Time |
|-----------|-------|-----------|-----------|--------|------|
| embed | AllMiniLML6V2 | ~80 MB | `test_fastembed_local_embedding` | PASS | 0.3s |
| embed | BGESmallENV15 | ~130 MB | `test_fastembed_bge_small_embedding` | PASS | 0.4s |

### candle

| Interface | Model | Model Size | Test Name | Result | Time |
|-----------|-------|-----------|-----------|--------|------|
| embed | all-MiniLM-L6-v2 | ~80 MB | `test_candle_local_embedding` | PASS | 0.2s |
| embed | bge-small-en-v1.5 | ~130 MB | `test_candle_bge_small_embedding` | PASS | 0.3s |
| embed | bge-base-en-v1.5 | ~440 MB | `test_candle_bge_base_embedding` | PASS | 0.8s |
| embed | all-MiniLM-L6-v2 (via Runtime) | ~80 MB | `test_runtime_candle_embed` | PASS | 0.2s |

### mistralrs — Embedding

| Interface | Modality | Model | Model Size | Test Name | Result | Time |
|-----------|----------|-------|-----------|-----------|--------|------|
| embed | text | google/embeddinggemma-300m | ~1.2 GB | `test_mistralrs_local_embedding_gemma3` | PASS | 41s |
| embed | text | Qwen/Qwen3-Embedding-0.6B | ~1.2 GB | `test_mistralrs_local_embedding_qwen3` | PASS | 121s |
| embed | text | nomic-ai/nomic-embed-text-v1.5 | ~550 MB | `test_mistralrs_local_embedding_bert_arch_unsupported` | PASS (expected FAIL) | 0.03s |

> **Note**: nomic-embed-text-v1.5 uses NomicBertModel which is unsupported by mistralrs 0.7 embedding. The test verifies the correct error message is returned.

### mistralrs — Generation

| Interface | Modality | Model | Format | Model Size | Test Name | Result | Time |
|-----------|----------|-------|--------|-----------|-----------|--------|------|
| generate | text | Qwen/Qwen2.5-0.5B-Instruct | safetensors | ~1 GB | `test_mistralrs_local_generation` | PASS | 103s |
| generate | text | Qwen/Qwen3-0.6B | safetensors | ~1.2 GB | `test_mistralrs_local_generation_qwen3` | **FAIL** | 1878s |
| generate | text | HuggingFaceTB/SmolLM2-135M-Instruct | safetensors (F32) | ~270 MB | `test_mistralrs_local_generation_smollm2_f32` | PASS | 146s |
| generate | text | bartowski/SmolLM2-135M-Instruct-GGUF | GGUF (Q4_K_M) | ~80 MB | `test_mistralrs_local_generation_smollm2_gguf` | PASS | 28s |
| generate | text | bartowski/Qwen_Qwen3-0.6B-GGUF | GGUF (Q4_K_M) | ~400 MB | `test_mistralrs_local_generation_qwen3_gguf` | PASS | 217s |
| generate | text | HuggingFaceTB/SmolLM2-135M-Instruct | safetensors (F32) | ~270 MB | `test_mistralrs_text_generation_multimodal_fields` | PASS | 353s |
| generate | vision | google/gemma-3n-E2B-it | safetensors | ~9 GB | `test_mistralrs_gemma3n_object_detection` | PASS | 1259s |
| generate | text | google/gemma-3n-E2B-it | safetensors | ~9 GB | `test_mistralrs_gemma3n_text_generation` | PASS | 851s |

> **FAIL detail**: `test_mistralrs_local_generation_qwen3` — Qwen3-0.6B raw safetensors generated empty text after 31 min on CPU. The model loaded successfully but inference timed out. This is likely a CPU-only performance issue; the GGUF variant of the same model passes.

### mistralrs — Heavy Models (Skipped on CPU)

These tests require GPU or take prohibitively long on CPU. They were not run to completion.

| Interface | Modality | Model | Model Size | Test Name | Status | Notes |
|-----------|----------|-------|-----------|-----------|--------|-------|
| generate | image | black-forest-labs/FLUX.1-schnell | ~23 GB | `test_mistralrs_diffusion_generation` | SKIP | Model loads OK; inference impractical on CPU (60+ min, not completed) |
| generate | vision | Qwen/Qwen2-VL-2B-Instruct | ~4 GB | `test_mistralrs_vision_generation` | SKIP | Not run (CPU-only) |
| generate | audio | nari-labs/Dia-1.6B | ~3 GB | `test_mistralrs_speech_generation` | SKIP | Not run (CPU-only) |

---

## Regular Tests (Non-Ignored) — 174 Passed

These tests use mock providers and require no model downloads or API keys.

### By Test Suite

| Suite | Tests | Result |
|-------|-------|--------|
| api::tests | 10 | 10 PASS |
| cache::tests | 7 | 7 PASS |
| provider::mistralrs::tests | 11 | 11 PASS |
| reliability::tests | 2 | 2 PASS |
| runtime::tests | 3 | 3 PASS |
| api_validation_test | 18 | 18 PASS |
| deduplication_test | 6 | 6 PASS |
| embedding_model_test | 9 | 9 PASS |
| error_handling_test | 15 | 15 PASS |
| generator_model_test | 17 | 17 PASS |
| options_validation_mistralrs_test | 14 | 14 PASS |
| options_validation_test | 4 | 4 PASS |
| provider_capability_test | 8 | 8 PASS |
| real_providers_test (mock) | 2 | 2 PASS |
| reliability_test | 5 | 5 PASS |
| reranker_model_test | 8 | 8 PASS |
| runtime_lifecycle_test | 19 | 19 PASS |
| warmup_policy_test | 13 | 13 PASS |

---

## How to Run

### Regular tests (fast, no downloads)
```bash
cargo nextest run --features provider-candle,provider-fastembed,provider-mistralrs
```

### All expensive local tests (downloads models)
```bash
EXPENSIVE_TESTS=1 cargo nextest run \
  --features provider-candle,provider-fastembed,provider-mistralrs \
  --run-ignored all --no-capture --no-fail-fast \
  -E 'test(~fastembed) | test(~candle) | test(~mistralrs)'
```

### Expensive tests excluding heavy GPU models
```bash
EXPENSIVE_TESTS=1 cargo nextest run \
  --features provider-candle,provider-fastembed,provider-mistralrs \
  --run-ignored all --no-capture --no-fail-fast \
  -E '(test(~fastembed) | test(~candle) | test(~mistralrs)) & not test(~diffusion) & not test(~vision) & not test(~speech)'
```
