# Feature Flags

Uni-Xervo's feature surface is small and orthogonal. Three independent axes:

1. **Providers** — which model backends to compile in.
2. **ORT linking** — bundled or BYO. Only matters if you use ONNX.
3. **GPU acceleration** — additive; off by default.

## Defaults

`uni-xervo = "0.18"` enables all three local backends and all nine remote providers on CPU:

```text
provider-candle, provider-mistralrs, provider-onnx,
provider-openai, provider-gemini, provider-vertexai, provider-mistral,
provider-anthropic, provider-voyageai, provider-cohere, provider-azure-openai,
provider-llamacpp
```

`provider-whisper-cpp` is **not** in defaults: it compiles whisper.cpp's
C/C++ source via CMake and needs a working C/C++ toolchain on the build
host, so it stays opt-in. Most users don't need to think about features.
Pass `default-features = false` when you want a leaner build.

## Providers

### Local

| Feature | Provider ID | Tasks |
| --- | --- | --- |
| `provider-candle` | `local/candle` | embed |
| `provider-mistralrs` | `local/mistralrs` | embed, generate (text, vision, diffusion, speech), **document_extract** (olmOCR-2 on the vision pipeline) |
| `provider-onnx` | `local/onnx` | raw, rerank, embed, **embed_sparse, embed_multi_vector, embed_image, nlp, ocr, document_extract** |
| `provider-onnx-dynamic` | `local/onnx` | same as `provider-onnx` (BYO ORT linking) |
| `provider-whisper-cpp` | `local/whisper-cpp` | transcribe (opt-in; needs CMake + C/C++ toolchain at build time) |

`provider-onnx` and `provider-onnx-dynamic` are mutually exclusive — pick one.

For `local/onnx`, `document_extract` ships as a scaffold today (catalog
wiring, options, and a reusable greedy autoreg decoder are
production-ready; the inference path returns `Unavailable` until an
upstream canonical ONNX export of a target VLM ships). All other tasks
on this provider are fully wired — including `ocr`, which gained an
optional DBNet detection stage for full-page detect→recognize. The live
`document_extract` path today is **olmOCR-2 on `provider-mistralrs`** (its
vision pipeline).

### Remote

| Feature | Provider ID | Tasks |
| --- | --- | --- |
| `provider-openai` | `remote/openai` | embed, generate |
| `provider-gemini` | `remote/gemini` | embed, generate, **embed_multimodal** |
| `provider-vertexai` | `remote/vertexai` | embed, generate |
| `provider-mistral` | `remote/mistral` | embed, generate |
| `provider-anthropic` | `remote/anthropic` | generate |
| `provider-voyageai` | `remote/voyageai` | embed, rerank |
| `provider-cohere` | `remote/cohere` | embed, rerank, generate, **embed_multimodal** |
| `provider-azure-openai` | `remote/azure-openai` | embed, generate |
| `provider-llamacpp` | `remote/llamacpp` | embed (llama.cpp `llama-server`, tokenizer-aware truncation) |

All remote providers share a single `reqwest` dependency, so enabling all of them costs roughly the same as enabling one. `provider-cohere` and `provider-gemini` additionally pull `base64` for encoding raw image / audio bytes into the multimodal embed request payload.

## ORT linking

Only matters if you use `provider-onnx*`.

| Feature | Behavior |
| --- | --- |
| `provider-onnx` | ONNX Runtime is statically linked into your binary at build time (pyke fetches the prebuilt). Self-contained, zero runtime configuration. Default. |
| `provider-onnx-dynamic` | ONNX Runtime is loaded at runtime via `dlopen`. You supply the shared library at deploy time and set `ORT_DYLIB_PATH`. Use for sandboxed CIs that can't reach the network at build time, custom ORT builds, or vendor-supplied builds (AMD ROCm, Intel OpenVINO, Microsoft DirectML, Qualcomm QNN). |

`build.rs` panics if both are enabled.

## GPU acceleration

CPU is always available — there is no "CPU feature." GPU features are purely additive: at runtime, ORT registers the GPU EP first and silently falls back to CPU per-op if the GPU path isn't available.

| Feature | Targets | What it activates |
| --- | --- | --- |
| `gpu-cuda` | Linux + Windows, NVIDIA | ORT CUDA EP (when ORT is in the build) + CUDA kernels for candle / mistralrs. Pyke fetches the CUDA-flavored ORT bundle automatically; sidecar EP libs are staged into `target/<profile>/` via `ort/copy-dylibs`. |
| `gpu-metal` | macOS + iOS | ORT CoreML EP (when ORT is in the build, hits Apple GPU + Neural Engine) + Metal kernels for candle / mistralrs. The CoreML EP is in pyke's standard macOS bundle — no extra fetch. |

Build host requirements:

- `gpu-cuda` needs `nvcc` and `CUDA_COMPUTE_CAP` set for candle / mistralrs PTX generation. ORT side is handled by pyke.
- `gpu-metal` needs an Apple build target — `build.rs` panics on non-Apple targets.

Don't enable both — they target different platforms. For other vendors (AMD ROCm, Intel OpenVINO, Microsoft DirectML, Qualcomm QNN, TensorRT, WebGPU), use `provider-onnx-dynamic` with a vendor-supplied ORT build at runtime via `ORT_DYLIB_PATH`, and request the EP per alias by setting `execution_providers` to e.g. `["rocm", "cpu"]`.

For NVIDIA / Linux runtime library setup (cuDNN, `LD_LIBRARY_PATH`) and how to verify the GPU EP actually loaded, see the [GPU setup guide](../guides/gpu-setup.md).

## Runtime EP selection (ORT only)

When a catalog spec doesn't set `execution_providers`, the default list is feature-aware:

| Compiled with… | Default EP list |
| --- | --- |
| `gpu-cuda` | `[Cuda, Cpu]` — CUDA preferred, CPU fallback |
| `gpu-metal` (no `gpu-cuda`) | `[CoreMl, Cpu]` — CoreML preferred, CPU fallback |
| neither | `[Cpu]` |

User-supplied `execution_providers` strings are checked in two stages:

- **Catalog options validation** (at runtime build/register time) accepts only `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`. Any other string — including `"rocm"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"` — is rejected here with a `RuntimeError::Config`.
- **Session build** (when the ORT session is created) additionally requires the right feature/runtime backing for the EP you asked for. The vendor EPs (`directml`, `rocm`, `openvino`, `qnn`, `tensorrt`, `webgpu`) require the `provider-onnx-dynamic` feature plus a vendor-supplied ONNX Runtime library via `ORT_DYLIB_PATH`; `cuda` requires `gpu-cuda` and `coreml` requires `gpu-metal`. Requesting one of these without its backing yields a clear `RuntimeError::Config`.

## Common build recipes

```toml
# Default — everything except GPU.
uni-xervo = "0.18"

# Add NVIDIA GPU (Linux / Windows).
uni-xervo = { version = "0.18", features = ["gpu-cuda"] }

# Add Apple GPU + Neural Engine (macOS / iOS).
uni-xervo = { version = "0.18", features = ["gpu-metal"] }

# Lean — only candle.
uni-xervo = { version = "0.18", default-features = false, features = ["provider-candle"] }

# Remote-only — no native deps at all.
uni-xervo = { version = "0.18", default-features = false, features = [
  "provider-openai",
  "provider-anthropic",
] }

# Local stack with ONNX Runtime.
uni-xervo = { version = "0.18", default-features = false, features = [
  "provider-candle",
  "provider-onnx",
] }

# BYO ONNX Runtime (ROCm, OpenVINO, custom builds, sandboxed CI).
uni-xervo = { version = "0.18", default-features = false, features = [
  "provider-candle",
  "provider-mistralrs",
  "provider-onnx-dynamic",
] }
# Then at runtime: ORT_DYLIB_PATH=/path/to/libonnxruntime.so ./your-binary

# Add local speech-to-text via whisper.cpp (opt-in).
uni-xervo = { version = "0.18", features = ["provider-whisper-cpp"] }
# Build host needs cmake + a C/C++ toolchain.
```

## Runtime registration reminder

Enabling features compiles provider code; it does not auto-register providers.

Register each provider in `ModelRuntime::builder()` before `build()`.
