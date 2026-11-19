# GPU setup

This guide covers what you need at run time to actually use a GPU
build of Uni-Xervo, and how to confirm the GPU is being exercised.
For build-time flags and `Cargo.toml` recipes see
[Installation](../getting-started/installation.md) and
[Feature Flags](../reference/feature-flags.md). For per-spec
placement (mixing GPU and CPU models in the same catalog) see the
[ONNX guide](onnx.md#mixing-gpu-and-cpu-placement-in-one-catalog).

## What runs where

Uni-Xervo has three GPU surfaces, all opt-in:

| Build axis | ORT execution provider | candle kernels |
| --- | --- | --- |
| default (no GPU feature) | CPU | CPU |
| `gpu-cuda` | CUDA EP, falling back to CPU | CUDA |
| `gpu-metal` (Apple only) | CoreML EP, falling back to CPU | Metal |

The default EP list when a `ModelAliasSpec` doesn't set
`execution_providers` is feature-aware: `[cuda, cpu]` under
`gpu-cuda`, `[coreml, cpu]` under `gpu-metal`, `[cpu]` otherwise. This
is set in `default_execution_providers()` in `src/provider/onnx_ep.rs`.

candle follows the same build feature: enabling `gpu-cuda` activates
its CUDA kernels, `gpu-metal` activates Metal. It has no per-spec
device option — it is not ORT-backed, so it doesn't take
`execution_providers` either, and the build features decide. A
`remote/llamacpp` alias places nothing: inference happens in the
`llama-server` process, so its device is a server flag
(`--n-gpu-layers` and friends).

## NVIDIA / Linux runtime requirements

The bundled ONNX Runtime (`ort 2.0.0-rc.15` via pyke, ONNX Runtime
1.31) ships one pre-built CUDA EP, built for **CUDA 13 / cuDNN 9**.
`libonnxruntime_providers_cuda.so` links:

- `libcublas.so.13`, `libcublasLt.so.13`
- `libcudart.so.13`, `libcurand.so.10`

and loads `libcudnn.so.9` at runtime, the first time a model needs it.
The dynamic loader resolves these by SONAME, so you need a CUDA 13
runtime plus a **cuDNN 9 built for CUDA 13**, both reachable on
`LD_LIBRARY_PATH`. A CUDA 12 install does not satisfy this. NVIDIA
ships cuDNN 9 once per CUDA major version; use the CUDA 13 build. There
is no CUDA 12 build of the bundled runtime to choose instead: `ort-sys`
rc.15 downloads only CUDA 13 artifacts. To run on CUDA 12, use
`provider-onnx-dynamic` with an ONNX Runtime built for CUDA 12 (see
[below](#provider-onnx-dynamic-byo-ort)).

Two recipes:

### System packages

Install the CUDA 13 toolkit and the cuDNN 9 package for CUDA 13 from
the NVIDIA repos (on Debian / Ubuntu, `libcudnn9-cuda-13`). Then:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

Adjust the path to wherever your distribution puts the CUDA 13
runtime and cuDNN. On Fedora / RHEL that is typically
`/usr/local/cuda-13.x/targets/x86_64-linux/lib`. The toolkit's
`/usr/local/cuda/lib64` holds cuBLAS and the runtime but not cuDNN,
which installs separately.

### Pip wheels (no root, easy in CI)

Pip-distributed NVIDIA wheels work just as well and don't need
sudo:

```sh
pip install nvidia-cudnn-cu13 nvidia-cublas nvidia-cuda-runtime

# The wheels are namespace packages (no __file__), so ask for __path__.
CUDNN_DIR=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
CUDA_DIR=$(python -c "import nvidia.cu13; print(nvidia.cu13.__path__[0])")/lib
export LD_LIBRARY_PATH=$CUDNN_DIR:$CUDA_DIR:$LD_LIBRARY_PATH
```

The CUDA 13 cuBLAS and runtime wheels install into `nvidia/cu13/lib`;
the cuDNN wheel into `nvidia/cudnn/lib`. With a CUDA 13 toolkit already
installed, the cuDNN wheel is the only part you need.

### What a missing library looks like

Nothing fails at startup. The CUDA libraries are opened when a session
is created, so a missing CUDA 13 runtime makes the CUDA EP fail to
register, and a missing cuDNN fails only the models with nodes that need
it (attention, convolution). In both cases the load-time check moves the
model to CPU and inference keeps working. The first sign is ONNX
Runtime's own log line, for example:

```text
[E:onnxruntime:...] cuDNN is unavailable or disabled for CUDA Execution Provider:
dlopen failed for libcudnn.so: libcudnn.so: cannot open shared object file
```

See [Verifying the GPU EP actually loaded](#verifying-the-gpu-ep-actually-loaded)
below.

### Driver compatibility

CUDA 13 needs an R580 or later NVIDIA driver (580.65.06 for CUDA 13.0
on Linux x86_64). `nvidia-smi` shows the driver version at the top.

### GPU memory

A model whose session cannot get GPU memory at creation also falls back
to CPU, through the same path. Loading several large models at once (for
example from tests running in parallel) can exhaust an 8 GB card, where
each loads fine alone.

## Apple / macOS runtime requirements

Nothing extra. The CoreML execution provider ships inside the
bundled ORT and the candle Metal kernels are
self-contained. macOS 13 (Ventura) or later is recommended for
current CoreML feature coverage.

### Accelerator output is checked at load

Some models run wrongly on CoreML with ORT's default NeuralNetwork format.
The acceptance matrix (`scripts/acceptance.sh coreml`, results in
`docs/TEST_MATRIX.md`) found two: PP-OCR recognition returns wrong text
with no error, and the ModernBERT ColBERT models fail on real input. These
defects come from CoreML's model conversion, not from the GPU: they occur
on CoreML's CPU compute unit too.

You don't need a per-model list for this. When a `local/onnx` session
attaches an accelerator, the load runs a fixed probe input through it and
through a CPU session of the same model, and keeps the accelerator only if
the outputs agree. A session that disagrees falls back to CPU and logs a
`warn` with the measured error. For OCR, each stage is checked on its own:
on CoreML the detector stays accelerated and the recognizer moves to CPU,
which reads correctly at about 160 ms a page against 430 ms on CPU alone.
When CoreML or ONNX Runtime fixes a defect, the check passes and the model
moves back to the accelerator with no configuration change.

`accelerator_check` controls it per alias: `"compare"` (the default),
`"probe"` (only check that the probe runs) or `"off"`. See the
[ONNX provider reference](../reference/providers/onnx.md#common-keys-all-tasks).
Because the alias reports the providers that every one of its sessions
uses, an OCR alias whose recognizer fell back reports `["cpu"]` even though
its detector runs on CoreML.

## Verifying the GPU EP actually loaded

ORT's CUDA EP is built with `fail_silently()`, so a failed CUDA init
falls through to the next EP in the list (typically CPU) **without**
an error or panic. Inference still works — just slower. There is no
in-band signal unless you go look for it. Three checks, in order of
intrusiveness:

### 1. Programmatic — `active_execution_providers()`

`active_execution_providers()` lives on the `ModelInfo` supertrait, so
every typed handle (`RerankerModel`, `RawTensorModel`, `EmbeddingModel`, …)
exposes it. For a `local/onnx` model it returns the EPs the model runs on
after loading, in priority order (empty for remote / non-ONNX models):

```rust
let model = runtime.reranker("rerank/bge").await?;
println!("active EPs: {:?}", model.active_execution_providers());
// ["cuda", "cpu"] when CUDA took the model; ["cpu"] when it fell back.
```

It reflects every way a model ends up on CPU:

- the binary lacks `gpu-cuda`, so the `cuda` entry is dropped at build time;
- the CUDA EP failed to register (no device, driver or CUDA 13 runtime) and
  was skipped;
- the session could not be created on CUDA (GPU memory exhausted, say) and
  was rebuilt on CPU;
- the load-time check failed or disagreed with CPU (a missing cuDNN fails
  attention nodes here) and moved the model to CPU.

So the first entry being `cuda` is the check to make. Two limits: a
`raw` or `nlp` model is not probed at load, so a CUDA failure that only
shows at inference leaves it reporting `cuda`; and an OCR alias reports the
providers common to its two stages. Checks 2 and 3 confirm from outside.

`scripts/acceptance.sh cuda` and `coreml` make this check after every load
(`tests/common/accelerator.rs`), so a run where a model fell back fails.

Source: `src/traits.rs` (the `ModelInfo` supertrait) and
`src/provider/onnx_ep.rs` (`register_execution_providers`,
`verify_execution_providers`).

### 2. Out-of-band — `nvidia-smi`

While inference is running, watch GPU memory:

```sh
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1
```

A loaded ORT CUDA model uses at least ~150 MiB of VRAM for the
session (more for larger models — the BGE + Qwen3 expensive test
suite peaks at ~2.3 GiB). CPU fallback shows the idle baseline,
typically <50 MiB. If memory never moves above baseline during
inference, the GPU EP is not engaged.

### 3. ORT logs

Bumping the log level surfaces EP registration failures:

```sh
RUST_LOG=ort=warn cargo run …
```

`info` shows successful EP registration; `warn` keeps the noise low
but still surfaces init failures.

## `provider-onnx-dynamic` (BYO ORT)

The recipes above apply to the **bundled** ORT build (`provider-onnx`
+ `gpu-cuda`). Under `provider-onnx-dynamic` the user supplies the
ORT shared library at deploy time via `ORT_DYLIB_PATH` and is
responsible for the entire ORT runtime, including its CUDA / cuDNN
dependencies. See
[`docs/migrations/0.9.0-feature-surface.md`](https://github.com/rustic-ai/uni-xervo/blob/main/docs/migrations/0.9.0-feature-surface.md)
for the BYO setup.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Inference works but VRAM stays at idle baseline | The model fell back to CPU | `active_execution_providers()` shows `["cpu"]`; ONNX Runtime's `E:onnxruntime` log lines give the reason. Install the CUDA 13 runtime and cuDNN 9 for CUDA 13 and add them to `LD_LIBRARY_PATH`. |
| `E:onnxruntime` log line `cuDNN is unavailable ... dlopen failed for libcudnn.so`; attention and convolution models report `["cpu"]` | cuDNN 9 for CUDA 13 not on `LD_LIBRARY_PATH` | Run one of the install recipes above. |
| The model reports `["cpu"]`; with a CPU-free list (`["cuda"]`) the load fails with `execution provider "cuda" failed to register ... Failed to load library .../libonnxruntime_providers_cuda.so` | CUDA 13 runtime (`libcublas.so.13`, `libcudart.so.13`) not found, for example only CUDA 12 installed | Install the CUDA 13 toolkit, or the `nvidia-cublas` and `nvidia-cuda-runtime` wheels. |
| `E:onnxruntime` log line `Failed to allocate memory` at session creation | GPU memory exhausted, often by several models loading at once | Load fewer models concurrently, or use a card with more memory. |
| Build error mentioning `nvcc` or `CUDA_COMPUTE_CAP` | Build-time CUDA toolkit missing | Install the CUDA toolkit. Set `CUDA_COMPUTE_CAP=89` (Ada / RTX 40-series), `86` (Ampere), `80` (A100), or whichever matches your GPU. |
| `build.rs` panics on a non-Apple host with `gpu-metal` enabled | `gpu-metal` only builds on Apple targets | Remove the feature, or build on macOS / iOS. |
