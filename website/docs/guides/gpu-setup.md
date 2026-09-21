# GPU setup

This guide covers what you need at run time to actually use a GPU
build of Uni-Xervo, and how to confirm the GPU is being exercised.
For build-time flags and `Cargo.toml` recipes see
[Installation](../getting-started/installation.md) and
[Feature Flags](../reference/feature-flags.md). For per-spec
placement (mixing GPU and CPU models in the same catalog) see the
[ONNX guide](onnx.md#mixing-gpu-and-cpu-placement-in-one-catalog) and
the [mistralrs device-placement reference](../reference/providers/mistralrs.md#device-placement).

## What runs where

Uni-Xervo has three GPU surfaces, all opt-in:

| Build axis | ORT execution provider | candle / mistralrs kernels |
| --- | --- | --- |
| default (no GPU feature) | CPU | CPU |
| `gpu-cuda` | CUDA EP, falling back to CPU | CUDA |
| `gpu-metal` (Apple only) | CoreML EP, falling back to CPU | Metal |

The default EP list when a `ModelAliasSpec` doesn't set
`execution_providers` is feature-aware: `[cuda, cpu]` under
`gpu-cuda`, `[coreml, cpu]` under `gpu-metal`, `[cpu]` otherwise. This
is set in `default_execution_providers()` in `src/provider/onnx_ep.rs`.

candle and mistralrs follow the same build feature: enabling
`gpu-cuda` activates their CUDA kernels, `gpu-metal` activates Metal.
For `local/mistralrs` the per-spec switch is
[`force_cpu`](../reference/providers/mistralrs.md#device-placement)
(it's candle-backed, not ORT-backed, so it doesn't take
`execution_providers`).

## NVIDIA / Linux runtime requirements

The bundled ONNX Runtime (`ort 2.0.0-rc.12` via pyke) ships pre-built
CUDA EP binaries that link against the **CUDA 12 / cuDNN 9** ABI:

- `libcudnn.so.9`
- `libcublas.so.12`, `libcublasLt.so.12`
- `libcudart.so.12`, `libnvrtc.so.12`

The dynamic loader resolves these by SONAME, so you need *any* CUDA
12 install plus cuDNN 9 reachable on `LD_LIBRARY_PATH`. Two recipes:

### System packages

Install the CUDA 12 toolkit (any 12.x — 12.4, 12.6, 12.8 all work)
and cuDNN 9 from the NVIDIA repos. Then:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

Adjust the path to wherever your distribution puts the CUDA 12
runtime. On Fedora / RHEL the directory is typically
`/usr/local/cuda-12.x/targets/x86_64-linux/lib`; on Debian / Ubuntu
it's often `/usr/lib/x86_64-linux-gnu/` once the `nvidia-cuda-toolkit`
and `libcudnn9-cuda-12` packages are installed.

### Pip wheels (no root, easy in CI)

Pip-distributed NVIDIA wheels work just as well and don't need
sudo:

```sh
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

CUDNN_DIR=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib
CUBLAS_DIR=$(python -c "import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))")/lib
CUDART_DIR=$(python -c "import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))")/lib

export LD_LIBRARY_PATH=$CUDNN_DIR:$CUBLAS_DIR:$CUDART_DIR:$LD_LIBRARY_PATH
```

This is also the easiest way to pin a specific CUDA 12 toolkit
inside a container or venv without touching `/usr/local`.

### About newer CUDA toolkits on the system

A CUDA 13 install in `/usr/local/cuda` (and on the default
`ldconfig` search path) is harmless on its own — `libcublas.so.13`
has a different SONAME and won't be picked up when ORT asks for
`libcublas.so.12`. The failure mode is the *absence* of CUDA 12
libs, not the presence of CUDA 13 ones. The symptom is a silent
fallback to CPU; see [Verifying the GPU EP actually loaded](#verifying-the-gpu-ep-actually-loaded)
below.

### Driver compatibility

The kernel-mode NVIDIA driver only needs to be new enough for the
CUDA 12 user-mode runtime — driver 525 or later. `nvidia-smi` shows
the driver version at the top.

## Apple / macOS runtime requirements

Nothing extra. The CoreML execution provider ships inside the
bundled ORT and the candle / mistralrs Metal kernels are
self-contained. macOS 13 (Ventura) or later is recommended for
current CoreML feature coverage.

## Verifying the GPU EP actually loaded

ORT's CUDA EP is built with `fail_silently()`, so a failed CUDA init
falls through to the next EP in the list (typically CPU) **without**
an error or panic. Inference still works — just slower. There is no
in-band signal unless you go look for it. Three checks, in order of
intrusiveness:

### 1. Programmatic — `active_execution_providers()`

`active_execution_providers()` lives on the `ModelInfo` supertrait, so
every typed handle (`RerankerModel`, `RawTensorModel`, `EmbeddingModel`, …)
exposes it. It returns the EPs the ONNX session was *built with*, in priority
order (empty for remote / non-ONNX models):

```rust
let model = runtime.reranker("rerank/bge").await?;
println!("active EPs: {:?}", model.active_execution_providers());
// ["cuda", "cpu"] on a gpu-cuda build; ["cpu"] without it, because an
// unbacked `cuda` entry is dropped at build time rather than rejected.
```

This is the requested list minus anything this binary cannot construct, so it
does catch build *misconfiguration*: if you asked for `["cuda", "cpu"]` and
see `["cpu"]`, the binary lacks `gpu-cuda`.

It still is not what ORT finally **attached**. An EP listed here can fail to
register at runtime — no device, no driver — and be skipped silently, so a
CUDA init that fails on a `gpu-cuda` build still shows `["cuda", "cpu"]`.
Use checks 2 and 3 to prove the GPU EP actually loaded.

Source: `src/traits.rs` (the `ModelInfo` supertrait); the `local/onnx`
override is in `src/provider/local_onnx/rerank.rs:202`.

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
| Inference works but VRAM stays at idle baseline | CUDA EP failed to register; ORT silently fell back to CPU | Watch `nvidia-smi` or `RUST_LOG=ort=warn` (the EP-registration signal — `active_execution_providers()` only reports the *requested* list). Install cuDNN 9 + CUDA 12 runtime and add it to `LD_LIBRARY_PATH`. |
| `error while loading shared libraries: libcudnn.so.9` at process start | cuDNN 9 not on `LD_LIBRARY_PATH` | Run one of the install recipes above. |
| `error while loading shared libraries: libcublas.so.12` | CUDA 12 runtime missing (only CUDA 13+ installed) | Install `nvidia-cuda-runtime-cu12` (pip) or the CUDA 12 toolkit (system). |
| Build error mentioning `nvcc` or `CUDA_COMPUTE_CAP` | Build-time CUDA toolkit missing | Install the CUDA toolkit. Set `CUDA_COMPUTE_CAP=89` (Ada / RTX 40-series), `86` (Ampere), `80` (A100), or whichever matches your GPU. |
| `build.rs` panics on a non-Apple host with `gpu-metal` enabled | `gpu-metal` only builds on Apple targets | Remove the feature, or build on macOS / iOS. |
