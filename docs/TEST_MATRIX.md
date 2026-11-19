# Test Matrix

Results of the deployment acceptance matrix, `scripts/acceptance.sh`, which
runs every opt-in real-model suite for the local backends on one platform
target. Remote providers need API keys and are not part of it (see
`scripts/test-integration.sh`). How to run it, and what each suite covers,
is in [TESTING.md](../TESTING.md#deployment-acceptance-matrix).

Each run below is a record of one machine on one day, not a live status. To
add a platform, run `scripts/acceptance.sh <target>` there and copy its
`target/acceptance/<target>/report.md` in.

## Summary

| Platform | Target | Date | Commit | Suites passed | Failed | Skipped |
|---|---|---|---|---|---|---|
| macOS arm64 (Apple M2 Max) | `cpu` | 2026-11-19 | `0313805` + the fixes below | 13 | 0 | 1 (whisper.cpp: no cmake) |
| macOS arm64 (Apple M2 Max) | `coreml` | 2026-11-19 | `0313805` + the fixes below | 14 | 0 | 1 (whisper.cpp: no cmake) |
| Linux x86_64 (Intel Core Ultra 9 185H) | `cpu` | 2026-11-19 | `c371778` + the fixes below | 14 | 0 | 0 |
| Linux x86_64 + NVIDIA RTX 4070 Laptop (8 GB) | `cuda` | 2026-11-19 | `c371778` + the fixes below | 15 | 0 | 0 |
| Windows Server 2022 x86_64 (Azure D16s_v5, Xeon Platinum 8370C) | `cpu` | 2026-11-19 | `b793542` | 14 | 0 | 0 |

At `0313805` the Mac runs failed the kniv v7 UltraBERT goldens on both
targets and `onnx-models` on `coreml`; the suites marked † were rerun after
the fixes under [macOS arm64](#macos-arm64) in Known issues. The first
`coreml` run, at `a76262a`, failed `multi-vector` (3 of 6) and `ocr` (1 of
3); the accelerator check fixed both.

The macOS runs used Rust 1.100.0 and macOS's own `/bin/bash` 3.2.57 for
`scripts/acceptance.sh`; the Linux runs Rust 1.99.1 and bash 5. All used the
`ort` 2.0.0-rc.15 ONNX Runtime binding, with the models listed under
[Models](#models), and `ORT_DYLIB_PATH` naming the ONNX Runtime 1.31.1
wheel's library (`libonnxruntime.1.31.1.dylib` on macOS).

The first Linux `cuda` runs passed every suite while several models ran on
CPU; see [Linux / CUDA](#linux--cuda-gpu-cuda) under Known issues for why,
and the matrix changes that make that a failure.

## macOS arm64, `cpu`

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` | pass | 11 / 0 / 0 | 6 s | 9.7 GiB |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 2 s | 0.9 GiB |
| `rerank` | pass | 2 / 0 / 0 | 2 s | 0.3 GiB |
| `sparse` | pass | 4 / 0 / 0 | 2 s | 2.3 GiB |
| `multi-vector` | pass | 6 / 0 / 0 | 2 s | 2.3 GiB |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 7 s | 3.3 GiB |
| `ocr` | pass | 3 / 0 / 0 | 2 s | 0.5 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 3 s | 0.9 GiB |
| `nlp-kniv-v7` † (UltraBERT darwin-arm64 goldens, tolerant: 0 of 844 differ) | pass | 4 / 0 / 0 | 14 s | |
| `nlp-kniv-v7-deberta` † (DeBERTa darwin-arm64 goldens, tolerant: 0 of 810 differ) | pass | 4 / 0 / 0 | 17 s | |
| `nlp-kniv-v7-exact` † (on the goldens' own library: 0 of 844 differ) | pass | 1 / 0 / 0 | 12 s | |
| `ep-reporting` | pass | 7 / 0 / 0 | 0 s | 0.1 GiB |
| `ep-skip-dynamic` (ONNX Runtime 1.31.1 wheel library) | pass | 2 / 0 / 0 | 0 s | 0.1 GiB |
| `whisper` | skipped: cmake not on PATH | | | |

## macOS arm64, `coreml`

`local/onnx` built with `gpu-metal`, so every session's default provider list
is `["coreml", "cpu"]`. Of the 37 model loads in the accelerator-checked
suites, 31 reported `["coreml", "cpu"]` and 6 `["cpu"]`: the OCR alias and
the three ModernBERT ColBERT models and the two Qwen3 models, all of which
are allowed (see [Known issues](#macos-arm64)).

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` † (Qwen3 embedding and reranker on CPU, allowed) | pass | 11 / 0 / 0 | 42 s | |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 3 s | 1.4 GiB |
| `rerank` | pass | 2 / 0 / 0 | 3 s | 0.6 GiB |
| `sparse` | pass | 4 / 0 / 0 | 13 s | 10.4 GiB |
| `multi-vector` (ModernBERT models on CPU, allowed) | pass | 6 / 0 / 0 | 15 s | 10.4 GiB |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 38 s | 13.9 GiB |
| `ocr` (detector on CoreML, recognizer on CPU, allowed) | pass | 3 / 0 / 0 | 8 s | 1.9 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 1 s | 0.9 GiB |
| `nlp-kniv-v7` † (0 of 844 differ) | pass | 4 / 0 / 0 | 17 s | |
| `nlp-kniv-v7-deberta` † (0 of 810 differ) | pass | 4 / 0 / 0 | 26 s | |
| `nlp-kniv-v7-exact` † (exact: 0 of 844 differ) | pass | 1 / 0 / 0 | 12 s | |
| `ep-reporting` (reports `["coreml", "cpu"]`) | pass | 9 / 0 / 0 | 1 s | 0.1 GiB |
| `ep-skip-dynamic` | pass | 2 / 0 / 0 | 0 s | 0.1 GiB |
| `coreml-inference` (embed and rerank on CoreML only) | pass | 2 / 0 / 0 | 3 s | 0.8 GiB |
| `whisper` | skipped: cmake not on PATH | | | |

† Rerun alone with `--only` after the fixes; `--only` runs report no peak
RSS here. Times exclude the build. On `cpu`, peak RSS is for the whole test binary,
whose tests run in parallel, so it is an upper bound on any one model; on
`coreml` the tests run one at a time, which is why its times are longer.

## Windows x86_64, `cpu`

Windows Server 2022 Datacenter Azure Edition on an Azure `Standard_D16s_v5`
(16 vCPU, 64 GB), with the MSVC 2022 Build Tools, Rust 1.100.0, cmake 4.5.1
and LLVM 23.1.6 (libclang for whisper.cpp's bindings). `scripts/acceptance.sh`
ran under Git for Windows' bash 5.3.15, which has no `/usr/bin/time`, so peak
RSS is not measured. `ORT_DYLIB_PATH` named the ONNX Runtime 1.31.1 wheel's
`onnxruntime.dll`.

| Suite | Result | Tests passed / failed / skipped | Time |
|---|---|---|---|
| `onnx-models` | pass | 11 / 0 / 0 | 132 s |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 42 s |
| `rerank` | pass | 2 / 0 / 0 | 33 s |
| `sparse` | pass | 4 / 0 / 0 | 42 s |
| `multi-vector` | pass | 6 / 0 / 0 | 49 s |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 48 s |
| `ocr` | pass | 3 / 0 / 0 | 35 s |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 32 s |
| `nlp-kniv-v7` (UltraBERT linux-x86_64 goldens, tolerant: 5 of 834 differ, 0.60%) | pass | 4 / 0 / 0 | 80 s |
| `nlp-kniv-v7-deberta` (DeBERTa, tolerant: 3 of 745 differ, 0.40%) | pass | 4 / 0 / 0 | 116 s |
| `nlp-kniv-v7-exact` (tolerant off linux-x86_64: 6 of 834 differ, 0.72%) | pass | 1 / 0 / 0 | 15 s |
| `ep-reporting` | pass | 7 / 0 / 0 | 30 s |
| `ep-skip-dynamic` (ONNX Runtime 1.31.1 wheel's `onnxruntime.dll`) | pass | 2 / 0 / 0 | 1 s |
| `whisper` | pass | 1 / 0 / 0 | 7 s |

Times include part of each suite's compile on a cold cache, so they are
longer than the other platforms'. The per-commit gate (`scripts/test.sh`)
passed 971 tests after the two fixes under
[Windows](#windows-x86_64) in Known issues; the publish dry-run passed.

## Linux x86_64, `cpu`

Run with `ORT_DYLIB_PATH` naming the ONNX Runtime 1.31.1 wheel's library,
which enables `ep-skip-dynamic` and `nlp-kniv-v7-exact`.

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` | pass | 11 / 0 / 0 | 5 s | 9.4 GiB |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 1 s | 0.9 GiB |
| `rerank` | pass | 2 / 0 / 0 | 1 s | 0.2 GiB |
| `sparse` | pass | 4 / 0 / 0 | 3 s | 2.6 GiB |
| `multi-vector` | pass | 6 / 0 / 0 | 2 s | 2.2 GiB |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 6 s | 4.0 GiB |
| `ocr` | pass | 3 / 0 / 0 | 7 s | 0.8 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 9 s | 0.6 GiB |
| `nlp-kniv-v7` (UltraBERT goldens, tolerant: 0.24% differ) | pass | 4 / 0 / 0 | 18 s | 4.0 GiB |
| `nlp-kniv-v7-deberta` (DeBERTa goldens, tolerant: 0.54% differ) | pass | 4 / 0 / 0 | 33 s | 8.9 GiB |
| `nlp-kniv-v7-exact` (UltraBERT goldens on their own library: 0 differ) | pass | 1 / 0 / 0 | 12 s | 2.4 GiB |
| `ep-reporting` | pass | 7 / 0 / 0 | 0 s | 0.1 GiB |
| `ep-skip-dynamic` (ONNX Runtime 1.31.1 wheel library) | pass | 2 / 0 / 0 | 0 s | 0.1 GiB |
| `whisper` | pass | 1 / 0 / 0 | 1 s | 0.2 GiB |

## Linux x86_64, `cuda`

`local/onnx` built with `gpu-cuda` (CUDA 13.0 toolkit, driver 620.70, cuDNN
9.19 for CUDA 13 from the `nvidia-cudnn-cu13` wheel on `LD_LIBRARY_PATH`), so
every session's default provider list is `["cuda", "cpu"]`. All 37 model
loads in the accelerator-checked suites reported `["cuda", "cpu"]`; the kniv
v7 suites pin `["cpu"]` because their goldens are CPU results.

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` | pass | 11 / 0 / 0 | 24 s | 3.5 GiB |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 2 s | 1.0 GiB |
| `rerank` | pass | 2 / 0 / 0 | 2 s | 0.8 GiB |
| `sparse` | pass | 4 / 0 / 0 | 7 s | 2.2 GiB |
| `multi-vector` | pass | 6 / 0 / 0 | 7 s | 2.1 GiB |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 12 s | 2.5 GiB |
| `ocr` (both stages on CUDA) | pass | 3 / 0 / 0 | 9 s | 1.3 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 9 s | 0.6 GiB |
| `nlp-kniv-v7` | pass | 4 / 0 / 0 | 19 s | 2.2 GiB |
| `nlp-kniv-v7-deberta` | pass | 4 / 0 / 0 | 39 s | 4.5 GiB |
| `nlp-kniv-v7-exact` | pass | 1 / 0 / 0 | 11 s | 2.4 GiB |
| `ep-reporting` (reports `["cuda", "cpu"]`) | pass | 7 / 0 / 0 | 0 s | 0.5 GiB |
| `ep-skip-dynamic` | pass | 2 / 0 / 0 | 0 s | 0.1 GiB |
| `cuda-inference` (embed and rerank on CUDA only) | pass | 2 / 0 / 0 | 2 s | 0.8 GiB |
| `whisper` | pass | 1 / 0 / 0 | 1 s | 0.2 GiB |

Accelerator targets run tests one at a time, so these times are serial; the
CPU target's run in parallel.

## Known issues

### CoreML (`gpu-metal`)

The first `coreml` run (commit `a76262a`) found two CoreML defects with
ORT's default CoreML model format (NeuralNetwork). The same models pass on
CPU, and both defects also occur on CoreML's CPU compute unit, so they come
from CoreML's model conversion, not the GPU. Seen on an M2 Max, macOS 27.2.1.

- **PP-OCR recognition read text wrong, without any error.** On the
  `maintenance_log.png` fixture, 3 of 4 lines came back garbled, for example
  `"PUm&UOEC(g"` for "Pump Station Maintenance Log".
- **ModernBERT ColBERT models failed at inference.** `lightonai/GTE-ModernColBERT-v1`
  and `mixedbread-ai/mxbai-edge-colbert-v0-17m` and `-32m` attached CoreML,
  passed a one-word load-time probe, then failed on real input with "Unable
  to compute the prediction using a neural network model".

The load-time accelerator check (`accelerator_check`, on by default) now
handles both with no per-model configuration. It runs a probe through the
accelerated session and a CPU session and keeps the accelerator only when
they agree:

- the OCR recognizer disagrees and moves to CPU, while the detector stays on
  CoreML;
- the ModernBERT probe errors on CoreML, so those aliases load on CPU.

Relative L2 error between CoreML and CPU on the probe, against the 0.2
tolerance:

| Model | Relative error | Result |
|---|---|---|
| BGE small and large, the BERT rerankers, SPLADE, answerai ColBERT | 1e-7 to 6e-6 | stays on CoreML |
| BGE-M3 (dense, sparse, ColBERT) | 0 | stays on CoreML |
| PP-OCRv5 detector | 0.052 | stays on CoreML |
| PP-OCRv5 recognizer | 0.66 | falls back to CPU |
| ModernBERT ColBERT | probe errors | falls back to CPU |
| Qwen3 0.6B embedding and reranker | probe errors (see below) | falls back to CPU |

With that split, OCR reads the fixture correctly at about 160 ms a page,
against 430 ms with both stages on CPU.

Switching CoreML to the ML Program format was tried and not adopted: neither
PP-OCR model builds on it, and BERT-family models (BGE small, the MiniLM
reranker, SPLADE) abort the whole process inside Metal (`'mps.matmul' op
contracting dimensions differ`). A crash cannot be caught or fallen back from.

### macOS arm64

Found by the `0313805` runs, the first with the accelerator assertions and
the kniv v7 golden suites on a Mac:

- **Qwen3 0.6B embedding and reranker run on CPU under `gpu-metal`.** Both
  fail the load-time probe on CoreML and fall back:

  ```
  WARN uni_xervo::provider::onnx_ep: ONNX model failed its load-time probe on the requested execution providers; falling back to CPU alias=embed/qwen3-06b failed=coreml error=... coreml_execution_provider.cc:226 operator() Input (past_key_values.0.key) has a dynamic shape ({-1,8,-1,128}) but the runtime shape ({2,8,0,128}) has zero elements. This is not supported by the CoreML EP.
  ```

  `rerank/qwen3-06b` gives the same error. The decoder is fed an empty
  (length-0) KV cache, which the CoreML provider rejects. The earlier
  `coreml` runs passed only because the matrix did not yet check where models
  ran. The model runs correctly on CPU, and the load-time check is what
  moves it there, so both aliases are on the CoreML allow list.
- **kniv v7 goldens differ by CPU architecture.** Against the linux-x86_64
  goldens, 22 of 830 UltraBERT decisions differed (2.65%, over the 2%
  budget) and 3 of 734 DeBERTa ones (0.41%), identically on the bundled
  build and on the 1.31.1 arm64 wheel, and on `cpu` and `coreml` (kniv v7
  pins `["cpu"]`). 18 of the 22 were fact margins; the rest near-tie
  decisions (a sentiment, an extra `ARGM-ADV` argument, an extra entity, an
  `ARG1`/`ARG2` label). No token or sentence differed. Since both builds
  agree with each other on arm64, the gap is the architecture's INT8
  kernels, so the Mac now has its own golden set, generated on this machine
  with the 1.31.1 wheel (`<tag>/darwin-arm64/`, chosen by the test on that
  platform). Against it, 0 decisions differ for either release, in tolerant
  mode on the bundled build and in exact mode on the wheel (UltraBERT 844,
  DeBERTa 810 decisions; fact margins within 2.7e-3).

### Windows x86_64

Found by the first Windows run, at `b793542`. The acceptance matrix passed;
the per-commit gate failed two unit tests, both fixed:

- **A rooted path without a drive letter was read as a Hugging Face repo.**
  On Windows `/tmp/model.onnx` is not absolute (it has no prefix), so the
  ONNX provider took it for a repo id. A rooted path is now always local.
- **The runtime verify test for an unreachable endpoint reported a timeout.**
  Windows takes about 2 s to refuse a connection to a closed local port (it
  retries the SYN), longer than the test's 1 s load timeout. The test now
  allows 10 s; the classification itself was right.

The kniv v7 goldens stay within budget on Windows x86_64 (0.60% and 0.40%),
so it needs no golden set of its own.

### Linux / CUDA (`gpu-cuda`)

Each of these let a `cuda` run pass while models ran on CPU, before the
matrix checked where models run:

- **cuDNN for CUDA 13 missing from `LD_LIBRARY_PATH`.** The bundled ONNX
  Runtime is a CUDA 13 build and loads cuDNN at run time. With only a CUDA 12
  cuDNN installed, every model with attention or convolution nodes failed its
  load-time probe and fell back. See the GPU setup guide.
- **GPU memory exhausted by parallel loads.** The tests of one suite loaded
  their models at once and exhausted the 8 GB card; each session that could
  not be created was rebuilt on CPU (BGE-M3, Qwen3). Run one at a time, none
  ran out.
- **Four BGE-M3 sessions in one runtime.** `hybrid_matches_per_task_paths`
  loaded the hybrid and three per-task BGE-M3 models together (2.2 GB of
  weights each); the third could not be created on CUDA. The test now loads
  each in its own runtime.

The kniv v7 goldens differ by ONNX Runtime build, not by platform alone: on
identical inputs the bundled (pyke) 1.31.0 build and Microsoft's 1.31.1 wheel
compute some INT8 shapes differently, both as close to the fp32 graph. Hence
the exact and tolerant modes described in
`crates/uni-xervo/tests/fixtures/kniv-v7/README.md`.

## Models

The Hugging Face revisions the runs resolved (from the model cache). Set an
alias's `revision` to one of these to reproduce a run.

| Repository | Commit | Snapshot size |
|---|---|---|
| `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | 1.1 GB |
| `Xenova/bge-large-en-v1.5` | `dfeef6070b90658e1b391a6940efdb0925c1de6f` | 1.2 GB |
| `Xenova/bge-small-en-v1.5` | `ea104dacec62c0de699686887e3f920caeb4f3e3` | 128 MB |
| `aapot/bge-m3-onnx` | `76a603396f5eb9f03ed51bbab8f4893fcea7b2fe` | 2.1 GB |
| `answerdotai/answerai-colbert-small-v1` | `934fa8bb4ce2284f4c2baa232d81aca4d076fa5e` | 128 MB |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 87 MB |
| `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` | `68af00aef58ef6dc4bb695612f2fd68d1f03fa27` | 294 MB |
| `dragonscale-ai/kniv-v7-tiny-en` | `c6335931a9c911faaa6939a6b765e0c17d260a81` | 119 MB |
| `dragonscale-ai/kniv-v7-xlarge-u-en` (pinned by the goldens) | `2733fe0d660f433b0e0d164e8f1f6ea58574954d` | 1.9 GB |
| `dragonscale-ai/kniv-v7-xlarge-d-en` (pinned by the goldens) | `5545dfe5641872dae072f1c4d091422e033d1613` | 3.5 GB |
| `lightonai/GTE-ModernColBERT-v1` | `25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639` | 573 MB |
| `mixedbread-ai/mxbai-edge-colbert-v0-17m` | `592c6417c1c6687572043408ed1ae5196bce16b1` | 68 MB |
| `mixedbread-ai/mxbai-edge-colbert-v0-32m` | `bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66` | 129 MB |
| `monkt/paddleocr-onnx` | `7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65` | 91 MB |
| `onnx-community/Qwen3-Embedding-0.6B-ONNX` | `c25a394dd583836952667c12f008335071b3f43d` | 2.2 GB |
| `onnx-community/Qwen3-Reranker-0.6B-ONNX` | `9995c50e2310679108a55f5ccd16ba8be9f17c20` | 960 MB |
| `prithivida/Splade_PP_en_v1` | `762be6a7206e2f299182705972a65e5c46e62be2` | 508 MB |
| `prithivida/Splade_PP_en_v2` | `f0d4aa214dcb60c274052a52c0497535e3aec64c` | 508 MB |
| `protectai/bert-base-NER-onnx` | `84d28ab2bc28401b4353b26033eb043c8c8a0739` | 412 MB |
| `smokxy/sequence_classification_onnx` | `ea7a6b194062bd82e8d92a909ce4e4543f203273` | 256 MB |

## Not covered

- **DirectML on Windows:** the matrix has no `directml` target, and the
  Azure subscription used had no GPU quota.
- **Exact kniv v7 goldens beyond linux-x86_64 and darwin-arm64:** only those
  two platforms have golden sets; elsewhere the comparison is tolerant
  against the linux-x86_64 set (Linux on arm64 may need its own).
- **whisper.cpp on macOS:** needs cmake to build (`brew install cmake`).
- **Document extraction on `local/onnx`:** there is no ONNX export of a target
  document model yet. `remote/llamacpp` serves it, and needs a running server.
- **Retrieval quality on real corpora:** that is a deployment concern. The
  suites check ranking on fixed fixtures only.
