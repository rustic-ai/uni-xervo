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
| macOS arm64 (Apple Silicon) | `cpu` | 2026-11-19 | `a76262a` | 11 | 0 | 1 (whisper.cpp: no cmake) |
| macOS arm64 (Apple Silicon) | `coreml` | 2026-11-19 | `d837a30` + the accelerator check | 12 | 0 | 1 (whisper.cpp: no cmake) |
| Linux x86_64 + NVIDIA | `cuda` | not yet run | | | | |

The first `coreml` run, at `a76262a`, failed `multi-vector` (3 of 6) and
`ocr` (1 of 3); see [Known issues](#known-issues) for what was wrong and how
the accelerator check fixed it.

Both runs used Rust 1.100.0 and the `ort` 2.0.0-rc.15 ONNX Runtime binding,
with the models listed under [Models](#models).

## macOS arm64, `cpu`

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` (BGE small/large, Qwen3 embedding, rerankers, kniv v1 NLP, …) | pass | 11 / 0 / 0 | 4 s | 10.1 GiB |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 2 s | 0.9 GiB |
| `rerank` | pass | 2 / 0 / 0 | 2 s | 0.2 GiB |
| `sparse` (SPLADE v1/v2, BGE-M3 lexical) | pass | 4 / 0 / 0 | 2 s | 2.3 GiB |
| `multi-vector` (ColBERT family, BGE-M3 ColBERT) | pass | 6 / 0 / 0 | 2 s | 2.0 GiB |
| `hybrid-bge-m3` (widths, parity with per-task models) | pass | 3 / 0 / 0 | 5 s | 7.3 GiB |
| `ocr` (PP-OCRv5 detection + recognition) | pass | 3 / 0 / 0 | 2 s | 0.5 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 2 s | 0.9 GiB |
| `nlp-kniv-v7` (kniv-v7-tiny-en, model-agnostic tests) | pass | 2 / 0 / 0 | 4 s | 0.2 GiB |
| `ep-reporting` | pass | 7 / 0 / 0 | 0 s | 0.1 GiB |
| `ep-skip-dynamic` (real ORT 1.30 library) | pass | 2 / 0 / 0 | 1 s | 0.1 GiB |
| `whisper` | skipped: `whisper-rs-sys` needs cmake, not installed | | | |

## macOS arm64, `coreml`

`local/onnx` built with `gpu-metal`, so every session's default provider list
is `["coreml", "cpu"]`.

| Suite | Result | Tests passed / failed / skipped | Time | Peak RSS |
|---|---|---|---|---|
| `onnx-models` | pass | 11 / 0 / 0 | 27 s | 16.3 GiB |
| `onnx-hf-download` | pass | 2 / 0 / 0 | 15 s | 2.1 GiB |
| `rerank` | pass | 2 / 0 / 0 | 2 s | 0.7 GiB |
| `sparse` | pass | 4 / 0 / 0 | 7 s | 9.8 GiB |
| `multi-vector` (ModernBERT models fall back to CPU) | pass | 6 / 0 / 0 | 7 s | 12.0 GiB |
| `hybrid-bge-m3` | pass | 3 / 0 / 0 | 36 s | 17.5 GiB |
| `ocr` (detector on CoreML, recognizer on CPU) | pass | 3 / 0 / 0 | 8 s | 1.5 GiB |
| `pdf-to-ocr` | pass | 1 / 0 / 0 | 2 s | 0.9 GiB |
| `nlp-kniv-v7` | pass | 2 / 0 / 0 | 4 s | 0.2 GiB |
| `ep-reporting` (reports `["coreml", "cpu"]`) | pass | 9 / 0 / 0 | 1 s | 0.1 GiB |
| `ep-skip-dynamic` | pass | 2 / 0 / 0 | 1 s | 0.1 GiB |
| `coreml-inference` (embed and rerank on CoreML only) | pass | 2 / 0 / 0 | 2 s | 1.1 GiB |
| `whisper` | skipped: no cmake | | | |

Times exclude the build. Peak RSS is for the whole test binary, whose tests
run in parallel and each load their own models, so it is an upper bound on
any one model, not a per-model figure.

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
| BGE small and large, Qwen3, the rerankers, SPLADE, answerai ColBERT | 1e-7 to 6e-6 | stays on CoreML |
| BGE-M3 (dense, sparse, ColBERT) | 0 | stays on CoreML |
| PP-OCRv5 detector | 0.052 | stays on CoreML |
| PP-OCRv5 recognizer | 0.66 | falls back to CPU |
| ModernBERT ColBERT | probe errors | falls back to CPU |

With that split, OCR reads the fixture correctly at about 160 ms a page,
against 430 ms with both stages on CPU.

Switching CoreML to the ML Program format was tried and not adopted: neither
PP-OCR model builds on it, and BERT-family models (BGE small, the MiniLM
reranker, SPLADE) abort the whole process inside Metal (`'mps.matmul' op
contracting dimensions differ`). A crash cannot be caught or fallen back from.

## Models

The Hugging Face revisions both runs resolved (from the model cache). Set an
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

- **CUDA:** no NVIDIA host was available. Run `scripts/acceptance.sh cuda` on
  one; the `cuda-inference` suite then runs too.
- **whisper.cpp:** needs cmake to build (`brew install cmake`).
- **kniv v7 golden tests:** need the xlarge releases (see
  `tests/local_onnx_nlp_v7_golden_expensive_test.rs`). The matrix runs the v7
  suite's model-agnostic tests on the tiny release instead.
- **Document extraction on `local/onnx`:** there is no ONNX export of a target
  document model yet. `remote/llamacpp` serves it, and needs a running server.
- **Retrieval quality on real corpora:** that is a deployment concern. The
  suites check ranking on fixed fixtures only.
