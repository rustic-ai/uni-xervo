# ONNX Runtime Guide

`local/onnx` is Uni-Xervo's raw tensor runtime for ONNX models. This guide
covers the `raw` task. The same provider also serves higher-level tasks —
`embed`, `rerank`, [`embed_sparse`](sparse-embeddings.md),
[`embed_multi_vector`](multi-vector-embeddings.md),
[`embed_hybrid`](multi-vector-embeddings.md#bge-m3-three-heads-one-export),
`embed_image`, [`nlp`](nlp.md), and [`ocr`](ocr.md) — which interpret outputs
for you; see the
[task trait surface](multimodal-traits.md) for the full list.

Use it when you want Uni-Xervo to handle:

- alias-based model resolution,
- Hugging Face snapshot download and caching,
- ONNX Runtime session loading,
- tensor signature introspection,
- batch validation and splitting,
- timeout/retry/warmup wrappers.

Your application still owns preprocessing and postprocessing.

## What `local/onnx` is for

`local/onnx` is best for:

- custom numeric models,
- tabular regression/classification,
- HF transformer exports where you already control tokenization,
- NER / classification / embedding pipelines that need raw tensor access,
- any ONNX graph where app code should interpret outputs.

It is not a high-level task wrapper like Python `transformers`.

## Catalog shape

Use:

- `provider_id: "local/onnx"`
- `task: "raw"`

Example:

```json
{
  "alias": "raw/minilm",
  "task": "raw",
  "provider_id": "local/onnx",
  "model_id": "nixiesearch/all-MiniLM-L6-v2-onnx",
  "options": {
    "artifact": "model.onnx"
  }
}
```

## Local path vs Hugging Face repo

`model_id` can be either:

- a local filesystem path to an ONNX file, or
- a Hugging Face repo ID

For HF-backed aliases, Uni-Xervo snapshots the full repo revision into cache before loading the selected artifact. That means repo-local sidecars such as tokenizer files, config files, and external data files are available next to the ONNX model on disk.

If the repo contains multiple `.onnx` files, set `options.artifact`.

## Runtime API

Resolve raw ONNX handles with `runtime.raw_tensor_model(alias)`.

```rust
use ndarray::{arr2, ArrayD};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{TensorBatch, TensorValue};

let runner = runtime.raw_tensor_model("raw/minilm").await?;

let mut batch = TensorBatch::new();
batch.insert(
    "input_ids".to_string(),
    TensorValue::I64(arr2(&[[101_i64, 7592, 2088, 102]]).into_dyn()),
);
batch.insert(
    "attention_mask".to_string(),
    TensorValue::I64(arr2(&[[1_i64, 1, 1, 1]]).into_dyn()),
);

let outputs = runner.run(&batch).await?;
let logits = outputs.get("last_hidden_state").unwrap();
```

Useful APIs:

- `runtime.raw_tensor_model(alias)`
- `runner.input_signature()`
- `runner.output_signature()`
- `runner.max_batch_size()`
- `runner.run(&batch)`
- `runner.run_batch(&batches)`

## Tensor contract

Uni-Xervo does not interpret model semantics. It only validates names, dtypes, and shapes against the ONNX graph signature.

Core types:

- `TensorBatch`: ordered tensor map keyed by input/output name
- `TensorValue`: raw `ndarray::ArrayD<T>` over supported ONNX tensor dtypes
- `TensorSpec`: name, dtype, and dimension metadata

Supported `TensorValue` variants cover:

- `f32`
- `f64`
- `i32`
- `i64`
- `bool`
- `string`

## Provider options

`local/onnx` accepts these `options` keys:

- `artifact`
- `max_batch_size`
- `execution_providers`
- `graph_optimization_level`
- `inter_op_num_threads`
- `intra_op_num_threads`

Example:

```json
{
  "artifact": "model.onnx",
  "max_batch_size": 32,
  "execution_providers": ["cuda", "cpu"],
  "graph_optimization_level": "extended",
  "intra_op_num_threads": 4
}
```

## Execution providers

`execution_providers` is optional.

Defaults:

- CPU-only builds: `["cpu"]`
- `gpu-cuda` builds: ORT-backed providers prefer `["cuda", "cpu"]`
- `gpu-metal` builds: ORT-backed providers prefer `["coreml", "cpu"]`

Supported names:

- `cpu`
- `cuda`
- `coreml`

Vendor names available under `provider-onnx-dynamic`: `rocm`, `directml`,
`openvino`, `qnn`, `tensorrt`, `webgpu` (each requires a matching ORT library at
`ORT_DYLIB_PATH`).

### Mixing GPU and CPU placement in one catalog

`execution_providers` is set per `ModelAliasSpec`, so a single
`ModelRuntime` can run some models on GPU and others on CPU even when
the binary was built with `gpu-cuda` (or `gpu-metal`). The placement
choice consumes the build's GPU capability — it does not require a
different binary.

```rust
ModelRuntime::builder()
    .register_provider(LocalOnnxProvider::new())
    .catalog(vec![
        // Cheap embedder: keep on CPU to leave VRAM for the reranker.
        ModelAliasSpec {
            alias: "embed/bge-small".into(),
            task: ModelTask::Embed,
            provider_id: "local/onnx".into(),
            model_id: "BGESmallENV15".into(),
            options: serde_json::json!({"execution_providers": ["cpu"]}),
            // …
        },
        // Heavy reranker: run on GPU with CPU as ORT-internal fallback.
        ModelAliasSpec {
            alias: "rerank/bge-large".into(),
            task: ModelTask::Rerank,
            provider_id: "local/onnx".into(),
            model_id: "BAAI/bge-reranker-base".into(),
            options: serde_json::json!({
                "execution_providers": ["cuda", "cpu"],
            }),
            // …
        },
    ])
    .build()
    .await?;
```

The order in `execution_providers` is priority; ORT silently falls
through per-op when an EP can't run a particular kernel.

`execution_providers` is the only per-spec device control in the
crate: `local/candle` takes its device from the build features alone,
and `remote/llamacpp` runs inference in the server process, where the
device is a `llama-server` flag rather than a catalog option. The two
can still be mixed freely in one catalog.

On a build with neither `gpu-cuda` nor `gpu-metal`,
`execution_providers` is a no-op — the only available device is CPU.

## Developer workflow

Typical flow for HF transformer exports:

1. Configure `local/onnx` in the catalog.
2. Resolve `runtime.raw_tensor_model(alias)`.
3. Load `tokenizer.json` / `config.json` from the same cached snapshot if needed.
4. Build `TensorBatch` inputs in app code.
5. Run inference.
6. Interpret outputs in app code.

See the live test coverage in `tests/local_onnx_hf_e2e_test.rs` for real classification and NER examples.
