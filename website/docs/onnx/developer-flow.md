# ONNX Developer Flow

This page describes the full developer experience for `local/onnx`.

## 1. Enable the provider

`provider-onnx` is on by default, so the minimal setup is just:

```toml
[dependencies]
uni-xervo = "0.18"
tokio = { version = "1", features = ["full"] }
ndarray = "0.17"
```

If you want a lean ONNX-only build (no candle, no mistralrs, no remote providers):

```toml
uni-xervo = { version = "0.18", default-features = false, features = ["provider-onnx"] }
```

For CUDA-enabled ORT builds (Linux / Windows + NVIDIA), add `gpu-cuda`:

```toml
uni-xervo = { version = "0.18", features = ["gpu-cuda"] }
```

For Apple GPU + Neural Engine via the CoreML EP, add `gpu-metal`:

```toml
uni-xervo = { version = "0.18", features = ["gpu-metal"] }
```

## 2. Decide how the model is addressed

`model_id` can be either:

- a local path to a `.onnx` file, or
- a Hugging Face repo ID

Examples:

```json
{
  "model_id": "./models/sentiment.onnx"
}
```

```json
{
  "model_id": "smokxy/sequence_classification_onnx"
}
```

HF-backed aliases download a full repo snapshot into cache before loading the selected model.

## 3. Add a catalog alias

`local/onnx` always uses:

- `provider_id: "local/onnx"`
- `task: "raw"`

Example:

```json
{
  "alias": "raw/classifier",
  "task": "raw",
  "provider_id": "local/onnx",
  "model_id": "smokxy/sequence_classification_onnx",
  "warmup": "lazy",
  "options": {
    "artifact": "model.onnx",
    "execution_providers": ["cpu"]
  }
}
```

If the HF repo contains multiple `.onnx` files, set `options.artifact`.

## 4. Register the provider

```rust
use uni_xervo::provider::local_onnx::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

let runtime = ModelRuntime::builder()
    .register_provider(LocalOnnxProvider::new())
    .catalog_from_file("model-catalog.json")?
    .build()
    .await?;
```

## 5. Resolve the runner

```rust
let runner = runtime.raw_tensor_model("raw/classifier").await?;
```

This gives you an `Arc<dyn RawTensorModel>`.

Useful methods:

- `input_signature()`
- `output_signature()`
- `max_batch_size()`
- `run(&batch)`
- `run_batch(&batches)`

## 6. Inspect signatures before wiring inputs

```rust
println!("{:#?}", runner.input_signature());
println!("{:#?}", runner.output_signature());
```

This is the first thing to do when integrating a new model. It tells you:

- input/output names,
- tensor dtypes,
- rank and dimensions,
- whether symbolic or dynamic dimensions are present.

## 7. Build `TensorBatch`

Inputs and outputs use `TensorBatch`, an ordered map from tensor name to `TensorValue`.

```rust
use ndarray::arr2;
use uni_xervo::traits::{TensorBatch, TensorValue};

let mut batch = TensorBatch::new();
batch.insert(
    "input_ids".to_string(),
    TensorValue::I64(arr2(&[[101_i64, 1045, 2293, 3185, 102]]).into_dyn()),
);
batch.insert(
    "attention_mask".to_string(),
    TensorValue::I64(arr2(&[[1_i64, 1, 1, 1, 1]]).into_dyn()),
);
```

Supported tensor value families:

- `f32`
- `f64`
- `i32`
- `i64`
- `bool`
- `string`

## 8. Run inference

```rust
let outputs = runner.run(&batch).await?;
```

Or for batched execution:

```rust
let per_item_outputs = runner.run_batch(&[batch1, batch2, batch3]).await?;
```

Uni-Xervo validates:

- required input names,
- dtype matches,
- fixed-shape dimensions,
- batch stacking compatibility for `run_batch()`.

## 9. Interpret outputs in application code

Uni-Xervo does not decide what output tensors mean.

Typical application-side logic:

- classifier: read logits, apply argmax, map class IDs to labels
- NER: read token logits, align tokens to offsets, rebuild spans
- embeddings: pool or normalize hidden states if the export requires it
- tabular regression: read a scalar output tensor

That is by design. `local/onnx` stops at validated raw execution.

## 10. HF transformer-style flow

For a typical Hugging Face ONNX export:

1. Put the HF repo ID in `model_id`.
2. Set `options.artifact` when the repo contains multiple `.onnx` files.
3. Resolve `runtime.raw_tensor_model(alias)`.
4. Use your tokenizer/preprocessing layer in app code.
5. Convert the prepared arrays into `TensorBatch`.
6. Run inference.
7. Decode outputs in app code.

The live repo tests exercise this pattern for:

- sequence classification
- token classification / NER

See `tests/local_onnx_hf_e2e_test.rs`.

## 11. Prefetching

Use `uni-prefetch` when you want model artifacts available before first request:

```bash
cargo run --bin prefetch -- model-catalog.json
```

For HF-backed `local/onnx` aliases, this materializes the full repo snapshot into cache ahead of time.

## DX summary

The intended DX is:

- Uni-Xervo owns runtime and artifact orchestration
- your app owns task semantics

That keeps `local/onnx` lightweight, composable, and useful for both simple numeric models and harder custom pipelines.
