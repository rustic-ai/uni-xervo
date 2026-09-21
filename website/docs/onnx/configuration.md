# ONNX Configuration

## Required catalog fields

Every ONNX alias needs:

- `alias`
- `task` — one of `raw`, `rerank`, `embed`, `embed_sparse`, `embed_multi_vector`, `embed_hybrid`, `embed_image`, `nlp`, `ocr`, `document_extract`
- `provider_id: "local/onnx"`
- `model_id`

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

## `model_id`

Supported forms:

- absolute local path
- relative local path
- Hugging Face repo ID

HF repos are snapshotted into cache before ONNX Runtime loads the artifact.

## Provider options

`local/onnx` supports a common set of options across every task:

- `artifact`
- `max_batch_size`
- `execution_providers`
- `graph_optimization_level`
- `inter_op_num_threads`
- `intra_op_num_threads`
- `cache_dir`

Most tasks also accept task-specific keys (validated per task):

- `embed`: `tokenizer_path`, `pooling` (`cls`/`mean`/`max`/`last-token`), `normalize`, `dimensions`, `max_seq_len`, `token_type_ids`, `output_name`
- `rerank`: `max_seq_len`, `style` (`cross-encoder`/`generative`), `instruction`
- `embed_sparse`: `tokenizer_path`, `sparse_method` (`mlm`/`lexical`), `output_name`, `output_index`, `max_seq_len`, `token_type_ids`, `top_k`
- `embed_multi_vector`: `tokenizer_path`, `dimensions`, `normalize`, `drop_special_tokens`, `output_name`, `output_index`, `max_seq_len`, `token_type_ids`
- `embed_image`: `onnx_path`, `image_size`, `dimensions`, `normalization` (`siglip`/`imagenet`), `pool` (`none`/`mean`), `normalize`, `output_name`
- `nlp`: `onnx_path`, `tokenizer_path`, `label_maps_path`, `max_seq_len`
- `ocr`: `onnx_path`, `char_dict_path`, `image_height`, `image_width`, `normalization`, `blank_class`, `output_name`, plus an optional DBNet detection stage (`det_onnx_path`, `det_model_id`, `det_limit_side`, `det_bin_threshold`, `det_box_score_threshold`, `det_unclip_ratio`, `det_min_box_size`, `det_input_name`, `det_output_name`)
- `document_extract`: `style` (`granite-docling`/`mineru`/`olmocr`), `onnx_path`, `tokenizer_path`, `max_seq_len`

## `artifact`

Use `artifact` when the HF repo contains more than one `.onnx` file.

```json
{
  "artifact": "model.onnx"
}
```

If a repo has exactly one `.onnx` file, `artifact` can be omitted.

## `execution_providers`

`execution_providers` is an ordered list.

Examples:

```json
{
  "execution_providers": ["cpu"]
}
```

```json
{
  "execution_providers": ["cuda", "cpu"]
}
```

The list is a priority order, and entries this binary cannot construct are
dropped rather than fatal — so the example above runs on CUDA in a `gpu-cuda`
build and on CPU otherwise, with a `WARN` logged when `cuda` is dropped.
`active_execution_providers()` reports the list that survived, so it reads
`["cpu"]` on a build without `gpu-cuda`.

Dropping only applies while something remains. `{"execution_providers": ["cuda"]}`
with no `cpu` entry is a `RuntimeError::Config` on a build lacking `gpu-cuda` —
use that form when you want a missing GPU feature to fail loudly.

Supported names:

- `cpu`
- `cuda`
- `coreml`
- `directml`

These four are what **catalog validation** accepts, and it requires the array
form. The EP layer beneath it also understands `rocm`, `openvino`, `qnn`,
`tensorrt` and `webgpu`, but those are reachable only by calling
`ModelProvider::load` directly, and each additionally needs the
`provider-onnx-dynamic` feature plus a vendor ONNX Runtime via `ORT_DYLIB_PATH`.

Defaults:

- CPU-only builds: `["cpu"]`
- `gpu-cuda` builds: ORT-backed providers prefer `["cuda", "cpu"]`

## Batch tuning

`max_batch_size` controls the ceiling for dynamic-batch models.

```json
{
  "max_batch_size": 32
}
```

For non-batch models, Uni-Xervo falls back to sequential execution in `run_batch()`.

## ORT tuning

Example:

```json
{
  "graph_optimization_level": "extended",
  "inter_op_num_threads": 2,
  "intra_op_num_threads": 4
}
```

Accepted optimization levels:

- `disable`
- `basic`
- `extended`
- `all`

## Validation behavior

Uni-Xervo validates ONNX options at runtime build/register time.

It rejects:

- unknown option keys,
- wrong value types,
- unsupported execution provider names,
- invalid batch/thread counts.

Schema files:

- `schemas/model-catalog.schema.json`
- `schemas/provider-options/onnx.schema.json`
