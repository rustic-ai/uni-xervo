# Sparse Embeddings

`SparseEmbeddingModel` is Uni-Xervo's learned-sparse text embedding task: it
turns text into a small set of weighted vocabulary terms — a `Vec<(u32, f32)>`
of `(term_id, weight)` pairs — rather than one dense vector. Learned sparse
(SPLADE family, the BGE-M3 sparse head) keeps the inverted-index compatibility
of BM25 while learning term weights that beat it in and out of domain. It is the
lexical half of hybrid **dense + sparse** first-stage retrieval.

The `local/onnx` provider implements it. Uni-Xervo produces the vectors; the
inverted index and impact/WAND scoring are a downstream concern — for in-process
ranking use [`uni_xervo::score::sparse_dot`](#scoring-sparse-vectors).

## Two methods: `mlm` vs `lexical`

SPLADE and BGE-M3 sparse compute weights differently and are **not**
interchangeable. The `sparse_method` option (or the preset) selects the recipe:

| | `mlm` (SPLADE) | `lexical` (BGE-M3 sparse) |
|---|---|---|
| Output read | MLM logits over the full vocab | One weight per token |
| Transform | `log(1 + ReLU(x))`, masked max-pool | `ReLU`, max over equal token ids |
| Term ids | Vocabulary indices (term **expansion**) | Input token ids (no expansion) |
| `vocab_size()` | Model vocabulary | Tokenizer vocabulary |

"Term expansion" means SPLADE can weight terms that never appeared in the text
(that is why it beats BM25); the lexical head only re-weights the literal input
terms. A consumer building an inverted index must not treat the two as the same.

## Catalog shape

```json
{
  "alias": "embed_sparse/splade",
  "task": "embed_sparse",
  "provider_id": "local/onnx",
  "model_id": "prithivida/Splade_PP_en_v1",
  "options": { "top_k": 64 }
}
```

`prithivida/Splade_PP_en_v1` (Apache-2.0) ships an in-graph MLM head, so no
separate projection weights are needed. The preset supplies `sparse_method`,
`vocab_size`, and file layout; options can override. `top_k` keeps only the
highest-weight terms per vector to bound index size.

### Model options

| Preset | Method | Vocab | Notes |
|---|---|---|---|
| `prithivida/Splade_PP_en_v1` | `mlm` | 30522 | Apache-2.0 SPLADE++; in-graph ONNX (default) |
| `prithivida/Splade_PP_en_v2` | `mlm` | 30522 | Improved SPLADE++ training; ships `onnx/` |
| `aapot/bge-m3-onnx` (`BGEM3Sparse`) | `lexical` | 250002 | BGE-M3 sparse head; multilingual; community export |

The current quality leader, **OpenSearch `neural-sparse-encoding-doc-v3-gte`**
(~0.546 BEIR, Apache-2.0, *doc-side inference-free*), is **not** preset: it ships
only safetensors on the Hub, so it needs an ONNX export first — then it drops
into the `mlm` path (its output is the same 30522-d vocab token weights).

See the [local/onnx provider reference](../reference/providers/onnx.md) for the
full option list (`sparse_method`, `output_name`, `output_index`, `max_seq_len`,
`token_type_ids`, `top_k`).

## Running sparse embedding

```rust
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::provider::LocalOnnxProvider;

let runtime = ModelRuntime::builder()
    .register_provider(LocalOnnxProvider::new())
    .catalog_from_file("catalog.json")?
    .build()
    .await?;

let model = runtime.sparse_embedder("embed_sparse/splade").await?;
let result = model.embed(&["how to bake sourdough"]).await?;
let vector = &result.vectors[0]; // Vec<(term_id, weight)>
```

A runnable version lives at `crates/uni-xervo/examples/embed_sparse.rs`:

```sh
cargo run --example embed_sparse --features provider-onnx
```

## Scoring sparse vectors

To rank a query against documents in-process, score two sparse vectors with
their dot product:

```rust
use uni_xervo::score::sparse_dot;

let score = sparse_dot(&query.vectors[0], &doc.vectors[0]);
```

Term ids are only comparable **within the same model** — two models with
different vocabularies produce incompatible sparse vectors.

## See also

- [Multi-vector / late interaction](multi-vector-embeddings.md) — the ColBERT
  sibling head, plus the `BGEM3Hybrid` single-pass task that emits the dense,
  sparse, and ColBERT heads of `aapot/bge-m3-onnx` from one forward pass.
- [local/onnx provider reference](../reference/providers/onnx.md) — full option schema.
