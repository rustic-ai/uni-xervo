# Multi-Vector (Late-Interaction) Embeddings

`MultiVectorEmbeddingModel` is Uni-Xervo's per-token embedding task: instead of
pooling a sequence into one dense vector, it emits **one vector per token**.
Scoring is done with MaxSim — the sum, over each query token, of its maximum
similarity to any document token — the late-interaction approach popularized by
ColBERT and the strongest known method for layout-rich document retrieval
(ColPali / ColQwen2).

The `local/onnx` provider implements it. Uni-Xervo produces the per-token
vectors; a native multi-vector index is a downstream concern. For in-process
reranking of a top-k candidate set, use
[`uni_xervo::score::max_sim`](#scoring-with-maxsim) — no index required.

## When to use multi-vector vs dense or sparse

| | Dense (`EmbeddingModel`) | Sparse (`SparseEmbeddingModel`) | Multi-vector (this) |
|---|---|---|---|
| Output | One vector per input | Weighted term ids | One vector **per token** |
| Scoring | Cosine / dot | Sparse dot | MaxSim (late interaction) |
| Strength | Cheap recall, ANN-friendly | Lexical, inverted-index | Token-level precision |
| Cost | Lowest | Low | Highest (seq× larger output) |

Multi-vector output is `seq_len`× larger than dense, so it shines as a precise
**reranking** signal over a small candidate set rather than a first-stage index.

## Catalog shape

```json
{
  "alias": "embed_mv/colbert",
  "task": "embed_multi_vector",
  "provider_id": "local/onnx",
  "model_id": "answerdotai/answerai-colbert-small-v1",
  "options": { "normalize": true }
}
```

`answerdotai/answerai-colbert-small-v1` (Apache-2.0) ships an in-graph 384→96
ColBERT projection (`vespa_colbert.onnx`), so Uni-Xervo only strips padding and
L2-normalizes each token vector — no separate projection weights. The preset
supplies dimensions, file layout, and output selection.

### Model options

| Preset | Dim | Backbone | Notes |
|---|---|---|---|
| `lightonai/GTE-ModernColBERT-v1` | 128 | ModernBERT | Apache-2.0; BEIR-leading open ColBERT; long-context |
| `answerdotai/answerai-colbert-small-v1` | 96 | BERT | Apache-2.0; strong quality-per-byte default |
| `mixedbread-ai/mxbai-edge-colbert-v0-32m` | 64 | ModernBERT | small; storage-efficient |
| `mixedbread-ai/mxbai-edge-colbert-v0-17m` | 48 | ModernBERT | **smallest** strong ColBERT; beats 130M ColBERTv2 |
| `aapot/bge-m3-onnx` (`BGEM3Colbert`) | 1024 | XLM-R | multilingual; community export, output #2 |

Smaller `dimensions` means a smaller MaxSim index — the mxbai 48/64-d models cut
per-token storage to a fraction of ColBERTv2's 128-d. Every preset above is
validated end-to-end against its live repo (per-token dim plus a MaxSim
relevance check); each ships a `model.onnx` with the ColBERT projection folded
in. The best *multilingual* ColBERT, `jina-colbert-v2`, is intentionally omitted
— it is CC BY-NC (non-commercial).

See the [local/onnx provider reference](../reference/providers/onnx.md) for the
full option list (`dimensions`, `normalize`, `drop_special_tokens`,
`output_name`, `output_index`, `max_seq_len`, `token_type_ids`).

## Running multi-vector embedding

```rust
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::score::max_sim;

let runtime = ModelRuntime::builder()
    .register_provider(LocalOnnxProvider::new())
    .catalog_from_file("catalog.json")?
    .build()
    .await?;

let model = runtime.multi_vector_embedder("embed_mv/colbert").await?;
let result = model.embed(&["how do tides work", "tides come from the moon"]).await?;
let query = &result.vectors[0]; // Vec<Vec<f32>> — one vector per token
let doc = &result.vectors[1];
let score = max_sim(query, doc);
```

A runnable version lives at `crates/uni-xervo/examples/embed_multivector.rs`:

```sh
cargo run --example embed_multivector --features provider-onnx
```

The result is *ragged*: padding (and, with `drop_special_tokens`, special)
tokens are stripped, so the token count varies per input.

## Scoring with MaxSim

[`max_sim`](../reference/index.md) sums each query token's best match across the
document's tokens. With L2-normalized vectors (the default) it is cosine MaxSim.
[`colbert_rerank`](../reference/index.md) scores a query against many documents
at once:

```rust
use uni_xervo::score::colbert_rerank;

// Each doc is its own &[Vec<f32>] list of per-token vectors.
let scores = colbert_rerank(query, &[&doc_a[..], &doc_b[..], &doc_c[..]]); // one score per doc
```

## BGE-M3: three heads, one export

`aapot/bge-m3-onnx` exports all three BGE-M3 heads in one graph — dense
(`dense_vecs`), sparse (`sparse_vecs`), and ColBERT (`colbert_vecs`). uni-xervo
exposes each as its own task: `BGEM3Colbert` (this guide), `BGEM3Sparse`, and
`BGEM3Dense` (a dense alternative to the official `BAAI/bge-m3` preset). The bare
`aapot/bge-m3-onnx` model id resolves to whichever head matches the task.

Registering BGE-M3 for all three per-task tasks loads **three** sessions and runs
**three** forward passes. To get all three heads from a **single** pass, use the
hybrid task instead:

```rust
use uni_xervo::traits::HeadSet;

// `BGEM3Hybrid` declares all three heads of the one graph.
let model = runtime.hybrid_embedder("embed_hybrid/bgem3").await?;
let out = model.embed(&["multilingual hybrid retrieval"], HeadSet::ALL).await?;

let dense = out.dense.unwrap();               // Vec<Vec<f32>>        (cosine)
let sparse = out.sparse.unwrap();             // Vec<SparseVector>    (lexical)
let colbert = out.multi_vector.unwrap();      // Vec<Vec<Vec<f32>>>   (MaxSim)
```

One weight load, one pass, three retrieval signals. See
[`HybridEmbeddingModel`](../reference/index.md) and the runnable
`crates/uni-xervo/examples/embed_hybrid.rs`. `HeadSet` selects a subset (e.g.
`HeadSet::DENSE | HeadSet::SPARSE`) when you don't need every head.

## See also

- [Sparse embeddings](sparse-embeddings.md) — the learned-sparse sibling head.
- [local/onnx provider reference](../reference/providers/onnx.md) — full option schema.
