# kniv v7 conformance fixtures

The reference runtime's own output, replayed by
`tests/local_onnx_nlp_v7_golden_expensive_test.rs`.

The Rust provider in `src/provider/local_onnx/nlp_v7` is a port of
`kniv_v7/runtime.py` from the model repository. The decoding there is not an
implementation detail — BIO Viterbi, Chu-Liu-Edmonds, predicate-aware SRL,
greedy span decoding and calibrated relation margins are what the published
accuracy figures are figures *for*. Unit tests pin each algorithm against
hand-built matrices; these fixtures are the only check that the whole
pipeline, wired to real graphs, answers what the reference answers.

## Layout

One directory per tag, each pinning one encoder at one precision. Quantisation
moves scores, so an INT8 run is compared against INT8 goldens and never
against fp32 ones.

| tag | model | encoder |
| --- | --- | --- |
| `ultrabert-int8` | `dragonscale-ai/kniv-v7-xlarge-u-en` | 1,024 subwords, mean pooling |
| `deberta-int8` | `dragonscale-ai/kniv-v7-xlarge-d-en` | 512 subwords, first-subword pooling |

Both are worth running: the pooling difference is a different set of graph
inputs, not a tuning knob.

Each file holds the case's input, the schema it was run against, and the
reference's per-window output verbatim.

## Regenerating

```bash
python scripts/gen-kniv-v7-goldens.py \
    --model /path/to/kniv-v7-xlarge-u-en \
    --runtime /path/to/kniv-nlp-models \
    --variant int8 \
    --out crates/uni-xervo/tests/fixtures/kniv-v7/ultrabert-int8
```

Needs `numpy`, `onnxruntime` and `tokenizers`, and the `kniv_v7` package from
the model repository. Regenerate only when the reference runtime or a model
release changes — a diff here is a change in what the model says, so read it
rather than accepting it.

## Running

```bash
EXPENSIVE_TESTS=1 KNIV_V7_MODEL_DIR=/path/to/kniv-v7-xlarge-u-en \
  cargo test --features provider-onnx --test local_onnx_nlp_v7_golden_expensive_test

EXPENSIVE_TESTS=1 KNIV_V7_FIXTURE_TAG=deberta-int8 \
  KNIV_V7_MODEL_DIR=/path/to/kniv-v7-xlarge-d-en \
  cargo test --features provider-onnx --test local_onnx_nlp_v7_golden_expensive_test
```

Without `KNIV_V7_MODEL_DIR` the model is downloaded from the Hugging Face repo
matching the tag (~2–3 GB).
