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

The files at the top of each tag were generated on linux-x86_64. A platform
whose INT8 kernels decide differently has its own set in a subdirectory
named for it, which the test uses when it runs there:

| set | platform |
| --- | --- |
| `<tag>/` | `linux-x86_64`, and any platform without its own set |
| `<tag>/darwin-arm64/` | macOS on Apple Silicon |

On darwin-arm64 the bundled build and the 1.31.1 wheel agree with each
other, and both differ from the linux-x86_64 goldens in 2.65% of UltraBERT's
decisions, above the tolerant budget: the gap is the CPU architecture's INT8
kernels, not the build.

Each case file holds the case's input, the schema it was run against, and
the reference's per-window output verbatim. `provenance.json` records what
produced them:

| field | meaning |
| --- | --- |
| `repo`, `revision` | the Hugging Face release; the test loads exactly this revision |
| `onnxruntime`, `onnxruntime_library`, `onnxruntime_library_sha256` | the ONNX Runtime wheel and the exact library file that ran |
| `platform` | `sys.platform` and machine |
| `intra_op_threads` | threads per session (1: a fixed reduction order) |

## Exact and tolerant comparison

INT8 results depend on the ONNX Runtime *build*, not only its version. The
runtime bundled with `provider-onnx` (pyke's 1.31.0) and Microsoft's 1.31.1
wheel compute some shapes differently: on the same inputs and graph file the
encoder logits differ by up to 0.25, while both stay as close to the fp32
graph as each other. A decision near a tie can therefore flip between builds,
so the test has two modes:

- **Exact** — `provider-onnx-dynamic` with `ORT_DYLIB_PATH` naming the library
  whose SHA-256 is in `provenance.json`, and the pinned model. The sessions
  run on `intra_op_threads`, and every discrete decision must match. Measured
  for both tags: 0 differences on linux-x86_64 (839 and 798 decisions) and
  on darwin-arm64 against its own set (844 and 810), fact margins within
  2.3e-3.
- **Tolerant** — any other build. Structure (windows, tokens, sentences,
  word counts) must match exactly; at most 2% of the discrete decisions may
  differ. Measured on the bundled build against its platform's goldens:
  0.24% (UltraBERT) and 0.54% (DeBERTa) on linux-x86_64, 0% for both on
  darwin-arm64, all of them facts near the threshold or their margins.

Every difference is printed in both modes.

## Regenerating

```bash
# REV: the full 40-character commit to pin, from the model's history page
# (https://huggingface.co/dragonscale-ai/kniv-v7-xlarge-u-en/commits/main).
REV=...
hf download dragonscale-ai/kniv-v7-xlarge-u-en --revision "$REV" --exclude model.pt
pip install 'onnxruntime==1.31.1' numpy tokenizers
python scripts/gen-kniv-v7-goldens.py \
    --model ~/.cache/huggingface/hub/models--dragonscale-ai--kniv-v7-xlarge-u-en/snapshots/"$REV" \
    --repo dragonscale-ai/kniv-v7-xlarge-u-en --revision "$REV" \
    --variant int8 \
    --out crates/uni-xervo/tests/fixtures/kniv-v7/ultrabert-int8
```

The release ships its own `kniv_v7/` reference runtime, which the script
uses (`--runtime` overrides it). Regenerate when the reference runtime or a
model release changes, and read the diff rather than accepting it: it is a
change in what the model says. Generate the tag's own set on linux-x86_64
with the wheel above so the exact mode stays reproducible, and keep the ONNX
Runtime version the same as the one `ort` bundles.

When a tag is regenerated, regenerate each platform set with it, on that
platform, into `--out <tag>/<platform>`, for example on a Mac:

```bash
python scripts/gen-kniv-v7-goldens.py --model "$SNAPSHOT" \
    --repo dragonscale-ai/kniv-v7-xlarge-u-en --revision "$REV" --variant int8 \
    --out crates/uni-xervo/tests/fixtures/kniv-v7/ultrabert-int8/darwin-arm64
```

`provenance.json` then names that platform and its library
(`libonnxruntime.<version>.dylib` on macOS). The model cache the tests fill
holds only the files the provider loads; download the `kniv_v7/` folder at
the same revision (`hf download ... --include 'kniv_v7/*'`) and pass it as
`--runtime`.

## Running

```bash
# Tolerant, on the bundled runtime; the pinned model downloads on first use
# (about 2 GB for UltraBERT, 3.5 GB for DeBERTa).
EXPENSIVE_TESTS=1 cargo test --features provider-onnx \
  --test local_onnx_nlp_v7_golden_expensive_test
EXPENSIVE_TESTS=1 KNIV_V7_FIXTURE_TAG=deberta-int8 cargo test --features provider-onnx \
  --test local_onnx_nlp_v7_golden_expensive_test

# Exact, on the goldens' own library.
pip install 'onnxruntime==1.31.1'
# The wheel's library: libonnxruntime.so.<version> on Linux,
# libonnxruntime.<version>.dylib on macOS.
EXPENSIVE_TESTS=1 ORT_DYLIB_PATH=$(python -c "import onnxruntime, pathlib; \
  capi = pathlib.Path(onnxruntime.__file__).parent / 'capi'; \
  print(next(p for p in sorted(capi.iterdir()) if p.name.startswith('libonnxruntime.')))") \
  cargo test --no-default-features --features provider-onnx-dynamic \
  --test local_onnx_nlp_v7_golden_expensive_test
```

`KNIV_V7_MODEL_DIR` replaces the pinned model with a local release or another
repository; the comparison is then tolerant and unpinned.
