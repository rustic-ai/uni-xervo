# Structured NLP

`NlpModel` is Uni-Xervo's managed, multi-head structured-NLP task. One
`analyze` call runs a single transformer forward pass and returns
part-of-speech tags, named entities, a dependency parse, semantic-role frames,
and a sentence-level dialog-act classification — already decoded into typed
Rust structs, with provider-side tokenization, chunking, and label mapping
handled for you.

`local/onnx` is the only provider that ships `NlpModel`, and it serves two
generations of the kniv family. Which one you get follows from the model repo,
so a catalog entry is the only thing that changes:

| | **kniv v1 cascade** | **kniv v7** |
| --- | --- | --- |
| model | [`kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall) | [`kniv-v7-xlarge-u-en`](https://huggingface.co/dragonscale-ai/kniv-v7-xlarge-u-en), [`kniv-v7-xlarge-d-en`](https://huggingface.co/dragonscale-ai/kniv-v7-xlarge-d-en) |
| graphs | 1 (`onnx/cascade.onnx`), ~75 MB INT8 | 8, with fp32 / FP16 / INT8 encoders; ~1.9 GB INT8 |
| input per call | 128 subwords | a window of up to 512 words |
| heads | POS, NER, DEP, SRL, dialog act | the same, plus morphology, lemma, sentiment, coreference, typed entities and relations |
| decoding | per-token argmax | BIO Viterbi, one dependency tree per sentence, predicate-aware SRL |
| input shape | prose | prose **or** conversation turns |
| extraction | — | your own relations and entity types, supplied per request |

Start with v7 unless you need the small model's footprint: it is a different
model, not a bigger one, and the extra heads are what agent memory needs.

## Heads

| Head | `NlpTasks` flag | Where it lands | v1 |
| --- | --- | --- | --- |
| Part-of-speech | `POS` | `tokens[i].pos`, `windows[w].words[i].pos` | ✓ |
| Named-entity recognition | `NER` | `tokens[i].ner` (raw BIO) + `entities` (merged spans) | ✓ |
| Dependency parse | `DEP` | `tokens[i].dep` | ✓ |
| Semantic role labeling | `SRL` | `frames` | ✓ |
| Dialog acts | `CLS` | `speech_acts` | ✓ |
| Morphology | `MORPH` | `windows[w].words[i].morph` | |
| Lemmas | `LEMMA` | `windows[w].words[i].lemma` | |
| Sentiment | `SENTIMENT` | `windows[w].sentences[j].sentiment` | |
| Coreference | `COREF` | `windows[w].coref` | |
| Typed entities | `ENTITIES` | `windows[w].entities` | |
| Facts | `FACTS` | `windows[w].facts` | |

Callers request the subset they want via the `NlpTasks` bitflag; the provider
populates exactly those fields, leaves the rest empty, and — on v7 — skips the
graph calls nothing asked for. Three convenience sets are defined:
`NlpTasks::CASCADE` (the five v1 heads), `NlpTasks::LINGUISTIC` (every per-word
and per-sentence head), and `NlpTasks::ALL`. `FACTS` implies `ENTITIES`.

## Quick example

```rust
use uni_xervo::traits::{NlpRequest, NlpTasks};

let model = runtime.nlp_model("nlp/default").await?;

let req = NlpRequest::new("Marie Curie discovered radium in Paris.", NlpTasks::ALL);
let result = model.analyze(vec![req]).await?.remove(0);

// Merged named entities — no BIO bookkeeping required:
for e in &result.entities {
    println!("{} [{}] tokens {:?}", e.text, e.label, e.token_span);
}

// Dependency heads: `None` is the sentence root, `Some(i)` indexes `tokens`:
for tok in &result.tokens {
    match tok.dep.as_ref().and_then(|d| d.head) {
        None => println!("{:>12} ─▶ ROOT", tok.text),
        Some(h) => println!("{:>12} ─▶ {}", tok.text, result.tokens[h].text),
    }
}

// Gate on the full dialog-act distribution, self-described by the cls vocab:
if let (Some(act), Some(maps)) = (result.speech_acts.first(), model.label_maps()) {
    let informative = act
        .scores
        .iter()
        .zip(&maps.cls)
        .filter(|(_, label)| matches!(label.as_str(), "inform" | "status" | "request" | "offer"))
        .any(|(p, _)| *p >= 0.20);
    println!("informative: {informative}");
}

// Regroup subword tokens into words:
let words = result.tokens.last().map(|t| t.word_index + 1).unwrap_or(0);
println!("{} tokens across {} words", result.tokens.len(), words);
```

`analyze` takes a batch (`Vec<NlpRequest>`) and returns one `NlpResult` per
request, in order.

## Requesting heads

`NlpTasks` is a bitflag — combine the heads you need, or use `NlpTasks::ALL`:

`NlpRequest` is `#[non_exhaustive]`: build it with `new` (or `from_turns`) and
chain the setters, so later releases can add inputs without breaking callers.

```rust
// POS + NER only — DEP / SRL / CLS fields come back empty:
let req = NlpRequest::new("Marie Curie discovered radium in Paris.", NlpTasks::POS | NlpTasks::NER);

// A conversation, with a schema and a threshold override:
let turns = [
    Turn::new("user", "I moved to Lisbon last year and started at a bank."),
    Turn::new("assistant", "Congratulations on the move!"),
];
let req = NlpRequest::from_turns(&turns, NlpTasks::FACTS)
    .with_schema(&schema)
    .with_calibration(&overrides);
```

Restricting the task set skips that head's post-processing — and, on v7, the
graph call behind it — which saves wall time on hot ingest paths where, for
example, only NER is needed.

## Result types

`NlpModel::analyze` returns one `NlpResult` per request:

```rust
pub struct NlpResult {
    pub tokens: Vec<NlpToken>,        // per-token POS / NER / DEP, in input order
    pub sentences: Vec<NlpSentence>,  // boundaries with token-range indices
    pub frames: Vec<SrlFrame>,        // SRL frames (empty unless NlpTasks::SRL)
    pub speech_acts: Vec<SpeechAct>,  // dialog acts (empty unless NlpTasks::CLS)
    pub entities: Vec<NerEntity>,     // merged NER spans (empty unless NlpTasks::NER)
    pub windows: Vec<NlpWindow>,      // v7: the full per-window analysis
}
```

The first five fields are the v1 shape. A v7 model fills them too — as a
flattened projection of its windows — so a caller written against v1 keeps
working; everything the projection cannot express (coreference, typed
entities, facts and their attributes) is in `windows`.

### `NlpToken`

```rust
pub struct NlpToken {
    pub text: String,
    pub start: usize,            // UTF-8 byte offset in the original text
    pub end: usize,              // UTF-8 byte offset (exclusive)
    pub pos: Option<String>,     // Universal Dependencies tag; None if POS not requested
    pub ner: Option<String>,     // raw BIO tag ("B-PERSON", "O", …); None if NER not requested
    pub dep: Option<DepLink>,    // dependency arc; None if DEP not requested
    pub word_index: usize,       // which word this subword belongs to
}
```

Tokens are subword units. `word_index` is **dense and monotonically
non-decreasing** over `tokens`: subwords of one word share an index, and each
new word increments it by one starting at `0`. Use it to regroup subwords into
words without parsing tokenizer metaspace markers.

### `DepLink`

```rust
pub struct DepLink {
    pub head: Option<usize>,   // Some(i): 0-based index into NlpResult::tokens; None: root
    pub relation: String,      // "nsubj", "obj", "amod", …
}
```

`head` is a **global** index into `NlpResult::tokens` (not sentence- or
chunk-local). `None` means the token attaches to the sentence root.

### `SrlFrame` and `SrlRole`

```rust
pub struct SrlFrame {
    pub predicate_token: usize,            // index into NlpResult::tokens
    pub predicate_sense: Option<String>,   // e.g. "buy.01" when known
    pub roles: Vec<SrlRole>,               // argument spans, excluding the predicate
}

pub struct SrlRole {
    pub span: (usize, usize),  // inclusive [first, last] token indices
    pub label: String,         // "ARG0", "ARG1", "ARGM-TMP", …
}
```

`span` endpoints are both inclusive; a single-token argument has `first == last`.
The predicate token is never itself included in `roles`. The provider runs one
forward pass per detected verb to fill every frame — this multi-pass
orchestration is invisible to callers.

### `SpeechAct`

```rust
pub struct SpeechAct {
    pub sentence_index: usize,
    pub label: String,       // == argmax of `scores`
    pub confidence: f32,     // == scores[argmax]
    pub scores: Vec<f32>,    // full per-class softmax, parallel to NlpLabelMaps::cls
}
```

`scores` is the complete softmax over the dialog-act vocabulary, so consumers
can do thresholded multi-label gating (admit a sentence if *any* relevant class
clears a probability floor) rather than relying only on the top-1 `label`.

### `NerEntity`

```rust
pub struct NerEntity {
    pub text: String,                // surface text spanning the original bytes
    pub label: String,               // entity type, BIO prefix stripped (e.g. "PERSON")
    pub token_span: (usize, usize),  // inclusive [first, last] token indices
    pub char_span: (usize, usize),   // half-open [start, end) UTF-8 byte offsets
}
```

`entities` is a BIO-collapsed view of the per-token `ner` tags: a `B-` (or
orphan `I-`) opens a span, matching `I-` tags extend it, and an `O` tag or a
label change closes it. The raw per-token `ner` tags remain available on each
`NlpToken`.

## kniv v7: windows, entities and facts

A v7 model analyses **windows** of whole sentences — up to 512 words, never
crossing a paragraph or a turn — rather than a fixed token budget. Every index
inside a window (dependency heads, mention spans, role spans) is a *word*
index local to that window, which is why the full analysis lives in
`NlpResult::windows` instead of being forced into the flat token list.

```rust
pub struct NlpWindow {
    pub tokens: Vec<String>,                 // the window's words
    pub analysed_words: usize,               // words that fit the encoder
    pub sentences: Vec<WindowSentence>,      // spans, speaker, dialog acts, sentiment
    pub words: Vec<WordAnalysis>,            // pos, morph, lemma, head, deprel
    pub ner: Vec<WordSpan>,                  // OntoNotes spans
    pub srl: Vec<WindowFrame>,               // one frame per predicate — not only verbs
    pub coref: Vec<Vec<(usize, usize)>>,     // clusters of inclusive word spans
    pub entities: Vec<NlpEntity>,            // typed, grouped across mentions
    pub facts: Vec<NlpFact>,                 // entity → entity, with attributes
}
```

`analysed_words` is how many leading words fit the encoder's subword budget.
Words past it appear in `tokens` but carry no analysis, so a very long sentence
is reported truncated rather than silently dropped.

### The extraction schema

Entity types and relations are **not** baked into the model: you supply them
per request, by name and description, and they are scored across the whole
window. That is what makes the model usable for a schema it has never seen.

```rust
use uni_xervo::traits::{ExtractionSchema, LabelSpec, NlpRequest, NlpTasks, Turn};

let schema = ExtractionSchema {
    relations: vec![
        LabelSpec::described("lives in", "the place where the head resides"),
        LabelSpec::described("employer", "organization the head works for"),
    ],
    entity_types: Some(vec![
        LabelSpec::name("location"),
        LabelSpec::name("organization"),
        LabelSpec::name("person"),
    ]),
    top_per_pair: None,          // None uses the model's calibrated rel_top (4)
};

let turns = [
    Turn::new("user", "I moved to Lisbon last year and started at a bank."),
    Turn::new("assistant", "Congratulations on the move!"),
];
let req = NlpRequest::from_turns(&turns, NlpTasks::FACTS).with_schema(&schema);
let result = model.analyze(vec![req]).await?.remove(0);

for window in &result.windows {
    for fact in &window.facts {
        println!(
            "{} --{}--> {} ({}, {})",
            window.entities[fact.head].text,
            fact.relation,
            window.entities[fact.tail].text,
            fact.polarity.as_str(),
            fact.modality.as_str(),
        );
    }
}
// user --lives in--> Lisbon (positive, asserted)
// user --employer--> a bank (positive, asserted)
```

Schema label embeddings are computed once and cached per exact label list, so
a **stable** schema costs one extra graph pass for the life of the model. A
schema that changes per request pays that pass every time — build it once and
reuse it.

Guidance that matters more than it looks:

- **Name relations as a reader would**, and say in the description which side
  is the head: `"the place where the head resides"`.
- **Avoid near-synonyms in one schema** (`located in` *and*
  `headquarters location`). They split the decision between them; merge
  downstream if you need both.
- **Prefer specific relations.** A generic `related to` attracts instances of
  every specific one.
- Without temporal entity types (`date`, `time`) in the schema, `NlpFact::time`
  stays `None` — it can only point at an entity.

### Conversations

Passing `turns` instead of `text` does three things prose cannot: sentences
never cross a turn, the previous turn gives `"it"` and `"that one"` their
referents, and **each speaker becomes an entity in its own right**. That last
one is what lets `"I live in Lisbon"` be stored about the user:

```rust
for e in &window.entities {
    if e.speaker {
        println!("participant: {}", e.text);   // "user", "assistant"
    }
}
```

Speaker entities come first, in order of first appearance, and their `spans`
are the turn-start positions rather than mentions in the text.

### Fact attributes

```rust
pub struct NlpFact {
    pub head: usize,               // index into NlpWindow::entities
    pub tail: usize,
    pub relation: String,
    pub margin: f32,               // > 0 for a reported fact; larger is surer
    pub polarity: FactPolarity,    // Positive | Negative
    pub modality: FactModality,    // Asserted | Hypothetical | Planned | Desired | Questioned | Reported
    pub holder: Option<usize>,     // whoever holds the belief
    pub time: Option<usize>,       // a temporal entity, when the schema has one
}
```

Only `Asserted` and `Reported` are claims about the world. Consolidating
`Planned`, `Desired`, `Questioned` or `Hypothetical` into current state is a
mistake — `"next month I'm joining Northstar"` is not an employer, and
`"do you live in Lisbon?"` is not a residence. `Negative` polarity is evidence
*against* the same `(head, relation, tail)`, not a separate fact.

`margin` is not a probability. Map it to one with a sigmoid if you want a
confidence, and tune a relation's threshold through calibration rather than by
filtering on the margin.

### Calibration

A v7 release ships `calibration.json`: dev-tuned decoding thresholds that the
published quality figures were measured with. They are readable, and
overridable per request.

```rust
use uni_xervo::traits::NlpCalibrationOverrides;

if let Some(cal) = model.calibration() {
    println!("rel_top={} unseen_offset={}", cal.rel_top, cal.unseen_offset);
}

// One relation firing too eagerly: raise its own threshold, leave the rest.
let mut overrides = NlpCalibrationOverrides::default();
overrides.rel_offsets.insert("located in".to_string(), -1.0);
let req = NlpRequest::from_turns(&turns, NlpTasks::FACTS)
    .with_schema(&schema)
    .with_calibration(&overrides);
```

A fact is kept when its raw margin exceeds `-offset`, so a *lower* offset is a
higher bar. Validate any change on a labelled sample: these are the numbers the
model's reported accuracy depends on.

### Picking an encoder

| | `kniv-v7-xlarge-u-en` (UltraBERT) | `kniv-v7-xlarge-d-en` (DeBERTa) |
| --- | --- | --- |
| window | 1,024 subwords | 512 subwords |
| better at | zero-shot types, agent memory, long inputs | parsing accuracy, fixed-schema relations, plans vs assertions |
| INT8 speed, 200-word window | ~0.47 s | ~0.58 s |
| resident graphs, INT8 | ~2 GB | ~3 GB |

Both are Apache-2.0 and need no token. The provider reads each release's
`onnx/onnx_meta.json`, so the window size and subword pooling difference needs
no configuration.

### Cost

`FACTS` is the expensive flag: the relation pass scores every one of the
`E(E-1)` ordered entity pairs, so cost grows with the square of the entities in
a window. If latency matters, keep the schema's entity types narrow rather than
listing everything. The INT8 graphs use dynamic per-channel quantisation and
run on CPU — on a GPU, use `"variant": "fp16"`, since CUDA would run the INT8
graphs through its own CPU fallback.

## Label vocabularies

The model decodes against fixed per-head vocabularies, exposed directly so you
do not have to embed a parallel copy of the model's `label_maps.json`:

```rust
pub struct NlpLabelMaps {
    pub pos: Vec<String>,
    pub ner: Vec<String>,
    pub deprel: Vec<String>,
    pub srl: Vec<String>,
    pub cls: Vec<String>,        // parallel to SpeechAct::scores
    pub morph: Vec<String>,      // v7: UD feature bundles
    pub lemma: Vec<String>,      // v7: edit scripts, not words
    pub sentiment: Vec<String>,  // v7
    pub ent_type: Vec<String>,   // v7: the coarse types used without a schema
}

// Returns Some for models with a fixed label set (the ONNX cascade),
// None for models that don't expose vocabularies (remote / mock).
if let Some(maps) = model.label_maps() {
    // maps.cls[i] names the class whose probability is SpeechAct::scores[i]
}
```

Each vector is indexed by the model's internal class id, so `cls` lines up
index-for-index with `SpeechAct::scores`.

## Catalog entry

kniv v7:

```json
{
  "alias": "nlp/default",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-v7-xlarge-u-en",
  "warmup": "background",
  "options": {
    "variant": "int8,fp16",
    "execution_providers": ["cpu"],
    "intra_op_num_threads": 8
  }
}
```

The v1 cascade:

```json
{
  "alias": "nlp/cascade",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
  "warmup": "background",
  "options": {
    "onnx_path": "onnx/cascade.onnx",
    "max_seq_len": 128
  }
}
```

The generation is inferred from the repo — a repo that carries
`onnx/onnx_meta.json` is v7 — and `"generation": "v1" | "v7"` pins it
explicitly. The option sets do not overlap: `onnx_path`, `label_maps_path` and
`max_seq_len` are v1-only (v7 reads its graph set and window size from
`onnx_meta.json`), while `variant`, `calibration_path` and `label_bank_path`
are v7-only. Mixing them is rejected at catalog registration, before anything
is downloaded.

`variant` is a *fallback chain*, not a suffix: `"int8,fp16"` means "INT8 if
this graph publishes it, else FP16, else fp32". That is how the documented
INT8 configuration ends up running `plain-int8`, `marker-int8` and
`label-fp16` — the label graph has no INT8 file because it runs once per
schema and is cached.

See the [`local/onnx` provider reference](../reference/providers/onnx.md) for the
full NLP option-key table.

## Migration to 0.19.0

v7 support is additive to the result types, which are `#[non_exhaustive]`, but
`NlpRequest` changed shape.

- **`NlpRequest` is now `#[non_exhaustive]`** and carries `turns`, `schema` and
  `calibration` alongside `text` and `tasks`. Replace struct literals with
  `NlpRequest::new(text, tasks)` (or `from_turns`) plus the `with_*` setters.
  This is the last change of its kind: further inputs can now be added without
  breaking callers.
- **`NlpTasks::ALL` now covers the v7 heads too.** Code that wants exactly the
  old five should say `NlpTasks::CASCADE`; a v1 model reports `CASCADE` from
  `supported_tasks()` rather than `ALL`.
- **New, additive:** `NlpResult::windows`, `NlpModel::calibration()`, four more
  vocabularies on `NlpLabelMaps`, and the `MORPH` / `LEMMA` / `SENTIMENT` /
  `COREF` / `ENTITIES` / `FACTS` task flags. The v1 path is unchanged.

## Migration to 0.14.0

The 0.14.0 `NlpModel` surface is richer than 0.13.x. Two changes are breaking;
the rest is additive.

- **`DepLink.head` is now `Option<usize>`** (was `usize`). `Some(i)` is a 0-based
  global index into `NlpResult::tokens`; `None` is the sentence root. Match on
  `None` instead of inferring root from a sentinel value.
- **The result structs are `#[non_exhaustive]`.** Build them through `Default`
  (`NlpResult::default()` plus field assignment) rather than struct literals —
  this is what lets future heads be added without breaking callers. Reading
  existing fields is unaffected.
- **New, additive fields and methods:** `SpeechAct::scores` (full CLS
  distribution), `NlpResult::entities` (merged `NerEntity` spans),
  `NlpToken::word_index`, and `NlpModel::label_maps()`. Code that ignores them
  compiles unchanged.

## See also

- [`local/onnx` provider reference](../reference/providers/onnx.md) — NLP option
  keys and the canonical model details.
- `scripts/gen-kniv-v7-goldens.py` — regenerates the conformance fixtures the
  v7 provider is tested against, from the model's Python reference runtime.
- [Multimodal trait surface](multimodal-traits.md) — the full set of traits
  introduced alongside `NlpModel`.
- [API reference (rustdoc)](../api/uni_xervo/index.html) — complete trait and
  type signatures.
