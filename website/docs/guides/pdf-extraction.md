# Tiered PDF extraction (`uni-xervo-pdf`)

`uni-xervo-pdf` is an optional companion crate that turns a PDF into structured,
provenance-bearing text by **escalating per page only as far as needed** along a
capability ladder:

| Tier | What it answers | Backed by | Cost | Hallucinates? |
| --- | --- | --- | --- | :-: |
| `Native` | "what text is digitally embedded" | pure-Rust PDF text parse | ~free | no |
| `Ocr` | "what text is visually present" (scans) | [`local/onnx` OCR](../reference/providers/onnx.md#ocr-only-keys-task-ocr) (PP-OCR det+rec) | low (CPU) | no |
| `Vlm` | "what is the document *structure*" (tables/formulas/order) | [`local/mistralrs` olmOCR-2](../reference/providers/mistralrs.md#document-extraction-olmocr-2) | high (GPU) | **yes** |

## Why a separate crate

`uni-xervo` is an inference engine; the tiered router is *orchestration* (it
sequences models, rasterizes pages, parses PDFs) — none of which belongs inside
the engine. So the router lives in `uni-xervo-pdf`, which **composes** core's
existing model capabilities by alias (`runtime.ocr_model` /
`runtime.document_extractor`). The `Ocr` and `Vlm` rungs are ordinary
`uni-xervo` models — cached and instrumented by the runtime as usual.

It still feels native to the `embed` / `generate` / `nlp` family via an
**extension trait**: bring `PdfExt` into scope and call
`runtime.pdf_extractor(..)` like any other accessor.

## Features

- `hayro` (default) — pure-Rust PDF page rasterization (no FFI).
- `pdf-input` (default) — pure-Rust native text extraction via `lopdf`.

The OCR/VLM providers themselves are enabled on the `uni-xervo` dependency by
your app (`provider-onnx`, `provider-mistralrs`).

## Usage

```rust
use uni_xervo::runtime::ModelRuntime;
use uni_xervo_pdf::{PdfExt, PdfConfig, DocInput, DocExtractPolicy, Tier};

// A runtime whose catalog registers the rung models. PdfConfig::auto() looks
// for aliases "ocr/default" (an OcrModel) and "docext/default" (a
// DocumentExtractionModel); override PdfConfig.{ocr_alias,vlm_alias} to use
// your own names.
let runtime: std::sync::Arc<ModelRuntime> = /* build with OCR (+ optional VLM) aliases */;

// Family-native accessor (opt-in import of the extension trait).
let pdf = runtime.pdf_extractor(PdfConfig::auto()).await?;

// Auto-escalate from Native up to Vlm, per page.
let pages = pdf
    .extract(
        DocInput::Pdf { bytes: std::fs::read("doc.pdf")?, pages: None },
        DocExtractPolicy::auto_up_to(Tier::Vlm),
    )
    .await?;

for page in &pages {
    for block in &page.blocks {
        // Every block is traceable: which tier produced it, and how trusted.
        println!(
            "p{} [{:?} via {:?} conf={:?}] {}",
            page.page_number, block.kind, block.produced_by, block.confidence, block.content,
        );
    }
}
```

### Input — own or supply the pixels

`DocInput` resolves the "who rasterizes?" question per call:

- `DocInput::Pdf { bytes, pages }` — the crate parses + rasterizes (needs the
  `pdf-input` + `hayro` features).
- `DocInput::Pages(Vec<PageInput>)` — you already have page images (and,
  optionally, a pre-parsed text layer); no rasterizer needed.

### Policy — the reliability and cost knob

`DocExtractPolicy.level` is a `LevelPolicy`:

- `Fixed(tier)` — always exactly this tier.
- `Ceiling(tier)` — auto-escalate, but never above this tier. `Ceiling(Tier::Ocr)`
  **forbids the generative VLM entirely** (no hallucination risk).
- `Auto { min, max }` — escalate within the range per page.

`DocExtractPolicy.want` is `Text` or `Structure` (whether table/formula/layout
recovery is needed — only `Structure` reaches the `Vlm` tier).

The policy is also **hardware-adaptive**: if no VLM is configured/available,
`Auto`/`Ceiling` silently cap at `Ocr` rather than erroring — so a CPU-only
deployment still closes the scanned-PDF gap.

## Reliability

The whole point of the ladder is to keep a generative VLM from quietly writing a
wrong number into your downstream store:

- **Provenance** — every `TieredBlock` carries `produced_by: Tier` and a
  `Confidence` that is *honest about its source*: `Deterministic` (native /
  read-only OCR), `Measured(f32)` (a real OCR probability), or `Derived` (a
  heuristic for the VLM, which emits no native confidence).
- **Corroboration** — when a deterministic tier ran for the same page, the
  pipeline diffs the VLM's output against it (`VerifyPolicy`) and flags numeric
  divergence — a stronger guard than any confidence scalar.

Gate trust downstream on `produced_by` + `confidence` (e.g. "don't write a
`Vlm`-produced numeric block below confidence X without verification").
