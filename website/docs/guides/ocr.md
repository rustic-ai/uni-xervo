# OCR

`OcrModel` is Uni-Xervo's optical character recognition task: it turns an image
into text with per-block bounding boxes and confidences. The `local/onnx`
provider implements it with PP-OCR — a deterministic, CPU-friendly pipeline that
pairs a DBNet text **detector** with a CRNN/CTC **recognizer**. It is the
right tool when you need to *read* text off a page reliably and cheaply.

OCR is one rung of the document-understanding family. For layout-aware output —
tables, formulas, headings, reading order — reach for the generative
[`DocumentExtractionModel`](multimodal-traits.md) instead, or let
[tiered PDF extraction](pdf-extraction.md) escalate between them per page.

## When to use OCR vs. document extraction

| | `OcrModel` (`local/onnx`) | `DocumentExtractionModel` (`local/mistralrs`) |
|---|---|---|
| Output | Plain text + bounding boxes | Structured Markdown (tables, formulas, headings) |
| Determinism | Deterministic; will not invent text | Generative; can hallucinate |
| Confidence | Per-block score | None (corroborate against a lower tier) |
| Cost | Low; runs on CPU | High; 7B VLM, GPU recommended |

A common pattern is to run OCR as the cheap, trustworthy tier and only escalate
to a document VLM for pages that need structure — see
[tiered PDF extraction](pdf-extraction.md).

## Catalog shape

Verified end-to-end against `monkt/paddleocr-onnx` (Apache-2.0): PP-OCRv5 DBNet
detection plus an English recognizer. Drop `det_onnx_path` for single-stage
recognition of images that are already cropped to a text line.

```json
{
  "alias": "ocr/ppocr-en",
  "task": "ocr",
  "provider_id": "local/onnx",
  "model_id": "monkt/paddleocr-onnx",
  "options": {
    "onnx_path": "languages/english/rec.onnx",
    "char_dict_path": "languages/english/dict.txt",
    "image_height": 48,
    "image_width": 320,
    "normalization": "siglip",
    "det_onnx_path": "detection/v5/det.onnx"
  }
}
```

See the [local/onnx provider reference](../reference/providers/onnx.md#ocr-only-keys-task-ocr)
for the full option list. In short: recognition keys (`onnx_path`,
`char_dict_path`, `image_height`, `image_width`, `normalization`, `blank_class`)
are always required, and the optional `det_*` keys turn on the detection stage —
the presence of `det_onnx_path` is what switches single-stage recognition into
two-stage detect → recognize.

## Running OCR

```rust
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::traits::ImageInput;

let runtime = ModelRuntime::builder()
    .register_provider(LocalOnnxProvider::new())
    .catalog_from_file("catalog.json")?
    .build()
    .await?;

let ocr = runtime.ocr_model("ocr/ppocr-en").await?;
let image = ImageInput::Bytes {
    data: std::fs::read("scan.png")?,
    media_type: "image/png".to_string(),
};
let results = ocr.recognize(vec![image]).await?;
println!("{}", results[0].plain_text);
```

A runnable version lives at `crates/uni-xervo/examples/ocr.rs`:

```sh
cargo run --example ocr --features provider-onnx
```

`recognize` takes a batch of images and returns one result per image, in order.
Inputs must be `ImageInput::Bytes`; the provider does not fetch URLs (the caller
fetches and passes bytes).

## Interpreting results

```rust
pub struct OcrResult {
    pub blocks: Vec<OcrBlock>,  // per detected region
    pub plain_text: String,     // concatenation, always populated
}

pub struct OcrBlock {
    pub text: String,
    pub bbox: [f32; 4],   // [x0, y0, x1, y1] in image coordinates
    pub confidence: f32,  // in [0.0, 1.0]
}
```

For the simple "just give me the text" case, read `plain_text`. When you need
positions — highlighting, redaction, layout — iterate `blocks`.

## Detection vs. recognition-only

- **Two-stage (with `det_onnx_path`):** the detector finds text regions on a full
  page, each region is cropped and recognized, and results are returned in
  reading order. Use this for whole pages and scans.
- **Recognition-only (no `det_onnx_path`):** the whole image is treated as a
  single text line. Use this when upstream has already cropped to lines.

## See also

- [local/onnx provider reference](../reference/providers/onnx.md) — full OCR option schema.
- [Multimodal trait surface](multimodal-traits.md) — where `OcrModel` sits among the tasks.
- [Tiered PDF extraction](pdf-extraction.md) — OCR as a tier in the PDF pipeline.
