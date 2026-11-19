//! End-to-end (real model) test of the two-stage OCR path: PP-OCRv5 DBNet
//! detection + English CRNN/CTC recognition through `LocalOnnxProvider`.
//!
//! This is the *accuracy gate* for the ported DBNet detection postprocessing:
//! the pure postprocessing math is covered by deterministic unit tests in
//! `provider::local_onnx::det`, while this test exercises the whole pipeline
//! against a real probability map and recognizer.
//!
//! Gated by both:
//! - `#![cfg(feature = "provider-onnx")]`
//! - `EXPENSIVE_TESTS=1` (downloads ~96 MB of ONNX weights from HuggingFace).
//!
//! Run with:
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test --features provider-onnx \
//!     --test local_onnx_ocr_detection_expensive_test -- --ignored
//! ```
//!
//! Model: `monkt/paddleocr-onnx` (Apache-2.0) — PP-OCRv5 server detection
//! (`detection/v5/det.onnx`, input `x [N,3,H,W]` → prob map `[N,1,H,W]`) and
//! the English mobile recognizer (`languages/english/rec.onnx`, height 48,
//! 438 CTC classes, SigLIP-style `(x/255-0.5)/0.5` normalization).

#![cfg(feature = "provider-onnx")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::ImageInput;

fn should_run_expensive_tests() -> bool {
    match env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

const MODEL_ID: &str = "monkt/paddleocr-onnx";

fn ocr_detection_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "ocr/ppocr-en".to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            // Recognition (English CRNN, height 48, SigLIP-style normalization).
            "onnx_path": "languages/english/rec.onnx",
            "char_dict_path": "languages/english/dict.txt",
            "image_height": 48,
            "image_width": 320,
            "normalization": "siglip",
            "blank_class": 0,
            // Detection (PP-OCRv5 DBNet, same repo).
            "det_onnx_path": "detection/v5/det.onnx"
        }),
    }
}

fn image_bytes(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/ocr/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

async fn ocr_runtime() -> std::sync::Arc<ModelRuntime> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_detection_spec()])
        .build()
        .await
        .expect("build runtime")
}

/// The positive end-to-end case: a two-line text image is detected and read.
#[tokio::test]
#[ignore]
async fn ppocr_detects_and_recognizes_text_lines() {
    require_expensive_tests!();

    let runtime = ocr_runtime().await;
    let ocr = runtime
        .ocr_model("ocr/ppocr-en")
        .await
        .expect("load ocr model");

    let input = ImageInput::Bytes {
        data: image_bytes("text_lines.png"),
        media_type: "image/png".to_string(),
    };
    let results = ocr.recognize(vec![input]).await.expect("recognize");
    assert_eq!(results.len(), 1, "one result per input image");
    let result = &results[0];

    eprintln!("detected {} block(s):", result.blocks.len());
    for b in &result.blocks {
        eprintln!(
            "  bbox={:?} conf={:.3} text={:?}",
            b.bbox, b.confidence, b.text
        );
    }
    eprintln!("plain_text:\n{}", result.plain_text);

    // Detection found at least the two text lines.
    assert!(
        result.blocks.len() >= 2,
        "expected >= 2 detected text boxes, got {}",
        result.blocks.len()
    );
    // Every block carries a real (non-whole-image) bbox and a confidence.
    for b in &result.blocks {
        assert_ne!(
            b.bbox,
            [0.0, 0.0, 1.0, 1.0],
            "detection bbox, not whole-image"
        );
        assert!(b.confidence > 0.0, "recognition produced a confidence");
    }
    // Recognition produced the expected words (allow OCR imperfection: require
    // a few of the known tokens, case-insensitively).
    let text = result.plain_text.to_lowercase();
    let hits = ["hello", "world", "ocr", "test", "2026"]
        .iter()
        .filter(|t| text.contains(*t))
        .count();
    assert!(
        hits >= 2,
        "expected to recognize >= 2 known tokens, matched {hits} in {text:?}"
    );
}

/// The negative case: a blank page yields an empty result (no boxes), not an error.
#[tokio::test]
#[ignore]
async fn ppocr_blank_image_yields_empty_result() {
    require_expensive_tests!();

    let runtime = ocr_runtime().await;
    let ocr = runtime
        .ocr_model("ocr/ppocr-en")
        .await
        .expect("load ocr model");

    let input = ImageInput::Bytes {
        data: image_bytes("blank.png"),
        media_type: "image/png".to_string(),
    };
    let results = ocr.recognize(vec![input]).await.expect("recognize");
    assert_eq!(results.len(), 1);
    assert!(
        results[0].blocks.is_empty(),
        "blank page should detect no text, got {} block(s)",
        results[0].blocks.len()
    );
}

/// Long lines keep their characters and their spaces (rustic-ai/uni-xervo#68,
/// #69): each detected line must read back exactly as rendered.
///
/// The fixture is Arial at 92 px (44 pt at 150 dpi). Stretched to a fixed
/// 48×320, the longer lines had fewer CTC time steps than characters and came
/// back cut short, and the recognizer's trailing space class was dropped.
#[tokio::test]
#[ignore]
async fn ppocr_reads_long_lines_with_spaces() {
    require_expensive_tests!();

    let runtime = ocr_runtime().await;
    let ocr = runtime
        .ocr_model("ocr/ppocr-en")
        .await
        .expect("load ocr model");

    let input = ImageInput::Bytes {
        data: image_bytes("maintenance_log.png"),
        media_type: "image/png".to_string(),
    };
    let results = ocr.recognize(vec![input]).await.expect("recognize");
    let lines: Vec<&str> = results[0].blocks.iter().map(|b| b.text.as_str()).collect();
    eprintln!("lines: {lines:#?}");
    assert_eq!(
        lines,
        [
            "Pump Station Maintenance Log",
            "The north pump was replaced on March 12.",
            "Inspector Mira Okafor signed off the repair.",
            "Next inspection is due in September.",
        ]
    );
}
