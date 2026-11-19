//! Fixture-level tests for `local/onnx` × `Ocr`.

#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn ocr_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

#[tokio::test]
async fn local_onnx_advertises_ocr_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = LocalOnnxProvider::new().capabilities().supported_tasks;
    assert!(tasks.contains(&ModelTask::Ocr));
}

#[tokio::test]
async fn ocr_alias_registers_with_full_options() {
    let opts = serde_json::json!({
        "onnx_path": "model.onnx",
        "char_dict_path": "char_dict.txt",
        "image_height": 48,
        "image_width": 320,
        "normalization": "imagenet",
        "blank_class": 0
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec("ocr/test", "some/repo", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "valid options should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn ocr_alias_rejects_unknown_normalization() {
    let opts = serde_json::json!({
        "onnx_path": "model.onnx",
        "char_dict_path": "char_dict.txt",
        "normalization": "bogus"
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec("ocr/bad", "some/repo", opts)])
        .build()
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn ocr_alias_rejects_zero_image_height() {
    let opts = serde_json::json!({
        "onnx_path": "model.onnx",
        "char_dict_path": "char_dict.txt",
        "image_height": 0
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec("ocr/bad", "some/repo", opts)])
        .build()
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn ocr_alias_accepts_space_char_and_max_width() {
    let opts = serde_json::json!({
        "onnx_path": "model.onnx",
        "char_dict_path": "char_dict.txt",
        "use_space_char": true,
        "max_image_width": 2000
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec("ocr/test", "some/repo", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "valid options should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn ocr_alias_rejects_bad_space_char_and_max_width() {
    for opts in [
        serde_json::json!({
            "onnx_path": "model.onnx",
            "char_dict_path": "char_dict.txt",
            "use_space_char": "yes"
        }),
        serde_json::json!({
            "onnx_path": "model.onnx",
            "char_dict_path": "char_dict.txt",
            "max_image_width": 0
        }),
        serde_json::json!({
            "onnx_path": "model.onnx",
            "char_dict_path": "char_dict.txt",
            "blank_class": -1
        }),
    ] {
        let result = ModelRuntime::builder()
            .register_provider(LocalOnnxProvider::new())
            .catalog(vec![ocr_spec("ocr/bad", "some/repo", opts.clone())])
            .build()
            .await;
        assert!(result.is_err(), "should reject {opts}");
    }
}

/// A recognizer whose graph fixes its batch gets lines exactly that many at a
/// time, in order (fixed batch 1 is common in paddle2onnx exports). Runs
/// offline: the model sits in a pre-filled cache at a pinned commit.
#[tokio::test]
async fn fixed_batch_recognizer_reads_every_line_in_order() {
    use uni_xervo::traits::ImageInput;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    for batch in [1, 2] {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = cache
            .path()
            .join("models--fixture--ctc/snapshots")
            .join(COMMIT);
        std::fs::create_dir_all(&snapshot).unwrap();
        let fixture = format!(
            "{}/tests/fixtures/onnx/ctc_fixed_batch_{batch}.onnx",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::copy(&fixture, snapshot.join("rec.onnx")).unwrap();
        std::fs::write(snapshot.join("dict.txt"), "a\nb\n").unwrap();

        let mut spec = ocr_spec(
            "ocr/fixed",
            "fixture/ctc",
            serde_json::json!({
                "onnx_path": "rec.onnx",
                "char_dict_path": "dict.txt",
                "normalization": "siglip",
                "cache_dir": cache.path().to_str().unwrap(),
            }),
        );
        spec.revision = Some(COMMIT.to_string());
        let runtime = ModelRuntime::builder()
            .register_provider(LocalOnnxProvider::new())
            .catalog(vec![spec])
            .build()
            .await
            .unwrap();
        let ocr = runtime.ocr_model("ocr/fixed").await.unwrap();

        // Three lines of different aspect ratios: more than one batch, and a
        // short last batch for the batch-2 graph.
        let images: Vec<ImageInput> = [(40, 20), (400, 20), (90, 30)]
            .into_iter()
            .map(|(w, h)| {
                let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
                    w,
                    h,
                    image::Rgb([200, 200, 200]),
                ));
                let mut data = Vec::new();
                img.write_to(
                    &mut std::io::Cursor::new(&mut data),
                    image::ImageFormat::Png,
                )
                .unwrap();
                ImageInput::Bytes {
                    data,
                    media_type: "image/png".to_string(),
                }
            })
            .collect();
        let results = ocr
            .recognize(images)
            .await
            .unwrap_or_else(|e| panic!("batch {batch}: {e}"));
        let texts: Vec<&str> = results.iter().map(|r| r.plain_text.as_str()).collect();
        assert_eq!(texts, ["a b", "a b", "a b"], "batch {batch}");
    }
}

/// Integer options written as whole-number floats, as Python's `json` emits
/// them, build like integers.
#[tokio::test]
async fn ocr_alias_accepts_whole_number_floats() {
    let opts = serde_json::json!({
        "onnx_path": "model.onnx",
        "char_dict_path": "char_dict.txt",
        "image_height": 48.0,
        "image_width": 320.0,
        "blank_class": 0.0
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec("ocr/test", "some/repo", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "{}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}
