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
