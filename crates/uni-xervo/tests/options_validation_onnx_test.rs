#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn onnx_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "raw/onnx".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: "model.onnx".to_string(),
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
async fn builder_accepts_valid_onnx_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "artifact": "onnx/model.onnx",
            "max_batch_size": 8,
            "execution_providers": ["cpu"],
            "graph_optimization_level": "all",
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 2
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_cuda_onnx_execution_provider() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "execution_providers": ["cuda"]
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_onnx_execution_provider() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "execution_providers": ["rocm"]
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Execution provider 'rocm'")
    );
}

#[tokio::test]
async fn builder_rejects_non_string_onnx_artifact() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "artifact": 7
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Option 'artifact'")
    );
}

// ---------------------------------------------------------------------------
// OCR detection-stage option validation
// ---------------------------------------------------------------------------

fn ocr_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "ocr/test".to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: "test/repo".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

async fn build_ocr(
    options: serde_json::Value,
) -> Result<std::sync::Arc<ModelRuntime>, uni_xervo::error::RuntimeError> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec(options)])
        .build()
        .await
}

#[tokio::test]
async fn builder_accepts_ocr_without_detection() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt"
    }))
    .await;
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_ocr_with_detection_options() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_model_id": "other/det",
        "det_limit_side": 960,
        "det_bin_threshold": 0.3,
        "det_box_score_threshold": 0.6,
        "det_unclip_ratio": 1.5,
        "det_min_box_size": 3,
        "det_input_name": "x"
    }))
    .await;
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_unknown_det_key() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_bogus": 1
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_bin_threshold_out_of_range() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_bin_threshold": 1.5
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_unclip_ratio_negative() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_unclip_ratio": -1.0
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_bin_threshold_not_a_number() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_bin_threshold": "x"
    }))
    .await;
    assert!(runtime.is_err());
}

// ---------------------------------------------------------------------------
// NLP: the two kniv generations take different options
// ---------------------------------------------------------------------------

fn nlp_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "nlp/kniv".to_string(),
        task: ModelTask::Nlp,
        provider_id: "local/onnx".to_string(),
        model_id: "dragonscale-ai/kniv-v7-xlarge-u-en".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

/// Validation runs at catalog registration, before any download, so a
/// mismatched option is caught in milliseconds rather than after fetching a
/// couple of gigabytes of graphs.
async fn build_nlp(options: serde_json::Value) -> Result<(), String> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(options)])
        .build()
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn builder_accepts_kniv_v7_options() {
    build_nlp(serde_json::json!({
        "variant": "int8,fp16",
        "calibration_path": "calibration.json",
        "label_bank_path": "label_bank.json",
        "tokenizer_path": "tokenizer.json",
        "generation": "v7",
        "execution_providers": ["cpu"],
        "intra_op_num_threads": 8,
        "graph_optimization_level": "all"
    }))
    .await
    .expect("v7 options accepted");
}

#[tokio::test]
async fn builder_accepts_kniv_v1_options() {
    build_nlp(serde_json::json!({
        "onnx_path": "onnx/cascade-int8.onnx",
        "label_maps_path": "label_maps.json",
        "max_seq_len": 128,
        "generation": "v1"
    }))
    .await
    .expect("v1 options accepted");
}

#[tokio::test]
async fn builder_rejects_mixing_the_two_generations() {
    let err = build_nlp(serde_json::json!({
        "variant": "int8,fp16",
        "max_seq_len": 128
    }))
    .await
    .expect_err("v1 and v7 options together");
    assert!(
        err.contains("variant") && err.contains("max_seq_len"),
        "{err}"
    );
}

#[tokio::test]
async fn builder_rejects_v1_options_on_a_declared_v7_model() {
    for key in ["onnx_path", "label_maps_path", "max_seq_len"] {
        let value = if key == "max_seq_len" {
            serde_json::json!(128)
        } else {
            serde_json::json!("x")
        };
        let err = build_nlp(serde_json::json!({"generation": "v7", key: value}))
            .await
            .expect_err("rejected");
        assert!(err.contains(key), "{key}: {err}");
    }
}

#[tokio::test]
async fn builder_rejects_v7_options_on_a_declared_v1_model() {
    for key in ["variant", "calibration_path", "label_bank_path"] {
        let err = build_nlp(serde_json::json!({"generation": "v1", key: "x"}))
            .await
            .expect_err("rejected");
        assert!(err.contains(key), "{key}: {err}");
    }
}

#[tokio::test]
async fn builder_rejects_an_unknown_generation() {
    let err = build_nlp(serde_json::json!({"generation": "v6"}))
        .await
        .expect_err("rejected");
    assert!(err.contains("generation"), "{err}");
}

#[tokio::test]
async fn builder_rejects_non_string_v7_options() {
    for key in [
        "variant",
        "calibration_path",
        "label_bank_path",
        "generation",
    ] {
        let err = build_nlp(serde_json::json!({key: 1}))
            .await
            .expect_err("rejected");
        assert!(err.contains(key), "{key}: {err}");
    }
}

#[tokio::test]
async fn builder_rejects_an_unknown_nlp_option() {
    let err = build_nlp(serde_json::json!({"schema_path": "schema.json"}))
        .await
        .expect_err("rejected");
    assert!(err.contains("schema_path"), "{err}");
}
