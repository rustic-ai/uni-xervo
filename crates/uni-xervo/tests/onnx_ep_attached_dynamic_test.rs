//! A requested execution provider that fails to register is not reported
//! (rustic-ai/uni-xervo#64).
//!
//! Needs a real ONNX Runtime library, so it runs only on a
//! `provider-onnx-dynamic` build with `ORT_DYLIB_PATH` set, and skips
//! otherwise. A stock CPU ONNX Runtime build has no DirectML provider, so
//! `["directml", "cpu"]` registers DirectML, fails, skips it and must report
//! `["cpu"]`; before #64 it reported `["directml", "cpu"]`.
//!
//! ```sh
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib cargo test \
//!     --no-default-features --features provider-onnx-dynamic \
//!     --test onnx_ep_attached_dynamic_test
//! ```

#![cfg(feature = "provider-onnx-dynamic")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

#[tokio::test]
async fn an_accelerator_that_fails_to_register_is_not_reported() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("Skipping test - set ORT_DYLIB_PATH to an ONNX Runtime library");
        return;
    }
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/onnx/dynamic_batch_linear.onnx");
    let spec = ModelAliasSpec {
        alias: "raw/linear".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: model.to_str().unwrap().to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "execution_providers": ["directml", "cpu"] }),
    };
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();
    let raw = runtime.raw_tensor_model("raw/linear").await.unwrap();
    assert_eq!(raw.active_execution_providers(), ["cpu"]);
}

/// A CPU-free list whose accelerator fails to register is a load error, not a
/// silent CPU session.
#[tokio::test]
async fn a_strict_accelerator_that_fails_to_register_fails_the_load() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("Skipping test - set ORT_DYLIB_PATH to an ONNX Runtime library");
        return;
    }
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/onnx/dynamic_batch_linear.onnx");
    let spec = ModelAliasSpec {
        alias: "raw/strict".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: model.to_str().unwrap().to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "execution_providers": ["directml"] }),
    };
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();
    let err = match runtime.raw_tensor_model("raw/strict").await {
        Ok(_) => panic!("a CPU-free list whose provider cannot register must fail"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("\"directml\" failed to register"), "{err}");
}
