#![cfg(any(feature = "provider-onnx", feature = "provider-onnx-dynamic"))]
//! Unit tests for the runtime EP-resolution layer.
//!
//! These tests don't require a real ONNX model, an ORT runtime DLL, or any
//! network access — they only exercise the parsing and feature-aware default
//! logic in `crate::provider::onnx_ep`. The deeper integration tests that
//! load real models live in `local_onnx_*_hf_e2e_test.rs`.

// onnx_ep is `pub(crate)`, so we test through the surface of the providers
// that consume it: catalog spec parsing → load (which fails fast on missing
// runtime, but the EP list is constructed *before* the load attempt).

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::traits::ModelProvider;

fn rerank_spec_with_eps(eps: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "rerank/test".to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx".to_string(),
        model_id: "cross-encoder/ms-marco-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "execution_providers": eps }),
    }
}

/// Loading with a CUDA-only EP list when `gpu-cuda` isn't enabled at compile
/// time should surface a clear `Config` error before any model download.
#[cfg(not(feature = "gpu-cuda"))]
#[tokio::test]
async fn cuda_only_eps_fail_when_cuda_feature_disabled() {
    let provider = LocalOnnxProvider::new();
    let spec = rerank_spec_with_eps(serde_json::json!(["cuda"]));

    let err = provider
        .load(&spec)
        .await
        .expect_err("loading with cuda-only EPs must fail when gpu-cuda is off");

    // Should be a Config error from `feature_not_enabled`, not an HF download
    // failure. This proves the EP-list validation runs before any I/O.
    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("CUDA") && msg.contains("gpu-cuda"),
        "error should explain CUDA needs gpu-cuda, got: {msg}"
    );
}

/// CoreML-only EPs without `gpu-metal` should fail with the same Config
/// error pattern. Verifies the guard is feature-uniform across EPs.
#[cfg(not(feature = "gpu-metal"))]
#[tokio::test]
async fn coreml_only_eps_fail_when_metal_feature_disabled() {
    let provider = LocalOnnxProvider::new();
    let spec = rerank_spec_with_eps(serde_json::json!(["coreml"]));

    let err = provider
        .load(&spec)
        .await
        .expect_err("loading with coreml-only EPs must fail when gpu-metal is off");

    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
}

/// Vendor EP strings (rocm, directml, openvino, qnn, tensorrt, webgpu)
/// must surface a Config error pointing at `provider-onnx-dynamic` when
/// only the bundled `provider-onnx` feature is active. Mirrors the
/// CUDA/CoreML guard pattern.
#[cfg(all(feature = "provider-onnx", not(feature = "provider-onnx-dynamic")))]
#[tokio::test]
async fn vendor_eps_require_provider_onnx_dynamic() {
    let provider = LocalOnnxProvider::new();
    for ep in ["rocm", "directml", "openvino", "qnn", "tensorrt", "webgpu"] {
        let spec = rerank_spec_with_eps(serde_json::json!([ep]));
        let err = provider
            .load(&spec)
            .await
            .expect_err(&format!("{ep} should fail under bundled provider"));
        match err {
            RuntimeError::Config(msg) => {
                assert!(
                    msg.contains("provider-onnx-dynamic"),
                    "{ep} error should mention provider-onnx-dynamic, got: {msg}"
                );
            }
            other => panic!("{ep}: expected Config error, got {other:?}"),
        }
    }
}

/// String form of `execution_providers` (single EP name as a JSON string,
/// not array) should parse — we documented this in `parse_execution_providers_option`.
#[cfg(not(feature = "gpu-cuda"))]
#[tokio::test]
async fn execution_providers_accepts_string_form() {
    let provider = LocalOnnxProvider::new();
    let spec = rerank_spec_with_eps(serde_json::json!("cuda"));

    // Same as the array case: should fail with Config error before any I/O,
    // proving the string was parsed into the same internal representation.
    let err = provider
        .load(&spec)
        .await
        .expect_err("string-form cuda EP should also fail without gpu-cuda");
    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
}

/// The complement of `cuda_only_eps_fail_when_cuda_feature_disabled`: adding
/// an explicit `cpu` fallback makes the same list resolve instead of erroring.
///
/// Only assertable without network under `provider-onnx-dynamic`, where
/// `preflight_ort_dylib` fails immediately *after* EP validation and so still
/// stops short of any HF download. The point is which error we get: a dylib
/// complaint, never `gpu-cuda`.
#[cfg(all(feature = "provider-onnx-dynamic", not(feature = "gpu-cuda")))]
#[tokio::test]
async fn cuda_with_cpu_fallback_passes_ep_validation() {
    // With a real runtime configured, preflight succeeds and the load would
    // proceed to download a model — skip rather than reach for the network.
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        eprintln!("Skipping - ORT_DYLIB_PATH is set, load would reach the network");
        return;
    }

    let provider = LocalOnnxProvider::new();
    let spec = rerank_spec_with_eps(serde_json::json!(["cuda", "cpu"]));

    let err = provider
        .load(&spec)
        .await
        .expect_err("no ORT dylib is configured, so preflight must still fail");
    let msg = err.to_string();
    assert!(
        !msg.contains("gpu-cuda"),
        "cuda should have been dropped in favour of the cpu entry, not rejected: {msg}"
    );
}
