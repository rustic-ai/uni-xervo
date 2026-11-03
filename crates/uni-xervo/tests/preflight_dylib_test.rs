#![cfg(feature = "provider-onnx-dynamic")]
//! Regression test for the load-dynamic missing-dylib preflight.
//!
//! Background: when ort fails to load `libonnxruntime` (missing
//! `ORT_DYLIB_PATH`, wrong path, etc.) it panics from inside `setup_api`'s
//! `OnceLock` initializer, which a caller cannot turn into an error.
//! Through ort 2.0.0-rc.12 it was worse — the error-construction path called
//! back into `ort::api()` and deadlocked the thread forever
//! (pykeio/ort#560, fixed by `17ed7277` and released by the pinned rc.15).
//!
//! `provider::onnx_ep::preflight_ort_dylib` runs **before** any ort API
//! call and converts either failure mode into a clear `RuntimeError::Config`
//! that returns in milliseconds.
//!
//! This test:
//! 1. Sets `ORT_DYLIB_PATH` to a guaranteed-nonexistent path.
//! 2. Attempts to load the reranker provider.
//! 3. Asserts the call returns `Err(Config(...))` within a tight
//!    `tokio::time::timeout` (1 second), proving we no longer hang.
//!
//! Lives in its own integration test file so that the `setenv` it
//! performs doesn't race with other tests' env-var reads in the same
//! binary.

use std::time::Duration;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::traits::ModelProvider;

#[tokio::test]
async fn preflight_returns_config_error_fast_when_dylib_missing() {
    // Point ORT_DYLIB_PATH at a guaranteed-bad path so libloading fails
    // deterministically. SAFETY: this test file is its own integration-test
    // binary; no other test in this process reads ORT_DYLIB_PATH.
    unsafe {
        std::env::set_var(
            "ORT_DYLIB_PATH",
            "/nonexistent/path/to/libonnxruntime-this-must-not-exist.so",
        );
    }

    let spec = ModelAliasSpec {
        alias: "rerank/preflight-test".to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx".to_string(),
        model_id: "cross-encoder/ms-marco-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    };

    let provider = LocalOnnxProvider::new();

    // The whole point: this must NOT hang. Pre-workaround, it would block
    // forever. We give it 1s — preflight should return in single-digit ms.
    let result = tokio::time::timeout(Duration::from_secs(1), provider.load(&spec))
        .await
        .expect("preflight must return within 1s, not hang on the ort OnceLock deadlock");

    let err = result.expect_err("loading with a bad ORT_DYLIB_PATH must return Err, not Ok");

    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );

    let msg = err.to_string();
    assert!(
        msg.contains("ONNX Runtime") && msg.contains("nonexistent"),
        "error message should name the bad path and ORT, got: {msg}"
    );
    assert!(
        msg.contains("docs/migrations/0.9.0-feature-surface.md"),
        "error message should point at the migration doc, got: {msg}"
    );
}

/// Sanity check: clearing `ORT_DYLIB_PATH` falls back to platform default
/// names (`libonnxruntime.so` / `.dylib` / `onnxruntime.dll`). On a host
/// without ORT installed system-wide, this must also fail fast — not hang.
#[tokio::test]
async fn preflight_returns_config_error_fast_when_no_dylib_path_and_no_system_ort() {
    // SAFETY: own binary, no concurrent env reads.
    unsafe {
        std::env::remove_var("ORT_DYLIB_PATH");
    }

    let spec = ModelAliasSpec {
        alias: "rerank/preflight-default-path".to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx".to_string(),
        model_id: "cross-encoder/ms-marco-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    };

    let provider = LocalOnnxProvider::new();
    // Generous timeout because the test's *legitimate* slow path is a real
    // HF model download (when the host has a system ORT and the model isn't
    // cached). We're guarding against a different failure mode: the ort
    // OnceLock deadlock blocks **forever**, so any finite timeout detects it.
    // 60s is plenty for a network hiccup yet still flagrantly less than
    // forever.
    let result = tokio::time::timeout(Duration::from_secs(60), provider.load(&spec)).await;

    // Three legal outcomes:
    //   (a) timeout — implies the preflight didn't fail-fast; THIS IS A REGRESSION.
    //   (b) Err(Config(...)) — preflight caught missing dylib (typical case).
    //   (c) Ok(...) — host happens to have a system ORT on the loader path; preflight
    //       passed and the actual load succeeded. The test is informational only here.
    match result {
        Err(_elapsed) => {
            panic!("preflight hung past 60s — the deadlock workaround is broken");
        }
        Ok(Err(RuntimeError::Config(_))) => {
            // expected on a host without system ORT
        }
        Ok(Err(other)) => {
            // also fine — could be a different error (e.g. HF download failed); the
            // important thing is we didn't hang.
            eprintln!("preflight returned a non-Config error (acceptable, not a hang): {other:?}");
        }
        Ok(Ok(_)) => {
            // Host has system ORT and the model is cached. Also fine.
            eprintln!("preflight + load succeeded (host has system ORT on loader path)");
        }
    }
}
