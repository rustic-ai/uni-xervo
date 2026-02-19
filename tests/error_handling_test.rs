//! Tests for error variant coverage and propagation

use uni_xervo::api::ModelTask;
use uni_xervo::error::RuntimeError;
mod common;
use common::mock_support::{MockProvider, make_spec};
use uni_xervo::runtime::ModelRuntime;

#[test]
fn test_error_display_config() {
    let err = RuntimeError::Config("invalid setting".to_string());
    assert_eq!(err.to_string(), "Configuration error: invalid setting");
}

#[test]
fn test_error_display_provider_not_found() {
    let err = RuntimeError::ProviderNotFound("mock/missing".to_string());
    assert_eq!(err.to_string(), "Provider not found: mock/missing");
}

#[test]
fn test_error_display_capability_mismatch() {
    let err = RuntimeError::CapabilityMismatch("task not supported".to_string());
    assert_eq!(err.to_string(), "Capability mismatch: task not supported");
}

#[test]
fn test_error_display_load() {
    let err = RuntimeError::Load("download failed".to_string());
    assert_eq!(err.to_string(), "Load error: download failed");
}

#[test]
fn test_error_display_api_error() {
    let err = RuntimeError::ApiError("upstream failed".to_string());
    assert_eq!(err.to_string(), "API error: upstream failed");
}

#[test]
fn test_error_display_inference() {
    let err = RuntimeError::InferenceError("model crashed".to_string());
    assert_eq!(err.to_string(), "Inference error: model crashed");
}

#[test]
fn test_error_display_rate_limited() {
    let err = RuntimeError::RateLimited;
    assert_eq!(err.to_string(), "Rate limited");
}

#[test]
fn test_error_display_unauthorized() {
    let err = RuntimeError::Unauthorized;
    assert_eq!(err.to_string(), "Unauthorized");
}

#[test]
fn test_error_display_timeout() {
    let err = RuntimeError::Timeout;
    assert_eq!(err.to_string(), "Timeout");
}

#[test]
fn test_error_display_unavailable() {
    let err = RuntimeError::Unavailable;
    assert_eq!(err.to_string(), "Unavailable");
}

#[tokio::test]
async fn test_error_propagation_provider_load_failure() {
    let provider = MockProvider::failing();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let result = runtime.embedding("embed/test").await;
    assert!(result.is_err());

    if let Err(err) = result {
        assert!(err.to_string().contains("Mock load failure"));
    }
}

#[tokio::test]
async fn test_error_propagation_model_inference_failure() {
    use crate::common::mock_support::MockEmbeddingModel;
    use uni_xervo::traits::EmbeddingModel;

    let model = MockEmbeddingModel::new(384, "test".to_string()).with_failure(true);
    let result = model.embed(vec!["test"]).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        RuntimeError::InferenceError(msg) => assert!(msg.contains("Mock embedding failure")),
        _ => panic!("Expected Inference error"),
    }
}

#[tokio::test]
async fn test_error_propagation_missing_provider() {
    let spec = make_spec("embed/test", ModelTask::Embed, "nonexistent", "test-model");

    let runtime = ModelRuntime::builder().catalog(vec![spec]).build().await;
    assert!(runtime.is_err());

    let err_msg = runtime.err().unwrap().to_string();
    assert!(err_msg.contains("Unknown provider"));
    assert!(err_msg.contains("nonexistent"));
}

#[tokio::test]
async fn test_error_propagation_capability_mismatch() {
    let provider = MockProvider::embed_only();
    let spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/embed",
        "test-model",
    );

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let result = runtime.generator("generate/test").await;
    assert!(result.is_err());

    if let Err(err) = result {
        assert!(err.to_string().contains("does not support task"));
    }
}

#[tokio::test]
async fn test_error_is_debug() {
    let err = RuntimeError::Config("test".to_string());
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("Config"));
}
