//! Tests for provider capability reporting and health status

use uni_xervo::api::ModelTask;
mod common;
use common::mock_support::{MockProvider, make_spec};
use uni_xervo::traits::{ModelProvider, ProviderHealth};

#[tokio::test]
async fn test_provider_capability_reporting() {
    let provider = MockProvider::new("mock/test", vec![ModelTask::Embed, ModelTask::Generate]);
    let caps = provider.capabilities();

    assert_eq!(caps.supported_tasks.len(), 2);
    assert!(caps.supported_tasks.contains(&ModelTask::Embed));
    assert!(caps.supported_tasks.contains(&ModelTask::Generate));
    assert!(!caps.supported_tasks.contains(&ModelTask::Rerank));
}

#[tokio::test]
async fn test_provider_health_healthy() {
    let provider = MockProvider::embed_only();
    let health = provider.health().await;

    match health {
        ProviderHealth::Healthy => {}
        _ => panic!("Expected Healthy status"),
    }
}

#[tokio::test]
async fn test_provider_health_degraded() {
    let provider =
        MockProvider::embed_only().with_health(ProviderHealth::Degraded("slow".to_string()));
    let health = provider.health().await;

    match health {
        ProviderHealth::Degraded(msg) => assert_eq!(msg, "slow"),
        _ => panic!("Expected Degraded status"),
    }
}

#[tokio::test]
async fn test_provider_health_unhealthy() {
    let provider =
        MockProvider::embed_only().with_health(ProviderHealth::Unhealthy("down".to_string()));
    let health = provider.health().await;

    match health {
        ProviderHealth::Unhealthy(msg) => assert_eq!(msg, "down"),
        _ => panic!("Expected Unhealthy status"),
    }
}

#[tokio::test]
async fn test_task_mismatch_error() {
    let provider = MockProvider::embed_only(); // Only supports Embed
    let spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/embed",
        "test-model",
    );

    let result = provider.load(&spec).await;
    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(err.contains("does not support task"));
}

#[tokio::test]
async fn test_provider_id() {
    let provider = MockProvider::new("custom/provider", vec![ModelTask::Embed]);
    assert_eq!(provider.provider_id(), "custom/provider");
}

#[tokio::test]
async fn test_load_increments_counter() {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    assert_eq!(provider.load_count(), 0);

    let _ = provider.load(&spec).await;
    assert_eq!(provider.load_count(), 1);

    let _ = provider.load(&spec).await;
    assert_eq!(provider.load_count(), 2);
}

#[tokio::test]
async fn test_failing_provider() {
    let provider = MockProvider::failing();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");

    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Mock load failure")
    );
}
