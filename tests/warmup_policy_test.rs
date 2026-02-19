//! Tests for warmup policy lifecycle

use uni_xervo::api::{ModelTask, WarmupPolicy};
mod common;
use common::mock_support::{MockProvider, make_spec};
use uni_xervo::runtime::ModelRuntime;

#[tokio::test]
async fn test_eager_loads_on_build() {
    let provider = MockProvider::embed_only();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.warmup = WarmupPolicy::Eager;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();

    // Model should already be loaded
    let model = runtime.embedding("embed/test").await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_lazy_does_not_load_on_build() {
    let provider = MockProvider::embed_only();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.warmup = WarmupPolicy::Lazy;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();

    // Model loads on first access
    let model = runtime.embedding("embed/test").await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_background_loads_eventually() {
    let provider = MockProvider::embed_only();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.warmup = WarmupPolicy::Background;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();

    // Give background task time to load
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Model should be accessible
    let model = runtime.embedding("embed/test").await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_eager_failure_fails_build() {
    let provider = MockProvider::failing();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");
    spec.warmup = WarmupPolicy::Eager;
    spec.required = true;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_err());
    if let Err(err) = runtime {
        assert!(err.to_string().contains("Mock load failure"));
    }
}

#[tokio::test]
async fn test_eager_optional_failure_does_not_fail_build() {
    let provider = MockProvider::failing();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");
    spec.warmup = WarmupPolicy::Eager;
    spec.required = false;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();

    // Optional eager warmup failure should not fail startup; access still fails.
    assert!(runtime.embedding("embed/test").await.is_err());
}

#[tokio::test]
async fn test_background_failure_does_not_fail_build() {
    let provider = MockProvider::failing();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");
    spec.warmup = WarmupPolicy::Background;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    // Build succeeds even though background load will fail
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn test_lazy_failure_does_not_fail_build() {
    let provider = MockProvider::failing();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/failing", "test-model");
    spec.warmup = WarmupPolicy::Lazy;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    // Build succeeds, failure happens on first access
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn test_mixed_policies_in_single_catalog() {
    let provider = MockProvider::embed_only();

    let mut spec1 = make_spec("embed/eager", ModelTask::Embed, "mock/embed", "model1");
    spec1.warmup = WarmupPolicy::Eager;

    let mut spec2 = make_spec("embed/lazy", ModelTask::Embed, "mock/embed", "model2");
    spec2.warmup = WarmupPolicy::Lazy;

    let mut spec3 = make_spec("embed/background", ModelTask::Embed, "mock/embed", "model3");
    spec3.warmup = WarmupPolicy::Background;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec1, spec2, spec3])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();

    // All models should be accessible
    assert!(runtime.embedding("embed/eager").await.is_ok());
    assert!(runtime.embedding("embed/lazy").await.is_ok());
    assert!(runtime.embedding("embed/background").await.is_ok());
}

#[tokio::test]
async fn test_provider_warmup_tracking() {
    use uni_xervo::traits::ModelProvider;
    let provider = MockProvider::embed_only();
    assert_eq!(provider.warmup_count(), 0);

    provider.warmup().await.unwrap();
    assert_eq!(provider.warmup_count(), 1);

    provider.warmup().await.unwrap();
    assert_eq!(provider.warmup_count(), 2);
}

#[tokio::test]
async fn test_provider_warmup_on_build() {
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use uni_xervo::api::ModelAliasSpec;
    use uni_xervo::error::Result;
    use uni_xervo::traits::{
        LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
    };

    struct WarmupTracker(Arc<AtomicU32>);

    #[async_trait]
    impl ModelProvider for WarmupTracker {
        fn provider_id(&self) -> &'static str {
            "tracker"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supported_tasks: vec![],
            }
        }
        async fn load(&self, _: &ModelAliasSpec) -> Result<LoadedModelHandle> {
            unreachable!()
        }
        async fn health(&self) -> ProviderHealth {
            ProviderHealth::Healthy
        }
        async fn warmup(&self) -> Result<()> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let count = Arc::new(AtomicU32::new(0));
    let tracker = WarmupTracker(count.clone());

    let _ = ModelRuntime::builder()
        .register_provider(tracker)
        .warmup_policy(uni_xervo::api::WarmupPolicy::Eager)
        .build()
        .await
        .unwrap();

    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_provider_warmup_failure_is_load_error() {
    use async_trait::async_trait;
    use uni_xervo::api::ModelAliasSpec;
    use uni_xervo::error::{Result, RuntimeError};
    use uni_xervo::traits::{
        LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
    };

    struct FailingWarmup;

    #[async_trait]
    impl ModelProvider for FailingWarmup {
        fn provider_id(&self) -> &'static str {
            "failing_warmup"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supported_tasks: vec![],
            }
        }
        async fn load(&self, _: &ModelAliasSpec) -> Result<LoadedModelHandle> {
            unreachable!()
        }
        async fn health(&self) -> ProviderHealth {
            ProviderHealth::Healthy
        }
        async fn warmup(&self) -> Result<()> {
            Err(RuntimeError::InferenceError(
                "provider init failed".to_string(),
            ))
        }
    }

    let build_result = ModelRuntime::builder()
        .register_provider(FailingWarmup)
        .warmup_policy(WarmupPolicy::Eager)
        .build()
        .await;
    assert!(build_result.is_err());
    let err = build_result.err().unwrap();

    assert!(matches!(err, RuntimeError::Load(_)));
}

#[tokio::test]
async fn test_provider_warmup_background() {
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use uni_xervo::api::{ModelAliasSpec, WarmupPolicy};
    use uni_xervo::error::Result;
    use uni_xervo::traits::{
        LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
    };

    struct WarmupTracker(Arc<AtomicU32>);

    #[async_trait]
    impl ModelProvider for WarmupTracker {
        fn provider_id(&self) -> &'static str {
            "tracker_bg"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supported_tasks: vec![],
            }
        }
        async fn load(&self, _: &ModelAliasSpec) -> Result<LoadedModelHandle> {
            unreachable!()
        }
        async fn health(&self) -> ProviderHealth {
            ProviderHealth::Healthy
        }
        async fn warmup(&self) -> Result<()> {
            // Simulate some work
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let count = Arc::new(AtomicU32::new(0));
    let tracker = WarmupTracker(count.clone());

    let _ = ModelRuntime::builder()
        .register_provider(tracker)
        .warmup_policy(WarmupPolicy::Background)
        .build()
        .await
        .unwrap();

    // Should return immediately, count should still be 0
    assert_eq!(count.load(Ordering::SeqCst), 0);

    // Wait for warmup to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_model_warmup_called_on_eager() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    let tracker = Arc::new(AtomicU32::new(0));
    let provider = MockProvider::embed_only().with_model_warmup_tracker(tracker.clone());

    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.warmup = WarmupPolicy::Eager;

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let _ = runtime.embedding("embed/test").await.unwrap();

    // Model should have been warmed up exactly once
    assert_eq!(tracker.load(Ordering::SeqCst), 1);
}
