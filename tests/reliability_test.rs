use uni_xervo::api::{ModelAliasSpec, ModelTask, RetryConfig, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::runtime::ModelRuntime;
mod common;
use common::mock_support::MockProvider;

#[tokio::test]
async fn test_instrumented_embedding_timeout_enforced() {
    let provider = MockProvider::embed_only().with_model_delay(2000);

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![ModelAliasSpec {
            alias: "embed/timeout".to_string(),
            task: ModelTask::Embed,
            provider_id: "mock/embed".to_string(),
            model_id: "test-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: Some(1), // 1 second timeout
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        }])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/timeout").await.unwrap();

    let start = std::time::Instant::now();
    let res = model.embed(vec!["hello"]).await;

    assert!(res.is_err());
    match res.unwrap_err() {
        RuntimeError::Timeout => (),
        e => panic!("Expected Timeout error, got: {}", e),
    }

    let elapsed = start.elapsed();
    // It should have failed around 1 second, not 2 seconds.
    assert!(elapsed.as_secs() < 2);
}

use metrics_util::debugging::DebuggingRecorder;

#[tokio::test]
async fn test_instrumented_embedding_metrics() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let _ = metrics::set_global_recorder(recorder);

    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![ModelAliasSpec {
            alias: "embed/metrics".to_string(),
            task: ModelTask::Embed,
            provider_id: "mock/embed".to_string(),
            model_id: "test-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        }])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/metrics").await.unwrap();
    model.embed(vec!["hello"]).await.unwrap();

    let snapshot = snapshotter.snapshot();

    // Check for counter
    let counter_found = snapshot.into_vec().into_iter().any(|(ckey, _, _, _)| {
        let name = ckey.key().name();
        let mut labels = ckey.key().labels();

        name == "model_inference.total"
            && labels.any(|l| l.key() == "alias" && l.value() == "embed/metrics")
            && {
                let mut labels = ckey.key().labels(); // Get fresh iterator
                labels.any(|l| l.key() == "status" && l.value() == "success")
            }
    });
    assert!(counter_found, "Inference counter not found");
}

#[tokio::test]
async fn test_instrumented_embedding_retry_success() {
    let provider = MockProvider::embed_only().with_model_fail_count(2);

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![ModelAliasSpec {
            alias: "embed/retry".to_string(),
            task: ModelTask::Embed,
            provider_id: "mock/embed".to_string(),
            model_id: "test-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: Some(RetryConfig {
                max_attempts: 3,
                initial_backoff_ms: 10,
            }),
            options: serde_json::Value::Null,
        }])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/retry").await.unwrap();

    let res = model.embed(vec!["hello"]).await;
    assert!(
        res.is_ok(),
        "Expected success after retries, got: {:?}",
        res.err()
    );
}

#[tokio::test]
async fn test_instrumented_embedding_retry_failure() {
    let provider = MockProvider::embed_only().with_model_fail_count(5);

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![ModelAliasSpec {
            alias: "embed/retry-fail".to_string(),
            task: ModelTask::Embed,
            provider_id: "mock/embed".to_string(),
            model_id: "test-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: Some(RetryConfig {
                max_attempts: 3,
                initial_backoff_ms: 10,
            }),
            options: serde_json::Value::Null,
        }])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/retry-fail").await.unwrap();

    let res = model.embed(vec!["hello"]).await;
    assert!(res.is_err());
    match res.unwrap_err() {
        RuntimeError::RateLimited => (),
        e => panic!("Expected RateLimited error, got: {}", e),
    }
}

#[tokio::test]
async fn test_instrumented_embedding_success_within_timeout() {
    let provider = MockProvider::embed_only().with_model_delay(500);

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![ModelAliasSpec {
            alias: "embed/fast".to_string(),
            task: ModelTask::Embed,
            provider_id: "mock/embed".to_string(),
            model_id: "test-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: Some(2), // 2 second timeout
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        }])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/fast").await.unwrap();

    let res = model.embed(vec!["hello"]).await;
    assert!(res.is_ok());
}
