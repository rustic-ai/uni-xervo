//! Tests for ModelRuntime builder and lifecycle operations

use uni_xervo::api::ModelTask;
mod common;
use common::mock_support::{MockProvider, make_spec};
use uni_xervo::runtime::ModelRuntime;

#[tokio::test]
async fn test_builder_empty_catalog() {
    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn test_builder_single_spec() {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();
    assert!(runtime.contains_alias("embed/test").await);
}

#[tokio::test]
async fn test_builder_duplicate_alias_rejection() {
    let provider = MockProvider::embed_only();
    let spec1 = make_spec("embed/test", ModelTask::Embed, "mock/embed", "model1");
    let spec2 = make_spec("embed/test", ModelTask::Embed, "mock/embed", "model2");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec1, spec2])
        .build()
        .await;

    assert!(runtime.is_err());
    if let Err(err) = runtime {
        assert!(err.to_string().contains("Duplicate alias"));
    }
}

#[tokio::test]
async fn test_builder_invalid_alias_rejection() {
    let provider = MockProvider::embed_only();
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.alias = "no-slash".to_string(); // Invalid format

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_err());
    if let Err(err) = runtime {
        assert!(err.to_string().contains("task/name' format"));
    }
}

#[tokio::test]
async fn test_runtime_register() {
    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    assert!(!runtime.contains_alias("embed/test").await);

    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    runtime.register(spec).await.unwrap();

    assert!(runtime.contains_alias("embed/test").await);
}

#[tokio::test]
async fn test_runtime_register_duplicate_alias_rejected() {
    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    let spec1 = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    let spec2 = make_spec("embed/test", ModelTask::Embed, "mock/embed", "other-model");

    runtime.register(spec1).await.unwrap();
    let err = runtime.register(spec2).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("already exists"));
}

#[tokio::test]
async fn test_runtime_register_unknown_provider_rejected() {
    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    let spec = make_spec(
        "embed/test",
        ModelTask::Embed,
        "unknown/provider",
        "test-model",
    );
    let err = runtime.register(spec).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("Unknown provider"));
}

#[tokio::test]
async fn test_load_timeout_applies_to_provider_load() {
    let provider = MockProvider::embed_only().with_load_delay(2_000);
    let mut spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    spec.load_timeout = Some(1);

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let start = std::time::Instant::now();
    let result = runtime.embedding("embed/test").await;
    let elapsed = start.elapsed();

    assert!(matches!(
        result,
        Err(uni_xervo::error::RuntimeError::Timeout)
    ));
    assert!(elapsed.as_secs() < 2);
}

#[tokio::test]
async fn test_runtime_contains_alias() {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    assert!(runtime.contains_alias("embed/test").await);
    assert!(!runtime.contains_alias("embed/other").await);
}

#[tokio::test]
async fn test_resolve_embedding_model() {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let model = runtime.embedding("embed/test").await;
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.dimensions(), 384);
    assert_eq!(model.model_id(), "test-model");
}

#[tokio::test]
async fn test_resolve_generator_model() {
    let provider = MockProvider::generate_only();
    let spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/generate",
        "test-model",
    );

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let model = runtime.generator("generate/test").await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_resolve_reranker_model() {
    let provider = MockProvider::rerank_only();
    let spec = make_spec(
        "rerank/test",
        ModelTask::Rerank,
        "mock/rerank",
        "test-model",
    );

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let model = runtime.reranker("rerank/test").await;
    assert!(model.is_ok());
}

#[tokio::test]
async fn test_missing_provider_error() {
    let spec = make_spec("embed/test", ModelTask::Embed, "nonexistent", "test-model");

    let runtime = ModelRuntime::builder().catalog(vec![spec]).build().await;
    assert!(runtime.is_err());
    let err = runtime.err().unwrap();
    assert!(err.to_string().contains("Unknown provider"));
}

#[tokio::test]
async fn test_missing_alias_error() {
    let provider = MockProvider::embed_only();
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    let result = runtime.embedding("embed/nonexistent").await;
    assert!(result.is_err());
    if let Err(err) = result {
        assert!(err.to_string().contains("not found"));
    }
}

#[tokio::test]
async fn test_downcast_failure_wrong_model_type() {
    // Create a provider that returns an embedding model
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    // Try to resolve it as a generator (wrong type)
    let result = runtime.generator("embed/test").await;
    assert!(result.is_err());
    if let Err(err) = result {
        assert!(
            err.to_string()
                .contains("does not implement GeneratorModel")
        );
    }
}

#[tokio::test]
async fn test_multiple_providers() {
    let embed_provider = MockProvider::embed_only();
    let gen_provider = MockProvider::generate_only();

    let embed_spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "embed-model");
    let gen_spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/generate",
        "gen-model",
    );

    let runtime = ModelRuntime::builder()
        .register_provider(embed_provider)
        .register_provider(gen_provider)
        .catalog(vec![embed_spec, gen_spec])
        .build()
        .await
        .unwrap();

    assert!(runtime.embedding("embed/test").await.is_ok());
    assert!(runtime.generator("generate/test").await.is_ok());
}

#[tokio::test]
async fn test_prefetch_all_loads_all_models() {
    let provider = MockProvider::embed_only();
    let spec1 = make_spec("embed/a", ModelTask::Embed, "mock/embed", "model-a");
    let spec2 = make_spec("embed/b", ModelTask::Embed, "mock/embed", "model-b");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec1, spec2])
        .build()
        .await
        .unwrap();

    // Both models should load without error
    runtime.prefetch_all().await.unwrap();

    // Calling again is a no-op (already loaded)
    runtime.prefetch_all().await.unwrap();
}

#[tokio::test]
async fn test_prefetch_specific_aliases() {
    let embed_provider = MockProvider::embed_only();
    let gen_provider = MockProvider::generate_only();
    let spec1 = make_spec("embed/a", ModelTask::Embed, "mock/embed", "model-a");
    let spec2 = make_spec("embed/b", ModelTask::Embed, "mock/embed", "model-b");
    let spec3 = make_spec(
        "generate/c",
        ModelTask::Generate,
        "mock/generate",
        "model-c",
    );

    let runtime = ModelRuntime::builder()
        .register_provider(embed_provider)
        .register_provider(gen_provider)
        .catalog(vec![spec1, spec2, spec3])
        .build()
        .await
        .unwrap();

    // Only load embed/a and generate/c, skip embed/b
    runtime.prefetch(&["embed/a", "generate/c"]).await.unwrap();

    // embed/a and generate/c work; embed/b also works (lazy-loaded on access)
    assert!(runtime.embedding("embed/a").await.is_ok());
    assert!(runtime.generator("generate/c").await.is_ok());
    assert!(runtime.embedding("embed/b").await.is_ok());
}

#[tokio::test]
async fn test_prefetch_unknown_alias_errors() {
    let provider = MockProvider::embed_only();
    let spec = make_spec("embed/a", ModelTask::Embed, "mock/embed", "model-a");

    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let err = runtime.prefetch(&["embed/a", "embed/nonexistent"]).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("not found"));
}
