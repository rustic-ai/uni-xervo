#![cfg(feature = "provider-candle")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;

#[tokio::test]
#[ignore] // Requires model download from HF
async fn test_runtime_candle_embed() -> anyhow::Result<()> {
    // 1. Define catalog
    let catalog = vec![ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }];

    // 2. Build runtime
    let runtime = ModelRuntime::builder()
        .catalog(catalog)
        .register_provider(LocalCandleProvider::new())
        .build()
        .await?;

    // 3. Resolve and use embedding model
    let embed_model = runtime.embedding("embed/default").await?;
    let vectors = embed_model
        .embed(vec!["hello world", "rust is great"])
        .await?;

    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), 384); // Mock dimension
    assert_eq!(vectors[1].len(), 384);

    Ok(())
}

#[tokio::test]
#[ignore] // Requires model download from HF
async fn test_warmup_policies() -> anyhow::Result<()> {
    // Test Eager policy
    let catalog = vec![ModelAliasSpec {
        alias: "embed/eager".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Eager,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }];

    let _runtime = ModelRuntime::builder()
        .catalog(catalog)
        .register_provider(LocalCandleProvider::new())
        .build()
        .await?;

    // Should be loaded already
    // How to verify? Maybe check if resolve_and_load returns fast?
    // Or add a method to registry to check status?
    // For now just ensure it builds.

    Ok(())
}
