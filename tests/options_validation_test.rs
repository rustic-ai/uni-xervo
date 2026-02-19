#![cfg(feature = "provider-candle")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;

fn candle_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "all-minilm-l6-v2".to_string(),
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
async fn builder_rejects_unknown_candle_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![candle_spec(serde_json::json!({"unknown": true}))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Unknown option")
    );
}

#[tokio::test]
async fn builder_rejects_invalid_candle_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![candle_spec(serde_json::json!({"cache_dir": 123}))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a string")
    );
}

#[tokio::test]
async fn builder_accepts_valid_candle_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![candle_spec(
            serde_json::json!({"cache_dir": "/tmp/models"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn register_rejects_unknown_candle_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    let err = runtime
        .register(candle_spec(serde_json::json!({"unknown": true})))
        .await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("Unknown option"));
}
