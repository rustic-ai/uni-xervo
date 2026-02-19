#![cfg(feature = "provider-voyageai")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteVoyageAIProvider;
use uni_xervo::runtime::ModelRuntime;

fn voyageai_spec(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "remote/voyageai".to_string(),
        model_id: "voyage-3".to_string(),
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
async fn builder_rejects_unknown_voyageai_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVoyageAIProvider::new())
        .catalog(vec![voyageai_spec(
            ModelTask::Embed,
            serde_json::json!({"unknown": true}),
        )])
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
async fn builder_rejects_invalid_voyageai_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVoyageAIProvider::new())
        .catalog(vec![voyageai_spec(
            ModelTask::Embed,
            serde_json::json!({"api_key_env": 42}),
        )])
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
async fn builder_accepts_valid_voyageai_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVoyageAIProvider::new())
        .catalog(vec![voyageai_spec(
            ModelTask::Embed,
            serde_json::json!({"api_key_env": "MY_VOYAGE_KEY"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_null_voyageai_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVoyageAIProvider::new())
        .catalog(vec![voyageai_spec(
            ModelTask::Rerank,
            serde_json::Value::Null,
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}
