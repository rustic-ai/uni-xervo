#![cfg(feature = "provider-cohere")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteCohereProvider;
use uni_xervo::runtime::ModelRuntime;

fn cohere_spec(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "remote/cohere".to_string(),
        model_id: "embed-english-v3.0".to_string(),
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
async fn builder_rejects_unknown_cohere_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![cohere_spec(
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
async fn builder_rejects_invalid_cohere_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![cohere_spec(
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
async fn builder_rejects_invalid_cohere_input_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![cohere_spec(
            ModelTask::Embed,
            serde_json::json!({"input_type": 123}),
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
async fn builder_accepts_valid_cohere_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![cohere_spec(
            ModelTask::Embed,
            serde_json::json!({
                "api_key_env": "MY_COHERE_KEY",
                "input_type": "search_document"
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_null_cohere_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![cohere_spec(
            ModelTask::Rerank,
            serde_json::Value::Null,
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}
