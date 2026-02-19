#![cfg(feature = "provider-anthropic")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteAnthropicProvider;
use uni_xervo::runtime::ModelRuntime;

fn anthropic_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "gen/default".to_string(),
        task: ModelTask::Generate,
        provider_id: "remote/anthropic".to_string(),
        model_id: "claude-sonnet-4-5-20250929".to_string(),
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
async fn builder_rejects_unknown_anthropic_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAnthropicProvider::new())
        .catalog(vec![anthropic_spec(serde_json::json!({"unknown": true}))])
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
async fn builder_rejects_invalid_anthropic_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAnthropicProvider::new())
        .catalog(vec![anthropic_spec(serde_json::json!({"api_key_env": 42}))])
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
async fn builder_rejects_invalid_anthropic_version_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAnthropicProvider::new())
        .catalog(vec![anthropic_spec(
            serde_json::json!({"anthropic_version": 123}),
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
async fn builder_accepts_valid_anthropic_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAnthropicProvider::new())
        .catalog(vec![anthropic_spec(serde_json::json!({
            "api_key_env": "MY_ANTHROPIC_KEY",
            "anthropic_version": "2023-06-01"
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_null_anthropic_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAnthropicProvider::new())
        .catalog(vec![anthropic_spec(serde_json::Value::Null)])
        .build()
        .await;

    assert!(runtime.is_ok());
}
