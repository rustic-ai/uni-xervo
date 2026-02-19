#![cfg(feature = "provider-mistral")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteMistralProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistral_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "remote/mistral".to_string(),
        model_id: "mistral-embed".to_string(),
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
async fn builder_rejects_unknown_mistral_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(serde_json::json!({"unknown": true}))])
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
async fn builder_rejects_invalid_mistral_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(serde_json::json!({"api_key_env": 42}))])
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
async fn builder_accepts_valid_mistral_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(serde_json::json!({
            "api_key_env": "MY_MISTRAL_KEY"
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_null_mistral_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(serde_json::Value::Null)])
        .build()
        .await;

    assert!(runtime.is_ok());
}
