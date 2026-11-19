#![cfg(feature = "provider-mistral")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteMistralProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistral_spec(options: serde_json::Value) -> ModelAliasSpec {
    mistral_spec_task(ModelTask::Embed, options)
}

fn mistral_spec_task(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
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

#[tokio::test]
async fn builder_accepts_valid_embedding_dimensions() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(
            serde_json::json!({"embedding_dimensions": 512}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_zero_embedding_dimensions() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(
            serde_json::json!({"embedding_dimensions": 0}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be greater than 0")
    );
}

#[tokio::test]
async fn builder_rejects_string_embedding_dimensions() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec(
            serde_json::json!({"embedding_dimensions": "big"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a positive integer")
    );
}

#[tokio::test]
async fn builder_rejects_embedding_dimensions_for_generate() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteMistralProvider::new())
        .catalog(vec![mistral_spec_task(
            ModelTask::Generate,
            serde_json::json!({"embedding_dimensions": 512}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("applies only to task 'embed'")
    );
}
