#![cfg(feature = "provider-azure-openai")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteAzureOpenAIProvider;
use uni_xervo::runtime::ModelRuntime;

fn azure_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "remote/azure-openai".to_string(),
        model_id: "text-embedding-ada-002".to_string(),
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
async fn builder_rejects_unknown_azure_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![azure_spec(serde_json::json!({"unknown": true}))])
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
async fn builder_rejects_invalid_azure_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![azure_spec(serde_json::json!({"resource_name": 42}))])
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
async fn builder_rejects_invalid_azure_api_version_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![azure_spec(serde_json::json!({"api_version": false}))])
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
async fn builder_accepts_valid_azure_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![azure_spec(serde_json::json!({
            "api_key_env": "MY_AZURE_KEY",
            "resource_name": "my-resource",
            "api_version": "2024-10-21"
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_null_azure_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![azure_spec(serde_json::Value::Null)])
        .build()
        .await;

    assert!(runtime.is_ok());
}
