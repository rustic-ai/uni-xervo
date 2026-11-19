#![cfg(feature = "provider-gemini")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteGeminiProvider;
use uni_xervo::runtime::ModelRuntime;

fn gemini_spec(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "remote/gemini".to_string(),
        model_id: "embedding-001".to_string(),
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
async fn builder_rejects_unknown_gemini_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
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
async fn builder_accepts_valid_gemini_api_version() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
            ModelTask::Embed,
            serde_json::json!({"api_version": "v1"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_gemini_api_version_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
            ModelTask::Embed,
            serde_json::json!({"api_version": 123}),
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
async fn builder_accepts_valid_gemini_embedding_dimensions() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
            ModelTask::Embed,
            serde_json::json!({"embedding_dimensions": 256}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_zero_gemini_embedding_dimensions() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
            ModelTask::Embed,
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
async fn builder_rejects_gemini_embedding_dimensions_for_generate() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(
            ModelTask::Generate,
            serde_json::json!({"embedding_dimensions": 256}),
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

#[tokio::test]
async fn builder_accepts_null_gemini_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![gemini_spec(ModelTask::Embed, serde_json::Value::Null)])
        .build()
        .await;

    assert!(runtime.is_ok());
}
