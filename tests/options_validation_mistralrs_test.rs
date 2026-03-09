#![cfg(feature = "provider-mistralrs")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistralrs_spec(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task,
        provider_id: "local/mistralrs".to_string(),
        model_id: "google/embeddinggemma-300m".to_string(),
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
async fn builder_rejects_unknown_mistralrs_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"unknown_key": true}),
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
async fn builder_accepts_dtype_f32() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": "f32"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_dtype_f16() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": "f16"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_dtype_bf16() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": "bf16"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_dtype_auto() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": "auto"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_dtype_value() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": "int8"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    let err = runtime.err().unwrap().to_string();
    assert!(
        err.contains("invalid value") || err.contains("dtype"),
        "Expected dtype error message, got: {err}"
    );
}

#[tokio::test]
async fn builder_rejects_non_string_dtype() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::json!({"dtype": 32}),
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
async fn builder_accepts_null_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            ModelTask::Embed,
            serde_json::Value::Null,
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}
