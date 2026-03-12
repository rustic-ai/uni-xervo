#![cfg(feature = "provider-mistralrs")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistralrs_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/mistralrs".to_string(),
        model_id: "test-model".to_string(),
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
async fn builder_accepts_valid_dtype_option() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "f32"}))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_dtype_auto() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "auto"}))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_dtype_value() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "int8"}))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

#[tokio::test]
async fn builder_rejects_non_string_dtype() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": 16}))])
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
async fn builder_accepts_dtype_with_other_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            serde_json::json!({"dtype": "f32", "force_cpu": true}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}
