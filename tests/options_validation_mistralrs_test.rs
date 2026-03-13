#![cfg(feature = "provider-mistralrs")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistralrs_spec(options: serde_json::Value) -> ModelAliasSpec {
    mistralrs_spec_with_task(ModelTask::Embed, options)
}

fn mistralrs_spec_with_task(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
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

// ---------------------------------------------------------------------------
// Pipeline validation tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_accepts_valid_pipeline_vision() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_valid_pipeline_diffusion() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_valid_pipeline_speech() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "audio"}),
        )])
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
async fn builder_rejects_gguf_for_vision_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision", "gguf_files": ["model.gguf"]}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("not supported for the vision pipeline")
    );
}

#[tokio::test]
async fn builder_accepts_diffusion_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "diffusion_loader_type": "flux"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_diffusion_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "diffusion_loader_type": "invalid"}),
        )])
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
async fn builder_rejects_isq_for_diffusion_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "isq": "Q4K"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("not supported for the diffusion pipeline")
    );
}

#[tokio::test]
async fn builder_accepts_speech_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech", "speech_loader_type": "dia"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_speech_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech", "speech_loader_type": "invalid"}),
        )])
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
