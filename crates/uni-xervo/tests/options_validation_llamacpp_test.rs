#![cfg(feature = "provider-llamacpp")]

use serde_json::{Value, json};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteLlamaCppProvider;
use uni_xervo::runtime::ModelRuntime;

fn llamacpp_spec(task: ModelTask, options: Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "remote/llamacpp".to_string(),
        model_id: "bge-small-en-v1.5".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

fn valid() -> Value {
    json!({
        "base_url": "http://127.0.0.1:8080/v1",
        "max_input_tokens": 512,
        "embedding_dimensions": 384
    })
}

async fn build(task: ModelTask, options: Value) -> Result<(), String> {
    ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![llamacpp_spec(task, options)])
        .build()
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn assert_rejects(result: Result<(), String>, needle: &str) {
    match result {
        Ok(()) => panic!("expected rejection containing {needle:?}, but build succeeded"),
        Err(msg) => assert!(msg.contains(needle), "{msg:?} lacks {needle:?}"),
    }
}

#[tokio::test]
async fn accepts_minimal_required_options() {
    assert!(build(ModelTask::Embed, valid()).await.is_ok());
}

#[tokio::test]
async fn accepts_all_options_together() {
    let options = json!({
        "base_url": "http://127.0.0.1:8080/v1/",
        "tokenizer_base_url": "http://127.0.0.1:8080",
        "max_input_tokens": 512,
        "embedding_dimensions": 384,
        "api_key_env": "RUSTIC_LOCAL_LLM_API_KEY",
        "request_timeout_secs": 30
    });
    assert!(build(ModelTask::Embed, options).await.is_ok());
}

#[tokio::test]
async fn rejects_null_options() {
    assert_rejects(
        build(ModelTask::Embed, Value::Null).await,
        "requires options",
    );
}

#[tokio::test]
async fn rejects_non_object_options() {
    assert_rejects(
        build(ModelTask::Embed, json!("nope")).await,
        "must be a JSON object",
    );
}

#[tokio::test]
async fn rejects_unknown_key() {
    let mut o = valid();
    o["pooling"] = json!("cls");
    assert_rejects(build(ModelTask::Embed, o).await, "Unknown option 'pooling'");
}

#[tokio::test]
async fn rejects_each_missing_required_key() {
    for key in ["base_url", "max_input_tokens", "embedding_dimensions"] {
        let mut o = valid();
        o.as_object_mut().unwrap().remove(key);
        assert_rejects(
            build(ModelTask::Embed, o).await,
            &format!("Option '{key}' for provider 'remote/llamacpp' is required"),
        );
    }
}

#[tokio::test]
async fn rejects_bad_base_url_forms() {
    for (value, needle) in [
        (json!(42), "must be a string"),
        (json!(""), "non-empty URL"),
        (json!("   "), "non-empty URL"),
        (json!("localhost:8080/v1"), "absolute http(s) URL"),
        (json!("/v1"), "absolute http(s) URL"),
    ] {
        let mut o = valid();
        o["base_url"] = value;
        assert_rejects(build(ModelTask::Embed, o).await, needle);
    }
}

#[tokio::test]
async fn rejects_bad_tokenizer_base_url_forms() {
    for (value, needle) in [
        (json!(7), "must be a string"),
        (json!(""), "non-empty URL"),
        (json!("127.0.0.1:8080"), "absolute http(s) URL"),
    ] {
        let mut o = valid();
        o["tokenizer_base_url"] = value;
        assert_rejects(build(ModelTask::Embed, o).await, needle);
    }
}

#[tokio::test]
async fn rejects_bad_max_input_tokens() {
    for (value, needle) in [
        (json!(0), "greater than 0"),
        (json!(2), "at least 3"),
        (json!("512"), "positive integer"),
        (json!(-1), "positive integer"),
        (json!(1.5), "positive integer"),
    ] {
        let mut o = valid();
        o["max_input_tokens"] = value;
        assert_rejects(build(ModelTask::Embed, o).await, needle);
    }
}

#[tokio::test]
async fn rejects_bad_embedding_dimensions() {
    for (value, needle) in [
        (json!(0), "greater than 0"),
        (json!("384"), "positive integer"),
    ] {
        let mut o = valid();
        o["embedding_dimensions"] = value;
        assert_rejects(build(ModelTask::Embed, o).await, needle);
    }
}

#[tokio::test]
async fn rejects_zero_request_timeout() {
    let mut o = valid();
    o["request_timeout_secs"] = json!(0);
    assert_rejects(build(ModelTask::Embed, o).await, "greater than 0");
}

#[tokio::test]
async fn rejects_non_string_api_key_env() {
    let mut o = valid();
    o["api_key_env"] = json!(42);
    assert_rejects(build(ModelTask::Embed, o).await, "must be a string");
}

#[tokio::test]
async fn rejects_non_embed_tasks() {
    for task in [ModelTask::Rerank, ModelTask::Generate, ModelTask::Raw] {
        assert_rejects(build(task, valid()).await, "only supports task 'embed'");
    }
}

#[tokio::test]
async fn rejects_multimodal_tasks() {
    assert_rejects(
        build(ModelTask::EmbedImage, valid()).await,
        "does not support task",
    );
}
