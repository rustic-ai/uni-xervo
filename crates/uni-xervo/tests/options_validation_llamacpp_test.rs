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
async fn rejects_unsupported_tasks() {
    assert_rejects(
        build(ModelTask::Raw, valid()).await,
        "supports tasks 'embed', 'generate', 'rerank' and 'document_extract'",
    );
}

// --- generate -------------------------------------------------------------

/// Generation needs nothing but the endpoint — in particular no option for
/// vision, which depends on the server having been started with `--mmproj`.
#[tokio::test]
async fn generate_accepts_base_url_alone() {
    assert!(
        build(
            ModelTask::Generate,
            json!({ "base_url": "http://127.0.0.1:8080/v1" })
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn generate_accepts_the_shared_options() {
    assert!(
        build(
            ModelTask::Generate,
            json!({
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key_env": "LLAMACPP_KEY",
                "request_timeout_secs": 120
            })
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn generate_still_requires_base_url() {
    for options in [
        json!({}),
        Value::Null,
        json!({ "request_timeout_secs": 30 }),
    ] {
        assert_rejects(build(ModelTask::Generate, options).await, "base_url");
    }
}

#[tokio::test]
async fn generate_rejects_each_embed_only_option() {
    for (key, value) in [
        ("max_input_tokens", json!(512)),
        ("embedding_dimensions", json!(384)),
        ("tokenizer_base_url", json!("http://127.0.0.1:8080")),
    ] {
        let mut options = json!({ "base_url": "http://127.0.0.1:8080/v1" });
        options[key] = value;
        assert_rejects(
            build(ModelTask::Generate, options).await,
            "applies only to task 'embed'",
        );
    }
}

// --- rerank ---------------------------------------------------------------

/// Rerank needs nothing but the endpoint — in particular no option for the
/// response shape, which follows from the request key this crate sends.
#[tokio::test]
async fn rerank_accepts_base_url_alone() {
    assert!(
        build(
            ModelTask::Rerank,
            json!({ "base_url": "http://127.0.0.1:8081/v1" })
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn rerank_rejects_each_embed_only_option() {
    for (key, value) in [
        ("max_input_tokens", json!(512)),
        ("embedding_dimensions", json!(384)),
        ("tokenizer_base_url", json!("http://127.0.0.1:8081")),
    ] {
        let mut options = json!({ "base_url": "http://127.0.0.1:8081/v1" });
        options[key] = value;
        assert_rejects(
            build(ModelTask::Rerank, options).await,
            "applies only to task 'embed'",
        );
    }
}

#[tokio::test]
async fn rerank_rejects_a_response_format_option() {
    let mut options = json!({ "base_url": "http://127.0.0.1:8081/v1" });
    options["rerank_format"] = json!("tei");
    assert_rejects(build(ModelTask::Rerank, options).await, "Unknown option");
}

#[tokio::test]
async fn generate_rejects_unknown_keys_and_bad_shared_values() {
    let base = json!({ "base_url": "http://127.0.0.1:8080/v1" });

    let mut options = base.clone();
    options["pipeline"] = json!("vision");
    assert_rejects(build(ModelTask::Generate, options).await, "Unknown option");

    let mut options = base.clone();
    options["request_timeout_secs"] = json!(0);
    assert_rejects(build(ModelTask::Generate, options).await, "greater than 0");

    let mut options = base;
    options["base_url"] = json!("127.0.0.1:8080/v1");
    assert_rejects(build(ModelTask::Generate, options).await, "base_url");
}

#[tokio::test]
async fn rejects_multimodal_tasks() {
    assert_rejects(
        build(ModelTask::EmbedImage, valid()).await,
        "does not support task",
    );
}

// --- document_extract -----------------------------------------------------

/// `style` defaults to `olmocr`, so the endpoint alone is a valid alias.
#[tokio::test]
async fn document_extract_accepts_base_url_alone() {
    assert!(
        build(
            ModelTask::DocumentExtract,
            json!({ "base_url": "http://127.0.0.1:8082/v1" })
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn document_extract_accepts_every_documented_style() {
    for style in ["granite-docling", "mineru", "olmocr"] {
        let options = json!({ "base_url": "http://127.0.0.1:8082/v1", "style": style });
        assert!(
            build(ModelTask::DocumentExtract, options).await.is_ok(),
            "{style} must be accepted"
        );
    }
}

/// A typo fails when the catalog is read rather than on the first extraction,
/// and the message names the accepted values.
#[tokio::test]
async fn document_extract_rejects_an_unknown_style() {
    let options = json!({ "base_url": "http://127.0.0.1:8082/v1", "style": "docling" });
    assert_rejects(
        build(ModelTask::DocumentExtract, options).await,
        "must be one of granite-docling, mineru, olmocr",
    );
}

#[tokio::test]
async fn document_extract_rejects_a_non_string_style() {
    let options = json!({ "base_url": "http://127.0.0.1:8082/v1", "style": 3 });
    assert_rejects(
        build(ModelTask::DocumentExtract, options).await,
        "must be a string",
    );
}

#[tokio::test]
async fn document_extract_rejects_each_embed_only_option() {
    for (key, value) in [
        ("max_input_tokens", json!(512)),
        ("embedding_dimensions", json!(384)),
        ("tokenizer_base_url", json!("http://127.0.0.1:8082")),
    ] {
        let mut options = json!({ "base_url": "http://127.0.0.1:8082/v1" });
        options[key] = value;
        assert_rejects(
            build(ModelTask::DocumentExtract, options).await,
            "applies only to task 'embed'",
        );
    }
}

/// And the other way round: `style` would be silently ignored by every other
/// task, which is what this gate exists to prevent.
#[tokio::test]
async fn style_is_rejected_on_the_other_tasks() {
    for task in [ModelTask::Embed, ModelTask::Generate, ModelTask::Rerank] {
        let mut options = match task {
            ModelTask::Embed => valid(),
            _ => json!({ "base_url": "http://127.0.0.1:8080/v1" }),
        };
        options["style"] = json!("olmocr");
        assert_rejects(
            build(task, options).await,
            "applies only to task 'document_extract'",
        );
    }
}
