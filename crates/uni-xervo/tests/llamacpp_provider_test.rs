//! Mock-server tests for the `remote/llamacpp` provider.
//!
//! Every test spins up an in-process fake `llama-server` (see
//! `common::mock_http`) and drives the provider through a real `ModelRuntime`,
//! so catalog validation, loading, and the tokenize → embed pipeline are all
//! exercised over HTTP. Budgets are kept tiny (`max_input_tokens = 10`,
//! 4 dimensions) so boundaries are cheap to spell out.

#![cfg(feature = "provider-llamacpp")]

mod common;

use common::mock_http::{
    CLS, EXCEED_CONTEXT_MESSAGE, MockLlamaServer, NOT_SUPPORTED_MESSAGE, RouteResponse, SEP,
    TOO_LARGE_MESSAGE, char_token, chat_completion, chat_echo, embeddings_echo, jina_rerank,
    llama_error, rerank_by_position, tei_rerank, tokenize_by_chars,
};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::RemoteLlamaCppProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{
    ContentBlock, DocBlockKind, DocExtractOptions, DocOutputFormat, DocumentExtractionModel,
    EmbedResult, EmbeddingModel, GenerationOptions, GeneratorModel, ImageInput, Message,
    MessageRole, RerankerModel,
};

const MAX_TOKENS: u64 = 10;
const DIMS: u64 = 4;
const MODEL_ID: &str = "bge-small-en-v1.5";

static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn spec(task: ModelTask, options: Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/llamacpp".to_string(),
        task,
        provider_id: "remote/llamacpp".to_string(),
        model_id: MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

fn base_options(server: &MockLlamaServer) -> Value {
    json!({
        "base_url": server.base_url(),
        "max_input_tokens": MAX_TOKENS,
        "embedding_dimensions": DIMS,
    })
}

async fn embedder_with(options: Value) -> Arc<dyn EmbeddingModel> {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec(ModelTask::Embed, options)])
        .build()
        .await
        .expect("runtime builds");
    runtime
        .embedding("embed/llamacpp")
        .await
        .expect("model loads")
}

async fn embedder(server: &MockLlamaServer) -> Arc<dyn EmbeddingModel> {
    embedder_with(base_options(server)).await
}

fn embeddings_input(server: &MockLlamaServer) -> Vec<Value> {
    let reqs = server.requests_to("/v1/embeddings");
    assert_eq!(reqs.len(), 1, "expected exactly one embeddings request");
    reqs[0].body["input"]
        .as_array()
        .expect("input is an array")
        .clone()
}

fn ok(result: uni_xervo::error::Result<EmbedResult>) -> EmbedResult {
    match result {
        Ok(r) => r,
        Err(e) => panic!("expected Ok, got {e:?}"),
    }
}

// ---------------------------------------------------------------------------
// Input planning
// ---------------------------------------------------------------------------

#[tokio::test]
async fn short_text_skips_tokenize_and_is_sent_as_string() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;

    let r = ok(model.embed(&["hello"]).await);
    assert_eq!(r.vectors.len(), 1);
    assert_eq!(r.vectors[0].len(), DIMS as usize);

    assert!(server.requests_to("/tokenize").is_empty());
    assert_eq!(embeddings_input(&server), vec![json!("hello")]);
}

#[tokio::test]
async fn exactly_max_minus_two_chars_skips_tokenize() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;
    let text = "12345678"; // 8 chars == max - 2

    ok(model.embed(&[text]).await);
    assert!(server.requests_to("/tokenize").is_empty());
    assert_eq!(embeddings_input(&server), vec![json!(text)]);
}

#[tokio::test]
async fn max_minus_one_chars_tokenizes_and_truncates_to_token_array() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;
    let text = "123456789"; // 9 chars → 11 tokens with specials > 10

    ok(model.embed(&[text]).await);

    let tok = server.requests_to("/tokenize");
    assert_eq!(tok.len(), 1);
    assert_eq!(tok[0].body["content"], json!(text));
    assert_eq!(tok[0].body["add_special"], json!(true));
    assert_eq!(tok[0].body["parse_special"], json!(false));
    // Router mode dispatches /tokenize by model name.
    assert_eq!(tok[0].body["model"], json!(MODEL_ID));

    let input = embeddings_input(&server);
    let ids: Vec<u32> = serde_json::from_value(input[0].clone()).expect("token array");
    assert_eq!(ids.len(), MAX_TOKENS as usize);
    assert_eq!(ids[0], CLS);
    assert_eq!(*ids.last().unwrap(), SEP);
    let expected_content: Vec<u32> = (0..8).map(char_token).collect();
    assert_eq!(&ids[1..9], expected_content.as_slice());
}

#[tokio::test]
async fn text_that_needs_tokenize_but_fits_is_sent_as_string() {
    let server = MockLlamaServer::start().await;
    // Tokenizer says: exactly 10 tokens (fits) regardless of input.
    server.on_tokenize(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "tokens": (0..10).collect::<Vec<u32>>() }),
    });
    let model = embedder(&server).await;
    let text = "123456789";

    ok(model.embed(&[text]).await);
    assert_eq!(server.requests_to("/tokenize").len(), 1);
    assert_eq!(embeddings_input(&server), vec![json!(text)]);
}

#[tokio::test]
async fn exactly_max_plus_one_tokens_truncates() {
    let server = MockLlamaServer::start().await;
    server.on_tokenize(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "tokens": (0..11).collect::<Vec<u32>>() }),
    });
    let model = embedder(&server).await;

    ok(model.embed(&["123456789"]).await);
    let input = embeddings_input(&server);
    assert_eq!(input[0], json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]));
}

#[tokio::test]
async fn char_count_not_byte_length_drives_tokenize_decision() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;

    let cjk_8 = "你好世界你好世界"; // 8 chars, 24 bytes → no tokenize
    let cjk_9 = "你好世界你好世界你"; // 9 chars → tokenize
    let emoji_4 = "🦀🦀🦀🦀"; // 4 chars, 16 bytes → no tokenize
    let accented = "héllo wörld"; // 11 chars → tokenize

    ok(model.embed(&[cjk_8, cjk_9, emoji_4, accented]).await);

    let tok = server.requests_to("/tokenize");
    let contents: Vec<&str> = tok
        .iter()
        .map(|r| r.body["content"].as_str().unwrap())
        .collect();
    assert_eq!(tok.len(), 2);
    assert!(contents.contains(&cjk_9));
    assert!(contents.contains(&accented));

    let input = embeddings_input(&server);
    assert_eq!(input[0], json!(cjk_8));
    assert!(input[1].is_array());
    assert_eq!(input[2], json!(emoji_4));
    assert!(input[3].is_array());
}

#[tokio::test]
async fn empty_string_is_embedded_without_tokenize() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;

    let r = ok(model.embed(&[""]).await);
    assert_eq!(r.vectors.len(), 1);
    assert!(server.requests_to("/tokenize").is_empty());
    assert_eq!(embeddings_input(&server), vec![json!("")]);
}

#[tokio::test]
async fn empty_slice_makes_no_requests() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;

    let r = ok(model.embed(&[]).await);
    assert!(r.vectors.is_empty());
    assert!(r.usage.is_none());
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn batch_preserves_order_with_mixed_short_and_long_inputs() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;
    let long_a = "a".repeat(40);
    let long_b = "b".repeat(30);

    let r = ok(model.embed(&["short", &long_a, "mid", &long_b]).await);

    // Wire shape: [string, array, string, array].
    let input = embeddings_input(&server);
    assert_eq!(input.len(), 4);
    assert_eq!(input[0], json!("short"));
    assert!(input[1].is_array());
    assert_eq!(input[2], json!("mid"));
    assert!(input[3].is_array());
    assert_eq!(server.requests_to("/tokenize").len(), 2);

    // Echo handler encodes the position into each vector.
    assert_eq!(r.vectors.len(), 4);
    for (i, v) in r.vectors.iter().enumerate() {
        assert!(v.iter().all(|x| *x == i as f32), "vector {i} = {v:?}");
    }
    let usage = r.usage.expect("usage propagated");
    assert_eq!(usage.prompt_tokens, 12);
    assert_eq!(usage.total_tokens, 12);
    assert_eq!(usage.completion_tokens, 0);
}

#[tokio::test]
async fn request_body_carries_model_id() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;
    ok(model.embed(&["x"]).await);
    let reqs = server.requests_to("/v1/embeddings");
    assert_eq!(reqs[0].body["model"], json!(MODEL_ID));
    assert_eq!(reqs[0].header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// Response decoding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn out_of_order_index_is_honoured() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "data": [
            { "index": 1, "embedding": [1.0, 1.0, 1.0, 1.0] },
            { "index": 0, "embedding": [0.0, 0.0, 0.0, 0.0] },
        ]}),
    });
    let model = embedder(&server).await;
    let r = ok(model.embed(&["a", "b"]).await);
    assert_eq!(r.vectors[0], vec![0.0; 4]);
    assert_eq!(r.vectors[1], vec![1.0; 4]);
}

#[tokio::test]
async fn missing_index_falls_back_to_positional() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "data": [
            { "embedding": [5.0, 5.0, 5.0, 5.0] },
            { "embedding": [6.0, 6.0, 6.0, 6.0] },
        ]}),
    });
    let model = embedder(&server).await;
    let r = ok(model.embed(&["a", "b"]).await);
    assert_eq!(r.vectors[0], vec![5.0; 4]);
    assert_eq!(r.vectors[1], vec![6.0; 4]);
    assert!(r.usage.is_none());
}

#[tokio::test]
async fn wrong_dimension_is_an_error_naming_the_option() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "data": [{ "index": 0, "embedding": [1.0, 2.0, 3.0] }] }),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::ApiError(_)), "{err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("3-dimensional") && msg.contains("expected 4"),
        "{msg}"
    );
    assert!(msg.contains("embedding_dimensions"), "{msg}");
}

#[tokio::test]
async fn missing_data_count_mismatch_and_non_numeric_are_api_errors() {
    let cases: Vec<(Value, &str)> = vec![
        (json!({ "object": "list" }), "missing 'data'"),
        (
            json!({ "data": [{ "index": 0, "embedding": [1.0, 1.0, 1.0, 1.0] }] }),
            "expected 2 embeddings, got 1",
        ),
        (
            json!({ "data": [
                { "index": 0, "embedding": [1.0, "x", 1.0, 1.0] },
                { "index": 1, "embedding": [1.0, 1.0, 1.0, 1.0] },
            ]}),
            "is not a number",
        ),
    ];
    for (body, needle) in cases {
        let server = MockLlamaServer::start().await;
        let b = body.clone();
        server.on_embeddings(move |_| RouteResponse::Json {
            status: 200,
            body: b.clone(),
        });
        let model = embedder(&server).await;
        let err = model.embed(&["a", "b"]).await.expect_err("must fail");
        assert!(matches!(err, RuntimeError::ApiError(_)), "{err:?}");
        assert!(err.to_string().contains(needle), "{err} lacks {needle}");
    }
}

#[tokio::test]
async fn malformed_json_body_is_an_api_error() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Raw {
        status: 200,
        body: "{not json".to_string(),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::ApiError(_)), "{err:?}");
    assert!(err.to_string().contains("malformed JSON"), "{err}");
}

#[tokio::test]
async fn tokenize_response_without_tokens_array_is_an_api_error() {
    let server = MockLlamaServer::start().await;
    server.on_tokenize(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "tokens": [1, "two", 3] }),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["123456789"]).await.expect_err("must fail");
    assert!(err.to_string().contains("/tokenize"), "{err}");
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_500_too_large_maps_to_inference_error_not_retryable() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 500,
        body: llama_error(500, TOO_LARGE_MESSAGE, "server_error"),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
    assert!(!err.is_retryable());
    let msg = err.to_string();
    assert!(msg.contains("max_input_tokens=10"), "{msg}");
    assert!(msg.contains("too large to process"), "{msg}");
}

#[tokio::test]
async fn http_400_exceed_context_maps_to_inference_error() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 400,
        body: llama_error(400, EXCEED_CONTEXT_MESSAGE, "exceed_context_size"),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
}

#[tokio::test]
async fn other_http_errors_map_like_remote_providers() {
    let cases: Vec<(u16, Value, fn(&RuntimeError) -> bool)> = vec![
        (500, llama_error(500, "boom", "server_error"), |e| {
            matches!(e, RuntimeError::Unavailable)
        }),
        (503, json!({}), |e| matches!(e, RuntimeError::Unavailable)),
        (
            401,
            llama_error(401, "Invalid API Key", "authentication_error"),
            |e| matches!(e, RuntimeError::Unauthorized),
        ),
        (429, json!({}), |e| matches!(e, RuntimeError::RateLimited)),
        (
            404,
            llama_error(404, "no route", "not_found_error"),
            |e| matches!(e, RuntimeError::ApiError(m) if m.contains("404") && m.contains("no route")),
        ),
    ];
    for (status, body, check) in cases {
        let server = MockLlamaServer::start().await;
        let b = body.clone();
        server.on_embeddings(move |_| RouteResponse::Json {
            status,
            body: b.clone(),
        });
        let model = embedder(&server).await;
        let err = model.embed(&["a"]).await.expect_err("must fail");
        assert!(check(&err), "status {status}: {err:?}");
    }
}

#[tokio::test]
async fn tokenize_route_failures_are_mapped_too() {
    let server = MockLlamaServer::start().await;
    server.on_tokenize(|_| RouteResponse::Json {
        status: 500,
        body: llama_error(500, TOO_LARGE_MESSAGE, "server_error"),
    });
    let model = embedder(&server).await;
    let err = model.embed(&["123456789"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
    assert!(server.requests_to("/v1/embeddings").is_empty());

    server.on_tokenize(|_| RouteResponse::Json {
        status: 401,
        body: json!({}),
    });
    let err = model.embed(&["123456789"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::Unauthorized), "{err:?}");
}

#[tokio::test]
async fn hanging_embeddings_route_times_out() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Hang);
    let mut options = base_options(&server);
    options["request_timeout_secs"] = json!(1);
    let model = embedder_with(options).await;

    let started = Instant::now();
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::Timeout), "{err:?}");
    assert!(err.is_retryable());
    let elapsed = started.elapsed();
    assert!(elapsed >= Duration::from_millis(900), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

#[tokio::test]
async fn hanging_tokenize_route_times_out() {
    let server = MockLlamaServer::start().await;
    server.on_tokenize(|_| RouteResponse::Hang);
    let mut options = base_options(&server);
    options["request_timeout_secs"] = json!(1);
    let model = embedder_with(options).await;

    let err = model.embed(&["123456789"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::Timeout), "{err:?}");
    assert!(server.requests_to("/v1/embeddings").is_empty());
}

#[tokio::test]
async fn input_size_rejections_do_not_trip_the_circuit_breaker() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 500,
        body: llama_error(500, TOO_LARGE_MESSAGE, "server_error"),
    });
    let model = embedder(&server).await;

    for _ in 0..6 {
        let err = model.embed(&["a"]).await.expect_err("must fail");
        assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
    }
    // Every call reached the server: the breaker stayed closed.
    assert_eq!(server.requests_to("/v1/embeddings").len(), 6);
}

#[tokio::test]
async fn plain_server_errors_do_trip_the_circuit_breaker() {
    let server = MockLlamaServer::start().await;
    server.on_embeddings(|_| RouteResponse::Json {
        status: 500,
        body: llama_error(500, "boom", "server_error"),
    });
    let model = embedder(&server).await;

    for _ in 0..5 {
        let err = model.embed(&["a"]).await.expect_err("must fail");
        assert!(matches!(err, RuntimeError::Unavailable), "{err:?}");
    }
    let err = model.embed(&["a"]).await.expect_err("must fail");
    assert!(matches!(err, RuntimeError::Unavailable), "{err:?}");
    // Sixth call was short-circuited without touching the server.
    assert_eq!(server.requests_to("/v1/embeddings").len(), 5);
}

// ---------------------------------------------------------------------------
// Auth and URLs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_api_key_env_means_no_authorization_header() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await;
    ok(model.embed(&["short", "123456789"]).await);
    for req in server.requests() {
        assert!(req.header("authorization").is_none(), "{req:?}");
    }
}

#[tokio::test]
async fn api_key_env_sends_bearer_on_both_routes() {
    let _lock = ENV_LOCK.lock().await;
    // SAFETY: protected by ENV_LOCK
    unsafe { std::env::set_var("LLAMACPP_TEST_API_KEY", "sekrit") };

    let server = MockLlamaServer::start().await;
    let mut options = base_options(&server);
    options["api_key_env"] = json!("LLAMACPP_TEST_API_KEY");
    let model = embedder_with(options).await;
    ok(model.embed(&["short", "123456789"]).await);

    let tok = server.requests_to("/tokenize");
    let emb = server.requests_to("/v1/embeddings");
    assert_eq!(tok.len(), 1);
    assert_eq!(emb.len(), 1);
    assert_eq!(tok[0].header("authorization"), Some("Bearer sekrit"));
    assert_eq!(emb[0].header("authorization"), Some("Bearer sekrit"));

    // SAFETY: protected by ENV_LOCK
    unsafe { std::env::remove_var("LLAMACPP_TEST_API_KEY") };
}

#[tokio::test]
async fn api_key_env_missing_fails_at_load() {
    let _lock = ENV_LOCK.lock().await;
    // SAFETY: protected by ENV_LOCK
    unsafe { std::env::remove_var("LLAMACPP_TEST_MISSING_KEY") };

    let server = MockLlamaServer::start().await;
    let mut options = base_options(&server);
    options["api_key_env"] = json!("LLAMACPP_TEST_MISSING_KEY");
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec(ModelTask::Embed, options)])
        .build()
        .await
        .expect("validation passes; env is checked at load");
    let err = match runtime.embedding("embed/llamacpp").await {
        Ok(_) => panic!("load must fail"),
        Err(e) => e,
    };
    assert!(matches!(err, RuntimeError::Config(_)), "{err:?}");
    assert!(
        err.to_string().contains("LLAMACPP_TEST_MISSING_KEY"),
        "{err}"
    );
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn tokenizer_base_url_defaults_to_base_url_without_v1() {
    let server = MockLlamaServer::start().await;
    let model = embedder(&server).await; // base_url = {root}/v1, no tokenizer_base_url
    ok(model.embed(&["123456789"]).await);
    assert_eq!(server.requests_to("/tokenize").len(), 1);
    assert_eq!(server.requests_to("/v1/embeddings").len(), 1);
}

#[tokio::test]
async fn explicit_tokenizer_base_url_routes_tokenize_elsewhere() {
    let embed_server = MockLlamaServer::start().await;
    let tok_server = MockLlamaServer::start().await;
    tok_server.on_tokenize(tokenize_by_chars());
    embed_server.on_embeddings(embeddings_echo(DIMS as usize));

    let mut options = base_options(&embed_server);
    options["tokenizer_base_url"] = json!(tok_server.root_url());
    let model = embedder_with(options).await;
    ok(model.embed(&["123456789"]).await);

    assert_eq!(tok_server.requests_to("/tokenize").len(), 1);
    assert!(tok_server.requests_to("/v1/embeddings").is_empty());
    assert!(embed_server.requests_to("/tokenize").is_empty());
    assert_eq!(embed_server.requests_to("/v1/embeddings").len(), 1);
}

// ---------------------------------------------------------------------------
// Capability
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unsupported_task_is_rejected_at_catalog_validation() {
    let server = MockLlamaServer::start().await;
    let result = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec(ModelTask::Raw, base_options(&server))])
        .build()
        .await;
    let err = match result {
        Ok(_) => panic!("must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("supports tasks 'embed', 'generate', 'rerank' and 'document_extract'"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Generation — /v1/chat/completions
// ---------------------------------------------------------------------------

const GEN_ALIAS: &str = "gen/llamacpp";
const GEN_MODEL_ID: &str = "SmolVLM-500M-Instruct";

/// A generation alias. Note the absence of `max_input_tokens` and
/// `embedding_dimensions`: they are embed-only, and the validator rejects them
/// here (see `generate_alias_rejects_embed_only_options`).
fn gen_spec(server: &MockLlamaServer) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: GEN_ALIAS.to_string(),
        task: ModelTask::Generate,
        provider_id: "remote/llamacpp".to_string(),
        model_id: GEN_MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({ "base_url": server.base_url() }),
    }
}

async fn generator(server: &MockLlamaServer) -> Arc<dyn GeneratorModel> {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![gen_spec(server)])
        .build()
        .await
        .expect("runtime builds");
    runtime.generator(GEN_ALIAS).await.expect("generator loads")
}

#[tokio::test]
async fn generation_posts_to_chat_completions_and_maps_text_and_usage() {
    let server = MockLlamaServer::start().await;
    server.on_chat(|_| RouteResponse::Json {
        status: 200,
        body: chat_completion("the whole earth", 41, 4),
    });

    let model = generator(&server).await;
    let result = model
        .generate(
            &[Message::user("describe it")],
            GenerationOptions::default(),
        )
        .await
        .expect("generates");

    assert_eq!(result.text, "the whole earth");
    assert!(result.images.is_empty());
    assert!(result.audio.is_none());
    let usage = result.usage.expect("usage reported");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (41, 4, 45)
    );

    let requests = server.requests_to("/v1/chat/completions");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].body["model"], json!(GEN_MODEL_ID));
    assert_eq!(requests[0].body["stream"], json!(false));
}

#[tokio::test]
async fn generation_sends_only_the_sampling_options_that_are_set() {
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());
    let model = generator(&server).await;

    model
        .generate(&[Message::user("hi")], GenerationOptions::default())
        .await
        .expect("generates");
    let body = &server.requests_to("/v1/chat/completions")[0].body;
    for key in ["max_tokens", "temperature", "top_p"] {
        assert!(body.get(key).is_none(), "{key} should be absent: {body}");
    }

    model
        .generate(
            &[Message::user("hi")],
            GenerationOptions {
                max_tokens: Some(32),
                temperature: Some(0.5),
                top_p: Some(0.75),
                width: Some(512),
                height: Some(512),
            },
        )
        .await
        .expect("generates");
    let body = &server.requests_to("/v1/chat/completions")[1].body;
    assert_eq!(body["max_tokens"], json!(32));
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["top_p"], json!(0.75));
    // Diffusion-only knobs must not reach a chat server.
    assert!(body.get("width").is_none(), "{body}");
    assert!(body.get("height").is_none(), "{body}");
}

#[tokio::test]
async fn text_only_messages_reach_the_server_as_plain_strings() {
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());
    let model = generator(&server).await;

    model
        .generate(
            &[Message::system("be terse"), Message::user("hi")],
            GenerationOptions::default(),
        )
        .await
        .expect("generates");

    let messages = server.requests_to("/v1/chat/completions")[0].body["messages"].clone();
    assert_eq!(
        messages,
        json!([
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": "hi" },
        ])
    );
}

#[tokio::test]
async fn an_image_block_reaches_the_server_as_a_data_uri_content_part() {
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());
    let model = generator(&server).await;

    // The echo handler reports what it received, so a dropped image — the bug
    // this provider was built around — would come back as "describe" alone.
    let result = model
        .generate(
            &[Message {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text("describe".into()),
                    ContentBlock::Image(ImageInput::Bytes {
                        data: vec![0xDE, 0xAD, 0xBE, 0xEF],
                        media_type: "image/jpeg".into(),
                    }),
                ],
            }],
            GenerationOptions::default(),
        )
        .await
        .expect("generates");
    assert_eq!(result.text, "describe <image>");

    let parts = server.requests_to("/v1/chat/completions")[0].body["messages"][0]["content"]
        .as_array()
        .expect("content is a parts array")
        .clone();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], json!({ "type": "text", "text": "describe" }));
    assert_eq!(
        parts[1],
        json!({
            "type": "image_url",
            "image_url": { "url": "data:image/jpeg;base64,3q2+7w==" },
        })
    );
}

#[tokio::test]
async fn an_image_url_is_passed_through_verbatim() {
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());
    let model = generator(&server).await;

    model
        .generate(
            &[Message {
                role: MessageRole::User,
                content: vec![ContentBlock::Image(ImageInput::Url(
                    "https://example.invalid/earthrise.jpg".into(),
                ))],
            }],
            GenerationOptions::default(),
        )
        .await
        .expect("generates");

    let parts =
        server.requests_to("/v1/chat/completions")[0].body["messages"][0]["content"].clone();
    assert_eq!(
        parts[0]["image_url"]["url"],
        json!("https://example.invalid/earthrise.jpg")
    );
}

#[tokio::test]
async fn generation_sends_the_bearer_token_when_configured() {
    let _lock = ENV_LOCK.lock().await;
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());

    // SAFETY: protected by ENV_LOCK
    unsafe { std::env::set_var("UNI_XERVO_LLAMACPP_CHAT_KEY", "sekrit") };
    let mut spec = gen_spec(&server);
    spec.options["api_key_env"] = json!("UNI_XERVO_LLAMACPP_CHAT_KEY");
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("runtime builds");
    let model = runtime.generator(GEN_ALIAS).await.expect("loads");
    model
        .generate(&[Message::user("hi")], GenerationOptions::default())
        .await
        .expect("generates");
    // SAFETY: protected by ENV_LOCK
    unsafe { std::env::remove_var("UNI_XERVO_LLAMACPP_CHAT_KEY") };

    assert_eq!(
        server.requests_to("/v1/chat/completions")[0].header("authorization"),
        Some("Bearer sekrit")
    );
}

#[tokio::test]
async fn a_context_overflow_during_generation_is_not_retryable() {
    let server = MockLlamaServer::start().await;
    server.on_chat(|_| RouteResponse::Json {
        status: 400,
        body: llama_error(400, EXCEED_CONTEXT_MESSAGE, "exceed_context_size"),
    });
    let model = generator(&server).await;

    let err = model
        .generate(&[Message::user("hi")], GenerationOptions::default())
        .await
        .expect_err("must fail");
    assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
    assert!(!err.is_retryable(), "{err:?}");
    // Generation has no `max_input_tokens`, so the advice names server flags.
    let msg = err.to_string();
    assert!(!msg.contains("max_input_tokens"), "{msg}");
    assert!(msg.contains("--ctx-size"), "{msg}");
}

#[tokio::test]
async fn a_generation_response_without_choices_is_an_api_error() {
    let server = MockLlamaServer::start().await;
    server.on_chat(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "object": "chat.completion" }),
    });
    let model = generator(&server).await;

    let err = model
        .generate(&[Message::user("hi")], GenerationOptions::default())
        .await
        .expect_err("must fail");
    assert!(
        matches!(err, RuntimeError::ApiError(ref m) if m.contains("choices")),
        "{err:?}"
    );
}

#[tokio::test]
async fn a_hung_chat_request_times_out_at_the_configured_timeout() {
    let server = MockLlamaServer::start().await;
    server.on_chat(|_| RouteResponse::Hang);

    let mut spec = gen_spec(&server);
    spec.options["request_timeout_secs"] = json!(1);
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("runtime builds");
    let model = runtime.generator(GEN_ALIAS).await.expect("loads");

    let started = Instant::now();
    let err = model
        .generate(&[Message::user("hi")], GenerationOptions::default())
        .await
        .expect_err("must time out");
    assert!(matches!(err, RuntimeError::Timeout), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(10), "took too long");
}

#[tokio::test]
async fn generate_alias_rejects_embed_only_options() {
    let server = MockLlamaServer::start().await;
    for key in [
        "max_input_tokens",
        "embedding_dimensions",
        "tokenizer_base_url",
    ] {
        let mut spec = gen_spec(&server);
        spec.options[key] = json!(512);
        let result = ModelRuntime::builder()
            .register_provider(RemoteLlamaCppProvider::new())
            .catalog(vec![spec])
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("{key} must be rejected for generate"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("applies only to task 'embed'"),
            "{key}: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Rerank — /v1/rerank
// ---------------------------------------------------------------------------

const RERANK_ALIAS: &str = "rerank/llamacpp";
const RERANK_MODEL_ID: &str = "jina-reranker-v1-tiny-en";

fn rerank_spec(server: &MockLlamaServer) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: RERANK_ALIAS.to_string(),
        task: ModelTask::Rerank,
        provider_id: "remote/llamacpp".to_string(),
        model_id: RERANK_MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({ "base_url": server.base_url() }),
    }
}

async fn reranker_from(spec: ModelAliasSpec) -> Arc<dyn RerankerModel> {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("runtime builds");
    runtime
        .reranker(RERANK_ALIAS)
        .await
        .expect("reranker loads")
}

async fn reranker(server: &MockLlamaServer) -> Arc<dyn RerankerModel> {
    reranker_from(rerank_spec(server)).await
}

#[tokio::test]
async fn rerank_posts_the_documents_form_and_omits_top_n() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(rerank_by_position());
    let model = reranker(&server).await;

    let scored = model
        .rerank(
            "what is the capital of France?",
            &["alpha", "beta", "gamma"],
        )
        .await
        .expect("reranks");
    assert_eq!(scored.len(), 3);

    let requests = server.requests_to("/v1/rerank");
    assert_eq!(requests.len(), 1);
    let body = &requests[0].body;
    assert_eq!(body["model"], json!(RERANK_MODEL_ID));
    assert_eq!(body["query"], json!("what is the capital of France?"));
    assert_eq!(body["documents"], json!(["alpha", "beta", "gamma"]));
    // `top_n` would truncate; the trait contracts to score every document.
    assert!(body.get("top_n").is_none(), "{body}");
    // We always send `documents`, never `texts` — that choice is what fixes
    // the response shape.
    assert!(body.get("texts").is_none(), "{body}");
}

#[tokio::test]
async fn rerank_maps_indices_back_to_the_original_order() {
    let server = MockLlamaServer::start().await;
    // Deliberately shuffled, with the middle document the best match.
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: jina_rerank(vec![
            json!({ "index": 1, "relevance_score": 0.91 }),
            json!({ "index": 2, "relevance_score": 0.42 }),
            json!({ "index": 0, "relevance_score": 0.03 }),
        ]),
    });
    let model = reranker(&server).await;

    let scored = model
        .rerank("q", &["irrelevant", "the answer", "related"])
        .await
        .expect("reranks");
    assert_eq!(
        scored.iter().map(|d| d.index).collect::<Vec<_>>(),
        [1, 2, 0]
    );
    assert!((scored[0].score - 0.91).abs() < 1e-6);
    // `text` is never populated: the caller already holds the originals.
    assert!(scored.iter().all(|d| d.text.is_none()));
}

#[tokio::test]
async fn rerank_accepts_the_bare_array_tei_shape_too() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: tei_rerank(vec![
            json!({ "index": 1, "score": 0.8 }),
            json!({ "index": 0, "score": 0.2 }),
        ]),
    });
    let model = reranker(&server).await;

    let scored = model.rerank("q", &["a", "b"]).await.expect("reranks");
    assert_eq!(scored.iter().map(|d| d.index).collect::<Vec<_>>(), [1, 0]);
}

#[tokio::test]
async fn rerank_sorts_descending_even_when_the_server_does_not() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: jina_rerank(vec![
            json!({ "index": 0, "relevance_score": 0.1 }),
            json!({ "index": 1, "relevance_score": 0.9 }),
        ]),
    });
    let model = reranker(&server).await;

    let scored = model.rerank("q", &["a", "b"]).await.expect("reranks");
    assert_eq!(scored[0].index, 1);
    assert!(scored[0].score > scored[1].score);
}

#[tokio::test]
async fn reranking_no_documents_never_reaches_the_server() {
    let server = MockLlamaServer::start().await;
    let model = reranker(&server).await;

    // A real llama-server answers an empty `documents` array with HTTP 400.
    let scored = model.rerank("q", &[]).await.expect("reranks");
    assert!(scored.is_empty());
    assert!(server.requests_to("/v1/rerank").is_empty());
}

#[tokio::test]
async fn an_out_of_range_rerank_index_is_rejected() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: jina_rerank(vec![
            json!({ "index": 0, "relevance_score": 0.5 }),
            json!({ "index": 9, "relevance_score": 0.4 }),
        ]),
    });
    let model = reranker(&server).await;

    let err = model.rerank("q", &["a", "b"]).await.expect_err("must fail");
    assert!(
        matches!(err, RuntimeError::ApiError(ref m) if m.contains("index 9")),
        "{err:?}"
    );
}

#[tokio::test]
async fn a_duplicate_rerank_index_is_rejected() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: jina_rerank(vec![
            json!({ "index": 0, "relevance_score": 0.5 }),
            json!({ "index": 0, "relevance_score": 0.4 }),
        ]),
    });
    let model = reranker(&server).await;

    let err = model.rerank("q", &["a", "b"]).await.expect_err("must fail");
    assert!(
        matches!(err, RuntimeError::ApiError(ref m) if m.contains("more than once")),
        "{err:?}"
    );
}

#[tokio::test]
async fn a_dropped_rerank_result_is_rejected_rather_than_scored_zero() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: jina_rerank(vec![json!({ "index": 0, "relevance_score": 0.5 })]),
    });
    let model = reranker(&server).await;

    let err = model
        .rerank("q", &["a", "b", "c"])
        .await
        .expect_err("must fail");
    assert!(
        matches!(err, RuntimeError::ApiError(ref m) if m.contains("for 3 documents")),
        "{err:?}"
    );
}

/// A server started without `--reranking` answers 501 for every request. That
/// is a permanent misconfiguration: retrying cannot fix it, and it must not
/// count against the breaker either.
#[tokio::test]
async fn reranking_on_a_server_without_the_flag_is_not_retryable() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 501,
        body: llama_error(501, NOT_SUPPORTED_MESSAGE, "not_supported_error"),
    });
    let model = reranker(&server).await;

    // Well past the breaker's failure threshold: every call must still reach
    // the server rather than being short-circuited.
    for _ in 0..8 {
        let err = model.rerank("q", &["a"]).await.expect_err("must fail");
        assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
        assert!(!err.is_retryable(), "{err:?}");
        assert!(err.to_string().contains("--reranking"), "{err}");
    }
    assert_eq!(server.requests_to("/v1/rerank").len(), 8);
}

#[tokio::test]
async fn a_malformed_rerank_response_is_an_api_error() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Json {
        status: 200,
        body: json!({ "object": "list" }),
    });
    let model = reranker(&server).await;

    let err = model.rerank("q", &["a"]).await.expect_err("must fail");
    assert!(
        matches!(err, RuntimeError::ApiError(ref m) if m.contains("results")),
        "{err:?}"
    );
}

#[tokio::test]
async fn a_hung_rerank_request_times_out() {
    let server = MockLlamaServer::start().await;
    server.on_rerank(|_| RouteResponse::Hang);

    let mut spec = rerank_spec(&server);
    spec.options["request_timeout_secs"] = json!(1);
    let model = reranker_from(spec).await;

    let started = Instant::now();
    let err = model.rerank("q", &["a"]).await.expect_err("must time out");
    assert!(matches!(err, RuntimeError::Timeout), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(10), "took too long");
}

#[tokio::test]
async fn rerank_alias_rejects_embed_only_options() {
    let server = MockLlamaServer::start().await;
    for key in [
        "max_input_tokens",
        "embedding_dimensions",
        "tokenizer_base_url",
    ] {
        let mut spec = rerank_spec(&server);
        spec.options[key] = json!(512);
        let result = ModelRuntime::builder()
            .register_provider(RemoteLlamaCppProvider::new())
            .catalog(vec![spec])
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("{key} must be rejected for rerank"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("applies only to task 'embed'"),
            "{key}: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Document extraction — /v1/chat/completions with a fixed prompt
// ---------------------------------------------------------------------------

const DOCS_ALIAS: &str = "docext/llamacpp";
const DOCS_MODEL_ID: &str = "olmOCR-2-7B";

fn docs_spec(server: &MockLlamaServer) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: DOCS_ALIAS.to_string(),
        task: ModelTask::DocumentExtract,
        provider_id: "remote/llamacpp".to_string(),
        model_id: DOCS_MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({ "base_url": server.base_url() }),
    }
}

async fn extractor_from(spec: ModelAliasSpec) -> Arc<dyn DocumentExtractionModel> {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("runtime builds");
    runtime
        .document_extractor(DOCS_ALIAS)
        .await
        .expect("extractor loads")
}

async fn extractor(server: &MockLlamaServer) -> Arc<dyn DocumentExtractionModel> {
    extractor_from(docs_spec(server)).await
}

/// Build a runtime expecting catalog validation to reject the spec.
/// `ModelRuntime` is not `Debug`, so `expect_err` is unavailable.
async fn catalog_error(spec: ModelAliasSpec) -> RuntimeError {
    match ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await
    {
        Ok(_) => panic!("catalog validation must reject this spec"),
        Err(e) => e,
    }
}

fn doc_options() -> DocExtractOptions {
    DocExtractOptions {
        output: DocOutputFormat::Markdown,
        include_tables: true,
        include_formulas: true,
        include_bboxes: false,
    }
}

fn page(byte: u8) -> ImageInput {
    ImageInput::Bytes {
        data: vec![byte, byte, byte],
        media_type: "image/png".to_string(),
    }
}

/// The olmOCR prompt and the page travel together, text part first, and the
/// result comes back parsed rather than raw.
#[tokio::test]
async fn extraction_sends_the_prompt_and_the_page_and_parses_the_markdown() {
    let server = MockLlamaServer::start().await;
    server.on_chat(|_| RouteResponse::Json {
        status: 200,
        body: chat_completion("# Title\n\nSome body text.\n", 150, 12),
    });

    let model = extractor(&server).await;
    let pages = model
        .extract(vec![page(0xab)], doc_options())
        .await
        .expect("extracts");

    assert_eq!(pages.len(), 1);
    assert!(pages[0].plain_markdown.contains("Some body text"));
    let kinds: Vec<DocBlockKind> = pages[0].blocks.iter().map(|b| b.kind).collect();
    assert!(
        kinds.contains(&DocBlockKind::Heading) && kinds.contains(&DocBlockKind::Text),
        "{kinds:?}"
    );

    let reqs = server.requests_to("/v1/chat/completions");
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].body["model"], json!(DOCS_MODEL_ID));
    let parts = reqs[0].body["messages"][0]["content"]
        .as_array()
        .expect("a parts array, since the page is an image");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], json!("text"));
    assert!(
        parts[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Attached is one page of a document"),
        "{:?}",
        parts[0]["text"]
    );
    assert_eq!(parts[1]["type"], json!("image_url"));
    assert_eq!(
        parts[1]["image_url"]["url"],
        json!("data:image/png;base64,q6ur")
    );
}

/// olmOCR's sampling policy is part of the contract: near-greedy, with the
/// reference per-page cap.
#[tokio::test]
async fn extraction_pins_the_olmocr_sampling_policy() {
    let server = MockLlamaServer::start().await;
    server.on_chat(chat_echo());
    let model = extractor(&server).await;
    let _ = model.extract(vec![page(1)], doc_options()).await.unwrap();

    let body = &server.requests_to("/v1/chat/completions")[0].body;
    assert_eq!(body["max_tokens"], json!(8000));
    // Compared as f32: 0.1 is not representable, so the JSON number is the
    // widened f32 rather than the f64 literal.
    assert_eq!(body["temperature"].as_f64().unwrap() as f32, 0.1_f32);
    assert!(body.get("top_p").is_none(), "{body}");
}

/// One request per page, in input order — not one batched multi-image call.
#[tokio::test]
async fn each_page_is_its_own_request_and_order_is_preserved() {
    let server = MockLlamaServer::start().await;
    let mut n = 0;
    server.on_chat(move |_| {
        n += 1;
        RouteResponse::Json {
            status: 200,
            body: chat_completion(&format!("page {n} body"), 10, 3),
        }
    });

    let model = extractor(&server).await;
    let pages = model
        .extract(vec![page(1), page(2), page(3)], doc_options())
        .await
        .expect("extracts");

    assert_eq!(pages.len(), 3);
    for (i, p) in pages.iter().enumerate() {
        assert!(
            p.plain_markdown.contains(&format!("page {} body", i + 1)),
            "page {i}: {:?}",
            p.plain_markdown
        );
    }
    assert_eq!(server.requests_to("/v1/chat/completions").len(), 3);
}

/// Nothing to extract costs no request, and is not an error.
#[tokio::test]
async fn an_empty_page_list_makes_no_request() {
    let server = MockLlamaServer::start().await;
    let model = extractor(&server).await;

    let pages = model
        .extract(vec![], doc_options())
        .await
        .expect("extracts");
    assert!(pages.is_empty());
    assert!(server.requests_to("/v1/chat/completions").is_empty());
}

/// A page that fails fails the document: a transcript silently missing a page
/// is worse than an error.
#[tokio::test]
async fn a_failing_page_fails_the_whole_document() {
    let server = MockLlamaServer::start().await;
    let mut n = 0;
    server.on_chat(move |_| {
        n += 1;
        if n == 2 {
            RouteResponse::Json {
                status: 500,
                body: llama_error(500, EXCEED_CONTEXT_MESSAGE, "exceed_context_size"),
            }
        } else {
            RouteResponse::Json {
                status: 200,
                body: chat_completion("fine", 10, 2),
            }
        }
    });

    let model = extractor(&server).await;
    let err = model
        .extract(vec![page(1), page(2), page(3)], doc_options())
        .await
        .expect_err("must fail");
    assert!(matches!(err, RuntimeError::InferenceError(_)), "{err:?}");
    assert!(!err.is_retryable(), "{err:?}");
    // The third page is never attempted.
    assert_eq!(server.requests_to("/v1/chat/completions").len(), 2);
}

/// `style` picks the parser. The same text parses differently under olmOCR's
/// Markdown rules and Granite-Docling's DocTags.
#[tokio::test]
async fn style_selects_the_parser() {
    let doctags = "<heading>Quarterly report</heading><text>Revenue rose.</text>";
    let server = MockLlamaServer::start().await;
    server.on_chat(move |_| RouteResponse::Json {
        status: 200,
        body: chat_completion(doctags, 20, 9),
    });

    // Under the default olmOCR style the tags are not understood, so the raw
    // text survives into the Markdown view.
    let default_pages = extractor(&server)
        .await
        .extract(vec![page(1)], doc_options())
        .await
        .expect("extracts");
    assert!(
        default_pages[0].plain_markdown.contains("<heading>"),
        "{:?}",
        default_pages[0].plain_markdown
    );

    let mut spec = docs_spec(&server);
    spec.options["style"] = json!("granite-docling");
    let docling_pages = extractor_from(spec)
        .await
        .extract(vec![page(1)], doc_options())
        .await
        .expect("extracts");
    assert_eq!(
        docling_pages[0].blocks[0].kind,
        DocBlockKind::Heading,
        "{:?}",
        docling_pages[0].blocks
    );
    assert_eq!(docling_pages[0].blocks[0].content, "Quarterly report");
}

#[tokio::test]
async fn an_unknown_style_is_rejected_at_catalog_validation() {
    let server = MockLlamaServer::start().await;
    let mut spec = docs_spec(&server);
    spec.options["style"] = json!("docling");

    let err = catalog_error(spec).await;
    assert!(err.to_string().contains("granite-docling"), "{err}");
}

/// `style` belongs to this task alone — it would be silently ignored by the
/// others, which is exactly what the validator exists to prevent.
#[tokio::test]
async fn style_is_rejected_on_the_other_tasks() {
    let server = MockLlamaServer::start().await;
    let mut spec = gen_spec(&server);
    spec.options["style"] = json!("olmocr");

    let err = catalog_error(spec).await;
    assert!(
        err.to_string()
            .contains("applies only to task 'document_extract'"),
        "{err}"
    );
}

#[tokio::test]
async fn extraction_alias_rejects_embed_only_options() {
    let server = MockLlamaServer::start().await;
    for key in [
        "max_input_tokens",
        "embedding_dimensions",
        "tokenizer_base_url",
    ] {
        let mut spec = docs_spec(&server);
        spec.options[key] = json!(512);
        let err = catalog_error(spec).await;
        assert!(
            err.to_string().contains("applies only to task 'embed'"),
            "{key}: {err}"
        );
    }
}
