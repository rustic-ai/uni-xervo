//! Mock-server tests for the `remote/llamacpp` embedding provider.
//!
//! Every test spins up an in-process fake `llama-server` (see
//! `common::mock_http`) and drives the provider through a real `ModelRuntime`,
//! so catalog validation, loading, and the tokenize → embed pipeline are all
//! exercised over HTTP. Budgets are kept tiny (`max_input_tokens = 10`,
//! 4 dimensions) so boundaries are cheap to spell out.

#![cfg(feature = "provider-llamacpp")]

mod common;

use common::mock_http::{
    CLS, EXCEED_CONTEXT_MESSAGE, MockLlamaServer, RouteResponse, SEP, TOO_LARGE_MESSAGE,
    char_token, embeddings_echo, llama_error, tokenize_by_chars,
};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::RemoteLlamaCppProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{EmbedResult, EmbeddingModel};

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
async fn non_embed_task_is_rejected_at_catalog_validation() {
    let server = MockLlamaServer::start().await;
    let result = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec(ModelTask::Rerank, base_options(&server))])
        .build()
        .await;
    let err = match result {
        Ok(_) => panic!("must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("only supports task 'embed'"),
        "{err}"
    );
}
