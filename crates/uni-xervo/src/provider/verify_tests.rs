// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! `ModelProvider::verify` for every bundled remote provider, against a local
//! stub server (rustic-ai/uni-xervo#66).
//!
//! Each provider is driven through the same cases: success, a rejected
//! credential, the other error statuses, an unreachable endpoint and a missing
//! credential. Every case also checks that no error text contains the key.
//! Keys live in per-test environment variables with unique names, so tests
//! never race on a shared one.

use serde_json::{Value, json};

use super::stub_http::{Reply, StubServer, unreachable_url};
use crate::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use crate::inspect::{VerifyErrorKind, VerifyOutcome};
use crate::traits::ModelProvider;

use super::anthropic::RemoteAnthropicProvider;
use super::azure_openai::RemoteAzureOpenAIProvider;
use super::cohere::RemoteCohereProvider;
use super::gemini::RemoteGeminiProvider;
use super::llamacpp::RemoteLlamaCppProvider;
use super::mistral::RemoteMistralProvider;
use super::openai::RemoteOpenAIProvider;
use super::vertexai::RemoteVertexAIProvider;
use super::voyageai::RemoteVoyageAIProvider;

/// A recognisable fake key; no error may ever contain it.
const SECRET: &str = "sk-test-SECRET-0123456789";

/// One provider's verify request, as the stub must see it.
struct Case {
    name: &'static str,
    provider_id: &'static str,
    task: ModelTask,
    model: &'static str,
    method: &'static str,
    path: &'static str,
    /// The option naming the credential's env var.
    key_option: &'static str,
    /// Extra options; `{stub}` in a string value becomes the stub's URL.
    options: Value,
    /// Whether a 400 counts as verified (providers that post an empty body).
    empty_body: bool,
    /// Build the provider pointed at `url`.
    make: fn(&str) -> Box<dyn ModelProvider>,
    /// Check the credential was sent as this provider sends it.
    auth: fn(&super::stub_http::Seen) -> bool,
}

fn bearer(seen: &super::stub_http::Seen) -> bool {
    seen.header("authorization") == Some(&format!("Bearer {SECRET}"))
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "openai",
            provider_id: "remote/openai",
            task: ModelTask::Generate,
            model: "gpt-4o",
            method: "GET",
            path: "/v1/models",
            key_option: "api_key_env",
            options: json!({ "base_url": "{stub}/v1" }),
            empty_body: false,
            make: |_| Box::new(RemoteOpenAIProvider::new()),
            auth: bearer,
        },
        Case {
            name: "anthropic",
            provider_id: "remote/anthropic",
            task: ModelTask::Generate,
            model: "claude-x",
            method: "GET",
            path: "/v1/models/claude-x",
            key_option: "api_key_env",
            options: json!({}),
            empty_body: false,
            make: |url| Box::new(RemoteAnthropicProvider::with_endpoint_override(url)),
            auth: |s| {
                s.header("x-api-key") == Some(SECRET) && s.header("anthropic-version").is_some()
            },
        },
        Case {
            name: "azure",
            provider_id: "remote/azure-openai",
            task: ModelTask::Embed,
            model: "my-deployment",
            method: "POST",
            path: "/openai/deployments/my-deployment/embeddings",
            key_option: "api_key_env",
            options: json!({ "resource_name": "acme" }),
            empty_body: true,
            make: |url| Box::new(RemoteAzureOpenAIProvider::with_endpoint_override(url)),
            auth: |s| s.header("api-key") == Some(SECRET) && s.target.contains("api-version="),
        },
        Case {
            name: "cohere",
            provider_id: "remote/cohere",
            task: ModelTask::Rerank,
            model: "rerank-v3.5",
            method: "GET",
            path: "/v1/models/rerank-v3.5",
            key_option: "api_key_env",
            options: json!({}),
            empty_body: false,
            make: |url| Box::new(RemoteCohereProvider::with_endpoint_override(url)),
            auth: bearer,
        },
        Case {
            name: "gemini",
            provider_id: "remote/gemini",
            task: ModelTask::Embed,
            model: "text-embedding-004",
            method: "GET",
            path: "/v1beta/models/text-embedding-004",
            key_option: "api_key_env",
            options: json!({}),
            empty_body: false,
            make: |url| Box::new(RemoteGeminiProvider::with_endpoint_override(url)),
            // The key travels in the query string, never a header.
            auth: |s| {
                s.target.ends_with(&format!("?key={SECRET}")) && s.header("authorization").is_none()
            },
        },
        Case {
            name: "mistral",
            provider_id: "remote/mistral",
            task: ModelTask::Generate,
            model: "mistral-small",
            method: "GET",
            path: "/v1/models/mistral-small",
            key_option: "api_key_env",
            options: json!({}),
            empty_body: false,
            make: |url| Box::new(RemoteMistralProvider::with_endpoint_override(url)),
            auth: bearer,
        },
        Case {
            name: "vertexai",
            provider_id: "remote/vertexai",
            task: ModelTask::Generate,
            model: "gemini-2.0-flash",
            method: "POST",
            path: "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent",
            key_option: "api_token_env",
            options: json!({ "project_id": "p" }),
            empty_body: true,
            make: |url| Box::new(RemoteVertexAIProvider::with_endpoint_override(url)),
            auth: bearer,
        },
        Case {
            name: "voyageai",
            provider_id: "remote/voyageai",
            task: ModelTask::Rerank,
            model: "rerank-2",
            method: "POST",
            path: "/v1/rerank",
            key_option: "api_key_env",
            options: json!({}),
            empty_body: true,
            make: |url| Box::new(RemoteVoyageAIProvider::with_endpoint_override(url)),
            auth: bearer,
        },
        Case {
            name: "llamacpp",
            provider_id: "remote/llamacpp",
            task: ModelTask::Generate,
            model: "qwen",
            method: "POST",
            path: "/tokenize",
            key_option: "api_key_env",
            options: json!({ "base_url": "{stub}/v1" }),
            empty_body: false,
            make: |_| Box::new(RemoteLlamaCppProvider::new()),
            auth: bearer,
        },
    ]
}

/// The case's spec, its key in a fresh env var named after `tag`.
fn spec(case: &Case, url: &str, tag: &str, set_key: bool) -> ModelAliasSpec {
    let env = format!(
        "UNI_XERVO_VERIFY_{}_{}",
        case.name.to_uppercase(),
        tag.to_uppercase()
    );
    if set_key {
        // SAFETY: each test uses its own variable name, read only by itself.
        unsafe { std::env::set_var(&env, SECRET) };
    }
    let mut options = case.options.clone();
    if let Some(map) = options.as_object_mut() {
        for value in map.values_mut() {
            if let Some(s) = value.as_str() {
                *value = json!(s.replace("{stub}", url));
            }
        }
        map.insert(case.key_option.to_string(), json!(env));
    }
    ModelAliasSpec {
        alias: format!("test/{}", case.name),
        task: case.task,
        provider_id: case.provider_id.to_string(),
        model_id: case.model.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

/// llama.cpp checks `/health` first; every other provider makes one request.
fn routes(case: &Case, reply: Reply) -> Vec<(&'static str, &'static str, Reply)> {
    let mut routes = vec![(case.method, case.path, reply)];
    if case.name == "llamacpp" {
        routes.insert(0, ("GET", "/health", Reply::Status(200)));
    }
    routes
}

fn assert_no_secret(case: &Case, text: &str) {
    assert!(
        !text.contains(SECRET),
        "{}: error leaks the key: {text}",
        case.name
    );
}

#[tokio::test]
async fn success_sends_the_credential_as_the_provider_expects() {
    for case in cases() {
        let stub = StubServer::start(&routes(&case, Reply::Status(200))).await;
        let provider = (case.make)(&stub.url());
        let outcome = provider.verify(&spec(&case, &stub.url(), "ok", true)).await;
        assert_eq!(outcome, Ok(VerifyOutcome::Verified), "{}", case.name);
        let seen = stub.seen();
        let request = seen
            .last()
            .unwrap_or_else(|| panic!("{}: nothing sent", case.name));
        assert_eq!(request.method, case.method, "{}", case.name);
        assert_eq!(
            request.target.split('?').next().unwrap(),
            case.path,
            "{}",
            case.name
        );
        assert!(
            (case.auth)(request),
            "{}: credential not sent as expected: {request:?}",
            case.name
        );
    }
}

#[tokio::test]
async fn error_statuses_map_to_kinds_without_leaking_the_key() {
    for (code, kind) in [
        (401, VerifyErrorKind::Unauthorized),
        (403, VerifyErrorKind::Unauthorized),
        (404, VerifyErrorKind::NotFound),
        (429, VerifyErrorKind::RateLimited),
        (503, VerifyErrorKind::Unavailable),
    ] {
        for case in cases() {
            let stub = StubServer::start(&routes(&case, Reply::Status(code))).await;
            let provider = (case.make)(&stub.url());
            let failure = provider
                .verify(&spec(&case, &stub.url(), &format!("e{code}"), true))
                .await
                .expect_err(case.name);
            assert_eq!(failure.kind, kind, "{} HTTP {code}", case.name);
            assert_no_secret(&case, &failure.detail);
            assert_no_secret(&case, &format!("{failure:?}"));
        }
    }
}

#[tokio::test]
async fn a_rejected_empty_body_counts_as_verified_only_where_one_is_sent() {
    for case in cases() {
        let stub = StubServer::start(&routes(&case, Reply::Status(400))).await;
        let provider = (case.make)(&stub.url());
        let outcome = provider
            .verify(&spec(&case, &stub.url(), "e400", true))
            .await;
        if case.empty_body {
            assert_eq!(outcome, Ok(VerifyOutcome::Verified), "{}", case.name);
        } else if case.name == "gemini" {
            // Gemini answers a bad key with 400 API_KEY_INVALID.
            assert_eq!(outcome.unwrap_err().kind, VerifyErrorKind::Unauthorized);
        } else {
            assert_eq!(
                outcome.unwrap_err().kind,
                VerifyErrorKind::Other,
                "{}",
                case.name
            );
        }
    }
}

#[tokio::test]
async fn an_unreachable_endpoint_is_reported_as_such() {
    for case in cases() {
        let url = unreachable_url().await;
        let provider = (case.make)(&url);
        let failure = provider
            .verify(&spec(&case, &url, "down", true))
            .await
            .expect_err(case.name);
        assert_eq!(
            failure.kind,
            VerifyErrorKind::Unreachable,
            "{}: {failure:?}",
            case.name
        );
        assert_no_secret(&case, &failure.detail);
    }
}

#[tokio::test]
async fn a_missing_credential_names_its_variable() {
    for case in cases() {
        let stub = StubServer::start(&routes(&case, Reply::Status(200))).await;
        let provider = (case.make)(&stub.url());
        let spec = spec(&case, &stub.url(), "unset", false);
        let env = spec.options[case.key_option].as_str().unwrap().to_string();
        let failure = provider.verify(&spec).await.expect_err(case.name);
        assert_eq!(failure.kind, VerifyErrorKind::Config, "{}", case.name);
        assert!(
            failure.detail.contains(&env),
            "{}: {}",
            case.name,
            failure.detail
        );
        if case.name != "llamacpp" {
            assert!(
                stub.seen().is_empty(),
                "{}: sent a request without a key",
                case.name
            );
        }
    }
}

#[tokio::test]
async fn llamacpp_checks_health_and_skips_models_without_a_key() {
    let case = cases().into_iter().find(|c| c.name == "llamacpp").unwrap();
    let stub = StubServer::start(&[
        ("GET", "/health", Reply::Status(200)),
        ("GET", "/v1/models", Reply::Status(200)),
    ])
    .await;
    let mut no_key = spec(&case, &stub.url(), "nokey", false);
    no_key
        .options
        .as_object_mut()
        .unwrap()
        .remove("api_key_env");
    let provider = RemoteLlamaCppProvider::new();
    assert_eq!(provider.verify(&no_key).await, Ok(VerifyOutcome::Verified));
    let paths: Vec<String> = stub.seen().iter().map(|s| s.target.clone()).collect();
    assert_eq!(paths, ["/health"], "no key, so nothing checks one");

    // With a key, it is checked on /tokenize, which --api-key guards
    // (recent llama-server leaves /models and /health public).
    let keyed = StubServer::start(&[
        ("GET", "/health", Reply::Status(200)),
        ("POST", "/tokenize", Reply::Status(401)),
        ("GET", "/v1/models", Reply::Status(200)),
    ])
    .await;
    let failure = provider
        .verify(&spec(&case, &keyed.url(), "keyed", true))
        .await
        .unwrap_err();
    assert_eq!(failure.kind, VerifyErrorKind::Unauthorized);
    let paths: Vec<String> = keyed.seen().iter().map(|s| s.target.clone()).collect();
    assert_eq!(paths, ["/health", "/tokenize"]);

    // A server still loading its model answers /health with 503.
    let loading = StubServer::start(&[("GET", "/health", Reply::Status(503))]).await;
    let failure = provider
        .verify(&spec(&case, &loading.url(), "loading", true))
        .await
        .unwrap_err();
    assert_eq!(failure.kind, VerifyErrorKind::Unavailable);
    assert!(failure.detail.contains("loading"), "{}", failure.detail);
}

/// An Azure deployment that cannot serve the alias's task answers 400
/// OperationNotSupported: reachable, but not verified.
#[tokio::test]
async fn azure_reports_a_deployment_that_cannot_serve_the_task() {
    let case = cases().into_iter().find(|c| c.name == "azure").unwrap();
    let stub = StubServer::start(&[(
        "POST",
        "/openai/deployments/my-deployment/embeddings",
        Reply::Json(
            400,
            r#"{"error":{"code":"OperationNotSupported","message":"The embeddings operation does not work with the specified model"}}"#,
        ),
    )])
    .await;
    let provider = RemoteAzureOpenAIProvider::with_endpoint_override(&stub.url());
    let failure = provider
        .verify(&spec(&case, &stub.url(), "unsupported", true))
        .await
        .unwrap_err();
    assert_eq!(failure.kind, VerifyErrorKind::Config);
    assert!(
        failure.detail.contains("OperationNotSupported"),
        "{}",
        failure.detail
    );
    assert!(
        !failure.detail.contains("does not work"),
        "the body message is not echoed"
    );

    // Any other 400 (here, the empty body itself) means the request got through.
    let ok = StubServer::start(&[(
        "POST",
        "/openai/deployments/my-deployment/embeddings",
        Reply::Json(400, r#"{"error":{"code":"BadRequest"}}"#),
    )])
    .await;
    let provider = RemoteAzureOpenAIProvider::with_endpoint_override(&ok.url());
    assert_eq!(
        provider
            .verify(&spec(&case, &ok.url(), "badrequest", true))
            .await,
        Ok(VerifyOutcome::Verified)
    );
}

#[tokio::test]
async fn local_providers_have_nothing_to_verify() {
    use crate::mock::MockProvider;
    let spec = crate::mock::make_spec("embed/a", ModelTask::Embed, "mock/embed", "m");
    assert_eq!(
        MockProvider::embed_only().verify(&spec).await,
        Ok(VerifyOutcome::NotApplicable)
    );
}

// --- through ModelRuntime -------------------------------------------------

mod runtime {
    use super::*;
    use crate::error::RuntimeError;
    use crate::inspect::VerificationStatus;
    use crate::runtime::ModelRuntime;
    use crate::traits::ProviderHealth;
    use std::time::{Duration, Instant};

    fn anthropic_spec(alias: &str, model: &str, env: &str) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: alias.to_string(),
            task: ModelTask::Generate,
            provider_id: "remote/anthropic".to_string(),
            model_id: model.to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: Some(1),
            retry: None,
            options: json!({ "api_key_env": env }),
        }
    }

    fn set_key(env: &str) {
        // SAFETY: each test uses its own variable name.
        unsafe { std::env::set_var(env, SECRET) };
    }

    async fn runtime_with(url: &str, specs: Vec<ModelAliasSpec>) -> std::sync::Arc<ModelRuntime> {
        ModelRuntime::builder()
            .register_provider(RemoteAnthropicProvider::with_endpoint_override(url))
            .register_provider(crate::mock::MockProvider::embed_only())
            .catalog(specs)
            .build()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_hung_endpoint_times_out_within_the_load_timeout_naming_the_alias() {
        set_key("UNI_XERVO_RT_HANG");
        let stub = StubServer::start(&[("GET", "/v1/models/claude-x", Reply::Hang)]).await;
        let rt = runtime_with(
            &stub.url(),
            vec![anthropic_spec("chat/a", "claude-x", "UNI_XERVO_RT_HANG")],
        )
        .await;
        let started = Instant::now();
        let (alias, result) = rt.verify(&["chat/a"]).await.pop().unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(alias, "chat/a");
        match result.unwrap_err() {
            RuntimeError::Verification { alias, kind, .. } => {
                assert_eq!(alias, "chat/a");
                assert_eq!(kind, VerifyErrorKind::Timeout);
            }
            other => panic!("expected Verification, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_fails_fast_naming_the_alias() {
        set_key("UNI_XERVO_RT_DOWN");
        let url = unreachable_url().await;
        // Windows takes about 2 s to refuse a connection to a closed local
        // port (it retries the SYN), so a 1 s load timeout would end the check
        // first and report a timeout.
        let mut spec = anthropic_spec("chat/down", "claude-x", "UNI_XERVO_RT_DOWN");
        spec.load_timeout = Some(10);
        let rt = runtime_with(&url, vec![spec]).await;
        let err = rt
            .verify(&["chat/down"])
            .await
            .pop()
            .unwrap()
            .1
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("chat/down"), "{message}");
        assert!(matches!(
            err,
            RuntimeError::Verification {
                kind: VerifyErrorKind::Unreachable,
                ..
            }
        ));
        assert!(err.is_retryable(), "unreachable is worth retrying");
    }

    #[tokio::test]
    async fn a_wrong_key_is_unauthorized_and_not_echoed() {
        set_key("UNI_XERVO_RT_BADKEY");
        let stub = StubServer::start(&[("GET", "/v1/models/claude-x", Reply::Status(401))]).await;
        let rt = runtime_with(
            &stub.url(),
            vec![anthropic_spec("chat/a", "claude-x", "UNI_XERVO_RT_BADKEY")],
        )
        .await;
        let err = rt.verify(&["chat/a"]).await.pop().unwrap().1.unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::Verification {
                kind: VerifyErrorKind::Unauthorized,
                ..
            }
        ));
        assert!(!err.to_string().contains(SECRET));
        assert!(!format!("{err:?}").contains(SECRET));
    }

    /// Nothing contacts the endpoint until verify is asked for: not build
    /// (even with eager warmup), not prefetch, not health.
    #[tokio::test]
    async fn no_traffic_unless_verification_is_requested() {
        set_key("UNI_XERVO_RT_QUIET");
        let stub = StubServer::start(&[("GET", "/v1/models/claude-x", Reply::Status(200))]).await;
        let mut spec = anthropic_spec("chat/a", "claude-x", "UNI_XERVO_RT_QUIET");
        spec.warmup = WarmupPolicy::Eager;
        let rt = runtime_with(&stub.url(), vec![spec]).await;
        rt.prefetch_all().await.unwrap();
        let health = rt.provider_health("remote/anthropic").await.unwrap();
        assert!(matches!(health, ProviderHealth::Healthy));
        assert!(
            rt.aliases()
                .await
                .iter()
                .all(|s| s.verification == VerificationStatus::NotRequested)
        );
        assert!(
            stub.seen().is_empty(),
            "contacted without verify: {:?}",
            stub.seen()
        );

        let results = rt.verify_all().await;
        assert!(results[0].1.is_ok());
        assert_eq!(stub.seen().len(), 1, "verify sends exactly one request");
    }

    #[tokio::test]
    async fn verification_is_recorded_and_folded_into_health() {
        set_key("UNI_XERVO_RT_MIXED");
        let stub = StubServer::start(&[
            ("GET", "/v1/models/good", Reply::Status(200)),
            ("GET", "/v1/models/bad", Reply::Status(404)),
        ])
        .await;
        let rt = runtime_with(
            &stub.url(),
            vec![
                anthropic_spec("chat/good", "good", "UNI_XERVO_RT_MIXED"),
                anthropic_spec("chat/bad", "bad", "UNI_XERVO_RT_MIXED"),
                crate::mock::make_spec("embed/local", ModelTask::Embed, "mock/embed", "m"),
            ],
        )
        .await;

        let results = rt.verify(&["chat/bad", "embed/local", "chat/good"]).await;
        let names: Vec<&str> = results.iter().map(|(a, _)| a.as_str()).collect();
        assert_eq!(
            names,
            ["chat/bad", "embed/local", "chat/good"],
            "input order"
        );
        assert!(results[0].1.is_err());
        assert_eq!(
            results[1].1.as_ref().unwrap(),
            &VerifyOutcome::NotApplicable
        );
        assert_eq!(results[2].1.as_ref().unwrap(), &VerifyOutcome::Verified);

        let snap = |alias: &str, snaps: &[crate::inspect::AliasSnapshot]| {
            snaps
                .iter()
                .find(|s| s.alias == alias)
                .unwrap()
                .verification
                .clone()
        };
        let snaps = rt.aliases().await;
        assert!(matches!(
            snap("chat/good", &snaps),
            VerificationStatus::Verified { .. }
        ));
        assert!(matches!(
            snap("chat/bad", &snaps),
            VerificationStatus::Failed {
                kind: VerifyErrorKind::NotFound,
                ..
            }
        ));
        assert!(matches!(
            snap("embed/local", &snaps),
            VerificationStatus::NotApplicable { .. }
        ));

        match rt.provider_health("remote/anthropic").await.unwrap() {
            ProviderHealth::Degraded(why) => assert!(why.contains("1 of 2"), "{why}"),
            other => panic!("expected Degraded, got {other:?}"),
        }
        // Local providers' health is untouched by NotApplicable results.
        assert!(matches!(
            rt.provider_health("mock/embed").await.unwrap(),
            ProviderHealth::Healthy
        ));
        assert!(rt.provider_health("remote/none").await.is_none());

        // Once every verified alias fails, the provider is Unhealthy.
        rt.verify(&["chat/bad"]).await;
        let stub_all_bad = StubServer::start(&[]).await;
        let rt2 = runtime_with(
            &stub_all_bad.url(),
            vec![anthropic_spec("chat/x", "x", "UNI_XERVO_RT_MIXED")],
        )
        .await;
        rt2.verify_all().await;
        assert!(matches!(
            rt2.provider_health("remote/anthropic").await.unwrap(),
            ProviderHealth::Unhealthy(_)
        ));
    }

    #[tokio::test]
    async fn an_unknown_alias_is_reported_not_verified() {
        let rt = runtime_with("http://127.0.0.1:9", vec![]).await;
        let (_, result) = rt.verify(&["chat/missing"]).await.pop().unwrap();
        assert!(matches!(result, Err(RuntimeError::AliasNotFound { .. })));
    }
}
