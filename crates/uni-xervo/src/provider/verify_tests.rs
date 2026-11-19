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
            method: "GET",
            path: "/v1/publishers/google/models/gemini-2.0-flash",
            key_option: "api_token_env",
            options: json!({ "project_id": "p" }),
            empty_body: false,
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
            method: "GET",
            path: "/v1/models",
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
    assert_eq!(paths, ["/health"], "no key, so /models is not called");

    // A server still loading its model answers /health with 503.
    let loading = StubServer::start(&[("GET", "/health", Reply::Status(503))]).await;
    let failure = provider
        .verify(&spec(&case, &loading.url(), "loading", true))
        .await
        .unwrap_err();
    assert_eq!(failure.kind, VerifyErrorKind::Unavailable);
    assert!(failure.detail.contains("loading"), "{}", failure.detail);
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
