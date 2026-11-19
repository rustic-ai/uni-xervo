//! Runtime snapshots describe remote aliases without exposing secrets
//! (rustic-ai/uni-xervo#61): the endpoint is redacted and the credential is
//! named, never read. Builds lazily, so no key or network is needed.

#![cfg(all(
    feature = "provider-openai",
    feature = "provider-anthropic",
    feature = "provider-gemini",
    feature = "provider-llamacpp",
    feature = "provider-azure-openai"
))]

use serde_json::{Value, json};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::inspect::LoadState;
use uni_xervo::provider::{
    RemoteAnthropicProvider, RemoteAzureOpenAIProvider, RemoteGeminiProvider,
    RemoteLlamaCppProvider, RemoteOpenAIProvider,
};
use uni_xervo::runtime::ModelRuntime;

fn spec(
    alias: &str,
    task: ModelTask,
    provider: &str,
    model: &str,
    options: Value,
) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: provider.to_string(),
        model_id: model.to_string(),
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
async fn remote_aliases_are_described_without_secrets() {
    let rt = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .register_provider(RemoteAnthropicProvider::new())
        .register_provider(RemoteGeminiProvider::new())
        .register_provider(RemoteLlamaCppProvider::new())
        .register_provider(RemoteAzureOpenAIProvider::new())
        .catalog(vec![
            spec(
                "chat/proxy",
                ModelTask::Generate,
                "remote/openai",
                "gpt-4o",
                json!({
                    "base_url": "https://user:hunter2@proxy.example/v1/?token=secret",
                    "api_key_env": "PROXY_KEY"
                }),
            ),
            spec(
                "chat/claude",
                ModelTask::Generate,
                "remote/anthropic",
                "claude-x",
                Value::Null,
            ),
            spec(
                "embed/gemini",
                ModelTask::Embed,
                "remote/gemini",
                "text-embedding-004",
                Value::Null,
            ),
            spec(
                "chat/local",
                ModelTask::Generate,
                "remote/llamacpp",
                "qwen",
                json!({ "base_url": "http://127.0.0.1:8080/v1" }),
            ),
            spec(
                "chat/azure",
                ModelTask::Generate,
                "remote/azure-openai",
                "my-deployment",
                json!({ "resource_name": "acme" }),
            ),
        ])
        .build()
        .await
        .unwrap();

    let by = |alias: &str, snaps: &[uni_xervo::inspect::AliasSnapshot]| {
        snaps.iter().find(|s| s.alias == alias).unwrap().clone()
    };
    let snaps = rt.aliases().await;

    let proxy = by("chat/proxy", &snaps);
    assert_eq!(proxy.endpoint.as_deref(), Some("https://proxy.example/v1"));
    assert_eq!(proxy.credential_env.as_deref(), Some("PROXY_KEY"));
    let debug = format!("{snaps:?}");
    for secret in ["hunter2", "token=secret", "user:"] {
        assert!(!debug.contains(secret), "snapshot leaks {secret}: {debug}");
    }

    let claude = by("chat/claude", &snaps);
    assert_eq!(
        claude.endpoint.as_deref(),
        Some("https://api.anthropic.com/v1")
    );
    assert_eq!(claude.credential_env.as_deref(), Some("ANTHROPIC_API_KEY"));

    let gemini = by("embed/gemini", &snaps);
    assert_eq!(
        gemini.endpoint.as_deref(),
        Some("https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004")
    );
    assert_eq!(gemini.credential_env.as_deref(), Some("GEMINI_API_KEY"));

    let local = by("chat/local", &snaps);
    assert_eq!(local.endpoint.as_deref(), Some("http://127.0.0.1:8080/v1"));
    assert_eq!(local.credential_env, None, "llama-server without --api-key");

    let azure = by("chat/azure", &snaps);
    assert_eq!(
        azure.endpoint.as_deref(),
        Some("https://acme.openai.azure.com/openai/deployments/my-deployment")
    );

    // Nothing was loaded to describe them.
    assert!(snaps.iter().all(|s| s.state == LoadState::Registered));
}
