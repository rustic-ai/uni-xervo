#![cfg(feature = "provider-vertexai")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteVertexAIProvider;
use uni_xervo::runtime::ModelRuntime;

fn vertex_embed_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/default".to_string(),
        task: ModelTask::Embed,
        provider_id: "remote/vertexai".to_string(),
        model_id: "text-embedding-005".to_string(),
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
async fn builder_rejects_unknown_vertex_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVertexAIProvider::new())
        .catalog(vec![vertex_embed_spec(
            serde_json::json!({"unknown": true}),
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
async fn builder_rejects_invalid_vertex_option_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVertexAIProvider::new())
        .catalog(vec![vertex_embed_spec(
            serde_json::json!({"project_id": 42}),
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
async fn builder_rejects_invalid_vertex_embedding_dimension() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVertexAIProvider::new())
        .catalog(vec![vertex_embed_spec(
            serde_json::json!({"embedding_dimensions": 0}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be greater than 0")
    );
}

#[tokio::test]
async fn builder_rejects_vertex_embedding_dimension_for_generate() {
    let mut spec = vertex_embed_spec(serde_json::json!({"embedding_dimensions": 768}));
    spec.alias = "generate/default".to_string();
    spec.task = ModelTask::Generate;
    spec.model_id = "gemini-1.5-flash".to_string();

    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVertexAIProvider::new())
        .catalog(vec![spec])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("only valid for embed tasks")
    );
}

#[tokio::test]
async fn builder_accepts_valid_vertex_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteVertexAIProvider::new())
        .catalog(vec![vertex_embed_spec(serde_json::json!({
            "api_token_env": "VERTEX_AI_TOKEN",
            "project_id": "demo-project",
            "location": "us-central1",
            "publisher": "google",
            "embedding_dimensions": 768
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}
