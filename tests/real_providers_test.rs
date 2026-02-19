//! Integration tests for real providers
//!
//! These tests require actual model downloads and/or API keys.
//! Run with: EXPENSIVE_TESTS=1 cargo test -p uni-xervo --test real_providers_test -- --ignored
//!
//! Environment variables needed for remote providers:
//! - OPENAI_API_KEY: For OpenAI tests
//! - GEMINI_API_KEY: For Gemini tests
//! - VERTEX_AI_TOKEN: OAuth bearer token for Vertex AI tests
//! - VERTEX_AI_PROJECT: GCP project ID for Vertex AI tests
//! - MISTRAL_API_KEY: For Mistral tests
//! - ANTHROPIC_API_KEY: For Anthropic tests
//! - VOYAGE_API_KEY: For Voyage AI tests
//! - CO_API_KEY: For Cohere tests
//! - AZURE_OPENAI_API_KEY: For Azure OpenAI tests

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::env;
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::GenerationOptions;

/// Helper to check if expensive tests should run
fn should_run_expensive_tests() -> bool {
    env::var("EXPENSIVE_TESTS").is_ok()
}

/// Helper to skip test if EXPENSIVE_TESTS is not set
macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

/// Helper to check if API key is available
fn has_api_key(env_var: &str) -> bool {
    env::var(env_var).is_ok()
}

// =============================================================================
// LOCAL EMBEDDING TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_fastembed_local_embedding() {
    require_expensive_tests!();

    #[cfg(feature = "provider-fastembed")]
    {
        use uni_xervo::provider::fastembed::LocalFastEmbedProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(LocalFastEmbedProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/fastembed".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/fastembed".to_string(),
                model_id: "AllMiniLML6V2".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/fastembed")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Rust is amazing", "Machine learning"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384); // AllMiniLML6V2 is 384-dim

        // Verify embeddings are normalized (L2 norm ≈ 1.0)
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");

        // Verify cache dir was created under .uni_cache/fastembed/
        let cache_dir = std::path::Path::new(".uni_cache/fastembed/AllMiniLML6V2");
        assert!(
            cache_dir.exists(),
            "Cache dir should exist at {:?}",
            cache_dir
        );

        println!("✓ FastEmbed local embedding test passed");
    }

    #[cfg(not(feature = "provider-fastembed"))]
    {
        eprintln!("Skipping - provider-fastembed feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_fastembed_bge_small_embedding() {
    require_expensive_tests!();

    #[cfg(feature = "provider-fastembed")]
    {
        use uni_xervo::provider::fastembed::LocalFastEmbedProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(LocalFastEmbedProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/fastembed-bge".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/fastembed".to_string(),
                model_id: "BGESmallENV15".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/fastembed-bge")
            .await
            .expect("Failed to resolve embedding model");

        let embeddings = model
            .embed(vec!["Hello world"])
            .await
            .expect("Embedding failed");

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384); // BGESmallENV15 is 384-dim

        println!("✓ FastEmbed BGESmallENV15 embedding test passed");
    }

    #[cfg(not(feature = "provider-fastembed"))]
    {
        eprintln!("Skipping - provider-fastembed feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_candle_local_embedding() {
    require_expensive_tests!();

    #[cfg(feature = "provider-candle")]
    {
        use uni_xervo::provider::candle::LocalCandleProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(LocalCandleProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/candle".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/candle".to_string(),
                model_id: "all-MiniLM-L6-v2".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/candle")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Rust is great"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);

        // Verify cache dir was created under .uni_cache/candle/
        let cache_dir =
            std::path::Path::new(".uni_cache/candle/sentence-transformers--all-MiniLM-L6-v2");
        assert!(
            cache_dir.exists(),
            "Cache dir should exist at {:?}",
            cache_dir
        );

        println!("✓ Candle local embedding test passed");
    }

    #[cfg(not(feature = "provider-candle"))]
    {
        eprintln!("Skipping - provider-candle feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_candle_bge_small_embedding() {
    require_expensive_tests!();

    #[cfg(feature = "provider-candle")]
    {
        use uni_xervo::provider::candle::LocalCandleProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(LocalCandleProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/candle-bge".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/candle".to_string(),
                model_id: "bge-small-en-v1.5".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/candle-bge")
            .await
            .expect("Failed to resolve embedding model");

        let embeddings = model
            .embed(vec!["Hello world"])
            .await
            .expect("Embedding failed");

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384); // bge-small-en-v1.5 is 384-dim

        println!("✓ Candle bge-small-en-v1.5 embedding test passed");
    }

    #[cfg(not(feature = "provider-candle"))]
    {
        eprintln!("Skipping - provider-candle feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_candle_bge_base_embedding() {
    require_expensive_tests!();

    #[cfg(feature = "provider-candle")]
    {
        use uni_xervo::provider::candle::LocalCandleProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(LocalCandleProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/candle-bge-base".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/candle".to_string(),
                model_id: "bge-base-en-v1.5".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/candle-bge-base")
            .await
            .expect("Failed to resolve embedding model");

        let embeddings = model
            .embed(vec!["Hello world"])
            .await
            .expect("Embedding failed");

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 768); // bge-base-en-v1.5 is 768-dim

        println!("✓ Candle bge-base-en-v1.5 embedding test passed");
    }

    #[cfg(not(feature = "provider-candle"))]
    {
        eprintln!("Skipping - provider-candle feature not enabled");
    }
}

// =============================================================================
// REMOTE EMBEDDING TESTS
// =============================================================================

/// OpenAI does not support Rerank — verify it is rejected without a network call.
#[tokio::test]
#[cfg(feature = "provider-openai")]
async fn test_openai_rerank_capability_mismatch() {
    use uni_xervo::provider::openai::RemoteOpenAIProvider;
    use uni_xervo::traits::ModelProvider;

    let provider = RemoteOpenAIProvider::new();
    let spec = ModelAliasSpec {
        alias: "rerank/openai".to_string(),
        task: ModelTask::Rerank,
        provider_id: "remote/openai".to_string(),
        model_id: "gpt-4o-mini".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not support task")
    );
}

#[tokio::test]
#[ignore]
async fn test_openai_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("OPENAI_API_KEY") {
        eprintln!("Skipping - OPENAI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-openai")]
    {
        use uni_xervo::provider::openai::RemoteOpenAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteOpenAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/openai".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/openai".to_string(),
                model_id: "text-embedding-3-small".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/openai")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "AI is transforming technology"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 1536); // text-embedding-3-small is 1536-dim

        println!("✓ OpenAI remote embedding test passed");
    }

    #[cfg(not(feature = "provider-openai"))]
    {
        eprintln!("Skipping - provider-openai feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_openai_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("OPENAI_API_KEY") {
        eprintln!("Skipping - OPENAI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-openai")]
    {
        use uni_xervo::provider::openai::RemoteOpenAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteOpenAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/openai".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/openai".to_string(),
                model_id: "gpt-4o-mini".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/openai")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from OpenAI' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.0),
            top_p: None,
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        assert!(result.usage.is_some(), "Usage stats should be present");
        let usage = result.usage.unwrap();
        assert!(usage.total_tokens > 0);

        println!("✓ OpenAI remote generation test passed");
        println!("  Generated: {}", result.text);
        println!(
            "  Tokens: {} prompt + {} completion = {} total",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    #[cfg(not(feature = "provider-openai"))]
    {
        eprintln!("Skipping - provider-openai feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_gemini_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("GEMINI_API_KEY") {
        eprintln!("Skipping - GEMINI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-gemini")]
    {
        use uni_xervo::provider::gemini::RemoteGeminiProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteGeminiProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/gemini".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/gemini".to_string(),
                model_id: "embedding-001".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/gemini")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Google Gemini AI"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768); // embedding-001 is 768-dim

        println!("✓ Gemini remote embedding test passed");
    }

    #[cfg(not(feature = "provider-gemini"))]
    {
        eprintln!("Skipping - provider-gemini feature not enabled");
    }
}

/// Vertex AI does not support Rerank — verify it is rejected without a network call.
#[tokio::test]
#[cfg(feature = "provider-vertexai")]
async fn test_vertexai_rerank_capability_mismatch() {
    use uni_xervo::provider::vertexai::RemoteVertexAIProvider;
    use uni_xervo::traits::ModelProvider;

    // SAFETY: test-scoped env setup
    unsafe {
        std::env::set_var("VERTEX_AI_TOKEN", "test-token");
        std::env::set_var("VERTEX_AI_PROJECT", "test-project");
    }

    let provider = RemoteVertexAIProvider::new();
    let spec = ModelAliasSpec {
        alias: "rerank/vertex".to_string(),
        task: ModelTask::Rerank,
        provider_id: "remote/vertexai".to_string(),
        model_id: "gemini-1.5-flash".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not support task")
    );

    // SAFETY: test-scoped env cleanup
    unsafe {
        std::env::remove_var("VERTEX_AI_TOKEN");
        std::env::remove_var("VERTEX_AI_PROJECT");
    }
}

#[tokio::test]
#[ignore]
async fn test_vertexai_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("VERTEX_AI_TOKEN") || !has_api_key("VERTEX_AI_PROJECT") {
        eprintln!("Skipping - VERTEX_AI_TOKEN or VERTEX_AI_PROJECT not set");
        return;
    }

    #[cfg(feature = "provider-vertexai")]
    {
        use uni_xervo::provider::vertexai::RemoteVertexAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteVertexAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/vertex".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/vertexai".to_string(),
                model_id: "text-embedding-005".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({
                    "project_id": std::env::var("VERTEX_AI_PROJECT").unwrap(),
                    "location": "us-central1"
                }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/vertex")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Vertex AI embedding"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert!(embeddings[0].len() > 0);

        println!("✓ Vertex AI remote embedding test passed");
    }

    #[cfg(not(feature = "provider-vertexai"))]
    {
        eprintln!("Skipping - provider-vertexai feature not enabled");
    }
}

// =============================================================================
// REMOTE GENERATION TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_gemini_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("GEMINI_API_KEY") {
        eprintln!("Skipping - GEMINI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-gemini")]
    {
        use uni_xervo::provider::gemini::RemoteGeminiProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteGeminiProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/gemini".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/gemini".to_string(),
                model_id: "gemini-pro".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/gemini")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Gemini' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.1),
            top_p: Some(0.9),
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        println!("✓ Gemini remote generation test passed");
        println!("  Generated: {}", result.text);
    }

    #[cfg(not(feature = "provider-gemini"))]
    {
        eprintln!("Skipping - provider-gemini feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_vertexai_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("VERTEX_AI_TOKEN") || !has_api_key("VERTEX_AI_PROJECT") {
        eprintln!("Skipping - VERTEX_AI_TOKEN or VERTEX_AI_PROJECT not set");
        return;
    }

    #[cfg(feature = "provider-vertexai")]
    {
        use uni_xervo::provider::vertexai::RemoteVertexAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteVertexAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/vertex".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/vertexai".to_string(),
                model_id: "gemini-1.5-flash".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({
                    "project_id": std::env::var("VERTEX_AI_PROJECT").unwrap(),
                    "location": "us-central1"
                }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/vertex")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Vertex AI' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.1),
            top_p: Some(0.9),
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        println!("✓ Vertex AI remote generation test passed");
        println!("  Generated: {}", result.text);
    }

    #[cfg(not(feature = "provider-vertexai"))]
    {
        eprintln!("Skipping - provider-vertexai feature not enabled");
    }
}

// =============================================================================
// MULTI-PROVIDER INTEGRATION TEST
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_multi_provider_integration() {
    require_expensive_tests!();

    println!("\n=== Multi-Provider Integration Test ===");

    // This test combines local and remote providers in one runtime
    let mut builder = ModelRuntime::builder();
    let mut catalog: Vec<ModelAliasSpec> = Vec::new();

    // Add FastEmbed for local embedding
    #[cfg(feature = "provider-fastembed")]
    {
        use uni_xervo::provider::fastembed::LocalFastEmbedProvider;
        builder = builder.register_provider(LocalFastEmbedProvider::new());
        catalog.push(ModelAliasSpec {
            alias: "embed/local".to_string(),
            task: ModelTask::Embed,
            provider_id: "local/fastembed".to_string(),
            model_id: "AllMiniLML6V2".to_string(),
            revision: None,
            warmup: WarmupPolicy::Background,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        });
        println!("✓ Added FastEmbed local embedding");
    }

    // Add OpenAI for remote embedding (if API key available)
    #[cfg(feature = "provider-openai")]
    if has_api_key("OPENAI_API_KEY") {
        use uni_xervo::provider::openai::RemoteOpenAIProvider;
        builder = builder.register_provider(RemoteOpenAIProvider::new());
        catalog.push(ModelAliasSpec {
            alias: "embed/remote".to_string(),
            task: ModelTask::Embed,
            provider_id: "remote/openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        });
        println!("✓ Added OpenAI remote embedding");
    }

    // Add Gemini for remote generation (if API key available)
    #[cfg(feature = "provider-gemini")]
    if has_api_key("GEMINI_API_KEY") {
        use uni_xervo::provider::gemini::RemoteGeminiProvider;
        builder = builder.register_provider(RemoteGeminiProvider::new());
        catalog.push(ModelAliasSpec {
            alias: "generate/remote".to_string(),
            task: ModelTask::Generate,
            provider_id: "remote/gemini".to_string(),
            model_id: "gemini-pro".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        });
        println!("✓ Added Gemini remote generation");
    }

    if catalog.is_empty() {
        eprintln!("No providers available - skipping multi-provider test");
        return;
    }

    let runtime = builder
        .catalog(catalog)
        .build()
        .await
        .expect("Failed to build multi-provider runtime");

    // Test embedding if available
    #[cfg(feature = "provider-fastembed")]
    {
        let embed_model = runtime.embedding("embed/local").await;
        if let Ok(model) = embed_model {
            let texts = vec!["Multi-provider test", "Integration test"];
            let embeddings = model.embed(texts).await.expect("Embedding failed");
            assert_eq!(embeddings.len(), 2);
            println!("✓ Local embedding successful: {} vectors", embeddings.len());
        }
    }

    // Test remote embedding if available
    #[cfg(feature = "provider-openai")]
    if has_api_key("OPENAI_API_KEY") {
        let embed_model = runtime.embedding("embed/remote").await;
        if let Ok(model) = embed_model {
            let texts = vec!["Remote embedding test"];
            let embeddings = model.embed(texts).await.expect("Remote embedding failed");
            assert_eq!(embeddings.len(), 1);
            println!(
                "✓ Remote embedding successful: {} vectors",
                embeddings.len()
            );
        }
    }

    // Test generation if available
    #[cfg(feature = "provider-gemini")]
    if has_api_key("GEMINI_API_KEY") {
        let gen_model = runtime.generator("generate/remote").await;
        if let Ok(model) = gen_model {
            let messages = vec!["Say hello.".to_string()];
            let result = model
                .generate(&messages, GenerationOptions::default())
                .await
                .expect("Generation failed");
            assert!(!result.text.is_empty());
            println!("✓ Remote generation successful: {}", result.text);
        }
    }

    println!("\n✓ Multi-provider integration test passed!");
}

// =============================================================================
// RAG-STYLE WORKFLOW TEST
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_rag_workflow() {
    require_expensive_tests!();

    if !has_api_key("GEMINI_API_KEY") {
        eprintln!("Skipping RAG test - GEMINI_API_KEY not set");
        return;
    }

    println!("\n=== RAG Workflow Test ===");

    let mut builder = ModelRuntime::builder();
    let mut catalog: Vec<ModelAliasSpec> = Vec::new();

    // Use local embedding for document indexing
    #[cfg(feature = "provider-fastembed")]
    {
        use uni_xervo::provider::fastembed::LocalFastEmbedProvider;
        builder = builder.register_provider(LocalFastEmbedProvider::new());
        catalog.push(ModelAliasSpec {
            alias: "embed/documents".to_string(),
            task: ModelTask::Embed,
            provider_id: "local/fastembed".to_string(),
            model_id: "AllMiniLML6V2".to_string(),
            revision: None,
            warmup: WarmupPolicy::Eager,
            required: true,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        });
    }

    // Use remote LLM for answer generation
    #[cfg(feature = "provider-gemini")]
    {
        use uni_xervo::provider::gemini::RemoteGeminiProvider;
        builder = builder.register_provider(RemoteGeminiProvider::new());
        catalog.push(ModelAliasSpec {
            alias: "generate/answer".to_string(),
            task: ModelTask::Generate,
            provider_id: "remote/gemini".to_string(),
            model_id: "gemini-pro".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: true,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        });
    }

    let _runtime = builder
        .catalog(catalog)
        .build()
        .await
        .expect("Failed to build RAG runtime");

    // Step 1: Embed documents
    #[cfg(feature = "provider-fastembed")]
    {
        let embed_model = _runtime
            .embedding("embed/documents")
            .await
            .expect("Failed to get embedding model");

        let documents = vec![
            "The Rust programming language is designed for performance and safety.",
            "Rust has a strong type system and ownership model.",
            "WebAssembly allows running code in the browser at near-native speed.",
        ];

        let doc_embeddings = embed_model
            .embed(documents.clone())
            .await
            .expect("Failed to embed documents");

        println!("✓ Embedded {} documents", doc_embeddings.len());

        // Step 2: Embed query
        let query = "What are Rust's key features?";
        let query_embedding = embed_model
            .embed(vec![query])
            .await
            .expect("Failed to embed query");

        // Step 3: Simple cosine similarity to find relevant doc
        let mut similarities = Vec::new();
        for (idx, doc_emb) in doc_embeddings.iter().enumerate() {
            let similarity: f32 = query_embedding[0]
                .iter()
                .zip(doc_emb.iter())
                .map(|(a, b)| a * b)
                .sum();
            similarities.push((idx, similarity));
        }
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let most_relevant_idx = similarities[0].0;
        let most_relevant_doc = documents[most_relevant_idx];

        println!(
            "✓ Found most relevant document (similarity: {:.3})",
            similarities[0].1
        );

        // Step 4: Generate answer using retrieved context
        #[cfg(feature = "provider-gemini")]
        {
            let gen_model = _runtime
                .generator("generate/answer")
                .await
                .expect("Failed to get generator model");

            let prompt = format!(
                "Based on this context: '{}'\n\nAnswer this question: {}",
                most_relevant_doc, query
            );

            let result = gen_model
                .generate(&[prompt.to_string()], GenerationOptions::default())
                .await
                .expect("Failed to generate answer");

            println!("✓ Generated answer: {}", result.text);
            assert!(!result.text.is_empty());
        }

        println!("\n✓ RAG workflow test passed!");
    }

    #[cfg(not(feature = "provider-fastembed"))]
    {
        eprintln!("Skipping RAG test - provider-fastembed feature not enabled");
    }
}

// =============================================================================
// MISTRAL REMOTE TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_mistral_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("MISTRAL_API_KEY") {
        eprintln!("Skipping - MISTRAL_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-mistral")]
    {
        use uni_xervo::provider::mistral::RemoteMistralProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteMistralProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/mistral".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/mistral".to_string(),
                model_id: "mistral-embed".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/mistral")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Mistral AI"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert!(!embeddings[0].is_empty());

        println!("Mistral remote embedding test passed");
    }

    #[cfg(not(feature = "provider-mistral"))]
    {
        eprintln!("Skipping - provider-mistral feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_mistral_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("MISTRAL_API_KEY") {
        eprintln!("Skipping - MISTRAL_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-mistral")]
    {
        use uni_xervo::provider::mistral::RemoteMistralProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteMistralProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/mistral".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/mistral".to_string(),
                model_id: "mistral-small-latest".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/mistral")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Mistral' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.0),
            top_p: None,
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        println!("Mistral remote generation test passed: {}", result.text);
    }

    #[cfg(not(feature = "provider-mistral"))]
    {
        eprintln!("Skipping - provider-mistral feature not enabled");
    }
}

/// Mistral does not support Rerank — verify it is rejected without a network call.
#[tokio::test]
#[cfg(feature = "provider-mistral")]
async fn test_mistral_rerank_capability_mismatch() {
    use uni_xervo::provider::mistral::RemoteMistralProvider;
    use uni_xervo::traits::ModelProvider;

    // SAFETY: test-scoped env setup
    unsafe { std::env::set_var("MISTRAL_API_KEY", "test-key") };

    let provider = RemoteMistralProvider::new();
    let spec = ModelAliasSpec {
        alias: "rerank/mistral".to_string(),
        task: ModelTask::Rerank,
        provider_id: "remote/mistral".to_string(),
        model_id: "mistral-embed".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not support task")
    );

    // SAFETY: test-scoped env cleanup
    unsafe { std::env::remove_var("MISTRAL_API_KEY") };
}

// =============================================================================
// ANTHROPIC REMOTE TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_anthropic_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("ANTHROPIC_API_KEY") {
        eprintln!("Skipping - ANTHROPIC_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-anthropic")]
    {
        use uni_xervo::provider::anthropic::RemoteAnthropicProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteAnthropicProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/anthropic".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/anthropic".to_string(),
                model_id: "claude-sonnet-4-5-20250929".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/anthropic")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Anthropic' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.0),
            top_p: None,
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        assert!(result.usage.is_some(), "Usage stats should be present");
        println!("Anthropic remote generation test passed: {}", result.text);
    }

    #[cfg(not(feature = "provider-anthropic"))]
    {
        eprintln!("Skipping - provider-anthropic feature not enabled");
    }
}

/// Anthropic does not support Embed — verify it is rejected without a network call.
#[tokio::test]
#[cfg(feature = "provider-anthropic")]
async fn test_anthropic_embed_capability_mismatch() {
    use uni_xervo::provider::anthropic::RemoteAnthropicProvider;
    use uni_xervo::traits::ModelProvider;

    // SAFETY: test-scoped env setup
    unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };

    let provider = RemoteAnthropicProvider::new();
    let spec = ModelAliasSpec {
        alias: "embed/anthropic".to_string(),
        task: ModelTask::Embed,
        provider_id: "remote/anthropic".to_string(),
        model_id: "claude-sonnet-4-5-20250929".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not support task")
    );

    // SAFETY: test-scoped env cleanup
    unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
}

// =============================================================================
// VOYAGE AI REMOTE TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_voyageai_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("VOYAGE_API_KEY") {
        eprintln!("Skipping - VOYAGE_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-voyageai")]
    {
        use uni_xervo::provider::voyageai::RemoteVoyageAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteVoyageAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/voyageai".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/voyageai".to_string(),
                model_id: "voyage-3".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/voyageai")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Voyage AI embeddings"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert!(!embeddings[0].is_empty());

        println!("Voyage AI remote embedding test passed");
    }

    #[cfg(not(feature = "provider-voyageai"))]
    {
        eprintln!("Skipping - provider-voyageai feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_voyageai_remote_rerank() {
    require_expensive_tests!();

    if !has_api_key("VOYAGE_API_KEY") {
        eprintln!("Skipping - VOYAGE_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-voyageai")]
    {
        use uni_xervo::provider::voyageai::RemoteVoyageAIProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteVoyageAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "rerank/voyageai".to_string(),
                task: ModelTask::Rerank,
                provider_id: "remote/voyageai".to_string(),
                model_id: "rerank-2".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .reranker("rerank/voyageai")
            .await
            .expect("Failed to resolve reranker model");

        let docs = vec![
            "The cat sat on the mat",
            "Machine learning is a subset of AI",
            "Rust is a systems programming language",
        ];
        let results = model
            .rerank("programming language", &docs)
            .await
            .expect("Rerank failed");

        assert!(!results.is_empty());
        println!("Voyage AI remote rerank test passed");
        for r in &results {
            println!("  index={}, score={:.4}", r.index, r.score);
        }
    }

    #[cfg(not(feature = "provider-voyageai"))]
    {
        eprintln!("Skipping - provider-voyageai feature not enabled");
    }
}

// =============================================================================
// COHERE REMOTE TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_cohere_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("CO_API_KEY") {
        eprintln!("Skipping - CO_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-cohere")]
    {
        use uni_xervo::provider::cohere::RemoteCohereProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteCohereProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/cohere".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/cohere".to_string(),
                model_id: "embed-english-v3.0".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({"input_type": "search_document"}),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/cohere")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Cohere embeddings"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 1024);

        println!("Cohere remote embedding test passed");
    }

    #[cfg(not(feature = "provider-cohere"))]
    {
        eprintln!("Skipping - provider-cohere feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_cohere_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("CO_API_KEY") {
        eprintln!("Skipping - CO_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-cohere")]
    {
        use uni_xervo::provider::cohere::RemoteCohereProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteCohereProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/cohere".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/cohere".to_string(),
                model_id: "command-r-plus".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/cohere")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Cohere' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.0),
            top_p: None,
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        println!("Cohere remote generation test passed: {}", result.text);
    }

    #[cfg(not(feature = "provider-cohere"))]
    {
        eprintln!("Skipping - provider-cohere feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_cohere_remote_rerank() {
    require_expensive_tests!();

    if !has_api_key("CO_API_KEY") {
        eprintln!("Skipping - CO_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-cohere")]
    {
        use uni_xervo::provider::cohere::RemoteCohereProvider;

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteCohereProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "rerank/cohere".to_string(),
                task: ModelTask::Rerank,
                provider_id: "remote/cohere".to_string(),
                model_id: "rerank-english-v3.0".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .reranker("rerank/cohere")
            .await
            .expect("Failed to resolve reranker model");

        let docs = vec![
            "The cat sat on the mat",
            "Machine learning is a subset of AI",
            "Rust is a systems programming language",
        ];
        let results = model
            .rerank("programming language", &docs)
            .await
            .expect("Rerank failed");

        assert!(!results.is_empty());
        println!("Cohere remote rerank test passed");
        for r in &results {
            println!("  index={}, score={:.4}", r.index, r.score);
        }
    }

    #[cfg(not(feature = "provider-cohere"))]
    {
        eprintln!("Skipping - provider-cohere feature not enabled");
    }
}

// =============================================================================
// AZURE OPENAI REMOTE TESTS
// =============================================================================

#[tokio::test]
#[ignore]
async fn test_azure_openai_remote_embedding() {
    require_expensive_tests!();

    if !has_api_key("AZURE_OPENAI_API_KEY") {
        eprintln!("Skipping - AZURE_OPENAI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-azure-openai")]
    {
        use uni_xervo::provider::azure_openai::RemoteAzureOpenAIProvider;

        let resource_name =
            env::var("AZURE_OPENAI_RESOURCE").unwrap_or_else(|_| "my-resource".to_string());

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteAzureOpenAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/azure".to_string(),
                task: ModelTask::Embed,
                provider_id: "remote/azure-openai".to_string(),
                model_id: "text-embedding-ada-002".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({
                    "resource_name": resource_name
                }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/azure")
            .await
            .expect("Failed to resolve embedding model");

        let texts = vec!["Hello world", "Azure OpenAI"];
        let embeddings = model.embed(texts).await.expect("Embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert!(!embeddings[0].is_empty());

        println!("Azure OpenAI remote embedding test passed");
    }

    #[cfg(not(feature = "provider-azure-openai"))]
    {
        eprintln!("Skipping - provider-azure-openai feature not enabled");
    }
}

#[tokio::test]
#[ignore]
async fn test_azure_openai_remote_generation() {
    require_expensive_tests!();

    if !has_api_key("AZURE_OPENAI_API_KEY") {
        eprintln!("Skipping - AZURE_OPENAI_API_KEY not set");
        return;
    }

    #[cfg(feature = "provider-azure-openai")]
    {
        use uni_xervo::provider::azure_openai::RemoteAzureOpenAIProvider;

        let resource_name =
            env::var("AZURE_OPENAI_RESOURCE").unwrap_or_else(|_| "my-resource".to_string());

        let runtime = ModelRuntime::builder()
            .register_provider(RemoteAzureOpenAIProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/azure".to_string(),
                task: ModelTask::Generate,
                provider_id: "remote/azure-openai".to_string(),
                model_id: "gpt-4o".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({
                    "resource_name": resource_name
                }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/azure")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from Azure' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.0),
            top_p: None,
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(!result.text.is_empty());
        println!(
            "Azure OpenAI remote generation test passed: {}",
            result.text
        );
    }

    #[cfg(not(feature = "provider-azure-openai"))]
    {
        eprintln!("Skipping - provider-azure-openai feature not enabled");
    }
}

/// Azure OpenAI does not support Rerank — verify it is rejected without a network call.
#[tokio::test]
#[cfg(feature = "provider-azure-openai")]
async fn test_azure_openai_rerank_capability_mismatch() {
    use uni_xervo::provider::azure_openai::RemoteAzureOpenAIProvider;
    use uni_xervo::traits::ModelProvider;

    // SAFETY: test-scoped env setup
    unsafe { std::env::set_var("AZURE_OPENAI_API_KEY", "test-key") };

    let provider = RemoteAzureOpenAIProvider::new();
    let spec = ModelAliasSpec {
        alias: "rerank/azure".to_string(),
        task: ModelTask::Rerank,
        provider_id: "remote/azure-openai".to_string(),
        model_id: "gpt-4o".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"resource_name": "test-resource"}),
    };
    let result = provider.load(&spec).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not support task")
    );

    // SAFETY: test-scoped env cleanup
    unsafe { std::env::remove_var("AZURE_OPENAI_API_KEY") };
}

// =============================================================================
// MISTRALRS LOCAL TESTS
// =============================================================================

#[cfg(feature = "provider-mistralrs")]
mod mistralrs_tests {
    use super::*;
    use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
    use uni_xervo::traits::ModelProvider;

    #[tokio::test]
    async fn test_mistralrs_rerank_capability_mismatch() {
        let provider = LocalMistralRsProvider::new();
        let spec = ModelAliasSpec {
            alias: "rerank/test".to_string(),
            task: ModelTask::Rerank,
            provider_id: "local/mistralrs".to_string(),
            model_id: "some-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Null,
        };

        let result = provider.load(&spec).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("does not support task"),
            "Expected CapabilityMismatch error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_mistralrs_invalid_isq_config() {
        let provider = LocalMistralRsProvider::new();
        let spec = ModelAliasSpec {
            alias: "embed/test".to_string(),
            task: ModelTask::Embed,
            provider_id: "local/mistralrs".to_string(),
            model_id: "some-model".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({ "isq": "INVALID_TYPE" }),
        };

        let result = provider.load(&spec).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Unknown ISQ type"),
            "Expected Config error about ISQ type, got: {}",
            err
        );
    }

    /// mistralrs 0.7 `EmbeddingModelBuilder` only accepts models with
    /// `Gemma3TextModel` or `Qwen3ForCausalLM` in their `config.json`
    /// `architectures` field. Standard BERT-family embedding models
    /// (NomicBert, BGE, MiniLM, etc.) are unsupported and return an error.
    ///
    /// This test documents that known limitation: it asserts that attempting
    /// to load a BERT-based embedding model returns the expected error.
    #[tokio::test]
    #[ignore]
    async fn test_mistralrs_local_embedding_bert_arch_unsupported() {
        require_expensive_tests!();

        let runtime = ModelRuntime::builder()
            .register_provider(LocalMistralRsProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/mistralrs".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/mistralrs".to_string(),
                model_id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        match runtime.embedding("embed/mistralrs").await {
            Ok(_) => panic!("Expected an error for unsupported BERT architecture, but got Ok"),
            Err(err) => {
                assert!(
                    err.to_string().contains("Unsupported"),
                    "Expected unsupported-architecture error, got: {err}"
                );
                println!("✓ mistralrs embedding correctly rejects unsupported architecture: {err}");
                println!(
                    "  NOTE: mistralrs 0.7 embedding only supports Gemma3TextModel and Qwen3ForCausalLM"
                );
            }
        }
    }

    /// Embedding via mistralrs using google/embeddinggemma-300m.
    /// EmbeddingGemma is Google's purpose-built 308M embedding model based on
    /// Gemma3TextModel with bidirectional attention. It is the architecture
    /// mistralrs labels "embeddinggemma".
    #[tokio::test]
    #[ignore]
    async fn test_mistralrs_local_embedding_gemma3() {
        require_expensive_tests!();

        let runtime = ModelRuntime::builder()
            .register_provider(LocalMistralRsProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/mistralrs-gemma3".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/mistralrs".to_string(),
                model_id: "google/embeddinggemma-300m".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/mistralrs-gemma3")
            .await
            .expect("Failed to resolve embeddinggemma-300m model");

        let embeddings = model
            .embed(vec!["The quick brown fox", "jumps over the lazy dog"])
            .await
            .expect("Embedding failed");

        assert_eq!(embeddings.len(), 2, "Expected 2 embeddings");
        assert!(
            !embeddings[0].is_empty(),
            "Embedding should have non-zero dimensions"
        );
        assert_eq!(
            embeddings[0].len(),
            embeddings[1].len(),
            "Both embeddings must have same dimensions"
        );
        assert!(
            model.dimensions() > 0,
            "dimensions() should return non-zero value"
        );
        assert_eq!(model.model_id(), "google/embeddinggemma-300m");

        println!("✓ mistralrs embeddinggemma test passed");
        println!("  Model:      google/embeddinggemma-300m (Gemma3TextModel)");
        println!("  Dimensions: {}", model.dimensions());
    }

    /// Embedding via mistralrs using Qwen3-Embedding (Qwen3ForCausalLM architecture).
    /// Qwen3-Embedding-0.6B is a purpose-built embedding model with 0.6B parameters.
    #[tokio::test]
    #[ignore]
    async fn test_mistralrs_local_embedding_qwen3() {
        require_expensive_tests!();

        let runtime = ModelRuntime::builder()
            .register_provider(LocalMistralRsProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "embed/mistralrs-qwen3".to_string(),
                task: ModelTask::Embed,
                provider_id: "local/mistralrs".to_string(),
                model_id: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({ "isq": "Q4K" }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .embedding("embed/mistralrs-qwen3")
            .await
            .expect("Failed to resolve qwen3 embedding model");

        let embeddings = model
            .embed(vec!["The quick brown fox", "jumps over the lazy dog"])
            .await
            .expect("Embedding failed");

        assert_eq!(embeddings.len(), 2, "Expected 2 embeddings");
        assert!(
            !embeddings[0].is_empty(),
            "Embedding should have non-zero dimensions"
        );
        assert_eq!(
            embeddings[0].len(),
            embeddings[1].len(),
            "Both embeddings must have same dimensions"
        );
        assert!(
            model.dimensions() > 0,
            "dimensions() should return non-zero value"
        );
        assert_eq!(model.model_id(), "Qwen/Qwen3-Embedding-0.6B");

        println!("✓ mistralrs qwen3 embedding test passed");
        println!("  Model:      Qwen/Qwen3-Embedding-0.6B (Qwen3ForCausalLM)");
        println!("  Dimensions: {}", model.dimensions());
    }

    #[tokio::test]
    #[ignore]
    async fn test_mistralrs_local_generation() {
        require_expensive_tests!();

        let runtime = ModelRuntime::builder()
            .register_provider(LocalMistralRsProvider::new())
            .catalog(vec![ModelAliasSpec {
                alias: "generate/mistralrs".to_string(),
                task: ModelTask::Generate,
                provider_id: "local/mistralrs".to_string(),
                model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::json!({ "isq": "Q4K" }),
            }])
            .build()
            .await
            .expect("Failed to build runtime");

        let model = runtime
            .generator("generate/mistralrs")
            .await
            .expect("Failed to resolve generator model");

        let messages = vec!["Say 'Hello from mistral.rs' and nothing else.".to_string()];
        let options = GenerationOptions {
            max_tokens: Some(20),
            temperature: Some(0.1),
            top_p: Some(0.9),
        };

        let result = model
            .generate(&messages, options)
            .await
            .expect("Generation failed");

        assert!(
            !result.text.is_empty(),
            "Generated text should not be empty"
        );
        assert!(result.usage.is_some(), "Usage stats should be present");

        let usage = result.usage.unwrap();
        assert!(usage.total_tokens > 0, "Total tokens should be > 0");

        println!("mistralrs local generation test passed");
        println!("  Generated: {}", result.text);
        println!(
            "  Tokens: {} prompt + {} completion = {} total",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
}
