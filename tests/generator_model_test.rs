//! Tests for generator model operations

mod common;
use common::mock_support::{MockGeneratorModel, runtime_with_generator};
use uni_xervo::traits::{GenerationOptions, GeneratorModel};

#[tokio::test]
async fn test_generate_returns_configured_text() {
    let model = MockGeneratorModel::new("Hello, I am a mock AI.".to_string());
    let result = model
        .generate(&["Hi there!".to_string()], GenerationOptions::default())
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.text, "Hello, I am a mock AI.");
}

#[tokio::test]
async fn test_generate_returns_usage_info() {
    let model = MockGeneratorModel::new("Mock response".to_string());
    let result = model
        .generate(
            &["Tell me a story".to_string()],
            GenerationOptions::default(),
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.usage.is_some());

    let usage = response.usage.unwrap();
    assert_eq!(usage.completion_tokens, 2); // "Mock response" = 2 words
    assert!(usage.prompt_tokens > 0);
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[tokio::test]
async fn test_generate_with_options() {
    let model = MockGeneratorModel::new("Response".to_string());
    let options = GenerationOptions {
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
    };

    let result = model.generate(&["Question".to_string()], options).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_generate_empty_messages() {
    let model = MockGeneratorModel::new("Response".to_string());
    let result = model.generate(&[], GenerationOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.text, "Response");
}

#[tokio::test]
async fn test_generate_multiple_messages() {
    let model = MockGeneratorModel::new("Response".to_string());
    let result = model
        .generate(
            &[
                "Message 1".to_string(),
                "Message 2".to_string(),
                "Message 3".to_string(),
            ],
            GenerationOptions::default(),
        )
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_failure_propagation() {
    let model = MockGeneratorModel::new("Response".to_string()).with_failure(true);
    let result = model
        .generate(&["Test".to_string()], GenerationOptions::default())
        .await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Mock generator failure")
    );
}

#[tokio::test]
async fn test_call_counting() {
    let model = MockGeneratorModel::new("Response".to_string());
    assert_eq!(model.call_count(), 0);

    let _ = model
        .generate(&["Test1".to_string()], GenerationOptions::default())
        .await;
    assert_eq!(model.call_count(), 1);

    let _ = model
        .generate(&["Test2".to_string()], GenerationOptions::default())
        .await;
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn test_end_to_end_via_runtime() {
    let runtime = runtime_with_generator().await.unwrap();
    let model = runtime.generator("generate/test").await.unwrap();

    let result = model
        .generate(&["Hello, world!".to_string()], GenerationOptions::default())
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.text, "Mock response");
    assert!(response.usage.is_some());
}
