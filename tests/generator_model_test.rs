//! Tests for generator model operations

mod common;
use common::mock_support::{MockGeneratorModel, runtime_with_generator};
use uni_xervo::traits::{
    AudioOutput, ContentBlock, GeneratedImage, GenerationOptions, GeneratorModel, ImageInput,
    Message,
};

#[tokio::test]
async fn test_generate_returns_configured_text() {
    let model = MockGeneratorModel::new("Hello, I am a mock AI.".to_string());
    let result = model
        .generate(&[Message::user("Hi there!")], GenerationOptions::default())
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
            &[Message::user("Tell me a story")],
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
        ..Default::default()
    };

    let result = model.generate(&[Message::user("Question")], options).await;

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
                Message::user("Message 1"),
                Message::assistant("Message 2"),
                Message::user("Message 3"),
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
        .generate(&[Message::user("Test")], GenerationOptions::default())
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
        .generate(&[Message::user("Test1")], GenerationOptions::default())
        .await;
    assert_eq!(model.call_count(), 1);

    let _ = model
        .generate(&[Message::user("Test2")], GenerationOptions::default())
        .await;
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn test_end_to_end_via_runtime() {
    let runtime = runtime_with_generator().await.unwrap();
    let model = runtime.generator("generate/test").await.unwrap();

    let result = model
        .generate(
            &[Message::user("Hello, world!")],
            GenerationOptions::default(),
        )
        .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.text, "Mock response");
    assert!(response.usage.is_some());
}

#[tokio::test]
async fn test_generation_result_has_empty_multimodal_fields() {
    let model = MockGeneratorModel::new("Response".to_string());
    let result = model
        .generate(&[Message::user("Test")], GenerationOptions::default())
        .await
        .unwrap();

    assert!(result.images.is_empty());
    assert!(result.audio.is_none());
}

#[tokio::test]
async fn test_message_convenience_constructors() {
    use uni_xervo::traits::MessageRole;

    let user = Message::user("hello");
    assert_eq!(user.role, MessageRole::User);
    assert_eq!(user.text(), "hello");

    let assistant = Message::assistant("hi");
    assert_eq!(assistant.role, MessageRole::Assistant);
    assert_eq!(assistant.text(), "hi");

    let system = Message::system("you are helpful");
    assert_eq!(system.role, MessageRole::System);
    assert_eq!(system.text(), "you are helpful");
}

// ---------------------------------------------------------------------------
// Image generation (diffusion) tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_generator_returns_images() {
    let fake_png = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic bytes
    let model = MockGeneratorModel::new(String::new()).with_images(vec![GeneratedImage {
        data: fake_png.clone(),
        media_type: "image/png".to_string(),
    }]);

    let result = model
        .generate(
            &[Message::user("Generate a sunset")],
            GenerationOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(result.images.len(), 1);
    assert_eq!(result.images[0].data, fake_png);
    assert_eq!(result.images[0].media_type, "image/png");
    assert!(result.text.is_empty());
}

#[tokio::test]
async fn test_generator_returns_multiple_images() {
    let img1 = GeneratedImage {
        data: vec![1, 2, 3],
        media_type: "image/png".to_string(),
    };
    let img2 = GeneratedImage {
        data: vec![4, 5, 6],
        media_type: "image/jpeg".to_string(),
    };
    let model = MockGeneratorModel::new(String::new()).with_images(vec![img1, img2]);

    let result = model
        .generate(
            &[Message::user("Generate two images")],
            GenerationOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(result.images.len(), 2);
    assert_eq!(result.images[0].media_type, "image/png");
    assert_eq!(result.images[1].media_type, "image/jpeg");
}

#[tokio::test]
async fn test_generation_options_width_height() {
    let model = MockGeneratorModel::new(String::new());
    let options = GenerationOptions {
        width: Some(1024),
        height: Some(768),
        ..Default::default()
    };

    // The mock doesn't use width/height but this verifies the fields exist and compile
    let result = model.generate(&[Message::user("Generate")], options).await;
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// Audio generation (speech) tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_generator_returns_audio() {
    let audio = AudioOutput {
        pcm_data: vec![0.1, 0.2, -0.1, -0.2, 0.0],
        sample_rate: 44100,
        channels: 1,
    };
    let model = MockGeneratorModel::new(String::new()).with_audio(audio);

    let result = model
        .generate(&[Message::user("Say hello")], GenerationOptions::default())
        .await
        .unwrap();

    assert!(result.audio.is_some());
    let audio_out = result.audio.unwrap();
    assert_eq!(audio_out.pcm_data.len(), 5);
    assert_eq!(audio_out.sample_rate, 44100);
    assert_eq!(audio_out.channels, 1);
    assert!(result.text.is_empty());
}

// ---------------------------------------------------------------------------
// Vision input (image content blocks) tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_message_with_image_content_block() {
    let fake_image_bytes = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
    let msg = Message {
        role: uni_xervo::traits::MessageRole::User,
        content: vec![
            ContentBlock::Text("What is in this image?".to_string()),
            ContentBlock::Image(ImageInput::Bytes {
                data: fake_image_bytes.clone(),
                media_type: "image/jpeg".to_string(),
            }),
        ],
    };

    assert_eq!(msg.text(), "What is in this image?");
    assert_eq!(msg.content.len(), 2);

    // Verify the mock generator handles mixed content (text extracted, images ignored)
    let model = MockGeneratorModel::new("It shows a cat.".to_string());
    let result = model
        .generate(&[msg], GenerationOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text, "It shows a cat.");
    // Word count should only count text blocks
    let usage = result.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 5); // "What is in this image?"
}

#[tokio::test]
async fn test_message_with_url_image() {
    let msg = Message {
        role: uni_xervo::traits::MessageRole::User,
        content: vec![
            ContentBlock::Text("Describe this".to_string()),
            ContentBlock::Image(ImageInput::Url("https://example.com/photo.jpg".to_string())),
        ],
    };

    assert_eq!(msg.text(), "Describe this");

    let model = MockGeneratorModel::new("A photo".to_string());
    let result = model
        .generate(&[msg], GenerationOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text, "A photo");
    let usage = result.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 2); // "Describe this"
}

#[tokio::test]
async fn test_text_generation_still_works_with_roles() {
    let model = MockGeneratorModel::new("Sure, here is the answer.".to_string());
    let result = model
        .generate(
            &[
                Message::system("You are a helpful assistant."),
                Message::user("What is 2+2?"),
                Message::assistant("Let me think..."),
                Message::user("Please answer."),
            ],
            GenerationOptions {
                max_tokens: Some(100),
                temperature: Some(0.7),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(result.text, "Sure, here is the answer.");
    assert!(result.images.is_empty());
    assert!(result.audio.is_none());

    let usage = result.usage.unwrap();
    // All text from all messages counted as prompt tokens
    // "You are a helpful assistant. What is 2+2? Let me think... Please answer."
    assert!(usage.prompt_tokens > 0);
}
