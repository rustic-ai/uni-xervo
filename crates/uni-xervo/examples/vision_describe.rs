//! Vision example: describe an image with a small vision-language model.
//!
//! Sends an image plus a text prompt to a VLM served by a llama.cpp
//! `llama-server` and prints the generated description. Image description is
//! not a dedicated task — it is ordinary text generation
//! (`runtime.generator`) with an image attached to the message, so the same
//! path handles captioning, visual Q&A, and "extract X from this image".
//!
//! Whether images are accepted depends on the server having been started with
//! a multimodal projector, which is a property of the server process rather
//! than of the alias — so there is no option to switch vision on here. Start
//! the server with one:
//! ```sh
//! llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -c 4096 --port 8081
//! ```
//! (`-hf` fetches the projector alongside the model; with local files, pass
//! `--mmproj` explicitly.) Without a projector the call fails with a message
//! naming the missing multimodal support.
//!
//! Then run with:
//! ```sh
//! cargo run --example vision_describe --features provider-llamacpp
//! ```
//!
//! Override the endpoint with `LLAMACPP_CHAT_BASE_URL`.

#[cfg(feature = "provider-llamacpp")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::provider::RemoteLlamaCppProvider;
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::traits::{ContentBlock, GenerationOptions, ImageInput, Message, MessageRole};

#[cfg(feature = "provider-llamacpp")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("LLAMACPP_CHAT_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8081/v1".to_string());

    // 1. Describe the model. One `llama-server` process serves one model, so
    //    the alias is really a pointer at that server.
    let spec = ModelAliasSpec {
        alias: "vision/smolvlm".to_string(),
        task: ModelTask::Generate,
        provider_id: "remote/llamacpp".to_string(),
        model_id: "local".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "base_url": base_url }),
    };

    // 2. Build the runtime with the llama.cpp provider.
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample photo (replace with your own as needed).
    //    `ImageInput::Bytes` becomes a `data:` URI; `ImageInput::Url` is
    //    passed to the server verbatim.
    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/earthrise.jpg");
    let message = Message {
        role: MessageRole::User,
        content: vec![
            ContentBlock::Image(ImageInput::Bytes {
                data: std::fs::read(image_path)?,
                media_type: "image/jpeg".to_string(),
            }),
            ContentBlock::Text("Describe this image.".to_string()),
        ],
    };

    // 4. Generate. The description is returned in `result.text`.
    let vision = runtime.generator("vision/smolvlm").await?;
    let result = vision
        .generate(
            &[message],
            GenerationOptions {
                // On a model with a thinking phase the budget is spent on the
                // chain of thought first, so keep it generous.
                max_tokens: Some(512),
                ..Default::default()
            },
        )
        .await?;
    println!("{}", result.text);

    Ok(())
}

#[cfg(not(feature = "provider-llamacpp"))]
fn main() {
    eprintln!(
        "This example requires the `provider-llamacpp` feature.\n\
         Run with: cargo run --example vision_describe --features provider-llamacpp"
    );
}
