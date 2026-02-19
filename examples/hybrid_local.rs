//! Hybrid local example: combine Candle (embedding) and mistral.rs (generation)
//! in a single runtime.
//!
//! Run with:
//! ```sh
//! cargo run --example hybrid_local --features provider-mistralrs
//! ```

#[cfg(feature = "provider-mistralrs")]
use serde_json::json;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::provider::candle::LocalCandleProvider;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::traits::GenerationOptions;

#[cfg(feature = "provider-mistralrs")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure Candle for Embedding (BGE Small)
    let embed_spec = ModelAliasSpec {
        alias: "embed/bge".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    // 2. Configure Mistral.rs for Generation (Gemma 3)
    let gen_spec = ModelAliasSpec {
        alias: "chat/gemma3".to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: "google/gemma-3-270m".to_string(), // Gemma 3
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({
            "isq": "Q4K",
            "max_num_seqs": 4
        }),
    };

    // 3. Build Runtime with both providers
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![embed_spec, gen_spec])
        .build()
        .await?;

    // 4. Use Embedding
    let embedder = runtime.embedding("embed/bge").await?;
    let _vec = embedder.embed(vec!["Search query"]).await?;
    println!("BGE embedding generated.");

    // 5. Use Generation
    let generator = runtime.generator("chat/gemma3").await?;
    let res = generator
        .generate(
            &["Explain the importance of Rust safety.".to_string()],
            GenerationOptions::default(),
        )
        .await?;
    println!("Gemma 3 says: {}", res.text);

    Ok(())
}

#[cfg(not(feature = "provider-mistralrs"))]
fn main() {
    eprintln!(
        "This example requires the `provider-mistralrs` feature.\n\
         Run with: cargo run --example hybrid_local --features provider-mistralrs"
    );
}
