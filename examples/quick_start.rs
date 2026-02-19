//! Quick-start example: load a local Candle embedding model and embed a sentence.
//!
//! Run with:
//! ```sh
//! cargo run --example quick_start --features provider-candle
//! ```

#[cfg(feature = "provider-candle")]
use uni_xervo::api::{ModelAliasSpec, ModelTask};
#[cfg(feature = "provider-candle")]
use uni_xervo::provider::candle::LocalCandleProvider;
#[cfg(feature = "provider-candle")]
use uni_xervo::runtime::ModelRuntime;

#[cfg(feature = "provider-candle")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define your model catalog
    let spec = ModelAliasSpec {
        alias: "embed/local".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/candle".to_string(),
        model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: Default::default(), // Lazy by default
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    // 2. Build the runtime
    let runtime = ModelRuntime::builder()
        .register_provider(LocalCandleProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Get a typed handle to the model
    let model = runtime.embedding("embed/local").await?;

    // 4. Run inference
    let embeddings = model.embed(vec!["Hello, world!"]).await?;
    println!("Embedding vector length: {}", embeddings[0].len());

    Ok(())
}

#[cfg(not(feature = "provider-candle"))]
fn main() {
    eprintln!(
        "This example requires the `provider-candle` feature.\n\
         Run with: cargo run --example quick_start --features provider-candle"
    );
}
