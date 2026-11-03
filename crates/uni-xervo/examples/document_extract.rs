//! Document-extraction example: structured Markdown from a page image.
//!
//! Runs a GGUF document VLM — olmOCR-2, MinerU 2.5 or Granite-Docling — on a
//! llama.cpp `llama-server` and prints the page's Markdown. Unlike OCR, which
//! only transcribes text, a document VLM recovers tables, formulas, headings
//! and reading order. Because it is generative, the output can hallucinate —
//! corroborate against a deterministic OCR tier for reliability (see the
//! tiered PDF extraction guide).
//!
//! Start the server first, with a projector so page images are accepted:
//! ```sh
//! llama-server -m olmOCR-2-7B-Q4_K_M.gguf \
//!     --mmproj olmOCR-2-mmproj.gguf -c 8192 --port 8099
//! ```
//!
//! Then run with:
//! ```sh
//! cargo run --example document_extract --features provider-llamacpp
//! ```
//!
//! Override the endpoint with `LLAMACPP_DOCS_BASE_URL`, and the output dialect
//! with `LLAMACPP_DOCS_STYLE` (`olmocr`, `mineru`, `granite-docling`).

#[cfg(feature = "provider-llamacpp")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::provider::RemoteLlamaCppProvider;
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-llamacpp")]
use uni_xervo::traits::{DocExtractOptions, DocOutputFormat, ImageInput};

#[cfg(feature = "provider-llamacpp")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("LLAMACPP_DOCS_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8099/v1".to_string());
    let style = std::env::var("LLAMACPP_DOCS_STYLE").unwrap_or_else(|_| "olmocr".to_string());

    // 1. Describe the document extractor. `style` names the loaded model's
    //    output dialect so the right parser is applied — it has to match what
    //    the server actually loaded. Vision is a property of the server
    //    process (`--mmproj`), so there is no option for it here.
    let spec = ModelAliasSpec {
        alias: "docext/olmocr".to_string(),
        task: ModelTask::DocumentExtract,
        provider_id: "remote/llamacpp".to_string(),
        // A single-model server accepts any value; in router mode this must
        // match the preset section name.
        model_id: "local".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "base_url": base_url,
            "style": style,
            // A page of transcription is far slower than a chat turn.
            "request_timeout_secs": 600
        }),
    };

    // 2. Build the runtime with the llama.cpp provider.
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteLlamaCppProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample page (replace with your own as needed).
    let page_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/text_lines.png"
    );
    let page = ImageInput::Bytes {
        data: std::fs::read(page_path)?,
        media_type: "image/png".to_string(),
    };

    // 4. Extract. `DocExtractOptions` has no `Default` — set all fields.
    //    Pages are sent one request each, in order, so a multi-page document
    //    is one call here regardless of the server's context size.
    let options = DocExtractOptions {
        output: DocOutputFormat::Markdown,
        include_tables: true,
        include_formulas: true,
        include_bboxes: false,
    };
    let extractor = runtime.document_extractor("docext/olmocr").await?;
    let pages = extractor.extract(vec![page], options).await?;

    println!("extracted Markdown:\n{}", pages[0].plain_markdown);
    println!("\n{} block(s) in reading order", pages[0].blocks.len());

    Ok(())
}

#[cfg(not(feature = "provider-llamacpp"))]
fn main() {
    eprintln!(
        "This example requires the `provider-llamacpp` feature.\n\
         Run with: cargo run --example document_extract --features provider-llamacpp"
    );
}
