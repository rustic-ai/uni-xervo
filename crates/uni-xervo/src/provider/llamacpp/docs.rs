//! [`ModelTask::DocumentExtract`](crate::api::ModelTask::DocumentExtract) on
//! `POST /v1/chat/completions`.
//!
//! Document extraction is a vision generation call with a fixed prompt: a
//! rendered page image goes in, the model's Markdown-or-DocTags transcription
//! comes out, and [`crate::doc_parse`] turns that text into typed blocks. So
//! this module owns the prompt, the sampling policy and the parse step, and
//! reuses [`LlamaCppChatClient`](super::chat::LlamaCppChatClient) verbatim for
//! the wire format.
//!
//! The server must have been started with `--mmproj` and a document VLM —
//! olmOCR-2, MinerU 2.5 or Granite-Docling. As with plain generation, that is a
//! property of the server process rather than of the alias.
//!
//! # Options
//!
//! | key | type | required | meaning |
//! |-----|------|----------|---------|
//! | `style` | string | no | output dialect of the loaded model: `olmocr` (default), `mineru` or `granite-docling` |
//!
//! `style` selects the parser, not the model: it has to match whatever the
//! server actually loaded, because the three families emit different syntax for
//! the same page. The value is checked at catalog registration rather than at
//! load, so a typo fails when the catalog is read instead of on the first
//! extraction.
//!
//! # One page per request
//!
//! Pages are sent one at a time, which is what olmOCR-2 expects — it is
//! single-image — and what keeps a long document from being capped by one
//! shared `--ctx-size`. A page that fails fails the whole call: a partially
//! transcribed document silently missing page 7 is worse than an error.
//!
//! # `DocExtractOptions` is ignored
//!
//! [`DocExtractOptions::include_bboxes`] and `output_format` are deliberately
//! not honoured, matching the provider this replaced. A generative backend
//! cannot be asked for trustworthy bounding boxes — it would invent plausible
//! ones — and the output format follows from the model's own dialect, which the
//! `style` option already names. Blocks carry whatever positions the model
//! emitted (Granite-Docling's DocTags do include `loc` attributes) and `None`
//! otherwise.

use super::chat::LlamaCppChatClient;
use super::{LlamaCppCommon, PROVIDER_ID};
use crate::doc_parse::{DocStyle, style_from_str};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    ContentBlock, DocExtractOptions, DocExtractResult, DocumentExtractionModel, GenerationOptions,
    ImageInput, Message, MessageRole, ModelInfo,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

/// The olmOCR-2 prompt — image-only, no document-anchoring text.
///
/// olmOCR-2 dropped olmOCR-1's anchor-text requirement; this is the static
/// no-anchoring instruction. The model is given a single rendered page and
/// returns Markdown with a YAML front-matter header.
const OLMOCR_PROMPT: &str = "Attached is one page of a document that you must process. \
Just return the plain text representation of this document as if you were reading it naturally. \
Convert equations to LateX and tables to HTML. \
If there are any figures or charts, label them with the following markdown syntax \
`![Alt text...](page_startx_starty_width_height.png)`. \
Return your output as markdown, with a front matter section on top specifying values for the \
primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.";

/// First-pass sampling temperature for document extraction.
///
/// olmOCR's reference pipeline starts near-greedy and only escalates on retry;
/// this is the deterministic-leaning first pass. Temperature-escalating retries
/// are a future refinement layered above this single pass.
const DOC_EXTRACT_TEMPERATURE: f32 = 0.1;

/// Maximum tokens to generate per page (olmOCR's reference cap).
const DOC_EXTRACT_MAX_TOKENS: usize = 8000;

/// Default `style` when the alias does not set one.
pub(crate) const DEFAULT_STYLE: &str = "olmocr";

/// Accepted `style` values, for validation and error messages.
pub(crate) const STYLES: [&str; 3] = ["granite-docling", "mineru", "olmocr"];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved document-extraction configuration for one alias.
#[derive(Debug, Clone)]
pub(crate) struct DocsConfig {
    pub common: LlamaCppCommon,
    pub style: DocStyle,
}

impl DocsConfig {
    /// Parse the alias `options`. The catalog validator already enforces the
    /// same rules, but `ModelProvider::load` is public, so this is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        let style_str = options
            .get("style")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_STYLE);
        let style = style_from_str(style_str).ok_or_else(|| {
            RuntimeError::Config(format!(
                "Option 'style' for provider '{}' must be one of {} (got '{}')",
                PROVIDER_ID,
                STYLES.join(", "),
                style_str
            ))
        })?;
        Ok(Self {
            common: LlamaCppCommon::from_options(options)?,
            style,
        })
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Document extraction backed by a document VLM on a llama.cpp server.
pub struct LlamaCppDocumentExtractor {
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    style: DocStyle,
    chat: LlamaCppChatClient,
}

impl LlamaCppDocumentExtractor {
    pub(crate) fn new(
        client: Client,
        cb: crate::reliability::CircuitBreakerWrapper,
        model_id: String,
        cfg: DocsConfig,
    ) -> Self {
        Self {
            cb,
            model_id: model_id.clone(),
            style: cfg.style,
            chat: LlamaCppChatClient::new(client, cfg.common, model_id),
        }
    }

    /// Transcribe and parse every page in order, one request each.
    async fn extract_pipeline(&self, pages: Vec<ImageInput>) -> Result<Vec<DocExtractResult>> {
        let mut results = Vec::with_capacity(pages.len());
        for page in pages {
            let messages = [Message {
                role: MessageRole::User,
                // Text first: the document VLMs are trained with the
                // instruction ahead of the page.
                content: vec![
                    ContentBlock::Text(OLMOCR_PROMPT.to_string()),
                    ContentBlock::Image(page),
                ],
            }];
            let options = GenerationOptions {
                max_tokens: Some(DOC_EXTRACT_MAX_TOKENS),
                temperature: Some(DOC_EXTRACT_TEMPERATURE),
                top_p: None,
                width: None,
                height: None,
            };
            let generated = self.chat.complete(&messages, &options).await?;
            results.push(crate::doc_parse::parse_by_style(
                self.style,
                &generated.text,
            ));
        }
        Ok(results)
    }
}

#[async_trait]
impl DocumentExtractionModel for LlamaCppDocumentExtractor {
    async fn extract(
        &self,
        pages: Vec<ImageInput>,
        _options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }

        // One breaker call around the whole document rather than one per page:
        // a half-open probe should be a document, not a page, and a document
        // that fails midway has already failed.
        //
        // Deterministic rejections (`InferenceError`, produced only by
        // `map_llamacpp_status`) are a configuration problem rather than a
        // server outage, so they are returned as `Ok(Err(..))` — the breaker
        // sees a success and stays closed — and flattened here. Same contract
        // as `embed` and `chat`.
        self.cb
            .call(move || async move {
                match self.extract_pipeline(pages).await {
                    Err(e @ RuntimeError::InferenceError(_)) => Ok(Err(e)),
                    other => other.map(Ok),
                }
            })
            .await?
    }
}

impl ModelInfo for LlamaCppDocumentExtractor {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Tests (no HTTP)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options(extra: Value) -> Value {
        let mut o = json!({ "base_url": "http://127.0.0.1:8080/v1" });
        if let Value::Object(map) = extra {
            for (k, v) in map {
                o[k] = v;
            }
        }
        o
    }

    #[test]
    fn style_defaults_to_olmocr() {
        let cfg = DocsConfig::from_options(&options(json!({}))).unwrap();
        assert_eq!(cfg.style, DocStyle::OlmOcr);
    }

    #[test]
    fn every_documented_style_is_accepted() {
        for (name, expected) in [
            ("olmocr", DocStyle::OlmOcr),
            ("mineru", DocStyle::Mineru),
            ("granite-docling", DocStyle::GraniteDocling),
        ] {
            let cfg = DocsConfig::from_options(&options(json!({ "style": name }))).unwrap();
            assert_eq!(cfg.style, expected, "{name}");
            assert!(STYLES.contains(&name), "{name} missing from STYLES");
        }
    }

    #[test]
    fn an_unknown_style_names_the_accepted_values() {
        let err = DocsConfig::from_options(&options(json!({ "style": "docling" })))
            .expect_err("must fail");
        match err {
            RuntimeError::Config(msg) => {
                assert!(msg.contains("granite-docling"), "{msg}");
                assert!(msg.contains("docling"), "{msg}");
            }
            other => panic!("expected Config, got {other:?}"),
        }
    }

    /// `style` is read from a non-string value as if absent, rather than
    /// panicking — the validator is what rejects the shape.
    #[test]
    fn a_non_string_style_falls_back_to_the_default() {
        let cfg = DocsConfig::from_options(&options(json!({ "style": 7 }))).unwrap();
        assert_eq!(cfg.style, DocStyle::OlmOcr);
    }

    #[test]
    fn the_shared_options_still_apply() {
        let err = DocsConfig::from_options(&json!({ "style": "olmocr" })).expect_err("must fail");
        assert!(err.to_string().contains("base_url"), "{err}");
    }

    #[test]
    fn the_prompt_is_the_olmocr_no_anchoring_instruction() {
        // Pinned because a reworded prompt changes model output, which is the
        // kind of regression a parse test would not catch.
        assert!(OLMOCR_PROMPT.starts_with("Attached is one page of a document"));
        assert!(OLMOCR_PROMPT.contains("front matter section"));
        assert!(
            !OLMOCR_PROMPT.contains("RAW_TEXT_START"),
            "anchoring removed"
        );
    }
}
