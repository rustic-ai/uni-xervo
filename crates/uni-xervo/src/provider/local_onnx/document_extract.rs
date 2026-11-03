// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Document extraction task for `local/onnx`.
//!
//! Targets vision-language document parsers that ONNX-export — Granite-Docling
//! (reference, smallest, DocTags output), MinerU 2.5 (structured Markdown +
//! `$$`-delimited LaTeX), and olmOCR-2 (Markdown with inline `$`-delimited
//! LaTeX). The `style` option (default `"granite-docling"`) selects which
//! parser to apply to the VLM's text output; the parsers themselves live in
//! the shared [`crate::doc_parse`] module.
//!
//! # v1 scope (this release)
//!
//! The actual VLM inference loop (vision encoder forward + LLM decoder
//! generation) is deferred to a follow-up that picks concrete ONNX-exported
//! variants and validates them end-to-end. Until then, `extract()` returns
//! `RuntimeError::Unavailable` with a clear message. The olmOCR-2 path is
//! served by `remote/llamacpp` instead, which runs a GGUF document VLM on a
//! `llama-server` started with `--mmproj` — see
//! [`crate::provider::llamacpp`]. Catalog authors can register
//! `document_extract/*` aliases here now and they will pass validation.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::api::ModelAliasSpec;
use crate::doc_parse::{DocStyle, style_from_str};
use crate::error::{Result, RuntimeError};
use crate::traits::{DocExtractOptions, DocExtractResult, DocumentExtractionModel, ImageInput};

/// Entry point for `LocalOnnxProvider::load` when `spec.task == DocumentExtract`.
pub(super) async fn load_document_extractor(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn DocumentExtractionModel>> {
    let style_str = spec
        .options
        .get("style")
        .and_then(Value::as_str)
        .unwrap_or("granite-docling");
    let style = style_from_str(style_str).ok_or_else(|| {
        RuntimeError::Config(format!(
            "Document extractor '{}' has unknown `style` value '{style_str}'; \
             expected one of: granite-docling, mineru, olmocr",
            spec.alias
        ))
    })?;
    let model = OnnxDocumentExtractor {
        alias: spec.alias.clone(),
        model_id: spec.model_id.clone(),
        style,
    };
    Ok(Arc::new(model) as Arc<dyn DocumentExtractionModel>)
}

struct OnnxDocumentExtractor {
    alias: String,
    model_id: String,
    style: DocStyle,
}

impl crate::traits::ModelInfo for OnnxDocumentExtractor {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl DocumentExtractionModel for OnnxDocumentExtractor {
    async fn extract(
        &self,
        _pages: Vec<ImageInput>,
        _options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>> {
        // The style-aware output parsers (`crate::doc_parse`) and the reusable
        // autoregressive decoder (`autoreg::greedy_decode`) are both shipped
        // and tested. The blocker for end-to-end ONNX inference is that no
        // canonical ONNX export of the three target VLMs exists yet:
        //
        //   - Granite-Docling: official IBM repo is SafeTensors only.
        //   - MinerU 2.5: no public ONNX export at time of writing.
        //   - olmOCR-2: no public ONNX export — served via `remote/llamacpp`
        //     against a `llama-server` with a projector instead.
        //
        // When an official ONNX export lands, the wiring is small: download
        // ONNX + tokenizer, preprocess pages via `image::preprocess_batch`,
        // tokenize the style-specific prompt, run `autoreg::greedy_decode`
        // with a callback that does one ONNX forward per token, and feed the
        // decoded text into `crate::doc_parse::parse_by_style`.
        tracing::warn!(
            alias = %self.alias,
            model_id = %self.model_id,
            style = ?self.style,
            "local/onnx document_extract invoked but no official ONNX export of \
             the target VLM has been validated yet. The style-aware parsers \
             (`crate::doc_parse`) and the autoregressive decoder helper are \
             production-ready; wiring is gated on an upstream ONNX release. \
             Point a remote/llamacpp document_extract alias at a llama-server \
             started with --mmproj and a GGUF document VLM instead. Returning \
             Unavailable."
        );
        Err(RuntimeError::Unavailable)
    }
}
