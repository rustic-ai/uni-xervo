// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Local ONNX Runtime provider (`local/onnx`).
//!
//! Single backend, multiple tasks. The provider declares
//! [`ModelTask::Raw`] and [`ModelTask::Rerank`] in [`ProviderCapabilities`]
//! and dispatches inside [`ModelProvider::load`] — matching the pattern
//! used by `cohere.rs`, `mistralrs.rs`, etc.
//!
//! Task implementations live in private submodules:
//!
//! - `raw` — arbitrary ONNX tensor execution via [`RawTensorModel`].
//! - `rerank` — cross-encoder reranking via [`RerankerModel`].
//! - `embedding` — dense text embeddings via [`EmbeddingModel`].

// Reserved for downstream provider impls (Granite-Docling / MinerU /
// olmOCR / future decoder LMs). The module is built and unit-tested
// independently; #[allow(dead_code)] suppresses the unused-warning
// noise until the first consumer wires it.
#[allow(dead_code)]
mod autoreg;
mod common;
mod decoder_inputs;
mod det;
mod document_extract;
mod embed_hybrid;
mod embed_multi_vector;
mod embed_sparse;
mod embedding;
mod image;
mod image_embed;
mod nlp;
mod nlp_v7;
mod ocr;
mod presets;
mod raw;
mod rerank;
mod rerank_generative;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    DocumentExtractionModel, EmbeddingModel, HybridEmbeddingModel, ImageEmbeddingModel,
    LoadedModelHandle, ModelProvider, MultiVectorEmbeddingModel, NlpModel, OcrModel,
    ProviderCapabilities, ProviderHealth, RawTensorModel, RerankerModel, SparseEmbeddingModel,
};

pub struct LocalOnnxProvider {
    base_dir: Option<PathBuf>,
    sessions: DashMap<String, Arc<raw::LoadedOnnxSession>>,
}

impl Default for LocalOnnxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalOnnxProvider {
    pub fn new() -> Self {
        Self {
            base_dir: None,
            sessions: DashMap::new(),
        }
    }

    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }
}

#[async_trait]
impl ModelProvider for LocalOnnxProvider {
    fn provider_id(&self) -> &'static str {
        "local/onnx"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![
                ModelTask::Raw,
                ModelTask::Rerank,
                ModelTask::Embed,
                ModelTask::Nlp,
                ModelTask::EmbedImage,
                ModelTask::Ocr,
                ModelTask::DocumentExtract,
                ModelTask::EmbedSparse,
                ModelTask::EmbedMultiVector,
                ModelTask::EmbedHybrid,
            ],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Raw => {
                let runner: Arc<dyn RawTensorModel> =
                    raw::load_raw(spec, self.base_dir.as_deref(), &self.sessions).await?;
                Ok(Arc::new(runner) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let style = spec
                    .options
                    .get("style")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cross-encoder");
                let reranker: Arc<dyn RerankerModel> = match style {
                    "cross-encoder" => rerank::load_rerank(spec).await?,
                    "generative" => rerank_generative::load(spec).await?,
                    other => {
                        return Err(RuntimeError::Config(format!(
                            "Unknown rerank style '{other}' for alias '{}'; expected one of cross-encoder, generative",
                            spec.alias
                        )));
                    }
                };
                Ok(Arc::new(reranker) as LoadedModelHandle)
            }
            ModelTask::Embed => {
                let embedder: Arc<dyn EmbeddingModel> = embedding::load_embedding(spec).await?;
                Ok(Arc::new(embedder) as LoadedModelHandle)
            }
            ModelTask::Nlp => {
                // Two generations of the kniv NLP family share this task.
                // The v1 cascade is one graph of 128 subwords; v7 is eight
                // graphs over word-level windows, with coreference, typed
                // entities and schema relations. They have no code in
                // common beyond this dispatch.
                let model: Arc<dyn NlpModel> = match nlp_v7::detect_generation(spec).await? {
                    nlp_v7::Generation::V7 => nlp_v7::load_nlp_v7(spec).await?,
                    nlp_v7::Generation::V1 => nlp::load_nlp(spec).await?,
                };
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::EmbedImage => {
                let model: Arc<dyn ImageEmbeddingModel> =
                    image_embed::load_image_embedder(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::Ocr => {
                let model: Arc<dyn OcrModel> = ocr::load_ocr(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::DocumentExtract => {
                let model: Arc<dyn DocumentExtractionModel> =
                    document_extract::load_document_extractor(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::EmbedSparse => {
                let model: Arc<dyn SparseEmbeddingModel> =
                    embed_sparse::load_sparse_embedder(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::EmbedMultiVector => {
                let model: Arc<dyn MultiVectorEmbeddingModel> =
                    embed_multi_vector::load_multi_vector_embedder(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            ModelTask::EmbedHybrid => {
                let model: Arc<dyn HybridEmbeddingModel> =
                    embed_hybrid::load_hybrid_embedder(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "ONNX provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}
