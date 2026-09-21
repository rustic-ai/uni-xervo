// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Single-pass hybrid embedding task for `local/onnx`.
//!
//! Runs a multi-output text encoder *once* and post-processes every requested
//! head from that single [`EncoderRun`](super::common::EncoderRun): the dense
//! head via [`finalize_dense`](super::embedding::finalize_dense), the sparse
//! head via the SPLADE/lexical post-processors, and the multi-vector head via
//! the ColBERT token collector. This is what BGE-M3 (`aapot/bge-m3-onnx`) is
//! designed for — three heads from one forward pass — instead of three separate
//! per-task sessions each reloading the 2.1 GB weights.
//!
//! Configuration is preset-driven: a [`presets::HybridPreset`] (matched on
//! `model_id`) declares which heads the graph exposes and each head's output
//! selection and recipe. Single-head models have no hybrid preset, so this task
//! errors for them — callers use the per-task resolvers instead.

use async_trait::async_trait;
use std::sync::Arc;

use super::common::{EncoderModel, EncoderSpec};
use super::embed_multi_vector::collect_token_vectors;
use super::embed_sparse::{post_process_lexical, post_process_mlm};
use super::embedding::finalize_dense;
use super::presets::{self, HybridPreset, SparseMethod};
use crate::api::ModelAliasSpec;
use crate::error::{Result, RuntimeError};
use crate::traits::{HeadSet, HybridEmbedResult, HybridEmbeddingModel};

/// Load the ONNX hybrid embedding model for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::EmbedHybrid`.
///
/// # Errors
/// Returns an error if `model_id` has no hybrid preset or the model fails to load.
pub(super) async fn load_hybrid_embedder(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn HybridEmbeddingModel>> {
    let embedder = OnnxHybridEmbedder::load(spec).await?;
    Ok(Arc::new(embedder) as Arc<dyn HybridEmbeddingModel>)
}

/// The heads a hybrid preset's graph exposes (one flag per `Some` sub-head).
fn available_heads_of(preset: &HybridPreset) -> HeadSet {
    let mut heads = HeadSet::empty();
    if preset.dense.is_some() {
        heads |= HeadSet::DENSE;
    }
    if preset.sparse.is_some() {
        heads |= HeadSet::SPARSE;
    }
    if preset.multi_vector.is_some() {
        heads |= HeadSet::MULTI_VECTOR;
    }
    heads
}

/// Resolved hybrid configuration after preset + options merge.
struct HybridConfig {
    encoder: EncoderSpec,
    /// The matched preset (`'static`): carries each head's output and recipe.
    preset: &'static HybridPreset,
    /// Optional cap on retained sparse terms per vector (largest weights).
    top_k: Option<usize>,
}

impl HybridConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        // Preset-driven: a hybrid task only exists for multi-output graphs that
        // declare their heads. Single-head models fall back to per-task handles.
        let preset = presets::lookup_hybrid(&spec.model_id).ok_or_else(|| {
            RuntimeError::Config(format!(
                "Model '{}' has no hybrid preset; the embed_hybrid task is only \
                 available for multi-output graphs (e.g. BGEM3Hybrid). Use the \
                 per-task embed / embed_sparse / embed_multi_vector resolvers instead.",
                spec.model_id
            ))
        })?;
        let opts = &spec.options;

        let onnx_path = opts
            .get("artifact")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| preset.onnx_path.to_string());

        let tokenizer_path = opts
            .get("tokenizer_path")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| preset.tokenizer_path.to_string());

        let max_seq_len = opts
            .get("max_seq_len")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(preset.max_seq_len);

        let top_k = opts
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Ok(Self {
            encoder: EncoderSpec {
                hf_repo: preset.hf_repo.to_string(),
                onnx_path,
                tokenizer_path,
                additional_files: preset
                    .additional_files
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                max_seq_len,
            },
            preset,
            top_k,
        })
    }
}

/// ONNX-backed single-pass hybrid embedder.
struct OnnxHybridEmbedder {
    encoder: EncoderModel,
    model_id: String,
    preset: &'static HybridPreset,
    available: HeadSet,
    top_k: Option<usize>,
}

impl OnnxHybridEmbedder {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = HybridConfig::resolve(spec)?;
        let available = available_heads_of(cfg.preset);
        let encoder = EncoderModel::load(spec, cfg.encoder).await?;
        Ok(Self {
            encoder,
            model_id: spec.model_id.clone(),
            preset: cfg.preset,
            available,
            top_k: cfg.top_k,
        })
    }
}

impl crate::traits::ModelInfo for OnnxHybridEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl HybridEmbeddingModel for OnnxHybridEmbedder {
    fn available_heads(&self) -> HeadSet {
        self.available
    }

    fn head_width(&self, head: HeadSet) -> Option<u32> {
        // Only a head this graph actually exposes has a width to report.
        if !self.available.contains(head) {
            return None;
        }
        match head {
            HeadSet::DENSE => self.preset.dense.as_ref().map(|h| h.dimensions),
            HeadSet::SPARSE => self.preset.sparse.as_ref().map(|h| h.vocab_size),
            HeadSet::MULTI_VECTOR => self.preset.multi_vector.as_ref().map(|h| h.dimensions),
            // A combination (or empty) names no single width.
            _ => None,
        }
    }

    async fn embed(&self, texts: &[&str], requested: HeadSet) -> Result<HybridEmbedResult> {
        // Only heads that are both requested and exposed by the graph are computed.
        let heads = requested.intersection(self.available);

        // Empty batch: each computed head is an empty Vec (never None), keeping
        // the "Some iff requested ∧ available" invariant exact.
        if texts.is_empty() {
            return Ok(HybridEmbedResult {
                dense: heads.contains(HeadSet::DENSE).then(Vec::new),
                sparse: heads.contains(HeadSet::SPARSE).then(Vec::new),
                multi_vector: heads.contains(HeadSet::MULTI_VECTOR).then(Vec::new),
                usage: None,
            });
        }

        // Nothing to compute (no requested head is available): skip the pass.
        if heads.is_empty() {
            return Ok(HybridEmbedResult::default());
        }

        // The single shared forward pass.
        let run = self.encoder.run(texts)?;
        let alias = self.encoder.alias();
        let mask = &run.attention_mask;

        let dense = if heads.contains(HeadSet::DENSE) {
            let h = self
                .preset
                .dense
                .as_ref()
                .expect("DENSE in available implies a dense head");
            let output = run.select(alias, h.output_name, Some(h.output_index))?;
            Some(finalize_dense(
                alias,
                output,
                mask,
                h.pooling,
                h.normalize,
                h.dimensions,
            )?)
        } else {
            None
        };

        let sparse = if heads.contains(HeadSet::SPARSE) {
            let h = self
                .preset
                .sparse
                .as_ref()
                .expect("SPARSE in available implies a sparse head");
            let output = run.select(alias, h.output_name, Some(h.output_index))?;
            let vectors = match h.method {
                SparseMethod::Mlm => post_process_mlm(alias, output, mask, self.top_k)?,
                SparseMethod::Lexical => {
                    post_process_lexical(alias, output, mask, &run.input_ids, self.top_k)?
                }
            };
            Some(vectors)
        } else {
            None
        };

        let multi_vector = if heads.contains(HeadSet::MULTI_VECTOR) {
            let h = self
                .preset
                .multi_vector
                .as_ref()
                .expect("MULTI_VECTOR in available implies a multi-vector head");
            let output = run.select(alias, h.output_name, Some(h.output_index))?;
            Some(collect_token_vectors(
                alias,
                output,
                mask,
                &run.input_ids,
                h.dimensions,
                h.normalize,
                h.drop_special_tokens,
            )?)
        } else {
            None
        };

        Ok(HybridEmbedResult {
            dense,
            sparse,
            multi_vector,
            usage: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgem3_hybrid_advertises_all_three_heads() {
        let preset = presets::lookup_hybrid("BGEM3Hybrid").expect("BGEM3Hybrid preset");
        assert_eq!(available_heads_of(preset), HeadSet::ALL);
    }
}
