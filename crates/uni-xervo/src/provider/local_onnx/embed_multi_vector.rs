// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Multi-vector (ColBERT / late-interaction) embedding task for `local/onnx`.
//!
//! Runs a text encoder and returns its per-token output *without* pooling: one
//! vector per retained token, per input. Padding tokens (and, optionally,
//! special tokens) are stripped, and each vector is optionally L2-normalized so
//! that downstream MaxSim is cosine similarity.
//!
//! Targets in-graph-projection exports — `answerai-colbert-small`
//! (`vespa_colbert.onnx`, 384->96) and the BGE-M3 ColBERT head — so no separate
//! projection weights are needed. Configuration merges a
//! [`presets::MultiVectorPreset`] (matched on `model_id`) with per-spec
//! `options`. Forward-pass plumbing is shared via [`super::common`].

use async_trait::async_trait;
use ndarray::{Axis, Ix3};
use std::sync::Arc;

use super::common::{EncoderModel, EncoderSpec};
use super::presets;
use crate::api::ModelAliasSpec;
use crate::error::{Result, RuntimeError};
use crate::traits::{MultiVectorEmbedResult, MultiVectorEmbeddingModel};

/// Load the ONNX multi-vector embedding model for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::EmbedMultiVector`.
///
/// # Errors
/// Returns an error if configuration is incomplete or the model fails to load.
pub(super) async fn load_multi_vector_embedder(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn MultiVectorEmbeddingModel>> {
    let embedder = OnnxMultiVectorEmbedder::load(spec).await?;
    Ok(Arc::new(embedder) as Arc<dyn MultiVectorEmbeddingModel>)
}

/// Resolved multi-vector configuration after preset + options merge.
struct MultiVectorConfig {
    encoder: EncoderSpec,
    output_name: Option<String>,
    output_index: usize,
    dimensions: u32,
    normalize: bool,
    drop_special_tokens: bool,
}

impl MultiVectorConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let preset = presets::lookup_multi_vector(&spec.model_id);
        let opts = &spec.options;

        let hf_repo = preset
            .map(|p| p.hf_repo.to_string())
            .unwrap_or_else(|| spec.model_id.clone());

        let onnx_path = opts
            .get("artifact")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| preset.map(|p| p.onnx_path.to_string()))
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Multi-vector model '{}' has no preset and no `artifact` option supplying the .onnx path",
                    spec.alias
                ))
            })?;

        let tokenizer_path = opts
            .get("tokenizer_path")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| preset.map(|p| p.tokenizer_path.to_string()))
            .unwrap_or_else(|| "tokenizer.json".to_string());

        let additional_files: Vec<String> = preset
            .map(|p| p.additional_files.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        let dimensions = opts
            .get("dimensions")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| preset.map(|p| p.dimensions))
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Multi-vector model '{}' has no preset and no `dimensions` option",
                    spec.alias
                ))
            })?;

        let normalize = opts
            .get("normalize")
            .and_then(|v| v.as_bool())
            .or_else(|| preset.map(|p| p.normalize))
            .unwrap_or(true);

        let max_seq_len = opts
            .get("max_seq_len")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| preset.map(|p| p.max_seq_len))
            .unwrap_or(512);

        let output_name = opts
            .get("output_name")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| preset.and_then(|p| p.output_name.map(str::to_string)));

        let output_index = opts
            .get("output_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| preset.map(|p| p.output_index))
            .unwrap_or(0);

        // Default `false`: most ColBERT exports already mask special-token
        // vectors or downstream MaxSim tolerates them. Opt in when needed.
        let drop_special_tokens = opts
            .get("drop_special_tokens")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            encoder: EncoderSpec {
                hf_repo,
                onnx_path,
                tokenizer_path,
                additional_files,
                max_seq_len,
            },
            output_name,
            output_index,
            dimensions,
            normalize,
            drop_special_tokens,
        })
    }
}

/// ONNX-backed multi-vector (ColBERT) embedder.
struct OnnxMultiVectorEmbedder {
    encoder: EncoderModel,
    model_id: String,
    output_name: Option<String>,
    output_index: usize,
    dimensions: u32,
    normalize: bool,
    drop_special_tokens: bool,
}

impl OnnxMultiVectorEmbedder {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = MultiVectorConfig::resolve(spec)?;
        let encoder = EncoderModel::load(spec, cfg.encoder).await?;
        Ok(Self {
            encoder,
            model_id: spec.model_id.clone(),
            output_name: cfg.output_name,
            output_index: cfg.output_index,
            dimensions: cfg.dimensions,
            normalize: cfg.normalize,
            drop_special_tokens: cfg.drop_special_tokens,
        })
    }
}

/// L2-normalize a single token vector in place. Leaves near-zero vectors as-is.
fn l2_normalize(vec: &mut [f32]) {
    const EPS: f32 = 1e-12;
    let norm = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > EPS {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

impl crate::traits::ModelInfo for OnnxMultiVectorEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.encoder.execution_providers()
    }
}

#[async_trait]
impl MultiVectorEmbeddingModel for OnnxMultiVectorEmbedder {
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    async fn embed(&self, texts: &[&str]) -> Result<MultiVectorEmbedResult> {
        if texts.is_empty() {
            return Ok(MultiVectorEmbedResult {
                vectors: vec![],
                usage: None,
            });
        }

        let run = self.encoder.run(texts)?;
        let alias = self.encoder.alias();
        let output = run.select(alias, self.output_name.as_deref(), Some(self.output_index))?;

        let vectors = collect_token_vectors(
            alias,
            output,
            &run.attention_mask,
            &run.input_ids,
            self.dimensions,
            self.normalize,
            self.drop_special_tokens,
        )?;

        Ok(MultiVectorEmbedResult {
            vectors,
            usage: None,
        })
    }
}

/// Strip padding (and optionally special) tokens and emit ragged per-token vectors.
///
/// `output` is the raw `[batch, seq, dim]` per-token tensor. Each retained token
/// (attention-mask == 1, and not a special token when `drop_special_tokens`) is
/// emitted as one vector, optionally L2-normalized.
///
/// # Errors
/// Returns an error if `output` is not 3-D or its last dimension does not match
/// the configured `dimensions` (the loud-failure guard against an export that
/// emits un-projected hidden states).
pub(super) fn collect_token_vectors(
    alias: &str,
    output: &ndarray::ArrayD<f32>,
    mask: &ndarray::Array2<i64>,
    input_ids: &ndarray::Array2<i64>,
    dimensions: u32,
    normalize: bool,
    drop_special_tokens: bool,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let hidden = output.view().into_dimensionality::<Ix3>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("expected 3-D per-token output [batch, seq, dim], got: {e}"),
        }
    })?;
    let (batch, seq, dim) = (hidden.shape()[0], hidden.shape()[1], hidden.shape()[2]);

    if dim as u32 != dimensions {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!(
                "per-token dimension mismatch: configured {dimensions} but model produced {dim}"
            ),
        });
    }

    let mut vectors = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut tokens = Vec::new();
        for s in 0..seq {
            if mask[[b, s]] == 0 {
                continue;
            }
            if drop_special_tokens && is_special_token(input_ids[[b, s]]) {
                continue;
            }
            let mut v: Vec<f32> = hidden
                .index_axis(Axis(0), b)
                .index_axis(Axis(0), s)
                .to_vec();
            if normalize {
                l2_normalize(&mut v);
            }
            tokens.push(v);
        }
        vectors.push(tokens);
    }
    Ok(vectors)
}

/// Heuristic for the common special-token ids shared by BERT / XLM-R tokenizers.
///
/// Covers `[PAD]=0`, `[CLS]/<s>=101/0`, `[SEP]/</s>=102/2`, `[UNK]=100`, and
/// XLM-R `<pad>=1`. This is a best-effort filter for `drop_special_tokens`; for
/// exact control supply pre-tokenized input or leave the flag off.
fn is_special_token(id: i64) -> bool {
    matches!(id, 0 | 1 | 2 | 100 | 101 | 102 | 103)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn spec(model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: "embed_mv/t".to_string(),
            task: crate::api::ModelTask::EmbedMultiVector,
            provider_id: "local/onnx".to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: crate::api::WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options,
        }
    }

    // ── collect_token_vectors ───────────────────────────────────────────

    #[test]
    fn collects_one_vector_per_unmasked_token() {
        // batch=1, seq=3, dim=2; pos1 masked out -> 2 tokens retained.
        let hidden = Array3::from_shape_vec((1, 3, 2), vec![1.0, 0.0, 9.0, 9.0, 0.0, 1.0])
            .unwrap()
            .into_dyn();
        let mask = Array2::from_shape_vec((1, 3), vec![1, 0, 1]).unwrap();
        let input_ids = Array2::from_shape_vec((1, 3), vec![50, 51, 52]).unwrap();

        let out = collect_token_vectors("t", &hidden, &mask, &input_ids, 2, false, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2); // the masked token is dropped
        assert_eq!(out[0][0], vec![1.0, 0.0]);
        assert_eq!(out[0][1], vec![0.0, 1.0]);
    }

    #[test]
    fn normalizes_each_token_to_unit_norm() {
        let hidden = Array3::from_shape_vec((1, 1, 2), vec![3.0, 4.0])
            .unwrap()
            .into_dyn();
        let mask = Array2::<i64>::ones((1, 1));
        let input_ids = Array2::from_shape_vec((1, 1), vec![50]).unwrap();
        let out = collect_token_vectors("t", &hidden, &mask, &input_ids, 2, true, false).unwrap();
        // [3,4] / 5 = [0.6, 0.8]
        assert!((out[0][0][0] - 0.6).abs() < 1e-6);
        assert!((out[0][0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn dimension_mismatch_guard_fails_loudly() {
        // Model produced dim 3 but caller configured 128 (the un-projected
        // hidden-states case). Must error, not silently return wrong vectors.
        let hidden = Array3::<f32>::zeros((1, 1, 3)).into_dyn();
        let mask = Array2::<i64>::ones((1, 1));
        let input_ids = Array2::from_shape_vec((1, 1), vec![50]).unwrap();
        let err = collect_token_vectors("t", &hidden, &mask, &input_ids, 128, true, false);
        assert!(err.is_err(), "dim mismatch must error");
    }

    #[test]
    fn drop_special_tokens_filters_cls_sep() {
        // ids: 101 ([CLS]), 50 (word), 102 ([SEP]) -> only the word survives.
        let hidden = Array3::from_shape_vec((1, 3, 2), vec![1.0, 1.0, 2.0, 0.0, 3.0, 3.0])
            .unwrap()
            .into_dyn();
        let mask = Array2::<i64>::ones((1, 3));
        let input_ids = Array2::from_shape_vec((1, 3), vec![101, 50, 102]).unwrap();
        let out = collect_token_vectors("t", &hidden, &mask, &input_ids, 2, false, true).unwrap();
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0], vec![2.0, 0.0]);
    }

    #[test]
    fn ragged_output_varies_token_count_per_input() {
        // Two inputs with different unmasked token counts.
        let hidden = Array3::<f32>::ones((2, 3, 2)).into_dyn();
        let mask = Array2::from_shape_vec((2, 3), vec![1, 1, 1, 1, 0, 0]).unwrap();
        let input_ids = Array2::from_shape_vec((2, 3), vec![50, 51, 52, 50, 51, 52]).unwrap();
        let out = collect_token_vectors("t", &hidden, &mask, &input_ids, 2, false, false).unwrap();
        assert_eq!(out[0].len(), 3);
        assert_eq!(out[1].len(), 1);
    }

    #[test]
    fn rejects_non_3d_output() {
        let two_d = Array2::<f32>::zeros((1, 2)).into_dyn();
        let mask = Array2::<i64>::ones((1, 1));
        let input_ids = Array2::from_shape_vec((1, 1), vec![1]).unwrap();
        assert!(collect_token_vectors("t", &two_d, &mask, &input_ids, 2, false, false).is_err());
    }

    #[test]
    fn l2_normalize_leaves_zero_vector() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn special_token_heuristic_matches_common_ids() {
        assert!(is_special_token(0) && is_special_token(101) && is_special_token(102));
        assert!(!is_special_token(50) && !is_special_token(2000));
    }

    // ── Config resolution (preset + options merge, failure paths) ────────

    #[test]
    fn resolve_uses_preset_dimensions() {
        let cfg = MultiVectorConfig::resolve(&spec(
            "answerdotai/answerai-colbert-small-v1",
            serde_json::Value::Null,
        ))
        .unwrap();
        assert_eq!(cfg.dimensions, 96);
        assert!(cfg.normalize);
    }

    #[test]
    fn resolve_options_override_preset() {
        let cfg = MultiVectorConfig::resolve(&spec(
            "lightonai/GTE-ModernColBERT-v1",
            serde_json::json!({ "normalize": false, "drop_special_tokens": true, "output_index": 0 }),
        ))
        .unwrap();
        assert_eq!(cfg.dimensions, 128);
        assert!(!cfg.normalize);
        assert!(cfg.drop_special_tokens);
    }

    #[test]
    fn resolve_without_preset_requires_artifact_and_dimensions() {
        let err = MultiVectorConfig::resolve(&spec("unknown/model", serde_json::Value::Null));
        assert!(err.is_err());
    }
}
