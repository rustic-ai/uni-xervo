// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Learned-sparse embedding task for `local/onnx`.
//!
//! Runs a text encoder and converts its raw output into a [`SparseVector`] per
//! input. Two recipes are supported via the preset / `sparse_method` option:
//!
//! - **`Mlm`** (SPLADE family): the model outputs MLM logits over the full
//!   vocabulary. We apply `log(1 + ReLU(x))`, mask padding, and max-pool over
//!   the sequence. Term ids are vocabulary indices, so the model performs term
//!   *expansion*.
//! - **`Lexical`** (BGE-M3 sparse head): the model outputs one weight per token.
//!   We key each position by its input token id and keep the max weight per id.
//!   No term expansion — keys are the literal input terms.
//!
//! Configuration merges a [`presets::SparsePreset`] (matched on `model_id`) with
//! per-spec `options`. Forward-pass and tokenization plumbing is shared via
//! [`super::common`].

use async_trait::async_trait;
use ndarray::Ix3;
use std::collections::HashMap;
use std::sync::Arc;

use super::common::{EncoderModel, EncoderSpec};
use super::presets::{self, SparseMethod};
use crate::api::ModelAliasSpec;
use crate::error::{Result, RuntimeError};
use crate::traits::{SparseEmbedResult, SparseEmbeddingModel, SparseVector};

/// Load the ONNX sparse-embedding model for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::EmbedSparse`.
///
/// # Errors
/// Returns an error if configuration is incomplete or the model fails to load.
pub(super) async fn load_sparse_embedder(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn SparseEmbeddingModel>> {
    let embedder = OnnxSparseEmbedder::load(spec).await?;
    Ok(Arc::new(embedder) as Arc<dyn SparseEmbeddingModel>)
}

/// Resolved sparse configuration after preset + options merge.
struct SparseConfig {
    encoder: EncoderSpec,
    method: SparseMethod,
    output_name: Option<String>,
    output_index: usize,
    vocab_size: u32,
    /// Optional cap on the number of retained terms per vector (largest weights).
    top_k: Option<usize>,
}

impl SparseConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let preset = presets::lookup_sparse(&spec.model_id);
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
                    "Sparse model '{}' has no preset and no `artifact` option supplying the .onnx path",
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

        let method = match opts.get("sparse_method").and_then(|v| v.as_str()) {
            Some("mlm") => SparseMethod::Mlm,
            Some("lexical") => SparseMethod::Lexical,
            Some(other) => {
                return Err(RuntimeError::Config(format!(
                    "Unknown sparse_method '{other}' for alias '{}'; expected mlm or lexical",
                    spec.alias
                )));
            }
            None => preset.map(|p| p.method).ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Sparse model '{}' has no preset and no `sparse_method` option",
                    spec.alias
                ))
            })?,
        };

        let vocab_size = preset.map(|p| p.vocab_size).unwrap_or(0);

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

        let top_k = opts
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Ok(Self {
            encoder: EncoderSpec {
                hf_repo,
                onnx_path,
                tokenizer_path,
                additional_files,
                max_seq_len,
            },
            method,
            output_name,
            output_index,
            vocab_size,
            top_k,
        })
    }
}

/// ONNX-backed learned-sparse embedder.
struct OnnxSparseEmbedder {
    encoder: EncoderModel,
    model_id: String,
    method: SparseMethod,
    output_name: Option<String>,
    output_index: usize,
    vocab_size: u32,
    top_k: Option<usize>,
}

impl OnnxSparseEmbedder {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = SparseConfig::resolve(spec)?;
        let encoder = EncoderModel::load(spec, cfg.encoder).await?;
        Ok(Self {
            encoder,
            model_id: spec.model_id.clone(),
            method: cfg.method,
            output_name: cfg.output_name,
            output_index: cfg.output_index,
            vocab_size: cfg.vocab_size,
            top_k: cfg.top_k,
        })
    }
}

impl crate::traits::ModelInfo for OnnxSparseEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.encoder.execution_providers()
    }
}

#[async_trait]
impl SparseEmbeddingModel for OnnxSparseEmbedder {
    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    async fn embed(&self, texts: &[&str]) -> Result<SparseEmbedResult> {
        if texts.is_empty() {
            return Ok(SparseEmbedResult {
                vectors: vec![],
                usage: None,
            });
        }

        let run = self.encoder.run(texts)?;
        let alias = self.encoder.alias();
        let output = run.select(alias, self.output_name.as_deref(), Some(self.output_index))?;
        let mask = &run.attention_mask;

        let vectors = match self.method {
            SparseMethod::Mlm => post_process_mlm(alias, output, mask, self.top_k)?,
            SparseMethod::Lexical => {
                post_process_lexical(alias, output, mask, &run.input_ids, self.top_k)?
            }
        };

        Ok(SparseEmbedResult {
            vectors,
            usage: None,
        })
    }
}

/// Truncate a sparse vector to the `top_k` largest weights, if configured.
fn apply_top_k(mut v: SparseVector, top_k: Option<usize>) -> SparseVector {
    if let Some(k) = top_k
        && v.len() > k
    {
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(k);
    }
    v
}

/// SPLADE: `log(1 + ReLU(logits))`, mask, max-pool over the sequence.
///
/// `output` is the raw `[batch, seq, vocab]` MLM-logits tensor. Term ids are
/// vocabulary indices (term expansion).
///
/// # Errors
/// Returns an error if `output` is not 3-dimensional.
pub(super) fn post_process_mlm(
    alias: &str,
    output: &ndarray::ArrayD<f32>,
    mask: &ndarray::Array2<i64>,
    top_k: Option<usize>,
) -> Result<Vec<SparseVector>> {
    let logits = output.view().into_dimensionality::<Ix3>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("expected 3-D SPLADE logits [batch, seq, vocab], got: {e}"),
        }
    })?;
    let (batch, seq, vocab) = (logits.shape()[0], logits.shape()[1], logits.shape()[2]);

    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        // Max over sequence positions of log(1 + ReLU(x)) for each vocab term.
        let mut weights = vec![0.0f32; vocab];
        for s in 0..seq {
            if mask[[b, s]] == 0 {
                continue;
            }
            for (v, w) in weights.iter_mut().enumerate() {
                let activated = logits[[b, s, v]].max(0.0).ln_1p();
                if activated > *w {
                    *w = activated;
                }
            }
        }
        let vector: SparseVector = weights
            .into_iter()
            .enumerate()
            .filter(|&(_, w)| w > 0.0)
            .map(|(v, w)| (v as u32, w))
            .collect();
        out.push(apply_top_k(vector, top_k));
    }
    Ok(out)
}

/// BGE-M3 lexical: per-token ReLU weight keyed by input token id, max over dups.
///
/// `output` is the per-token weights tensor, accepted as `[batch, seq]` or
/// `[batch, seq, 1]`. Term ids are the input token ids (no expansion); padding
/// positions and non-positive weights are dropped.
///
/// # Errors
/// Returns an error if `output` is neither `[batch, seq]` nor `[batch, seq, 1]`.
pub(super) fn post_process_lexical(
    alias: &str,
    output: &ndarray::ArrayD<f32>,
    mask: &ndarray::Array2<i64>,
    input_ids: &ndarray::Array2<i64>,
    top_k: Option<usize>,
) -> Result<Vec<SparseVector>> {
    // Accept [batch, seq] or [batch, seq, 1]; squeeze the trailing unit dim.
    let weights = match output.ndim() {
        2 => output.view().into_dimensionality::<ndarray::Ix2>().ok(),
        3 => output
            .view()
            .into_dimensionality::<Ix3>()
            .ok()
            .and_then(|v| {
                if v.shape()[2] == 1 {
                    Some(v.index_axis_move(ndarray::Axis(2), 0))
                } else {
                    None
                }
            }),
        _ => None,
    }
    .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
        alias: alias.to_string(),
        cause: "expected lexical weights [batch, seq] or [batch, seq, 1]".to_string(),
    })?;

    let (batch, seq) = (weights.shape()[0], weights.shape()[1]);
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut by_term: HashMap<u32, f32> = HashMap::new();
        for s in 0..seq {
            if mask[[b, s]] == 0 {
                continue;
            }
            let w = weights[[b, s]].max(0.0);
            if w <= 0.0 {
                continue;
            }
            let term = input_ids[[b, s]] as u32;
            let entry = by_term.entry(term).or_insert(0.0);
            if w > *entry {
                *entry = w;
            }
        }
        out.push(apply_top_k(by_term.into_iter().collect(), top_k));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn spec(model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: "embed_sparse/t".to_string(),
            task: crate::api::ModelTask::EmbedSparse,
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

    // ── SPLADE (mlm) post-processing ────────────────────────────────────

    #[test]
    fn mlm_activates_log1p_relu_and_max_pools_over_sequence() {
        // batch=1, seq=2, vocab=3. Negatives clamp to 0; max over the two
        // positions wins per term. Both positions unmasked.
        let logits = Array3::from_shape_vec(
            (1, 2, 3),
            vec![
                // s0: term0=1.0, term1=-5.0, term2=0.0
                1.0, -5.0, 0.0, // s1: term0=0.5, term1=2.0, term2=-1.0
                0.5, 2.0, -1.0,
            ],
        )
        .unwrap()
        .into_dyn();
        let mask = Array2::<i64>::ones((1, 2));

        let out = post_process_mlm("t", &logits, &mask, None).unwrap();
        assert_eq!(out.len(), 1);
        let mut v = out[0].clone();
        v.sort_by_key(|&(t, _)| t);
        // term0 = max(ln(2), ln(1.5)) = ln(2); term1 = max(ln1p(0), ln(3)) = ln(3).
        // term2 = max(ln1p(0), ln1p(0)) = 0 -> dropped.
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, 0);
        assert!((v[0].1 - 2.0f32.ln()).abs() < 1e-6);
        assert_eq!(v[1].0, 1);
        assert!((v[1].1 - 3.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn mlm_respects_attention_mask() {
        // s1 is masked out, so term1's large activation must not appear.
        let logits = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 9.0])
            .unwrap()
            .into_dyn();
        let mask = Array2::from_shape_vec((1, 2), vec![1, 0]).unwrap();
        let out = post_process_mlm("t", &logits, &mask, None).unwrap();
        // Only term0 from s0 survives.
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].0, 0);
    }

    #[test]
    fn mlm_top_k_keeps_largest_weights() {
        let logits = Array3::from_shape_vec((1, 1, 4), vec![3.0, 1.0, 2.0, 0.5])
            .unwrap()
            .into_dyn();
        let mask = Array2::<i64>::ones((1, 1));
        let out = post_process_mlm("t", &logits, &mask, Some(2)).unwrap();
        assert_eq!(out[0].len(), 2);
        // The two retained must be the largest-logit terms (0 and 2).
        let ids: Vec<u32> = out[0].iter().map(|&(t, _)| t).collect();
        assert!(ids.contains(&0) && ids.contains(&2));
    }

    #[test]
    fn mlm_rejects_non_3d_output() {
        let two_d = Array2::<f32>::zeros((1, 3)).into_dyn();
        let mask = Array2::<i64>::ones((1, 1));
        assert!(post_process_mlm("t", &two_d, &mask, None).is_err());
    }

    // ── BGE-M3 (lexical) post-processing ────────────────────────────────

    #[test]
    fn lexical_keys_by_token_id_and_maxes_duplicates() {
        // seq=3, token ids [5, 5, 9]; weights [0.2, 0.7, 0.4].
        // term 5 -> max(0.2, 0.7) = 0.7; term 9 -> 0.4.
        let weights = Array2::from_shape_vec((1, 3), vec![0.2, 0.7, 0.4])
            .unwrap()
            .into_dyn();
        let mask = Array2::<i64>::ones((1, 3));
        let input_ids = Array2::from_shape_vec((1, 3), vec![5, 5, 9]).unwrap();

        let out = post_process_lexical("t", &weights, &mask, &input_ids, None).unwrap();
        let mut v = out[0].clone();
        v.sort_by_key(|&(t, _)| t);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, 5);
        assert!((v[0].1 - 0.7).abs() < 1e-6);
        assert_eq!(v[1].0, 9);
    }

    #[test]
    fn lexical_drops_padding_and_nonpositive_weights() {
        // seq=3: pos0 masked, pos1 weight 0 (dropped), pos2 kept.
        let weights = Array2::from_shape_vec((1, 3), vec![5.0, 0.0, 0.3])
            .unwrap()
            .into_dyn();
        let mask = Array2::from_shape_vec((1, 3), vec![0, 1, 1]).unwrap();
        let input_ids = Array2::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
        let out = post_process_lexical("t", &weights, &mask, &input_ids, None).unwrap();
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].0, 3);
    }

    #[test]
    fn lexical_accepts_squeezable_3d_shape() {
        // [batch, seq, 1] is squeezed to [batch, seq].
        let weights = Array3::from_shape_vec((1, 2, 1), vec![0.5, 0.9])
            .unwrap()
            .into_dyn();
        let mask = Array2::<i64>::ones((1, 2));
        let input_ids = Array2::from_shape_vec((1, 2), vec![7, 8]).unwrap();
        let out = post_process_lexical("t", &weights, &mask, &input_ids, None).unwrap();
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn lexical_rejects_multi_channel_3d() {
        // [batch, seq, 2] is not a valid lexical-weights shape.
        let weights = Array3::<f32>::zeros((1, 2, 2)).into_dyn();
        let mask = Array2::<i64>::ones((1, 2));
        let input_ids = Array2::from_shape_vec((1, 2), vec![1, 2]).unwrap();
        assert!(post_process_lexical("t", &weights, &mask, &input_ids, None).is_err());
    }

    #[test]
    fn apply_top_k_truncates_to_largest() {
        let v = vec![(1, 0.1), (2, 0.9), (3, 0.5)];
        let out = apply_top_k(v, Some(2));
        assert_eq!(out.len(), 2);
        let ids: Vec<u32> = out.iter().map(|&(t, _)| t).collect();
        assert!(ids.contains(&2) && ids.contains(&3));
    }

    #[test]
    fn apply_top_k_noop_when_under_limit() {
        let v = vec![(1, 0.1), (2, 0.9)];
        assert_eq!(apply_top_k(v.clone(), Some(5)), v);
        assert_eq!(apply_top_k(v.clone(), None), v);
    }

    // ── Config resolution (preset + options merge, failure paths) ────────

    #[test]
    fn resolve_uses_preset_method_and_vocab() {
        let cfg =
            SparseConfig::resolve(&spec("prithivida/Splade_PP_en_v1", serde_json::Value::Null))
                .unwrap();
        assert_eq!(cfg.method, SparseMethod::Mlm);
        assert_eq!(cfg.vocab_size, 30522);
    }

    #[test]
    fn resolve_option_overrides_preset_method() {
        let cfg = SparseConfig::resolve(&spec(
            "prithivida/Splade_PP_en_v1",
            serde_json::json!({ "sparse_method": "lexical", "top_k": 5 }),
        ))
        .unwrap();
        assert_eq!(cfg.method, SparseMethod::Lexical);
        assert_eq!(cfg.top_k, Some(5));
    }

    #[test]
    fn resolve_rejects_unknown_sparse_method() {
        let err = SparseConfig::resolve(&spec(
            "prithivida/Splade_PP_en_v1",
            serde_json::json!({ "sparse_method": "bm25" }),
        ));
        assert!(err.is_err());
    }

    #[test]
    fn resolve_without_preset_requires_artifact_and_method() {
        // Unknown model, no options → no .onnx path and no method → error.
        let err = SparseConfig::resolve(&spec("unknown/model", serde_json::Value::Null));
        assert!(err.is_err());
    }
}
