// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Cross-encoder rerank task for `local/onnx`.
//!
//! Loads ONNX cross-encoder models such as `cross-encoder/ms-marco-MiniLM-L6-v2`
//! and `BAAI/bge-reranker-base`, handles tokenization (via the `tokenizers`
//! crate), and runs batched ONNX inference. Expects models that accept
//! `input_ids` + `attention_mask` (and optionally `token_type_ids` —
//! some BERT-family exports include the segment-id input, others, e.g.
//! `BAAI/bge-reranker-base` and XLM-R-based cross-encoders, omit it; the
//! presence is auto-detected from `session.inputs()`) and produce a single
//! logit per (query, document) pair (output shape `[batch, 1]` or `[batch]`).
//!
//! Scores are returned as **raw logits** — apply sigmoid or softmax in
//! the caller if you need a normalized `[0, 1]` domain.

use async_trait::async_trait;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, effective_ep_list,
    parse_execution_providers_option,
};
use crate::traits::{RerankerModel, ScoredDoc};

/// Default max sequence length for BERT-based cross-encoders.
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Load the ONNX cross-encoder reranker model for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::Rerank`.
pub(super) async fn load_rerank(spec: &ModelAliasSpec) -> Result<Arc<dyn RerankerModel>> {
    let reranker = OnnxCrossEncoder::load(spec).await?;
    Ok(Arc::new(reranker) as Arc<dyn RerankerModel>)
}

/// ONNX cross-encoder that tokenizes (query, document) pairs and runs
/// batched inference to produce relevance scores.
struct OnnxCrossEncoder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
    alias: String,
    model_id: String,
    /// Resolved execution-provider list as stable string ids
    /// (e.g. `["cuda", "cpu"]`). Surfaced through
    /// [`ModelInfo::active_execution_providers`].
    requested_eps: Vec<String>,
    /// Whether the ONNX graph declares a `token_type_ids` input.
    /// Some BERT-family exports (e.g. MiniLM cross-encoder) include
    /// it; others (e.g. `BAAI/bge-reranker-base`, XLM-R-based
    /// cross-encoders) omit it. Discovered from `session.inputs()`
    /// at load time so we don't feed an unexpected input at run.
    expects_token_type_ids: bool,
}

impl OnnxCrossEncoder {
    /// Load the ONNX model and tokenizer from a HuggingFace repo.
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let max_seq_len = spec
            .options
            .get("max_seq_len")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);

        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        let requested_eps: Vec<String> = effective_ep_list(execution_providers.as_deref())
            .into_iter()
            .map(|ep| ep.as_str().to_string())
            .collect();

        // Validate the requested EP list FIRST (before preflight or I/O).
        // Misconfigurations (e.g. CUDA requested without `gpu-cuda` enabled)
        // fail fast with a precise "feature not enabled" error rather than
        // a generic dylib-missing error or a wasted HF download.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        // Pre-flight (load-dynamic only): verify the ONNX Runtime dylib is
        // loadable before any ort API call. Sidesteps the upstream
        // load-dynamic OnceLock deadlock (pykeio/ort#560) when the dylib
        // is missing. Compiled out under `provider-onnx` (bundled CPU)
        // where the lib is statically linked into the binary.
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-reranker", &spec.model_id, &spec.options);
        let (model_path, tokenizer_path) = download_model_files(
            &spec.alias,
            &spec.model_id,
            spec.revision.as_deref(),
            &cache_dir,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            model_path = %model_path.display(),
            "Loading ONNX cross-encoder"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        let expects_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_seq_len,
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            requested_eps,
            expects_token_type_ids,
        })
    }

    /// Tokenize a batch of (query, document) pairs into padded tensors.
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids)` as i64 2D arrays,
    /// each with shape `[batch_size, padded_seq_len]`.
    fn tokenize_batch(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>)> {
        let batch_size = documents.len();

        // Tokenize each (query, doc) pair
        let encodings: Vec<tokenizers::Encoding> = documents
            .iter()
            .map(|doc| {
                self.tokenizer.encode((query, *doc), true).map_err(|e| {
                    RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Tokenization failed: {e}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Determine padded sequence length (capped at max_seq_len)
        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        // Build padded tensors
        let mut input_ids = Array2::<i64>::zeros((batch_size, padded_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, padded_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, padded_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let seq_len = ids.len().min(self.max_seq_len);

            for j in 0..seq_len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                token_type_ids[[i, j]] = types[j] as i64;
            }
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }
}

impl crate::traits::ModelInfo for OnnxCrossEncoder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.requested_eps.clone()
    }
}

#[async_trait]
impl RerankerModel for OnnxCrossEncoder {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        let (input_ids, attention_mask, token_type_ids) = self.tokenize_batch(query, docs)?;

        // Run inference (Session::run is blocking, so we hold the lock briefly)
        let logits = {
            let mut session =
                self.session
                    .lock()
                    .map_err(|e| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Session lock poisoned: {e}"),
                    })?;

            let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("{ctx}: {e}"),
            };

            let input_ids_tensor = ort::value::Tensor::from_array(input_ids.into_dyn())
                .map_err(|e| map_err(e, "input_ids tensor"))?;
            let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask.into_dyn())
                .map_err(|e| map_err(e, "attention_mask tensor"))?;

            let mut inputs: Vec<(String, ort::value::DynTensor)> = vec![
                ("input_ids".to_string(), input_ids_tensor.upcast()),
                ("attention_mask".to_string(), attention_mask_tensor.upcast()),
            ];
            if self.expects_token_type_ids {
                let token_type_ids_tensor =
                    ort::value::Tensor::from_array(token_type_ids.into_dyn())
                        .map_err(|e| map_err(e, "token_type_ids tensor"))?;
                inputs.push(("token_type_ids".to_string(), token_type_ids_tensor.upcast()));
            }

            // Get output name before run() to avoid borrow conflict
            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "logits".to_string());

            let outputs = session
                .run(inputs)
                .map_err(|e| map_err(e, "ONNX inference"))?;

            // Extract logits from first output — shape is typically [batch, 1] or [batch]
            let output =
                outputs
                    .get(&output_name)
                    .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Missing output tensor '{output_name}'"),
                    })?;
            let view = output.try_extract_array::<f32>().map_err(|e| {
                RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("Failed to extract output array: {e}"),
                }
            })?;

            // Handle both [batch, 1] and [batch] output shapes
            let scores: Vec<f32> = if view.ndim() == 2 {
                view.axis_iter(Axis(0)).map(|row| row[[0]]).collect()
            } else {
                view.iter().copied().collect()
            };

            scores
        };

        // Build ScoredDoc results sorted by score descending
        let mut scored: Vec<ScoredDoc> = logits
            .into_iter()
            .enumerate()
            .map(|(index, score)| ScoredDoc {
                index,
                score,
                text: None,
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored)
    }
}

/// Download the ONNX model and tokenizer files from HuggingFace.
///
/// Returns `(model_path, tokenizer_path)`. Tries `onnx/model.onnx` first
/// (the layout used by `optimum`-exported models) and falls back to
/// `model.onnx` at the repo root.
async fn download_model_files(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        })?;

    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::model(model_id.to_string()),
    };
    let api_repo = api.repo(repo);

    // Download model file — try `onnx/model.onnx` first, then `model.onnx`
    let model_path =
        match api_repo.get("onnx/model.onnx").await {
            Ok(path) => path,
            Err(_) => api_repo.get("model.onnx").await.map_err(|e| {
                RuntimeError::OnnxDownloadFailure {
                    alias: alias.to_string(),
                    cause: format!(
                        "Could not download ONNX model (tried onnx/model.onnx and model.onnx): {e}"
                    ),
                }
            })?,
        };

    // Download tokenizer
    let tokenizer_path =
        api_repo
            .get("tokenizer.json")
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download tokenizer.json: {e}"),
            })?;

    Ok((model_path, tokenizer_path))
}

/// Build an ORT session with sensible defaults for reranker inference.
///
/// Shared between the cross-encoder and generative reranker code paths —
/// both want the same optimization level and EP-dispatch behaviour.
///
/// `execution_providers` is the parsed user-supplied list (or `None` to use
/// the feature-aware defaults from
/// [`crate::provider::onnx_ep::default_execution_providers`]).
pub(super) fn build_session(
    path: &Path,
    spec: &ModelAliasSpec,
    execution_providers: Option<&[OnnxExecutionProvider]>,
) -> Result<Session> {
    let builder = Session::builder().map_err(|e| RuntimeError::OnnxLoadFailure {
        alias: spec.alias.clone(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    })?;

    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;

    let dispatch = build_execution_providers(execution_providers, &spec.alias, "local/onnx")?;
    builder =
        builder
            .with_execution_providers(dispatch)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;

    builder
        .commit_from_file(path)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
}
