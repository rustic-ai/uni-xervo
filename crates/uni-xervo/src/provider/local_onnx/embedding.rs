// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Dense embedding task for `local/onnx`.
//!
//! Loads an ONNX text-encoder (BERT/MiniLM/MPNet/E5/BGE/etc.), tokenizes
//! input strings via the `tokenizers` crate, runs batched inference, and
//! reduces last-hidden-state to a single vector per input via the
//! configured pooling strategy. Output is optionally L2-normalized.
//!
//! Configuration comes from two sources, merged in this order (later wins):
//!
//! 1. A built-in [`presets`](super::presets) entry matched on `model_id`.
//! 2. Per-spec `options` (artifact path, pooling, dimensions, normalize,
//!    max_seq_len, token_type_ids, output_name, execution_providers, etc.).
//!
//! Models that lack a preset must supply enough options to fully describe
//! the model; otherwise loading fails with [`RuntimeError::Config`].

use crate::hf::HfModelRepo;
use async_trait::async_trait;
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::presets::{self, PoolingKind};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, ort_environment,
    parse_execution_providers_option,
};
use crate::traits::{EmbedResult, EmbeddingModel};

/// Default truncation cap when neither preset nor options specify one.
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Tokenizer output for one batch: `(input_ids, attention_mask, token_type_ids?)`.
type TokenizedBatch = (Array2<i64>, Array2<i64>, Option<Array2<i64>>);

/// Load the ONNX embedding model for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::Embed`.
pub(super) async fn load_embedding(spec: &ModelAliasSpec) -> Result<Arc<dyn EmbeddingModel>> {
    let embedder = OnnxEmbedder::load(spec).await?;
    Ok(Arc::new(embedder) as Arc<dyn EmbeddingModel>)
}

/// Resolved embedding configuration after preset + options merge.
struct EmbedConfig {
    /// HF repo to download from. Comes from the preset (which maps short
    /// aliases like `"BGESmallENV15"` to canonical repos like
    /// `"BAAI/bge-small-en-v1.5"`) or falls back to `spec.model_id`.
    hf_repo: String,
    onnx_path: String,
    tokenizer_path: String,
    /// Additional files that must accompany the `.onnx` graph in the cache
    /// directory (e.g. external-data `.onnx_data` sidecars). Sourced from
    /// the matched preset; empty for pass-through models.
    additional_files: Vec<String>,
    dimensions: u32,
    pooling: PoolingKind,
    normalize: bool,
    max_seq_len: usize,
    token_type_ids: bool,
    output_name: Option<String>,
}

impl EmbedConfig {
    /// Merge a preset (if any) with per-spec options into a single config.
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let preset = presets::lookup(&spec.model_id);
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
                    "Embedding model '{}' has no preset and no `artifact` option supplying the .onnx path within the repo",
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

        let pooling = match opts.get("pooling").and_then(|v| v.as_str()) {
            Some("cls") => PoolingKind::Cls,
            Some("mean") => PoolingKind::Mean,
            Some("max") => PoolingKind::Max,
            Some("last-token") => PoolingKind::LastToken,
            Some(other) => {
                return Err(RuntimeError::Config(format!(
                    "Unknown pooling '{other}' for alias '{}'; expected one of cls|mean|max|last-token",
                    spec.alias
                )));
            }
            None => preset.map(|p| p.pooling).ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Embedding model '{}' has no preset and no `pooling` option",
                    spec.alias
                ))
            })?,
        };

        let dimensions = opts
            .get("dimensions")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| preset.map(|p| p.dimensions))
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Embedding model '{}' has no preset and no `dimensions` option",
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
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);

        let token_type_ids = opts
            .get("token_type_ids")
            .and_then(|v| v.as_bool())
            .or_else(|| preset.map(|p| p.token_type_ids))
            .unwrap_or(true);

        let output_name = opts
            .get("output_name")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| preset.and_then(|p| p.output_name.map(str::to_string)));

        Ok(Self {
            hf_repo,
            onnx_path,
            tokenizer_path,
            additional_files,
            dimensions,
            pooling,
            normalize,
            max_seq_len,
            token_type_ids,
            output_name,
        })
    }
}

/// ONNX-backed text embedder.
struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    alias: String,
    model_id: String,
    dimensions: u32,
    pooling: PoolingKind,
    normalize: bool,
    max_seq_len: usize,
    token_type_ids: bool,
    output_name: Option<String>,
    /// Whether the session declares a `position_ids` input. Decoder-LM
    /// embedders (Qwen3-Embedding, NV-Embed) require it; classic
    /// encoder embedders (BGE, MiniLM, MPNet) don't have it. Detected
    /// from `session.inputs()` at load time.
    expects_position_ids: bool,
    /// Schemas for any `past_key_values.*` inputs the session expects.
    /// Empty for encoder-style embedders. For decoder-style embedders
    /// these placeholders feed zero-shaped KV tensors per call so the
    /// model performs a clean prefill with no history.
    past_kv_schemas: Vec<super::decoder_inputs::InputSchema>,
}

impl OnnxEmbedder {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = EmbedConfig::resolve(spec)?;

        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;

        // Validate the requested EP list FIRST (before preflight or I/O).
        // Misconfigurations (e.g. CUDA without `gpu-cuda`) fail fast with a
        // precise error rather than a generic dylib-missing one.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        // Preflight (load-dynamic only) sidesteps pykeio/ort#560 deadlock.
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        // Cache key uses the resolved HF repo so short-alias and full-repo
        // catalog entries share a download (e.g. `"BGESmallENV15"` and
        // `"BAAI/bge-small-en-v1.5"` resolve to the same on-disk cache).
        let cache_dir = resolve_cache_dir("onnx-embed", &cfg.hf_repo, &spec.options);
        let (model_path, tokenizer_path) = download_model_files(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
            &cfg.tokenizer_path,
            &cfg.additional_files,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            hf_repo = %cfg.hf_repo,
            model_path = %model_path.display(),
            pooling = ?cfg.pooling,
            dimensions = cfg.dimensions,
            "Loading ONNX embedder"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        // Inspect session inputs once. Encoder-style embedders (BGE,
        // MiniLM, MPNet, ModernBERT) declare only input_ids /
        // attention_mask / [token_type_ids] — both fields below stay
        // empty/false. Decoder-style embedders (Qwen3-Embedding,
        // NV-Embed) additionally declare `position_ids` and a stack of
        // `past_key_values.*` placeholders that a one-shot embed call
        // feeds with empty zero tensors.
        let (expects_position_ids, past_kv_schemas) =
            inspect_decoder_extras(&session, &spec.alias)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            dimensions: cfg.dimensions,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
            max_seq_len: cfg.max_seq_len,
            token_type_ids: cfg.token_type_ids,
            output_name: cfg.output_name,
            expects_position_ids,
            past_kv_schemas,
        })
    }

    /// Encode a batch of texts into padded `[batch, padded]` i64 tensors.
    fn tokenize_batch(&self, texts: &[&str]) -> Result<TokenizedBatch> {
        let batch_size = texts.len();

        let encodings: Vec<tokenizers::Encoding> = texts
            .iter()
            .map(|text| {
                self.tokenizer.encode(*text, true).map_err(|e| {
                    RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Tokenization failed: {e}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, padded_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, padded_len));
        let mut token_type_ids = if self.token_type_ids {
            Some(Array2::<i64>::zeros((batch_size, padded_len)))
        } else {
            None
        };

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let seq_len = ids.len().min(self.max_seq_len);

            for j in 0..seq_len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                if let Some(tt) = token_type_ids.as_mut() {
                    tt[[i, j]] = types[j] as i64;
                }
            }
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }
}

impl crate::traits::ModelInfo for OnnxEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl EmbeddingModel for OnnxEmbedder {
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult> {
        if texts.is_empty() {
            return Ok(EmbedResult {
                vectors: vec![],
                usage: None,
            });
        }

        let (input_ids, attention_mask, token_type_ids) = self.tokenize_batch(texts)?;
        let mask_for_pool = attention_mask.clone();

        let hidden = {
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

            let batch_size = input_ids.shape()[0];
            let seq_len = input_ids.shape()[1];

            let input_ids_tensor = ort::value::Tensor::from_array(input_ids.into_dyn())
                .map_err(|e| map_err(e, "input_ids tensor"))?;
            let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask.into_dyn())
                .map_err(|e| map_err(e, "attention_mask tensor"))?;

            let mut inputs: Vec<(String, ort::value::DynTensor)> = vec![
                ("input_ids".to_string(), input_ids_tensor.upcast()),
                ("attention_mask".to_string(), attention_mask_tensor.upcast()),
            ];

            if let Some(tt) = token_type_ids {
                let tt_tensor = ort::value::Tensor::from_array(tt.into_dyn())
                    .map_err(|e| map_err(e, "token_type_ids tensor"))?;
                inputs.push(("token_type_ids".to_string(), tt_tensor.upcast()));
            }

            if self.expects_position_ids {
                let mut position_ids = Array2::<i64>::zeros((batch_size, seq_len));
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        position_ids[[b, s]] = s as i64;
                    }
                }
                let pos_tensor = ort::value::Tensor::from_array(position_ids.into_dyn())
                    .map_err(|e| map_err(e, "position_ids tensor"))?;
                inputs.push(("position_ids".to_string(), pos_tensor.upcast()));
            }

            for schema in &self.past_kv_schemas {
                let kv =
                    super::decoder_inputs::build_empty_past_kv(schema, batch_size, &self.alias)?;
                inputs.push((schema.name.clone(), kv));
            }

            // Resolve output name. If the user supplied one, honor it.
            // Otherwise default to the first session output — for HF's standard
            // BERT-family ONNX exports this is `last_hidden_state`. Models that
            // expose a different first output (e.g. a pre-pooled
            // `sentence_embedding`) need an explicit `output_name` option, and
            // the rank-handling below will treat their 2-D outputs as already
            // pooled. Computed before run() to avoid a session borrow conflict.
            let chosen_output = if let Some(name) = self.output_name.as_deref() {
                name.to_string()
            } else {
                session
                    .outputs()
                    .first()
                    .map(|o| o.name().to_string())
                    .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: "ONNX session has no outputs".to_string(),
                    })?
            };

            let outputs = session
                .run(inputs)
                .map_err(|e| map_err(e, "ONNX inference"))?;

            let output =
                outputs
                    .get(&chosen_output)
                    .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Missing output tensor '{chosen_output}'"),
                    })?;

            let view = output.try_extract_array::<f32>().map_err(|e| {
                RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("Failed to extract output array: {e}"),
                }
            })?;

            // Own the extracted view before exiting the lock block — it borrows
            // from `outputs`, which drops here. The rank branch (already-pooled
            // `[batch, hidden]` pass-through vs `[batch, seq, hidden]` pooling)
            // lives in `finalize_dense`, shared with the hybrid embedder.
            view.to_owned()
        };

        let vectors = finalize_dense(
            &self.alias,
            &hidden,
            &mask_for_pool,
            self.pooling,
            self.normalize,
            self.dimensions,
        )?;
        Ok(EmbedResult {
            vectors,
            usage: None,
        })
    }
}

/// Turn a raw graph output into pooled, normalized, dimension-checked dense rows.
///
/// Shared by [`OnnxEmbedder::embed`] and the hybrid embedder so both consume a
/// dense head identically. A 2-D `[batch, hidden]` output is treated as
/// already-pooled and passes through; a 3-D `[batch, seq, hidden]` output is
/// pooled with `pooling` over `mask`. With `normalize`, rows are L2-normalized.
/// The row width is validated against `dimensions`.
///
/// # Errors
/// Returns [`RuntimeError::OnnxInvocationFailure`] if `output` is not rank 2 or
/// 3, cannot be viewed at that rank, or its width differs from `dimensions`.
pub(super) fn finalize_dense(
    alias: &str,
    output: &ndarray::ArrayD<f32>,
    mask: &Array2<i64>,
    pooling: PoolingKind,
    normalize: bool,
    dimensions: u32,
) -> Result<Vec<Vec<f32>>> {
    let mut pooled = match output.ndim() {
        3 => {
            let hidden = output
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| RuntimeError::OnnxInvocationFailure {
                    alias: alias.to_string(),
                    cause: format!("Expected 3-D hidden states, got: {e}"),
                })?
                .to_owned();
            pool(&hidden, mask, pooling)
        }
        2 => output
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("Expected 2-D pooled embeddings, got: {e}"),
            })?
            .to_owned(),
        other => {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("Unsupported output rank {other} for embedding (expected 2 or 3)"),
            });
        }
    };

    if normalize {
        l2_normalize_rows(&mut pooled);
    }

    // Validate dimensionality matches the configured/preset value to catch
    // mis-specified models early rather than corrupting downstream caches.
    let actual_dim = pooled.shape()[1] as u32;
    if actual_dim != dimensions {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!(
                "Embedding dimension mismatch: configured {dimensions} but model produced {actual_dim}"
            ),
        });
    }

    Ok(pooled.axis_iter(Axis(0)).map(|row| row.to_vec()).collect())
}

/// Apply the configured pooling to last-hidden-state.
fn pool(hidden: &ndarray::Array3<f32>, mask: &Array2<i64>, kind: PoolingKind) -> Array2<f32> {
    let (batch, seq, dim) = (hidden.shape()[0], hidden.shape()[1], hidden.shape()[2]);
    let mut out = Array2::<f32>::zeros((batch, dim));

    // Guard against an empty sequence dim. `Cls` would index
    // `hidden[[b, 0, d]]` and `LastToken` would compute `seq - 1` —
    // both would panic. Return all-zeros (already initialised) so the
    // caller's downstream L2-normalize / dim-check produces a clean
    // error path instead of a panic. In practice tokenization yields
    // at least one token (the prompt isn't empty); this is a defensive
    // guard rather than a hot path.
    if seq == 0 {
        return out;
    }

    match kind {
        PoolingKind::Cls => {
            for b in 0..batch {
                for d in 0..dim {
                    out[[b, d]] = hidden[[b, 0, d]];
                }
            }
        }
        PoolingKind::Mean => {
            const EPS: f32 = 1e-12;
            for b in 0..batch {
                let mut count: f32 = 0.0;
                for s in 0..seq {
                    if mask[[b, s]] != 0 {
                        count += 1.0;
                        for d in 0..dim {
                            out[[b, d]] += hidden[[b, s, d]];
                        }
                    }
                }
                let denom = count.max(EPS);
                for d in 0..dim {
                    out[[b, d]] /= denom;
                }
            }
        }
        PoolingKind::Max => {
            for b in 0..batch {
                let mut initialized = false;
                for s in 0..seq {
                    if mask[[b, s]] == 0 {
                        continue;
                    }
                    if !initialized {
                        for d in 0..dim {
                            out[[b, d]] = hidden[[b, s, d]];
                        }
                        initialized = true;
                    } else {
                        for d in 0..dim {
                            if hidden[[b, s, d]] > out[[b, d]] {
                                out[[b, d]] = hidden[[b, s, d]];
                            }
                        }
                    }
                }
            }
        }
        PoolingKind::LastToken => {
            // Decoder-style embedders (Qwen3-Embedding, NV-Embed) use the
            // hidden state at the rightmost non-pad position as the
            // sentence representation. Falls back to seq-1 for the
            // pathological all-zero-mask row to avoid producing all
            // zeros. `saturating_sub` is a belt-and-braces guard — the
            // `seq == 0` early-return at the top of the function
            // already covers the case where the subtraction would
            // otherwise underflow.
            for b in 0..batch {
                let last = (0..seq)
                    .rev()
                    .find(|&s| mask[[b, s]] != 0)
                    .unwrap_or(seq.saturating_sub(1));
                for d in 0..dim {
                    out[[b, d]] = hidden[[b, last, d]];
                }
            }
        }
    }

    out
}

/// L2-normalize each row in place. Rows with norm below epsilon are left at zero.
fn l2_normalize_rows(matrix: &mut Array2<f32>) {
    const EPS: f32 = 1e-12;
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > EPS {
            row.mapv_inplace(|v| v / norm);
        }
    }
}

// ---------------------------------------------------------------------------
// HF Hub download + ORT session build
//
// These mirror the equivalents in `rerank.rs`. Factoring out into a shared
// `common.rs` module is a worthwhile follow-up once embedding is validated
// end-to-end; for now we keep them local to make the embedding path readable
// without cross-file jumps.
// ---------------------------------------------------------------------------

pub(super) async fn download_model_files(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
    tokenizer_path: &str,
    additional_files: &[String],
) -> Result<(PathBuf, PathBuf)> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        }
    })?;

    let model_file =
        api_repo
            .get(onnx_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download ONNX model '{onnx_path}': {e}"),
            })?;

    // Pull any external-data sidecars or auxiliary tensor files into the
    // same cache snapshot as the .onnx file. ONNX Runtime resolves these
    // via relative paths embedded in the graph, so they must land alongside.
    for extra in additional_files {
        api_repo
            .get(extra)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download additional file '{extra}': {e}"),
            })?;
    }

    let tokenizer_file =
        api_repo
            .get(tokenizer_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download tokenizer '{tokenizer_path}': {e}"),
            })?;

    Ok((model_file, tokenizer_file))
}

pub(super) fn build_session(
    path: &Path,
    spec: &ModelAliasSpec,
    execution_providers: Option<&[OnnxExecutionProvider]>,
) -> Result<Session> {
    let builder =
        Session::builder(ort_environment()?).map_err(|e| RuntimeError::OnnxLoadFailure {
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

/// Inspect a built session for the decoder-LM-only inputs an embedder
/// might need to feed: `position_ids` (caller-built `0..seq` per row)
/// and `past_key_values.*` (empty zero placeholders).
///
/// Returns `(expects_position_ids, past_kv_schemas)`. For encoder-style
/// embedders (BGE, MiniLM, MPNet, ModernBERT) both are empty/false.
pub(super) fn inspect_decoder_extras(
    session: &Session,
    alias: &str,
) -> Result<(bool, Vec<super::decoder_inputs::InputSchema>)> {
    let schema = super::decoder_inputs::build_input_schema(session, alias)?;
    let expects_position_ids = schema
        .iter()
        .any(|s| s.role == super::decoder_inputs::InputRole::PositionIds);
    let past_kv: Vec<_> = schema
        .into_iter()
        .filter(|s| s.role == super::decoder_inputs::InputRole::PastKeyValue)
        .collect();
    Ok((expects_position_ids, past_kv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// All-ones `[rows, cols]` attention mask (every position unmasked).
    fn ones_mask(rows: usize, cols: usize) -> Array2<i64> {
        Array2::<i64>::ones((rows, cols))
    }

    #[test]
    fn finalize_dense_passes_through_2d_pooled_output() {
        // 2-D output is already pooled (the `dense_vecs` path): returned verbatim.
        let out =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 2.0, 0.0, 3.0, 4.0]).unwrap();
        let rows = finalize_dense("t", &out, &ones_mask(2, 1), PoolingKind::Cls, false, 3).unwrap();
        assert_eq!(rows, vec![vec![1.0, 2.0, 2.0], vec![0.0, 3.0, 4.0]]);
    }

    #[test]
    fn finalize_dense_cls_pools_3d_hidden_states() {
        // [1, 2, 2] hidden states; CLS takes position 0 → [1.0, 2.0].
        let out = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 9.0, 9.0]).unwrap();
        let rows = finalize_dense("t", &out, &ones_mask(1, 2), PoolingKind::Cls, false, 2).unwrap();
        assert_eq!(rows, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn finalize_dense_mean_pools_over_unmasked_positions() {
        // [1, 2, 2]; mean over both positions → [(1+3)/2, (2+4)/2] = [2, 3].
        let out = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rows =
            finalize_dense("t", &out, &ones_mask(1, 2), PoolingKind::Mean, false, 2).unwrap();
        assert_eq!(rows, vec![vec![2.0, 3.0]]);
    }

    #[test]
    fn finalize_dense_normalizes_rows_to_unit_norm() {
        // [3, 4] → L2 norm 5 → [0.6, 0.8].
        let out = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0, 4.0]).unwrap();
        let rows = finalize_dense("t", &out, &ones_mask(1, 1), PoolingKind::Cls, true, 2).unwrap();
        let norm = (rows[0][0].powi(2) + rows[0][1].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        assert!((rows[0][0] - 0.6).abs() < 1e-6);
        assert!((rows[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn finalize_dense_rejects_dimension_mismatch() {
        let out = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let err =
            finalize_dense("t", &out, &ones_mask(1, 1), PoolingKind::Cls, false, 4).unwrap_err();
        assert!(
            format!("{err}").contains("dimension mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn finalize_dense_rejects_unsupported_rank() {
        // Rank 1 and rank 4 are both outside the 2/3 the path accepts.
        let r1 = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let e1 =
            finalize_dense("t", &r1, &ones_mask(1, 1), PoolingKind::Cls, false, 3).unwrap_err();
        assert!(
            format!("{e1}").contains("Unsupported output rank"),
            "got: {e1}"
        );

        let r4 = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![1.0, 2.0]).unwrap();
        let e4 =
            finalize_dense("t", &r4, &ones_mask(1, 1), PoolingKind::Cls, false, 2).unwrap_err();
        assert!(
            format!("{e4}").contains("Unsupported output rank"),
            "got: {e4}"
        );
    }
}
