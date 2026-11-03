// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Image embedding task for `local/onnx`.
//!
//! Targets ViT-based image embedders (SigLIP / SigLIP-2, CLIP, OpenCLIP) that
//! ONNX-export with a single tensor output of either `[batch, dim]`
//! (pre-pooled image embedding) or `[batch, n_patches, dim]` (mean-pool the
//! patch dim caller-side). Pre-pooled outputs are common; if a model exports
//! the patch tensor, set `pool: "mean"`.
//!
//! Configuration comes entirely from per-spec `options`:
//!
//! - `onnx_path`           (required) — path to the `.onnx` file within the
//!   HuggingFace repo, e.g. `"onnx/model.onnx"`.
//! - `image_size`          (required) — square input edge in pixels (e.g. 384
//!   for SigLIP-2-So400m-patch16-384).
//! - `dimensions`          (required) — output embedding dim (e.g. 1152 for
//!   SigLIP-2-So400m).
//! - `normalization`       (optional, default `"siglip"`) — `"siglip"` →
//!   `(0.5, 0.5, 0.5)` mean & std; `"imagenet"` → ImageNet stats.
//! - `pool`                (optional, default `"none"`) — `"none"` if the
//!   model emits a pre-pooled `[batch, dim]`; `"mean"` to mean-pool a
//!   `[batch, tokens, dim]` patch sequence.
//! - `normalize`           (optional, default `true`) — L2-normalize each row.
//! - `output_name`         (optional) — name of the output tensor to pull;
//!   defaults to the model's first output.
//! - `execution_providers` — same shape as other tasks.

use crate::hf::HfModelRepo;
use async_trait::async_trait;
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::image::{Normalization, preprocess_batch};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, ort_environment,
    parse_execution_providers_option,
};
use crate::traits::{EmbedResult, ImageEmbeddingModel, ImageInput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    None,
    Mean,
}

/// Entry point for `LocalOnnxProvider::load` when `spec.task == EmbedImage`.
pub(super) async fn load_image_embedder(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn ImageEmbeddingModel>> {
    let model = OnnxImageEmbedder::load(spec).await?;
    Ok(Arc::new(model) as Arc<dyn ImageEmbeddingModel>)
}

struct ImageEmbedConfig {
    hf_repo: String,
    onnx_path: String,
    image_size: u32,
    dimensions: u32,
    normalization: Normalization,
    pool: PoolKind,
    normalize: bool,
    output_name: Option<String>,
}

impl ImageEmbedConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let opts = &spec.options;
        let onnx_path = opts
            .get("onnx_path")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Image embedder '{}' requires option `onnx_path` (.onnx path within the HF repo)",
                    spec.alias
                ))
            })?;
        let image_size = opts
            .get("image_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Image embedder '{}' requires option `image_size` (target edge in pixels)",
                    spec.alias
                ))
            })? as u32;
        let dimensions = opts
            .get("dimensions")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Image embedder '{}' requires option `dimensions` (output embedding dim)",
                    spec.alias
                ))
            })? as u32;
        let normalization = match opts
            .get("normalization")
            .and_then(Value::as_str)
            .unwrap_or("siglip")
        {
            "siglip" => Normalization::SIGLIP,
            "imagenet" => Normalization::IMAGENET,
            other => {
                return Err(RuntimeError::Config(format!(
                    "Image embedder '{}' has unknown `normalization` value '{other}'; \
                     expected one of: siglip, imagenet",
                    spec.alias
                )));
            }
        };
        let pool = match opts.get("pool").and_then(Value::as_str).unwrap_or("none") {
            "none" => PoolKind::None,
            "mean" => PoolKind::Mean,
            other => {
                return Err(RuntimeError::Config(format!(
                    "Image embedder '{}' has unknown `pool` value '{other}'; \
                     expected one of: none, mean",
                    spec.alias
                )));
            }
        };
        let normalize = opts
            .get("normalize")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let output_name = opts
            .get("output_name")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok(Self {
            hf_repo: spec.model_id.clone(),
            onnx_path,
            image_size,
            dimensions,
            normalization,
            pool,
            normalize,
            output_name,
        })
    }
}

struct OnnxImageEmbedder {
    session: Mutex<Session>,
    alias: String,
    model_id: String,
    image_size: u32,
    dimensions: u32,
    normalization: Normalization,
    pool: PoolKind,
    normalize: bool,
    output_name: Option<String>,
}

impl OnnxImageEmbedder {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = ImageEmbedConfig::resolve(spec)?;
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-image-embed", &cfg.hf_repo, &spec.options);
        let model_path = download_image_artifact(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            image_size = cfg.image_size,
            dimensions = cfg.dimensions,
            "Loading ONNX image embedder"
        );

        let session = build_session(&model_path, spec, execution_providers.as_deref())?;
        Ok(Self {
            session: Mutex::new(session),
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            image_size: cfg.image_size,
            dimensions: cfg.dimensions,
            normalization: cfg.normalization,
            pool: cfg.pool,
            normalize: cfg.normalize,
            output_name: cfg.output_name,
        })
    }
}

impl crate::traits::ModelInfo for OnnxImageEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl ImageEmbeddingModel for OnnxImageEmbedder {
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    async fn embed(&self, images: Vec<ImageInput>) -> Result<EmbedResult> {
        if images.is_empty() {
            return Ok(EmbedResult {
                vectors: Vec::new(),
                usage: None,
            });
        }

        let tensor = preprocess_batch(&images, self.image_size, self.normalization)?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Session lock poisoned: {e}"),
            })?;

        let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let pixel_tensor = ort::value::Tensor::from_array(tensor.into_dyn())
            .map_err(|e| map_err(e, "pixel tensor"))?;
        let inputs: Vec<(String, ort::value::DynTensor)> = vec![
            // Common SigLIP/CLIP export name: `pixel_values`. If the first
            // input has a different name, the model will reject this and the
            // user must supply an alternate.
            ("pixel_values".to_string(), pixel_tensor.upcast()),
        ];

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

        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| map_err(e, "extract output"))?;

        let shape_err = |ctx: &str, e: ndarray::ShapeError| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let mut pooled: Array2<f32> = match view.ndim() {
            2 => view
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| shape_err("2-D output", e))?
                .to_owned(),
            3 => {
                let arr3 = view
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| shape_err("3-D output", e))?
                    .to_owned();
                match self.pool {
                    PoolKind::Mean => arr3.mean_axis(Axis(1)).ok_or_else(|| {
                        RuntimeError::OnnxInvocationFailure {
                            alias: self.alias.clone(),
                            cause: "Empty patch dim during mean pool".to_string(),
                        }
                    })?,
                    PoolKind::None => {
                        return Err(RuntimeError::OnnxInvocationFailure {
                            alias: self.alias.clone(),
                            cause: "3-D output requires `pool: \"mean\"` (got `pool: \"none\"`)"
                                .to_string(),
                        });
                    }
                }
            }
            other => {
                return Err(RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("Unsupported output rank {other} for image embedding"),
                });
            }
        };

        if self.normalize {
            l2_normalize_rows(&mut pooled);
        }

        let actual_dim = pooled.shape()[1] as u32;
        if actual_dim != self.dimensions {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!(
                    "Image embedding dim mismatch: configured {} but model produced {}",
                    self.dimensions, actual_dim
                ),
            });
        }

        let vectors: Vec<Vec<f32>> = pooled.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
        Ok(EmbedResult {
            vectors,
            usage: None,
        })
    }
}

fn l2_normalize_rows(matrix: &mut Array2<f32>) {
    const EPS: f32 = 1e-12;
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > EPS {
            row.mapv_inplace(|v| v / norm);
        }
    }
}

async fn download_image_artifact(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
) -> Result<PathBuf> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        }
    })?;
    api_repo
        .get(onnx_path)
        .await
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: format!("Could not download '{onnx_path}': {e}"),
        })
}

fn build_session(
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
