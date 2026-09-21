// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Raw ONNX tensor execution task for `local/onnx`.
//!
//! Loads arbitrary ONNX models and exposes their tensor I/O through the
//! [`RawTensorModel`] trait. Sessions are cached per alias on the parent
//! [`LocalOnnxProvider`](super::LocalOnnxProvider).

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, effective_ep_list,
    parse_execution_providers_option,
};
use crate::traits::{DimSize, RawTensorModel, TensorBatch, TensorDtype, TensorSpec, TensorValue};
use async_trait::async_trait;
use dashmap::DashMap;
use hf_hub::{
    Cache, Repo, RepoType,
    api::{RepoInfo, tokio::ApiBuilder},
};
use ndarray::{ArrayD, Axis, stack};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynTensor, Tensor, TensorElementType, ValueType};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(super) struct LoadedOnnxSession {
    session: std::sync::Mutex<Session>,
    input_signature: Vec<TensorSpec>,
    output_signature: Vec<TensorSpec>,
    batch_support: BatchDimSupport,
    /// Resolved execution-provider list as stable string ids
    /// (e.g. `["cuda", "cpu"]`). See
    /// [`crate::traits::ModelInfo::active_execution_providers`].
    requested_eps: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum BatchDimSupport {
    Static(usize),
    Dynamic { ceiling: usize },
    None,
}

#[derive(Clone)]
struct LocalRawTensorModel {
    alias: String,
    model_id: String,
    session: Arc<LoadedOnnxSession>,
}

#[derive(Debug, Clone)]
struct LocalOnnxOptions {
    artifact: Option<String>,
    max_batch_size: usize,
    execution_providers: Option<Vec<OnnxExecutionProvider>>,
    graph_optimization_level: GraphOptimizationLevel,
    inter_op_num_threads: Option<usize>,
    intra_op_num_threads: Option<usize>,
}

#[derive(Debug, Clone)]
enum ModelSource {
    LocalPath(PathBuf),
    HuggingFaceRepo(String),
}

/// Load (or fetch from cache) the raw ONNX runner for `spec`.
///
/// Called from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::Raw`.
///
/// Concurrency: the `sessions.get` / `sessions.insert` pair below is a
/// check-then-insert pattern that, in isolation, would race when two
/// callers ask for the same alias concurrently. In practice this is safe
/// because [`crate::runtime::ModelRuntime`] serializes provider loads
/// per-alias via a per-`ModelRuntimeKey` loader mutex (see
/// `runtime::ModelRuntime` and its `loader_locks` field), so by the time
/// the second caller reaches `provider.load(spec)` the first call has
/// already populated the cache. If `LocalOnnxProvider::load` ever gets
/// called outside that coordination, the worst case is duplicate session
/// builds for the same alias — wasteful but not unsafe (last writer wins
/// in `DashMap::insert`, both callers walk away with a valid `Arc`).
pub(super) async fn load_raw(
    spec: &ModelAliasSpec,
    base_dir: Option<&Path>,
    sessions: &DashMap<String, Arc<LoadedOnnxSession>>,
) -> Result<Arc<dyn RawTensorModel>> {
    if let Some(existing) = sessions.get(&spec.alias) {
        return Ok(Arc::new(LocalRawTensorModel {
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            session: existing.clone(),
        }) as Arc<dyn RawTensorModel>);
    }

    let options = LocalOnnxOptions::from_value(&spec.options)?;
    let requested_eps: Vec<String> = effective_ep_list(options.execution_providers.as_deref())
        .into_iter()
        .map(|ep| ep.as_str().to_string())
        .collect();

    // Validate the requested EP list FIRST (before preflight or I/O).
    // Misconfigurations (e.g. CUDA requested without `gpu-cuda` enabled)
    // surface as a precise "feature not enabled" error rather than a
    // generic dylib-missing error or a wasted HF download. Mirrors the
    // ordering in the rerank task.
    let _ = build_execution_providers(
        options.execution_providers.as_deref(),
        &spec.alias,
        "local/onnx",
    )?;

    // Pre-flight (load-dynamic only): verify the ONNX Runtime dylib is
    // loadable before any ort API call. Sidesteps the upstream
    // load-dynamic OnceLock deadlock (pykeio/ort#560) when the dylib
    // is missing. Compiled out under `provider-onnx` (bundled CPU)
    // where the lib is statically linked into the binary.
    #[cfg(feature = "provider-onnx-dynamic")]
    preflight_ort_dylib(&spec.alias, "local/onnx")?;

    let path = resolve_model_path(base_dir, spec, &options).await?;
    let session = build_session(&path, &options, &spec.alias)?;
    let (input_signature, output_signature) = extract_signatures(&session, &spec.alias)?;
    let batch_support = detect_batch_support(&input_signature, options.max_batch_size);

    let loaded = Arc::new(LoadedOnnxSession {
        session: std::sync::Mutex::new(session),
        input_signature,
        output_signature,
        batch_support,
        requested_eps,
    });

    sessions.insert(spec.alias.clone(), loaded.clone());

    Ok(Arc::new(LocalRawTensorModel {
        alias: spec.alias.clone(),
        model_id: spec.model_id.clone(),
        session: loaded,
    }) as Arc<dyn RawTensorModel>)
}

impl crate::traits::ModelInfo for LocalRawTensorModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.session.requested_eps.clone()
    }
}

#[async_trait]
impl RawTensorModel for LocalRawTensorModel {
    async fn run(&self, inputs: &TensorBatch) -> Result<TensorBatch> {
        validate_single_inputs(&self.alias, inputs, &self.session.input_signature)?;
        let ort_inputs = build_ort_inputs(&self.alias, inputs, &self.session.input_signature)?;
        let mut session = self.session.session.lock().unwrap();
        let outputs = session
            .run(ort_inputs)
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: e.to_string(),
            })?;
        pack_outputs(&self.alias, outputs, &self.session.output_signature)
    }

    async fn run_batch(&self, inputs: &[TensorBatch]) -> Result<Vec<TensorBatch>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        match self.session.batch_support {
            BatchDimSupport::None => {
                let mut out = Vec::with_capacity(inputs.len());
                for input in inputs {
                    out.push(self.run(input).await?);
                }
                Ok(out)
            }
            BatchDimSupport::Static(expected) => {
                if inputs.len() != expected {
                    return Err(RuntimeError::OnnxBatchStackingFailure {
                        alias: self.alias.clone(),
                        cause: format!(
                            "expected exactly {expected} items for a static batch model, got {}",
                            inputs.len()
                        ),
                    });
                }
                run_batched(&self.alias, &self.session, inputs)
            }
            BatchDimSupport::Dynamic { ceiling } => {
                if inputs.len() > ceiling {
                    return Err(RuntimeError::OnnxBatchStackingFailure {
                        alias: self.alias.clone(),
                        cause: format!(
                            "batch size {} exceeds configured maximum {}",
                            inputs.len(),
                            ceiling
                        ),
                    });
                }
                run_batched(&self.alias, &self.session, inputs)
            }
        }
    }

    fn max_batch_size(&self) -> usize {
        match self.session.batch_support {
            BatchDimSupport::Static(n) => n,
            BatchDimSupport::Dynamic { ceiling } => ceiling,
            BatchDimSupport::None => 1,
        }
    }

    fn input_signature(&self) -> &[TensorSpec] {
        &self.session.input_signature
    }

    fn output_signature(&self) -> &[TensorSpec] {
        &self.session.output_signature
    }
}

impl LocalOnnxOptions {
    fn from_value(value: &Value) -> Result<Self> {
        let map = value.as_object();
        let artifact = map
            .and_then(|m| m.get("artifact"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let max_batch_size = map
            .and_then(|m| m.get("max_batch_size"))
            .and_then(Value::as_u64)
            .unwrap_or(64) as usize;
        let execution_providers =
            parse_execution_providers_option(map.and_then(|m| m.get("execution_providers")))?;

        let graph_optimization_level = match map
            .and_then(|m| m.get("graph_optimization_level"))
            .and_then(Value::as_str)
            .unwrap_or("basic")
        {
            "disable" => GraphOptimizationLevel::Disable,
            "basic" => GraphOptimizationLevel::Level1,
            "extended" => GraphOptimizationLevel::Level2,
            "all" => GraphOptimizationLevel::All,
            _ => GraphOptimizationLevel::Level1,
        };

        let inter_op_num_threads = map
            .and_then(|m| m.get("inter_op_num_threads"))
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let intra_op_num_threads = map
            .and_then(|m| m.get("intra_op_num_threads"))
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        Ok(Self {
            artifact,
            max_batch_size,
            execution_providers,
            graph_optimization_level,
            inter_op_num_threads,
            intra_op_num_threads,
        })
    }
}

async fn resolve_model_path(
    base_dir: Option<&Path>,
    spec: &ModelAliasSpec,
    options: &LocalOnnxOptions,
) -> Result<PathBuf> {
    match classify_model_source(base_dir, &spec.model_id) {
        ModelSource::LocalPath(path) => {
            if path.exists() {
                Ok(path)
            } else {
                Err(RuntimeError::OnnxModelNotFound {
                    alias: spec.alias.clone(),
                    path,
                })
            }
        }
        ModelSource::HuggingFaceRepo(repo_id) => {
            resolve_hf_model_path(spec, options, &repo_id).await
        }
    }
}

fn classify_model_source(base_dir: Option<&Path>, model_id: &str) -> ModelSource {
    let path = PathBuf::from(model_id);
    if path.is_absolute() {
        return ModelSource::LocalPath(path);
    }

    if model_id.starts_with("./") || model_id.starts_with("../") {
        return ModelSource::LocalPath(match base_dir {
            Some(base_dir) => base_dir.join(path),
            None => path,
        });
    }

    let resolved = match base_dir {
        Some(base_dir) => base_dir.join(&path),
        None => path.clone(),
    };
    if resolved.exists() {
        return ModelSource::LocalPath(resolved);
    }

    ModelSource::HuggingFaceRepo(model_id.to_string())
}

async fn resolve_hf_model_path(
    spec: &ModelAliasSpec,
    options: &LocalOnnxOptions,
    repo_id: &str,
) -> Result<PathBuf> {
    let cache_dir = resolve_cache_dir("onnx", repo_id, &spec.options);
    let cache = Cache::new(cache_dir.clone());
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: spec.alias.clone(),
            cause: e.to_string(),
        })?;
    let repo = match &spec.revision {
        Some(revision) => {
            Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.clone())
        }
        None => Repo::model(repo_id.to_string()),
    };
    let api_repo = api.repo(repo.clone());
    let repo_info = api_repo
        .info()
        .await
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: spec.alias.clone(),
            cause: e.to_string(),
        })?;
    let artifact = select_onnx_artifact(&spec.alias, &repo_info, options.artifact.as_deref())?;

    materialize_hf_snapshot(&api_repo, &spec.alias, &repo_info).await?;
    resolve_snapshot_artifact_path(&cache, &repo, &repo_info, &spec.alias, &artifact)
}

fn select_onnx_artifact(
    alias: &str,
    repo_info: &RepoInfo,
    explicit_artifact: Option<&str>,
) -> Result<String> {
    if let Some(artifact) = explicit_artifact {
        if repo_info.siblings.iter().any(|s| s.rfilename == artifact) {
            return Ok(artifact.to_string());
        }
        return Err(RuntimeError::OnnxArtifactSelectionFailure {
            alias: alias.to_string(),
            cause: format!("artifact '{artifact}' was not found in the Hugging Face repo"),
        });
    }

    let mut candidates = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| name.ends_with(".onnx"))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.dedup();

    match candidates.len() {
        0 => Err(RuntimeError::OnnxArtifactSelectionFailure {
            alias: alias.to_string(),
            cause: "no .onnx artifact found in Hugging Face repo".to_string(),
        }),
        1 => Ok(candidates.remove(0)),
        _ => Err(RuntimeError::OnnxArtifactSelectionFailure {
            alias: alias.to_string(),
            cause: format!(
                "multiple .onnx artifacts found ({}); set options.artifact explicitly",
                candidates.join(", ")
            ),
        }),
    }
}

async fn materialize_hf_snapshot(
    api_repo: &hf_hub::api::tokio::ApiRepo,
    alias: &str,
    repo_info: &RepoInfo,
) -> Result<()> {
    for sibling in &repo_info.siblings {
        api_repo
            .get(&sibling.rfilename)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("failed to download repo file '{}': {e}", sibling.rfilename),
            })?;
    }

    Ok(())
}

fn resolve_snapshot_artifact_path(
    cache: &Cache,
    repo: &Repo,
    repo_info: &RepoInfo,
    alias: &str,
    artifact: &str,
) -> Result<PathBuf> {
    let cache_repo = cache.repo(repo.clone());
    let path = cache_repo.pointer_path(&repo_info.sha).join(artifact);
    if path.exists() {
        Ok(path)
    } else {
        Err(RuntimeError::OnnxArtifactSelectionFailure {
            alias: alias.to_string(),
            cause: format!("artifact '{artifact}' was not materialized into the repo snapshot"),
        })
    }
}

fn build_session(path: &Path, options: &LocalOnnxOptions, alias: &str) -> Result<Session> {
    let mut builder = Session::builder().map_err(|e| RuntimeError::OnnxLoadFailure {
        alias: alias.to_string(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    })?;

    builder = builder
        .with_optimization_level(options.graph_optimization_level)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;

    if let Some(v) = options.inter_op_num_threads {
        builder = builder
            .with_inter_threads(v)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: alias.to_string(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
    }

    if let Some(v) = options.intra_op_num_threads {
        builder = builder
            .with_intra_threads(v)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: alias.to_string(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
    }

    let execution_providers =
        build_execution_providers(options.execution_providers.as_deref(), alias, "local/onnx")?;
    builder = builder
        .with_execution_providers(execution_providers)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;

    builder
        .commit_from_file(path)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
}

fn extract_signatures(
    session: &Session,
    alias: &str,
) -> Result<(Vec<TensorSpec>, Vec<TensorSpec>)> {
    let inputs = session
        .inputs()
        .iter()
        .map(|input| {
            Ok(TensorSpec {
                name: input.name().to_string(),
                dtype: convert_dtype(input.dtype(), alias)?,
                shape: convert_shape(input.dtype(), alias)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let outputs = session
        .outputs()
        .iter()
        .map(|output| {
            Ok(TensorSpec {
                name: output.name().to_string(),
                dtype: convert_dtype(output.dtype(), alias)?,
                shape: convert_shape(output.dtype(), alias)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((inputs, outputs))
}

fn convert_dtype(value_type: &ValueType, alias: &str) -> Result<TensorDtype> {
    let Some(dtype) = value_type.tensor_type() else {
        return Err(RuntimeError::OnnxSignatureIntrospectionFailure {
            alias: alias.to_string(),
            cause: "non-tensor inputs/outputs are not supported".to_string(),
        });
    };

    match dtype {
        TensorElementType::Float32 => Ok(TensorDtype::F32),
        TensorElementType::Float64 => Ok(TensorDtype::F64),
        TensorElementType::Int32 => Ok(TensorDtype::I32),
        TensorElementType::Int64 => Ok(TensorDtype::I64),
        TensorElementType::Bool => Ok(TensorDtype::Bool),
        TensorElementType::String => Ok(TensorDtype::String),
        other => Err(RuntimeError::OnnxSignatureIntrospectionFailure {
            alias: alias.to_string(),
            cause: format!("unsupported tensor dtype {other:?}"),
        }),
    }
}

fn convert_shape(value_type: &ValueType, alias: &str) -> Result<Vec<DimSize>> {
    let ValueType::Tensor {
        shape,
        dimension_symbols,
        ..
    } = value_type
    else {
        return Err(RuntimeError::OnnxSignatureIntrospectionFailure {
            alias: alias.to_string(),
            cause: "non-tensor inputs/outputs are not supported".to_string(),
        });
    };

    Ok(shape
        .iter()
        .enumerate()
        .map(|(idx, dim)| match *dim {
            dim if dim >= 0 => DimSize::Fixed(dim as usize),
            _ => match dimension_symbols.get(idx).cloned() {
                Some(name) if !name.is_empty() => DimSize::Named(name),
                _ => DimSize::Dynamic,
            },
        })
        .collect())
}

fn detect_batch_support(input_signature: &[TensorSpec], ceiling: usize) -> BatchDimSupport {
    let Some(first_input) = input_signature.first() else {
        return BatchDimSupport::None;
    };
    if first_input.shape.len() <= 1 {
        return BatchDimSupport::None;
    }
    let Some(first_dim) = first_input.shape.first() else {
        return BatchDimSupport::None;
    };

    match first_dim {
        DimSize::Fixed(n) => BatchDimSupport::Static(*n),
        DimSize::Dynamic | DimSize::Named(_) => BatchDimSupport::Dynamic { ceiling },
    }
}

fn validate_single_inputs(
    alias: &str,
    inputs: &TensorBatch,
    signature: &[TensorSpec],
) -> Result<()> {
    for spec in signature {
        let Some(value) = inputs.get(&spec.name) else {
            return Err(RuntimeError::OnnxInputMissing {
                alias: alias.to_string(),
                required_input: spec.name.clone(),
            });
        };
        validate_tensor_against_signature(alias, &spec.name, value, &spec.shape, spec.dtype)?;
    }

    for name in inputs.tensors.keys() {
        if !signature.iter().any(|spec| spec.name == *name) {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("unexpected input tensor '{name}'"),
            });
        }
    }

    Ok(())
}

fn validate_tensor_against_signature(
    alias: &str,
    input_name: &str,
    value: &TensorValue,
    shape: &[DimSize],
    dtype: TensorDtype,
) -> Result<()> {
    if value.dtype() != dtype {
        return Err(RuntimeError::OnnxInputTypeMismatch {
            alias: alias.to_string(),
            input_name: input_name.to_string(),
            expected: dtype,
            got: value.dtype(),
        });
    }

    if value.shape().len() != shape.len() {
        return Err(RuntimeError::OnnxInputShapeMismatch {
            alias: alias.to_string(),
            input_name: input_name.to_string(),
            expected: shape.to_vec(),
            got: value.shape().to_vec(),
        });
    }

    for (expected, got) in shape.iter().zip(value.shape()) {
        if let DimSize::Fixed(expected) = expected
            && expected != got
        {
            return Err(RuntimeError::OnnxInputShapeMismatch {
                alias: alias.to_string(),
                input_name: input_name.to_string(),
                expected: shape.to_vec(),
                got: value.shape().to_vec(),
            });
        }
    }

    Ok(())
}

fn build_ort_inputs(
    alias: &str,
    inputs: &TensorBatch,
    signature: &[TensorSpec],
) -> Result<Vec<(String, DynTensor)>> {
    let mut tensors = inputs.tensors.clone();
    let mut ort_inputs = Vec::with_capacity(signature.len());
    for spec in signature {
        let value =
            tensors
                .shift_remove(&spec.name)
                .ok_or_else(|| RuntimeError::OnnxInputMissing {
                    alias: alias.to_string(),
                    required_input: spec.name.clone(),
                })?;

        ort_inputs.push((spec.name.clone(), tensor_value_to_dyn(value)?));
    }
    Ok(ort_inputs)
}

fn tensor_value_to_dyn(value: TensorValue) -> Result<DynTensor> {
    match value {
        TensorValue::F32(v) => Ok(Tensor::from_array(v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
        TensorValue::F64(v) => Ok(Tensor::from_array(v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
        TensorValue::I32(v) => Ok(Tensor::from_array(v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
        TensorValue::I64(v) => Ok(Tensor::from_array(v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
        TensorValue::Bool(v) => Ok(Tensor::from_array(v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
        TensorValue::String(v) => Ok(Tensor::from_string_array(&v)
            .map_err(|e| RuntimeError::InferenceError(e.to_string()))?
            .upcast()),
    }
}

fn pack_outputs(
    alias: &str,
    outputs: ort::session::SessionOutputs<'_>,
    signature: &[TensorSpec],
) -> Result<TensorBatch> {
    let mut batch = TensorBatch::new();
    for spec in signature {
        let value = outputs
            .get(&spec.name)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("missing output tensor '{}'", spec.name),
            })?;
        batch.insert(spec.name.clone(), dyn_tensor_to_value(alias, spec, value)?);
    }
    Ok(batch)
}

fn dyn_tensor_to_value(
    alias: &str,
    spec: &TensorSpec,
    value: &ort::value::DynValue,
) -> Result<TensorValue> {
    match spec.dtype {
        TensorDtype::F32 => value
            .try_extract_array::<f32>()
            .map(|v| TensorValue::F32(v.to_owned()))
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
        TensorDtype::F64 => value
            .try_extract_array::<f64>()
            .map(|v| TensorValue::F64(v.to_owned()))
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
        TensorDtype::I32 => value
            .try_extract_array::<i32>()
            .map(|v| TensorValue::I32(v.to_owned()))
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
        TensorDtype::I64 => value
            .try_extract_array::<i64>()
            .map(|v| TensorValue::I64(v.to_owned()))
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
        TensorDtype::Bool => value
            .try_extract_array::<bool>()
            .map(|v| TensorValue::Bool(v.to_owned()))
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
        TensorDtype::String => value
            .try_extract_string_array()
            .map(TensorValue::String)
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: e.to_string(),
            }),
    }
}

fn run_batched(
    alias: &str,
    session: &Arc<LoadedOnnxSession>,
    inputs: &[TensorBatch],
) -> Result<Vec<TensorBatch>> {
    let stacked = stack_batches(alias, inputs, &session.input_signature)?;
    let ort_inputs = build_ort_inputs(alias, &stacked, &session.input_signature)?;
    let mut ort_session = session.session.lock().unwrap();
    let outputs = ort_session
        .run(ort_inputs)
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        })?;
    let output_batch = pack_outputs(alias, outputs, &session.output_signature)?;
    split_batch(alias, output_batch, &session.output_signature)
}

fn stack_batches(
    alias: &str,
    batches: &[TensorBatch],
    signature: &[TensorSpec],
) -> Result<TensorBatch> {
    let batch_count = batches.len();
    let mut stacked = TensorBatch::new();
    validate_no_unexpected_inputs(alias, batches, signature)?;

    for spec in signature {
        let mut values = Vec::with_capacity(batch_count);
        for batch in batches {
            let Some(value) = batch.get(&spec.name) else {
                return Err(RuntimeError::OnnxInputMissing {
                    alias: alias.to_string(),
                    required_input: spec.name.clone(),
                });
            };
            validate_batched_sample(alias, &spec.name, value, spec)?;
            values.push(value.clone());
        }

        stacked.insert(
            spec.name.clone(),
            stack_tensor_values(alias, &spec.name, values)?,
        );
    }

    Ok(stacked)
}

fn validate_batched_sample(
    alias: &str,
    input_name: &str,
    value: &TensorValue,
    spec: &TensorSpec,
) -> Result<()> {
    if value.dtype() != spec.dtype {
        return Err(RuntimeError::OnnxInputTypeMismatch {
            alias: alias.to_string(),
            input_name: input_name.to_string(),
            expected: spec.dtype,
            got: value.dtype(),
        });
    }

    let expected_rank = spec.shape.len().saturating_sub(1);
    if value.shape().len() != expected_rank {
        return Err(RuntimeError::OnnxInputShapeMismatch {
            alias: alias.to_string(),
            input_name: input_name.to_string(),
            expected: spec.shape.clone(),
            got: value.shape().to_vec(),
        });
    }

    for (expected, got) in spec.shape.iter().skip(1).zip(value.shape()) {
        if let DimSize::Fixed(expected) = expected
            && expected != got
        {
            return Err(RuntimeError::OnnxInputShapeMismatch {
                alias: alias.to_string(),
                input_name: input_name.to_string(),
                expected: spec.shape.clone(),
                got: value.shape().to_vec(),
            });
        }
    }

    Ok(())
}

fn validate_no_unexpected_inputs(
    alias: &str,
    batches: &[TensorBatch],
    signature: &[TensorSpec],
) -> Result<()> {
    let allowed_inputs = signature
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<std::collections::HashSet<_>>();

    for (batch_index, batch) in batches.iter().enumerate() {
        if let Some(unexpected_input) = batch
            .tensors
            .keys()
            .find(|name| !allowed_inputs.contains(name.as_str()))
        {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!(
                    "unexpected input '{}' in batch at index {}",
                    unexpected_input, batch_index
                ),
            });
        }
    }

    Ok(())
}

fn stack_tensor_values(
    alias: &str,
    input_name: &str,
    values: Vec<TensorValue>,
) -> Result<TensorValue> {
    match values.first() {
        Some(TensorValue::F32(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::F32(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::F32),
        Some(TensorValue::F64(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::F64(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::F64),
        Some(TensorValue::I32(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::I32(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::I32),
        Some(TensorValue::I64(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::I64(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::I64),
        Some(TensorValue::Bool(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::Bool(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::Bool),
        Some(TensorValue::String(_)) => stack_array(values, alias, input_name, |v| match v {
            TensorValue::String(a) => Ok(a),
            _ => Err(RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: format!("tensor '{input_name}' had mixed dtypes in batch"),
            }),
        })
        .map(TensorValue::String),
        None => Err(RuntimeError::OnnxBatchStackingFailure {
            alias: alias.to_string(),
            cause: format!("tensor '{input_name}' had no values to stack"),
        }),
    }
}

fn stack_array<T: Clone>(
    values: Vec<TensorValue>,
    alias: &str,
    input_name: &str,
    convert: impl Fn(TensorValue) -> Result<ArrayD<T>>,
) -> Result<ArrayD<T>> {
    let arrays = values
        .into_iter()
        .map(convert)
        .collect::<Result<Vec<_>>>()?;
    let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
    stack(Axis(0), &views).map_err(|e| RuntimeError::OnnxBatchStackingFailure {
        alias: alias.to_string(),
        cause: format!("failed to stack tensor '{input_name}': {e}"),
    })
}

fn split_batch(
    alias: &str,
    batch: TensorBatch,
    signature: &[TensorSpec],
) -> Result<Vec<TensorBatch>> {
    let batch_len = match batch.tensors.values().next() {
        Some(value) => value.shape().first().copied().ok_or_else(|| {
            RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause: "batched output tensor is missing a leading batch dimension".to_string(),
            }
        })?,
        None => return Ok(vec![]),
    };

    let mut outputs = (0..batch_len)
        .map(|_| TensorBatch::new())
        .collect::<Vec<_>>();
    for spec in signature {
        let value = batch
            .get(&spec.name)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("missing output tensor '{}'", spec.name),
            })?;
        for (idx, split) in split_tensor_value(alias, &spec.name, value, batch_len)?
            .into_iter()
            .enumerate()
        {
            outputs[idx].insert(spec.name.clone(), split);
        }
    }
    Ok(outputs)
}

fn split_tensor_value(
    alias: &str,
    output_name: &str,
    value: &TensorValue,
    batch_len: usize,
) -> Result<Vec<TensorValue>> {
    if value.shape().first().copied() != Some(batch_len) {
        return Err(RuntimeError::OnnxBatchStackingFailure {
            alias: alias.to_string(),
            cause: format!(
                "output tensor '{output_name}' does not expose a split-able batch dimension"
            ),
        });
    }

    match value {
        TensorValue::F32(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::F32(a.to_owned().into_dyn()))
            .collect()),
        TensorValue::F64(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::F64(a.to_owned().into_dyn()))
            .collect()),
        TensorValue::I32(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::I32(a.to_owned().into_dyn()))
            .collect()),
        TensorValue::I64(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::I64(a.to_owned().into_dyn()))
            .collect()),
        TensorValue::Bool(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::Bool(a.to_owned().into_dyn()))
            .collect()),
        TensorValue::String(v) => Ok(v
            .axis_iter(Axis(0))
            .map(|a| TensorValue::String(a.to_owned().into_dyn()))
            .collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelSource, classify_model_source, select_onnx_artifact};
    use crate::provider::onnx_ep::{OnnxExecutionProvider, default_execution_providers};
    use hf_hub::api::{RepoInfo, Siblings};
    use std::path::{Path, PathBuf};

    #[test]
    fn classify_absolute_path_as_local() {
        match classify_model_source(None, "/tmp/model.onnx") {
            ModelSource::LocalPath(path) => assert_eq!(path, PathBuf::from("/tmp/model.onnx")),
            ModelSource::HuggingFaceRepo(_) => panic!("expected local path"),
        }
    }

    #[test]
    fn classify_relative_dot_path_as_local() {
        match classify_model_source(Some(Path::new("/models")), "./model.onnx") {
            ModelSource::LocalPath(path) => {
                assert_eq!(path, PathBuf::from("/models/./model.onnx"))
            }
            ModelSource::HuggingFaceRepo(_) => panic!("expected local path"),
        }
    }

    #[test]
    fn classify_hf_repo_id_when_not_path_like() {
        match classify_model_source(None, "onnx-community/all-MiniLM-L6-v2-onnx") {
            ModelSource::HuggingFaceRepo(repo_id) => {
                assert_eq!(repo_id, "onnx-community/all-MiniLM-L6-v2-onnx")
            }
            ModelSource::LocalPath(_) => panic!("expected Hugging Face repo id"),
        }
    }

    #[test]
    fn classify_hf_repo_id_even_if_suffix_looks_like_onnx_file() {
        match classify_model_source(None, "org/model.onnx") {
            ModelSource::HuggingFaceRepo(repo_id) => assert_eq!(repo_id, "org/model.onnx"),
            ModelSource::LocalPath(_) => panic!("expected Hugging Face repo id"),
        }
    }

    #[test]
    fn select_single_onnx_artifact() {
        let artifact = select_onnx_artifact(
            "raw/onnx",
            &RepoInfo {
                siblings: vec![
                    Siblings {
                        rfilename: "config.json".to_string(),
                    },
                    Siblings {
                        rfilename: "model.onnx".to_string(),
                    },
                ],
                sha: "abc".to_string(),
            },
            None,
        )
        .unwrap();

        assert_eq!(artifact, "model.onnx");
    }

    #[test]
    fn reject_ambiguous_onnx_artifact_selection() {
        let err = select_onnx_artifact(
            "raw/onnx",
            &RepoInfo {
                siblings: vec![
                    Siblings {
                        rfilename: "encoder.onnx".to_string(),
                    },
                    Siblings {
                        rfilename: "decoder.onnx".to_string(),
                    },
                ],
                sha: "abc".to_string(),
            },
            None,
        )
        .unwrap_err();

        assert!(err.to_string().contains("set options.artifact explicitly"));
    }

    #[test]
    fn explicit_artifact_must_exist() {
        let err = select_onnx_artifact(
            "raw/onnx",
            &RepoInfo {
                siblings: vec![Siblings {
                    rfilename: "encoder.onnx".to_string(),
                }],
                sha: "abc".to_string(),
            },
            Some("decoder.onnx"),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("artifact 'decoder.onnx' was not found")
        );
    }

    #[test]
    fn default_execution_providers_include_cpu() {
        let providers = default_execution_providers();
        assert!(providers.contains(&OnnxExecutionProvider::Cpu));
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn default_execution_providers_prefer_cuda_when_enabled() {
        let providers = default_execution_providers();
        assert_eq!(
            providers,
            vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
        );
    }
}
