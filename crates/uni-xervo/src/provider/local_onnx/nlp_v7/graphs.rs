// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! The eight graphs: resolution, sessions, and typed tensor plumbing.
//!
//! A kniv v7 release is eight ONNX graphs, three of which (`plain`, `label`,
//! `marker`) ship precision variants. `variant` is a *fallback chain*, not a
//! single suffix: `"int8,fp16"` means "INT8 if this graph has it, else FP16,
//! else fp32". That is how the documented INT8 configuration ends up running
//! `plain-int8`, `marker-int8` and `label-fp16` — the label graph has no INT8
//! file because it runs once per schema and is cached, so quantising it would
//! buy nothing.
//!
//! Every variant keeps fp32 tensors at its boundary, so nothing downstream
//! of this module needs to know which one is loaded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use ndarray::{Array1, Array2, Array3, ArrayD, ArrayViewD, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynTensor, Tensor};

use crate::api::ModelAliasSpec;
use crate::error::{Result, RuntimeError};
use crate::provider::onnx_ep::{OnnxExecutionProvider, build_execution_providers, ort_environment};

/// The graph names a release ships, in the order the pipeline reaches them.
pub(crate) const GRAPH_NAMES: [&str; 8] = [
    "plain",
    "dep_rel",
    "srl_roles",
    "coref_spans",
    "label",
    "span",
    "marker",
    "attr",
];

/// One graph output, in the dtype the graph declares.
#[derive(Debug, Clone)]
pub(crate) enum GraphTensor {
    /// `tensor(float)`.
    F32(ArrayD<f32>),
    /// `tensor(int64)`.
    I64(ArrayD<i64>),
}

/// A graph's outputs, by name.
#[derive(Debug, Default)]
pub(crate) struct Outputs {
    values: HashMap<String, GraphTensor>,
    graph: String,
}

impl Outputs {
    /// A float output, as a dynamic-rank view.
    pub fn f32(&self, name: &str) -> Result<ArrayViewD<'_, f32>> {
        match self.values.get(name) {
            Some(GraphTensor::F32(a)) => Ok(a.view()),
            Some(GraphTensor::I64(_)) => Err(RuntimeError::Config(format!(
                "kniv v7 graph '{}' output '{name}' is int64, expected float",
                self.graph
            ))),
            None => Err(RuntimeError::Config(format!(
                "kniv v7 graph '{}' produced no output '{name}'",
                self.graph
            ))),
        }
    }

    /// A float output with its leading batch dimension (always 1) dropped,
    /// as `[rows, cols]`.
    pub fn f32_batch0(&self, name: &str) -> Result<Array2<f32>> {
        let v = self.f32(name)?;
        let shape = v.shape().to_vec();
        if shape.len() != 3 {
            return Err(RuntimeError::Config(format!(
                "kniv v7 graph '{}' output '{name}' has rank {}, expected 3",
                self.graph,
                shape.len()
            )));
        }
        let (rows, cols) = (shape[1], shape[2]);
        let mut out = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                out[[r, c]] = v[[0, r, c]];
            }
        }
        Ok(out)
    }

    /// A float output of rank 2, as `[rows, cols]`.
    pub fn f32_2d(&self, name: &str) -> Result<Array2<f32>> {
        let v = self.f32(name)?;
        let shape = v.shape().to_vec();
        if shape.len() != 2 {
            return Err(RuntimeError::Config(format!(
                "kniv v7 graph '{}' output '{name}' has rank {}, expected 2",
                self.graph,
                shape.len()
            )));
        }
        let mut out = Array2::<f32>::zeros((shape[0], shape[1]));
        for r in 0..shape[0] {
            for c in 0..shape[1] {
                out[[r, c]] = v[[r, c]];
            }
        }
        Ok(out)
    }

    /// An int64 output of rank 2 or of rank 3 with a leading batch of 1.
    pub fn i64_2d(&self, name: &str) -> Result<Array2<i64>> {
        let v = match self.values.get(name) {
            Some(GraphTensor::I64(a)) => a.view(),
            Some(GraphTensor::F32(_)) => {
                return Err(RuntimeError::Config(format!(
                    "kniv v7 graph '{}' output '{name}' is float, expected int64",
                    self.graph
                )));
            }
            None => {
                return Err(RuntimeError::Config(format!(
                    "kniv v7 graph '{}' produced no output '{name}'",
                    self.graph
                )));
            }
        };
        let shape = v.shape().to_vec();
        let (base, rows, cols) = match shape.len() {
            2 => (false, shape[0], shape[1]),
            3 => (true, shape[1], shape[2]),
            n => {
                return Err(RuntimeError::Config(format!(
                    "kniv v7 graph '{}' output '{name}' has rank {n}, expected 2 or 3",
                    self.graph
                )));
            }
        };
        let mut out = Array2::<i64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                out[[r, c]] = if base { v[[0, r, c]] } else { v[[r, c]] };
            }
        }
        Ok(out)
    }

    /// Keep a float output as-is, for tensors that are fed straight back
    /// into another graph (`rel_dep`, `coref_words`, `pair`, ...).
    pub fn passthrough(&self, name: &str) -> Result<ArrayD<f32>> {
        Ok(self.f32(name)?.to_owned())
    }
}

/// An input tensor, ready to feed.
pub(crate) enum GraphInput {
    /// `tensor(int64)`.
    I64(ArrayD<i64>),
    /// `tensor(float)`.
    F32(ArrayD<f32>),
    /// `tensor(bool)`.
    Bool(ArrayD<bool>),
}

impl GraphInput {
    /// A `[1, n]` int64 row.
    pub fn row_i64(values: Vec<i64>) -> Self {
        let n = values.len();
        Self::I64(
            Array2::from_shape_vec((1, n), values)
                .expect("row shape")
                .into_dyn(),
        )
    }

    /// A `[1, n]` bool row.
    pub fn row_bool(values: Vec<bool>) -> Self {
        let n = values.len();
        Self::Bool(
            Array2::from_shape_vec((1, n), values)
                .expect("row shape")
                .into_dyn(),
        )
    }

    /// A flat `[n]` int64 vector.
    pub fn vec_i64(values: Vec<i64>) -> Self {
        Self::I64(Array1::from_vec(values).into_dyn())
    }

    /// A `[rows, cols]` int64 matrix from row-major rows.
    pub fn mat_i64(rows: Vec<Vec<i64>>) -> Self {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        let flat: Vec<i64> = rows.into_iter().flatten().collect();
        Self::I64(
            Array2::from_shape_vec((r, c), flat)
                .expect("matrix shape")
                .into_dyn(),
        )
    }

    /// A `[rows, cols]` bool matrix from row-major rows.
    pub fn mat_bool(rows: Vec<Vec<bool>>) -> Self {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        let flat: Vec<bool> = rows.into_iter().flatten().collect();
        Self::Bool(
            Array2::from_shape_vec((r, c), flat)
                .expect("matrix shape")
                .into_dyn(),
        )
    }

    /// An already-shaped float tensor.
    pub fn f32(array: ArrayD<f32>) -> Self {
        Self::F32(array)
    }

    /// A `[f, l, h]` float tensor.
    pub fn cube_f32(array: Array3<f32>) -> Self {
        Self::F32(array.into_dyn())
    }

    fn into_dyn_tensor(self, alias: &str, name: &str) -> Result<DynTensor> {
        let err = |e: ort::Error| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("kniv v7 input '{name}': {e}"),
        };
        Ok(match self {
            Self::I64(a) => Tensor::from_array(a).map_err(err)?.upcast(),
            Self::F32(a) => Tensor::from_array(a).map_err(err)?.upcast(),
            Self::Bool(a) => Tensor::from_array(a).map_err(err)?.upcast(),
        })
    }
}

/// The file each graph is loaded from, after variant resolution.
pub(crate) type GraphPaths = HashMap<String, PathBuf>;

/// Lazily-built sessions over a release's graphs.
///
/// One session per graph, each behind its own `Mutex` so one request's
/// `marker` pass does not block another's `plain` pass. Sessions are built on
/// first use: the entity-type and relation graphs are never touched by a
/// request that asks only for the linguistic layer.
pub(crate) struct Graphs {
    paths: GraphPaths,
    sessions: DashMap<String, Arc<Mutex<Session>>>,
    alias: String,
    execution_providers: Option<Vec<OnnxExecutionProvider>>,
    optimization_level: Option<String>,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
}

impl Graphs {
    /// Hold the resolved paths; sessions are built on demand.
    pub fn new(
        paths: GraphPaths,
        spec: &ModelAliasSpec,
        execution_providers: Option<Vec<OnnxExecutionProvider>>,
    ) -> Self {
        let opts = &spec.options;
        Self {
            paths,
            sessions: DashMap::new(),
            alias: spec.alias.clone(),
            execution_providers,
            optimization_level: opts
                .get("graph_optimization_level")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            intra_threads: opts
                .get("intra_op_num_threads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            inter_threads: opts
                .get("inter_op_num_threads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }

    /// The file `graph` loads from.
    pub fn path_of(&self, graph: &str) -> Option<&Path> {
        self.paths.get(graph).map(PathBuf::as_path)
    }

    fn session(&self, graph: &str) -> Result<Arc<Mutex<Session>>> {
        if let Some(existing) = self.sessions.get(graph) {
            return Ok(existing.clone());
        }
        let path = self.paths.get(graph).ok_or_else(|| {
            RuntimeError::Config(format!(
                "kniv v7 alias '{}' has no file for graph '{graph}'",
                self.alias
            ))
        })?;
        match self.sessions.entry(graph.to_string()) {
            Entry::Occupied(e) => Ok(e.get().clone()),
            Entry::Vacant(e) => {
                let built = Arc::new(Mutex::new(self.build_session(path)?));
                e.insert(built.clone());
                Ok(built)
            }
        }
    }

    fn build_session(&self, path: &Path) -> Result<Session> {
        let load_err = |e: &dyn std::fmt::Display| RuntimeError::OnnxLoadFailure {
            alias: self.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        };
        let mut builder = Session::builder(ort_environment()?).map_err(|e| load_err(&e))?;
        // Unlike the v1 path, the configured optimization level and thread
        // counts are honoured rather than validated and dropped.
        let level = match self.optimization_level.as_deref() {
            Some("disable") => GraphOptimizationLevel::Disable,
            Some("basic") => GraphOptimizationLevel::Level1,
            Some("extended") => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };
        builder = builder
            .with_optimization_level(level)
            .map_err(|e| load_err(&e))?;
        if let Some(n) = self.intra_threads {
            builder = builder.with_intra_threads(n).map_err(|e| load_err(&e))?;
        }
        if let Some(n) = self.inter_threads {
            builder = builder.with_inter_threads(n).map_err(|e| load_err(&e))?;
        }
        let dispatch = build_execution_providers(
            self.execution_providers.as_deref(),
            &self.alias,
            "local/onnx",
        )?;
        builder = builder
            .with_execution_providers(dispatch)
            .map_err(|e| load_err(&e))?;
        builder.commit_from_file(path).map_err(|e| load_err(&e))
    }

    /// Run `graph`, feeding only the inputs it declares.
    ///
    /// `declared` comes from `onnx_meta.json`, which is how one code path
    /// serves both encoders: UltraBERT's `plain` takes a `word_end` tensor
    /// for mean pooling and DeBERTa's does not, so the extra input is
    /// prepared either way and dropped here.
    pub fn run(
        &self,
        graph: &str,
        inputs: Vec<(&str, GraphInput)>,
        declared: &[String],
        wanted_outputs: &[String],
    ) -> Result<Outputs> {
        let mut feed: Vec<(String, DynTensor)> = Vec::with_capacity(inputs.len());
        for (name, value) in inputs {
            if !declared.iter().any(|d| d == name) {
                continue;
            }
            feed.push((name.to_string(), value.into_dyn_tensor(&self.alias, name)?));
        }

        let session = self.session(graph)?;
        let mut guard = session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("kniv v7 '{graph}' session lock poisoned: {e}"),
            })?;
        let outputs = guard
            .run(feed)
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("kniv v7 '{graph}' inference: {e}"),
            })?;

        let mut values = HashMap::with_capacity(wanted_outputs.len());
        for name in wanted_outputs {
            let Some(value) = outputs.get(name.as_str()) else {
                continue;
            };
            if let Ok(view) = value.try_extract_array::<f32>() {
                let shape = IxDyn(view.shape());
                let data: Vec<f32> = view.iter().copied().collect();
                values.insert(
                    name.clone(),
                    GraphTensor::F32(ArrayD::from_shape_vec(shape, data).map_err(|e| {
                        RuntimeError::OnnxInvocationFailure {
                            alias: self.alias.clone(),
                            cause: format!("kniv v7 '{graph}' output '{name}': {e}"),
                        }
                    })?),
                );
            } else if let Ok(view) = value.try_extract_array::<i64>() {
                let shape = IxDyn(view.shape());
                let data: Vec<i64> = view.iter().copied().collect();
                values.insert(
                    name.clone(),
                    GraphTensor::I64(ArrayD::from_shape_vec(shape, data).map_err(|e| {
                        RuntimeError::OnnxInvocationFailure {
                            alias: self.alias.clone(),
                            cause: format!("kniv v7 '{graph}' output '{name}': {e}"),
                        }
                    })?),
                );
            } else {
                return Err(RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("kniv v7 '{graph}' output '{name}' is neither float nor int64"),
                });
            }
        }
        Ok(Outputs {
            values,
            graph: graph.to_string(),
        })
    }
}

/// Split a `variant` option into its fallback chain.
///
/// `""` (or `None`) means fp32 only; `"int8,fp16"` means INT8 then FP16 then
/// fp32. Blank entries are dropped, so `"int8,"` is just `"int8"`.
pub(crate) fn variant_chain(variant: Option<&str>) -> Vec<String> {
    variant
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// The candidate file names for `graph`, in the order to try them.
///
/// Always ends with the unsuffixed fp32 file, which every release ships.
pub(crate) fn variant_candidates(graph: &str, chain: &[String]) -> Vec<String> {
    let mut out: Vec<String> = chain.iter().map(|v| format!("{graph}-{v}.onnx")).collect();
    out.push(format!("{graph}.onnx"));
    out
}

/// Resolve each graph to a file in a local release directory.
pub(crate) fn resolve_local_graphs(
    onnx_dir: &Path,
    chain: &[String],
    alias: &str,
) -> Result<GraphPaths> {
    let mut out = GraphPaths::new();
    for graph in GRAPH_NAMES {
        let mut found = None;
        for candidate in variant_candidates(graph, chain) {
            let path = onnx_dir.join(&candidate);
            if path.is_file() {
                found = Some(path);
                break;
            }
        }
        match found {
            Some(path) => {
                out.insert(graph.to_string(), path);
            }
            None => {
                return Err(RuntimeError::OnnxModelNotFound {
                    alias: alias.to_string(),
                    path: onnx_dir.join(format!("{graph}.onnx")),
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_chain_parses_fp32_single_and_fallback() {
        assert!(variant_chain(None).is_empty());
        assert!(variant_chain(Some("")).is_empty());
        assert_eq!(variant_chain(Some("fp16")), ["fp16"]);
        assert_eq!(variant_chain(Some("int8,fp16")), ["int8", "fp16"]);
        assert_eq!(variant_chain(Some(" int8 , , fp16 ")), ["int8", "fp16"]);
    }

    #[test]
    fn candidates_always_end_at_the_fp32_file() {
        let chain = variant_chain(Some("int8,fp16"));
        assert_eq!(
            variant_candidates("plain", &chain),
            ["plain-int8.onnx", "plain-fp16.onnx", "plain.onnx"]
        );
        assert_eq!(variant_candidates("dep_rel", &[]), ["dep_rel.onnx"]);
    }

    #[test]
    fn local_resolution_follows_the_chain_per_graph() {
        let dir = tempfile::tempdir().expect("tempdir");
        let onnx = dir.path();
        // A release shaped like the documented INT8 download set: INT8
        // encoders, an FP16 label graph, fp32 heads.
        for f in [
            "plain-int8.onnx",
            "marker-int8.onnx",
            "label-fp16.onnx",
            "dep_rel.onnx",
            "srl_roles.onnx",
            "coref_spans.onnx",
            "span.onnx",
            "attr.onnx",
        ] {
            std::fs::write(onnx.join(f), b"").expect("write");
        }
        let chain = variant_chain(Some("int8,fp16"));
        let paths = resolve_local_graphs(onnx, &chain, "nlp/v7").expect("resolve");
        assert!(paths["plain"].ends_with("plain-int8.onnx"));
        assert!(paths["marker"].ends_with("marker-int8.onnx"));
        // No INT8 label graph exists, so the chain falls through to FP16.
        assert!(paths["label"].ends_with("label-fp16.onnx"));
        // Head graphs ship fp32 only.
        assert!(paths["attr"].ends_with("attr.onnx"));
    }

    #[test]
    fn a_missing_graph_names_the_file_it_looked_for() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = resolve_local_graphs(dir.path(), &[], "nlp/v7").unwrap_err();
        assert!(
            matches!(err, RuntimeError::OnnxModelNotFound { .. }),
            "{err}"
        );
    }
}
