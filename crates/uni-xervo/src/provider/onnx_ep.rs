// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared ORT execution-provider selection used by both task implementations
//! of [`LocalOnnxProvider`](super::LocalOnnxProvider) (raw and rerank).
//!
//! The `execution_providers` option in a model alias spec is a list of
//! string identifiers. The always-recognized values are `"cpu"`, `"cuda"`,
//! `"coreml"`. When built with `provider-onnx-dynamic`, the vendor strings
//! `"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"`
//! are also accepted — they require the user to supply a matching ORT
//! library via `ORT_DYLIB_PATH`. This module parses those strings into an
//! internal enum and turns the list into the `Vec<ExecutionProviderDispatch>`
//! that `ort::Session` expects.
//!
//! # Default behavior
//!
//! When the spec doesn't specify `execution_providers`, we fall back to
//! a feature-aware default:
//!
//! - With `gpu-cuda`:  `[Cuda, Cpu]`   (CUDA preferred, CPU fallback).
//! - With `gpu-metal`: `[CoreMl, Cpu]` (CoreML preferred, CPU fallback).
//! - Otherwise:        `[Cpu]`.
//!
//! # Availability filtering
//!
//! A requested EP whose backing feature isn't compiled into this binary is
//! **dropped** from the list, with a warning, so long as something survives.
//! `["cuda", "cpu"]` on a build without `gpu-cuda` therefore runs on CPU —
//! the explicit `cpu` entry is the caller asking for exactly that.
//!
//! A list in which *nothing* survives is a hard `Config` error. Because CPU
//! is always linked in, that is precisely the CPU-free list whose GPU/vendor
//! EPs are all unavailable (`["cuda"]`, `["rocm"]`) — requesting a specific
//! accelerator and silently getting CPU would defeat the point.
//!
//! This is a compile-time question only. Whether the hardware is present at
//! *runtime* is ORT's business, handled by the strictness rules below.
//!
//! # Strict-vs-fallback semantics
//!
//! Applied to the list that survives filtering: when it came from the caller
//! and contains no `Cpu`, its last EP is built with `error_on_failure()` so
//! we don't silently fall back to CPU. Otherwise every EP is built with
//! `fail_silently()` so ORT can chain through the list as written.
//!
//! Filtering first matters here: `["cuda", "rocm"]` without `gpu-cuda`
//! narrows to `[rocm]`, which then becomes the strict last entry.

use ort::ep::ExecutionProviderDispatch;

use ort::ep::CPU;
#[cfg(feature = "gpu-cuda")]
use ort::ep::CUDA;
#[cfg(feature = "gpu-metal")]
use ort::ep::CoreML;
#[cfg(feature = "provider-onnx-dynamic")]
use ort::ep::{DirectML, OpenVINO, QNN, ROCm, TensorRT, WebGPU};

use crate::error::{Result, RuntimeError};

// ---------------------------------------------------------------------------
// Process-global ORT environment
// ---------------------------------------------------------------------------

/// The one [`Environment`](ort::environment::Environment) every session in this
/// process is built under.
///
/// ORT's environment is process-global by nature — it owns the logger, the
/// telemetry switch and any shared thread pool — and `ort` makes that explicit
/// as of 2.0.0-rc.13: `Session::builder` takes `&Environment` rather than
/// reaching for a hidden global. One `OnceLock` here keeps that shape without
/// threading an environment handle through every loader signature.
///
/// Execution providers are deliberately **not** configured on the environment,
/// even though `EnvironmentBuilder::with_execution_providers` would accept
/// them: EP selection is per alias in this crate (see this module's header),
/// so it stays on the session builder where a per-alias list belongs.
static ORT_ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();

/// Borrow the process-global ORT environment, creating it on first use.
///
/// Fails only if ORT itself cannot be initialised — most often a missing or
/// ABI-mismatched dynamic library on a `provider-onnx-dynamic` build.
pub(crate) fn ort_environment() -> Result<&'static ort::environment::Environment> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("uni-xervo")
        .build()
        .map_err(|e| RuntimeError::Config(format!("failed to initialise ONNX Runtime: {e}")))?;
    // A racing thread may have won; either handle is equally valid, so keep
    // whichever landed first and drop ours.
    Ok(ORT_ENV.get_or_init(|| env))
}

/// Start a session builder under the process-global environment with ORT's
/// automatic EP selection pinned to [`PreferCPU`](ort::session::builder::AutoDevicePolicy::PreferCPU).
///
/// Every ONNX session in this crate must be built through here. `ort`'s
/// `Session::builder` sets the auto-device policy to `MaxEfficiency`, which on
/// macOS registers the CoreML EP whenever the linked ONNX Runtime carries it —
/// and from ort 2.0.0-rc.15 (ORT 1.31) the prebuilt `aarch64-apple-darwin`
/// binary always does, with or without `gpu-metal`. CoreML then claims nodes
/// that no alias asked for, and rejects inputs CPU handles fine (zero-length
/// KV-cache tensors among them), and an explicit `execution_providers: ["cpu"]`
/// did not keep it out. Pinning `PreferCPU` leaves the per-alias list from
/// [`build_execution_providers`] as the sole authority over placement: CoreML
/// runs only when that list names it, which needs `gpu-metal`.
pub(crate) fn new_session_builder() -> Result<ort::Result<ort::session::builder::SessionBuilder>> {
    let env = ort_environment()?;
    Ok(ort::session::Session::builder(env).and_then(|b| {
        b.with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(Into::into)
    }))
}

// ---------------------------------------------------------------------------
// Load-time probe and CPU fallback
// ---------------------------------------------------------------------------

/// One small, real inference that proves a freshly built model runs on the
/// execution providers its session was built with.
///
/// ORT's own fallback happens once, at session creation: each node goes to
/// the first EP that *claims* it. An accelerator that accepts a node and then
/// fails to execute it (CoreML on a zero-length KV cache, for one) is never
/// retried on CPU — the inference call just errors. A probe moves that
/// failure to load time, where [`verify_execution_providers`] can still fall
/// back. It catches compile and op-support failures, not every shape: a model
/// that runs the probe can still fail on an input the probe did not cover.
#[async_trait::async_trait]
pub(crate) trait EpProbe: Sync {
    async fn probe(&self) -> Result<()>;
}

/// Whether a session built on `eps` needs a probe: only when something other
/// than CPU could claim nodes. CPU-only builds — the default — pay nothing.
pub(crate) fn needs_probe(eps: &[OnnxExecutionProvider]) -> bool {
    eps.iter().any(|ep| *ep != OnnxExecutionProvider::Cpu)
}

/// Probe `model` and, if its accelerator fails, fall back to CPU.
///
/// `eps` is the session's effective list ([`effective_ep_list`]). Returns the
/// list the model now runs on, for [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers):
///
/// - no probe needed (CPU only) → `eps` unchanged;
/// - probe passes → `eps` unchanged;
/// - probe fails and `eps` contains `cpu` → `rebuild_on_cpu` swaps in a
///   CPU-only session, the probe runs again on it, and the result is `[cpu]`.
///   Logged at `warn`, naming the EPs that failed and why;
/// - probe fails and `eps` has no `cpu` → a load error. A CPU-free list is
///   the caller asking for that accelerator or nothing (the same contract as
///   `error_on_failure()` in [`build_execution_providers`]), and failing here
///   beats failing on the first query.
pub(crate) async fn verify_execution_providers<M: EpProbe>(
    model: &M,
    eps: Vec<OnnxExecutionProvider>,
    alias: &str,
    model_path: &std::path::Path,
    rebuild_on_cpu: impl FnOnce(&M) -> Result<()> + Send,
) -> Result<Vec<OnnxExecutionProvider>> {
    if !needs_probe(&eps) {
        return Ok(eps);
    }
    let Err(probe_err) = model.probe().await else {
        return Ok(eps);
    };
    let failed: Vec<OnnxExecutionProvider> = eps
        .iter()
        .copied()
        .filter(|ep| *ep != OnnxExecutionProvider::Cpu)
        .collect();
    if !eps.contains(&OnnxExecutionProvider::Cpu) {
        return Err(RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: model_path.to_path_buf(),
            cause: format!(
                "load-time probe failed on execution providers [{}] and the list has no \
                 \"cpu\" entry to fall back to: {probe_err}",
                join_eps(&eps)
            ),
        });
    }
    tracing::warn!(
        alias = %alias,
        failed = %join_eps(&failed),
        error = %probe_err,
        "ONNX model failed its load-time probe on the requested execution providers; \
         falling back to CPU"
    );
    rebuild_on_cpu(model)?;
    model
        .probe()
        .await
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: model_path.to_path_buf(),
            cause: format!(
                "load-time probe failed on CPU after [{}] failed: {e}",
                join_eps(&failed)
            ),
        })?;
    Ok(vec![OnnxExecutionProvider::Cpu])
}

/// Replace the session behind `slot`, for a [`verify_execution_providers`]
/// `rebuild_on_cpu` callback.
pub(crate) fn replace_session(
    slot: &std::sync::Mutex<ort::session::Session>,
    session: ort::session::Session,
    alias: &str,
) -> Result<()> {
    let mut guard = slot
        .lock()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Session lock poisoned: {e}"),
        })?;
    *guard = session;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session building
// ---------------------------------------------------------------------------

/// How to build an ORT session for one alias, read once from its options.
///
/// Every `local/onnx` task builds its sessions through [`build_session`] with
/// one of these, so the shared session options (`execution_providers`,
/// `graph_optimization_level`, `intra_op_num_threads`,
/// `inter_op_num_threads`) mean the same thing for every task.
#[derive(Debug, Clone)]
pub(crate) struct SessionConfig {
    /// The caller's list, or `None` for the feature-aware default.
    pub(crate) execution_providers: Option<Vec<OnnxExecutionProvider>>,
    pub(crate) optimization_level: ort::session::builder::GraphOptimizationLevel,
    pub(crate) intra_threads: Option<usize>,
    pub(crate) inter_threads: Option<usize>,
}

impl SessionConfig {
    /// Read the shared session options from an alias's options.
    ///
    /// `default_level` applies when `graph_optimization_level` is unset, so
    /// each task keeps the level it has always used: `Level3` for most,
    /// `Level1` (`"basic"`) for `raw`.
    pub(crate) fn from_options(
        options: &serde_json::Value,
        default_level: ort::session::builder::GraphOptimizationLevel,
    ) -> Result<Self> {
        use ort::session::builder::GraphOptimizationLevel as Level;
        let execution_providers =
            parse_execution_providers_option(options.get("execution_providers"))?;
        let optimization_level = match options
            .get("graph_optimization_level")
            .and_then(serde_json::Value::as_str)
        {
            Some("disable") => Level::Disable,
            Some("basic") => Level::Level1,
            Some("extended") => Level::Level2,
            Some("all") => Level::All,
            // Option validation rejects any other string first.
            _ => default_level,
        };
        let threads = |key: &str| {
            options
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
        };
        Ok(Self {
            execution_providers,
            optimization_level,
            intra_threads: threads("intra_op_num_threads"),
            inter_threads: threads("inter_op_num_threads"),
        })
    }

    /// The same settings on CPU only: the rebuild [`verify_execution_providers`]
    /// falls back to.
    pub(crate) fn cpu_only(&self) -> Self {
        Self {
            execution_providers: Some(vec![OnnxExecutionProvider::Cpu]),
            ..self.clone()
        }
    }

    /// Fail fast on an execution-provider list this build cannot honour,
    /// before any download or session build. The same check
    /// [`build_session`] makes, moved to the start of a load.
    pub(crate) fn check_execution_providers(&self, alias: &str) -> Result<()> {
        build_execution_providers(self.execution_providers.as_deref(), alias, "local/onnx")
            .map(|_| ())
    }

    /// The execution providers a session built from this config would
    /// request (what the load-time probe starts from; a built session reports
    /// the providers that actually registered, see [`BuiltSession`]):
    /// the requested list minus any whose feature is not compiled in.
    pub(crate) fn effective_execution_providers(&self) -> Vec<OnnxExecutionProvider> {
        effective_ep_list(self.execution_providers.as_deref())
    }
}

/// A built session and the execution providers it was built with.
pub(crate) struct BuiltSession {
    pub(crate) session: ort::session::Session,
    /// The providers that registered with the session, in priority order
    /// (`[cpu]` after a CPU rebuild). What
    /// [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers)
    /// reports, until a load-time probe falls back to CPU.
    pub(crate) execution_providers: Vec<OnnxExecutionProvider>,
}

/// Build one ORT session from `path` with `cfg`.
///
/// The one place a `local/onnx` session is built, so every task honours the
/// same options and reports the execution providers it ended up with.
///
/// Providers are registered one at a time and only those that registered are
/// reported (an accelerator whose library or device is missing is skipped,
/// with a warning, unless it is the strict last entry of a CPU-free list).
/// If the session then fails to commit with an accelerator attached and the
/// list has a `cpu` entry, it is built again on CPU alone and reports
/// `["cpu"]`: some accelerators register fine and only fail once the model is
/// placed on them.
pub(crate) fn build_session(
    path: &std::path::Path,
    alias: &str,
    cfg: &SessionConfig,
) -> Result<BuiltSession> {
    let plan =
        planned_execution_providers(cfg.execution_providers.as_deref(), alias, "local/onnx")?;
    match commit_session(path, alias, cfg, &plan) {
        Ok(built) => Ok(built),
        Err(CommitError::AfterRegistration { error, attached })
            if retry_commit_on_cpu(&plan, &attached) =>
        {
            let on_cpu = commit_session(path, alias, cfg, &[(OnnxExecutionProvider::Cpu, false)])
                .map_err(CommitError::into_error);
            resolve_cpu_retry(alias, &attached, error, on_cpu)
        }
        Err(e) => Err(e.into_error()),
    }
}

/// Settle a CPU retry after a session failed to commit on `attached`.
///
/// A commit error does not say whether the accelerator caused it, so the
/// retry is attempted for any failure. If CPU succeeds, the accelerator was
/// the problem: warn and use the CPU session. If CPU fails too, the model
/// itself is at fault (corrupt file, missing sidecar), so return the
/// original error, which names the cause, and log nothing about a fallback.
fn resolve_cpu_retry<T>(
    alias: &str,
    attached: &[OnnxExecutionProvider],
    original: RuntimeError,
    on_cpu: Result<T>,
) -> Result<T> {
    match on_cpu {
        Ok(session) => {
            tracing::warn!(
                alias = %alias,
                failed = %join_eps(attached),
                error = %original,
                "ONNX session failed to build on the requested execution providers; \
                 built it on CPU instead"
            );
            Ok(session)
        }
        Err(_) => Err(original),
    }
}

/// The providers a session reports, from those that registered: everything
/// up to and including `cpu`.
///
/// ORT gives each node to the first provider in the list that claims it, and
/// the CPU provider claims every standard operator, so a provider listed
/// after `cpu` runs nothing. Reporting it would let `["cpu", "coreml"]` look
/// accelerated.
fn reported_providers(mut attached: Vec<OnnxExecutionProvider>) -> Vec<OnnxExecutionProvider> {
    if let Some(cpu) = attached
        .iter()
        .position(|ep| *ep == OnnxExecutionProvider::Cpu)
    {
        attached.truncate(cpu + 1);
    }
    attached
}

/// A failed [`commit_session`], telling apart a failure once providers had
/// registered, which a CPU retry may fix.
enum CommitError {
    Before(RuntimeError),
    AfterRegistration {
        error: RuntimeError,
        attached: Vec<OnnxExecutionProvider>,
    },
}

impl CommitError {
    fn into_error(self) -> RuntimeError {
        match self {
            Self::Before(e) | Self::AfterRegistration { error: e, .. } => e,
        }
    }
}

fn commit_session(
    path: &std::path::Path,
    alias: &str,
    cfg: &SessionConfig,
    plan: &[(OnnxExecutionProvider, bool)],
) -> std::result::Result<BuiltSession, CommitError> {
    let load_err = |e: &dyn std::fmt::Display| RuntimeError::OnnxLoadFailure {
        alias: alias.to_string(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    };
    let before = |e: RuntimeError| CommitError::Before(e);
    let mut builder = new_session_builder()
        .map_err(before)?
        .map_err(|e| before(load_err(&e)))?;
    builder = builder
        .with_optimization_level(cfg.optimization_level)
        .map_err(|e| before(load_err(&e)))?;
    if let Some(n) = cfg.intra_threads {
        builder = builder
            .with_intra_threads(n)
            .map_err(|e| before(load_err(&e)))?;
    }
    if let Some(n) = cfg.inter_threads {
        builder = builder
            .with_inter_threads(n)
            .map_err(|e| before(load_err(&e)))?;
    }
    let mut dispatch_err = None;
    let registration = register_execution_providers(plan, alias, path, |provider, strict| {
        let dispatch = match execution_provider_dispatch(provider, strict, alias, "local/onnx") {
            Ok(d) => d,
            Err(e) => {
                // Unreachable after planning: the plan holds only compiled-in
                // providers. Kept as an error rather than a panic.
                let message = e.to_string();
                dispatch_err = Some(e);
                return Err(message);
            }
        };
        dispatch.register(&mut builder).map_err(|e| e.to_string())
    })
    .map_err(before)?;
    if let Some(e) = dispatch_err {
        return Err(before(e));
    }
    let mut attached = reported_providers(registration.attached);
    if attached.is_empty() {
        // Nothing registered (every entry skipped): ORT runs the graph on its
        // built-in CPU provider.
        attached.push(OnnxExecutionProvider::Cpu);
    }
    // Keep downloads of the same repo from recreating the snapshot links
    // ORT is resolving (the graph's external-data sidecar among them).
    let _read_lock = crate::hf::lock_cached_repo_for_reading(path);
    match builder.commit_from_file(path) {
        Ok(session) => Ok(BuiltSession {
            session,
            execution_providers: attached,
        }),
        Err(e) => Err(CommitError::AfterRegistration {
            error: load_err(&e),
            attached,
        }),
    }
}

/// The providers in `a` that are also in `b`, in `a`'s order: what a model
/// built from several sessions runs on end to end.
pub(crate) fn common_execution_providers(
    a: &[OnnxExecutionProvider],
    b: &[OnnxExecutionProvider],
) -> Vec<OnnxExecutionProvider> {
    a.iter().copied().filter(|ep| b.contains(ep)).collect()
}

/// EP ids as strings, the shape [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers) reports.
pub(crate) fn ep_ids(eps: &[OnnxExecutionProvider]) -> Vec<String> {
    eps.iter().map(|ep| ep.as_str().to_string()).collect()
}

/// Parsed `execution_providers` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    CoreMl,
    Rocm,
    DirectMl,
    OpenVino,
    Qnn,
    TensorRt,
    WebGpu,
}

impl OnnxExecutionProvider {
    /// Parse a single execution-provider string. Returns `None` for
    /// unrecognized values so the caller can surface a precise
    /// configuration error rather than silently falling back to CPU.
    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "coreml" => Some(Self::CoreMl),
            "rocm" => Some(Self::Rocm),
            "directml" => Some(Self::DirectMl),
            "openvino" => Some(Self::OpenVino),
            "qnn" => Some(Self::Qnn),
            "tensorrt" => Some(Self::TensorRt),
            "webgpu" => Some(Self::WebGpu),
            _ => None,
        }
    }

    /// Human-readable label used in error messages.
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
            Self::Rocm => "ROCm",
            Self::DirectMl => "DirectML",
            Self::OpenVino => "OpenVINO",
            Self::Qnn => "QNN",
            Self::TensorRt => "TensorRT",
            Self::WebGpu => "WebGPU",
        }
    }

    /// Stable string id used by spec options and surfaced through
    /// [`RawTensorModel::active_execution_providers`](crate::traits::RawTensorModel::active_execution_providers).
    /// Round-trips with [`OnnxExecutionProvider::from_str`].
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::Rocm => "rocm",
            Self::DirectMl => "directml",
            Self::OpenVino => "openvino",
            Self::Qnn => "qnn",
            Self::TensorRt => "tensorrt",
            Self::WebGpu => "webgpu",
        }
    }

    /// True for execution providers that require `provider-onnx-dynamic`
    /// (a vendor-supplied ORT library loaded via `ORT_DYLIB_PATH`). The
    /// pyke-bundled binary used by `provider-onnx` doesn't include them.
    fn requires_dynamic(&self) -> bool {
        matches!(
            self,
            Self::Rocm
                | Self::DirectMl
                | Self::OpenVino
                | Self::Qnn
                | Self::TensorRt
                | Self::WebGpu
        )
    }

    /// Whether this EP can be built at all in the current binary.
    ///
    /// Purely a compile-time question — it asks whether the backing feature
    /// was enabled, not whether the hardware is present. Runtime availability
    /// is ORT's business and is handled by `fail_silently()` one layer down.
    ///
    /// Uses the `cfg!()` *expression* macro rather than `#[cfg]` attributes so
    /// this stays a single total match, which keeps it in lockstep with the
    /// dispatch arms in [`execution_provider_dispatch`]. The two are pinned
    /// together by `availability_predicate_matches_dispatch`.
    fn is_available(&self) -> bool {
        match self {
            // CPU is always linked in — see the unconditional `CPU` import.
            // This is load-bearing: it guarantees a list containing `cpu`
            // can never filter down to empty.
            Self::Cpu => true,
            Self::Cuda => cfg!(feature = "gpu-cuda"),
            Self::CoreMl => cfg!(feature = "gpu-metal"),
            Self::Rocm
            | Self::DirectMl
            | Self::OpenVino
            | Self::Qnn
            | Self::TensorRt
            | Self::WebGpu => cfg!(feature = "provider-onnx-dynamic"),
        }
    }
}

/// Split a requested EP list into the entries this build can construct and
/// the entries it cannot.
///
/// Order is preserved in both halves, so the surviving list keeps the
/// caller's priority ordering.
fn partition_available(
    requested: &[OnnxExecutionProvider],
) -> (Vec<OnnxExecutionProvider>, Vec<OnnxExecutionProvider>) {
    requested.iter().copied().partition(|ep| ep.is_available())
}

/// Resolve the EP list that will be used for a session: either the
/// caller-provided list verbatim, or the feature-aware default.
pub(crate) fn resolve_ep_list(
    configured: Option<&[OnnxExecutionProvider]>,
) -> Vec<OnnxExecutionProvider> {
    configured
        .map(<[OnnxExecutionProvider]>::to_vec)
        .unwrap_or_else(default_execution_providers)
}

/// The EP list a session will try to register: [`resolve_ep_list`] minus any
/// entry whose backing feature is not compiled in.
///
/// An upper bound on what [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers)
/// reports, which is the subset that actually registered (see
/// [`build_session`]).
///
/// Deliberately infallible. It is computed *before* the fail-fast
/// validation probe at each load site, and making this fallible would move
/// the origin of EP errors and blur the "validate before any I/O" ordering
/// that `onnx_ep_resolution_test.rs` pins. An empty return is unreachable in
/// practice: the probe rejects an all-unavailable list moments later.
pub(crate) fn effective_ep_list(
    configured: Option<&[OnnxExecutionProvider]>,
) -> Vec<OnnxExecutionProvider> {
    let (kept, _dropped) = partition_available(&resolve_ep_list(configured));
    kept
}

/// Index of the EP that should be built with `error_on_failure()`, if any.
///
/// Takes the **already-filtered** list: which entry is "last" changes once
/// unavailable EPs are dropped, and getting that wrong would silently turn a
/// CPU-free request into a quiet CPU fallback at ORT registration time.
///
/// Strict ⟺ the list came from the user (not the feature-aware default),
/// contains no `Cpu`, and this is its final entry. Kept as a pure, `ort`-free
/// function because `ExecutionProviderDispatch`'s `error_on_failure` field is
/// private and absent from its `Debug` impl — this is the only way to test
/// the behaviour.
fn strict_index(kept: &[OnnxExecutionProvider], configured: bool) -> Option<usize> {
    if !configured || kept.is_empty() || kept.contains(&OnnxExecutionProvider::Cpu) {
        return None;
    }
    Some(kept.len() - 1)
}

/// Default EP list when the spec doesn't specify one.
pub(crate) fn default_execution_providers() -> Vec<OnnxExecutionProvider> {
    #[cfg(feature = "gpu-cuda")]
    {
        vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
    }
    #[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
    {
        vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
    }
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    {
        vec![OnnxExecutionProvider::Cpu]
    }
}

/// Build the `Vec<ExecutionProviderDispatch>` to hand to
/// `Session::builder().with_execution_providers(...)`.
///
/// `configured` is the user-supplied list (or `None` to use defaults).
/// `provider_label` is the provider id string used only in error
/// messages (e.g. `"local/onnx"`) so failures point at the right alias.
///
/// Entries whose backing feature isn't compiled in are **dropped** rather
/// than fatal, provided at least one entry survives — so `["cuda", "cpu"]`
/// runs on CPU in a build without `gpu-cuda`, which is what an explicit
/// fallback entry asks for. Only an all-unavailable list is an error.
///
/// Note this runs twice per load (once as the fail-fast probe before any
/// I/O, once for real at session build; more for OCR, which builds several
/// sessions), so a dropped-EP warning is emitted more than once per alias.
/// The message is identical and idempotent; deduplicating it would mean
/// touching all eight load sites for no behavioural gain.
pub(crate) fn build_execution_providers(
    configured: Option<&[OnnxExecutionProvider]>,
    alias: &str,
    provider_label: &str,
) -> Result<Vec<ExecutionProviderDispatch>> {
    planned_execution_providers(configured, alias, provider_label)?
        .into_iter()
        .map(|(provider, strict)| {
            execution_provider_dispatch(provider, strict, alias, provider_label)
        })
        .collect()
}

/// The providers a session registers, in order, each paired with whether its
/// registration failure is an error (`strict`) rather than a skip.
///
/// The filtering and strictness rules of [`build_execution_providers`]
/// (see this module's header), without building the dispatches.
fn planned_execution_providers(
    configured: Option<&[OnnxExecutionProvider]>,
    alias: &str,
    provider_label: &str,
) -> Result<Vec<(OnnxExecutionProvider, bool)>> {
    let requested = resolve_ep_list(configured);
    let (kept, dropped) = partition_available(&requested);

    // The feature-aware default is cfg-generated, so it can only ever name
    // EPs this build can construct.
    debug_assert!(
        configured.is_some() || dropped.is_empty(),
        "default EP list must never contain an unavailable provider"
    );

    if kept.is_empty() {
        // Nothing survived, so the list was CPU-free and none of it is
        // available. Name the highest-priority entry: for the single-EP
        // lists this is the common case for, the message is unchanged.
        return Err(all_unavailable(&requested, alias, provider_label));
    }

    if !dropped.is_empty() {
        tracing::warn!(
            alias = %alias,
            provider = %provider_label,
            dropped = %join_eps(&dropped),
            using = %join_eps(&kept),
            "Requested execution providers are not available in this build; continuing without them"
        );
    }

    let strict_at = strict_index(&kept, configured.is_some());

    Ok(kept
        .into_iter()
        .enumerate()
        .map(|(index, provider)| (provider, strict_at == Some(index)))
        .collect())
}

/// Which planned providers registered with a session builder.
#[derive(Debug, Default, PartialEq)]
struct Registration {
    /// Registered, in priority order.
    attached: Vec<OnnxExecutionProvider>,
    /// Failed to register and were skipped, with ORT's reason.
    skipped: Vec<(OnnxExecutionProvider, String)>,
}

/// Register each planned provider in order through `register`, recording
/// which attached.
///
/// ORT's own `with_execution_providers` does the same loop but only logs a
/// skipped provider, so the session could not say what it runs on. Doing it
/// here keeps the module's strictness rules: a strict entry that fails is a
/// load error, any other is skipped with a warning.
fn register_execution_providers(
    plan: &[(OnnxExecutionProvider, bool)],
    alias: &str,
    path: &std::path::Path,
    mut register: impl FnMut(OnnxExecutionProvider, bool) -> std::result::Result<(), String>,
) -> Result<Registration> {
    let mut registration = Registration::default();
    for &(provider, strict) in plan {
        match register(provider, strict) {
            Ok(()) => registration.attached.push(provider),
            Err(cause) if strict => {
                return Err(RuntimeError::OnnxLoadFailure {
                    alias: alias.to_string(),
                    path: path.to_path_buf(),
                    cause: format!(
                        "execution provider \"{}\" failed to register and the list has no \
                         \"cpu\" entry to fall back to: {cause}",
                        provider.as_str()
                    ),
                });
            }
            Err(cause) => {
                tracing::warn!(
                    alias = %alias,
                    provider = provider.as_str(),
                    error = %cause,
                    "ONNX execution provider failed to register; continuing without it"
                );
                registration.skipped.push((provider, cause));
            }
        }
    }
    Ok(registration)
}

/// Whether a session whose commit failed should be built again on CPU alone:
/// an accelerator attached (so it may be what failed, e.g. CUDA with no usable
/// device), the plan has a `cpu` fallback, and the attempt was not already
/// CPU-only.
fn retry_commit_on_cpu(
    plan: &[(OnnxExecutionProvider, bool)],
    attached: &[OnnxExecutionProvider],
) -> bool {
    plan.iter().any(|(ep, _)| *ep == OnnxExecutionProvider::Cpu)
        && attached.iter().any(|ep| *ep != OnnxExecutionProvider::Cpu)
}

/// Comma-separated EP ids, for log fields and error messages.
fn join_eps(providers: &[OnnxExecutionProvider]) -> String {
    providers
        .iter()
        .map(|ep| ep.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Error for a requested list in which nothing is available.
///
/// Built on top of [`feature_not_enabled`] for the highest-priority entry so
/// single-EP lists keep their exact existing message, with a trailing hint
/// appended only when the list had more than one entry.
fn all_unavailable(
    requested: &[OnnxExecutionProvider],
    alias: &str,
    provider_label: &str,
) -> RuntimeError {
    let first = requested
        .first()
        .copied()
        .expect("an empty EP list resolves to the non-empty default");
    let base = feature_not_enabled(first, alias, provider_label);
    if requested.len() < 2 {
        return base;
    }
    RuntimeError::Config(format!(
        "{base} (none of the requested execution providers [{}] is available in \
         this build; add a \"cpu\" entry to allow fallback)",
        join_eps(requested)
    ))
}

/// Build a vendor-EP dispatch when `provider-onnx-dynamic` is active, or
/// return a Config error pointing the user at the right feature otherwise.
/// Implemented as a macro so the early-return flows through the caller's
/// `match` arm.
///
/// Since [`build_execution_providers`] now filters on
/// [`OnnxExecutionProvider::is_available`] before dispatching, the error
/// branch is unreachable in practice. It is kept as a fail-loud backstop: if
/// the availability predicate and these cfg arms ever drift apart, this
/// errors rather than silently mis-building a session.
macro_rules! vendor_dispatch {
    ($provider:ident, $alias:ident, $provider_label:ident, $ep:ident) => {{
        #[cfg(feature = "provider-onnx-dynamic")]
        {
            $ep::default().build()
        }
        #[cfg(not(feature = "provider-onnx-dynamic"))]
        {
            return Err(feature_not_enabled($provider, $alias, $provider_label));
        }
    }};
}

fn execution_provider_dispatch(
    provider: OnnxExecutionProvider,
    strict: bool,
    alias: &str,
    provider_label: &str,
) -> Result<ExecutionProviderDispatch> {
    let dispatch = match provider {
        OnnxExecutionProvider::Cpu => CPU::default().build(),
        OnnxExecutionProvider::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                CUDA::default().build()
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
        OnnxExecutionProvider::CoreMl => {
            #[cfg(feature = "gpu-metal")]
            {
                CoreML::default().build()
            }
            #[cfg(not(feature = "gpu-metal"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
        OnnxExecutionProvider::Rocm => vendor_dispatch!(provider, alias, provider_label, ROCm),
        OnnxExecutionProvider::DirectMl => {
            vendor_dispatch!(provider, alias, provider_label, DirectML)
        }
        OnnxExecutionProvider::OpenVino => {
            vendor_dispatch!(provider, alias, provider_label, OpenVINO)
        }
        OnnxExecutionProvider::Qnn => vendor_dispatch!(provider, alias, provider_label, QNN),
        OnnxExecutionProvider::TensorRt => {
            vendor_dispatch!(provider, alias, provider_label, TensorRT)
        }
        OnnxExecutionProvider::WebGpu => {
            vendor_dispatch!(provider, alias, provider_label, WebGPU)
        }
    };

    Ok(if strict {
        dispatch.error_on_failure()
    } else {
        dispatch.fail_silently()
    })
}

fn feature_not_enabled(
    provider: OnnxExecutionProvider,
    alias: &str,
    provider_label: &str,
) -> RuntimeError {
    if provider.requires_dynamic() {
        return RuntimeError::Config(format!(
            "Alias '{alias}' requested {} execution for {provider_label}, but vendor execution providers require the `provider-onnx-dynamic` feature plus a vendor-supplied ONNX Runtime library via ORT_DYLIB_PATH",
            provider.label()
        ));
    }
    let feature = match provider {
        OnnxExecutionProvider::Cuda => "gpu-cuda",
        OnnxExecutionProvider::CoreMl => "gpu-metal",
        OnnxExecutionProvider::Cpu => unreachable!("CPU is always available"),
        _ => unreachable!("vendor EPs handled above"),
    };
    RuntimeError::Config(format!(
        "Alias '{alias}' requested {} execution for {provider_label}, but {feature} is not enabled",
        provider.label()
    ))
}

/// Parse a `serde_json::Value` array into a list of
/// `OnnxExecutionProvider`s. Accepts either a JSON array of strings or
/// a single string.
///
/// Returns:
/// - `Ok(None)` when the value isn't present, or when an explicit empty
///   array is given (caller falls back to the feature-aware default).
/// - `Err(RuntimeError::Config)` when an entry is unrecognized or has the
///   wrong JSON shape — surfaces typos and stale docs at load time
///   instead of letting them silently degrade to CPU.
pub(crate) fn parse_execution_providers_option(
    value: Option<&serde_json::Value>,
) -> Result<Option<Vec<OnnxExecutionProvider>>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let raw: Vec<&str> = if let Some(arr) = value.as_array() {
        arr.iter()
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "execution_providers entries must be strings, got {v}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(s) = value.as_str() {
        vec![s]
    } else {
        return Err(RuntimeError::Config(format!(
            "execution_providers must be a string or array of strings, got {value}"
        )));
    };

    let providers: Vec<OnnxExecutionProvider> = raw
        .into_iter()
        .map(|s| {
            OnnxExecutionProvider::from_str(s).ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Unknown execution_providers entry `{s}`. \
                     Recognized values: cpu, cuda, coreml, rocm, directml, openvino, qnn, tensorrt, webgpu."
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if providers.is_empty() {
        Ok(None)
    } else {
        Ok(Some(providers))
    }
}

/// Default ONNX Runtime dylib filename for the current platform. Mirrors
/// what ort itself searches for when `ORT_DYLIB_PATH` is unset (see
/// `setup_api` in ort 2.0.0-rc.15 `src/lib.rs`).
///
/// Only present under `provider-onnx-dynamic`. Under `provider-onnx`
/// (bundled CPU) the lib is statically linked into the binary; there's
/// no dlopen and the preflight is meaningless.
#[cfg(feature = "provider-onnx-dynamic")]
fn default_dylib_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so"
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        "libonnxruntime.so"
    }
}

/// Pre-flight check that fails fast if the ONNX Runtime dynamic library
/// can't be loaded.
///
/// # Why this exists
///
/// A missing or unloadable dylib is unrecoverable inside `ort`: its
/// `setup_api` calls `panic!` from within a `OnceLock` initializer, so the
/// caller gets an unwind — or, on a `panic = "abort"` profile, no caller at
/// all — rather than an error it can report. Since this is the single most
/// likely misconfiguration of a `provider-onnx-dynamic` build, the dylib is
/// checked here *before* any code touches the ort API, turning that panic
/// into a `Config` error naming the alias, the path it tried and the fix.
///
/// Through `ort` 2.0.0-rc.12 the same path was worse still: the failure was
/// wrapped in `ort::Error::new(...)`, whose constructor called back into
/// `ort::api()` — the very `OnceLock` whose initializer had just failed — so
/// the thread blocked on that futex forever and the intended panic never
/// fired ([pykeio/ort#560], fixed by [`17ed7277`]). That deadlock is gone as
/// of the pinned rc.15, which leaves this check as the fast, actionable
/// failure rather than a workaround.
///
/// # Behavior
///
/// 1. If `ORT_DYLIB_PATH` is set, attempt `libloading::Library::new(path)`
///    and immediately drop the handle.
/// 2. Otherwise, attempt the platform-default name
///    (`libonnxruntime.so` / `.dylib` / `onnxruntime.dll`). The dynamic
///    loader walks `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / standard
///    system paths, so brew/system installs are auto-detected.
/// 3. On failure, return `RuntimeError::Config` naming the alias, the
///    attempted path, the OS error, and the migration-doc reference.
/// 4. On success, drop the handle. The lib stays in the loader's
///    in-memory cache (refcount), so ort's subsequent `dlopen` finds it
///    instantly without a second disk read.
///
/// Idempotent: safe to call from multiple loads. Cheap (one syscall on
/// the success path; one syscall + one error format on the failure path).
///
/// # Limitations
///
/// Does **not** catch ABI-version mismatches: a library that loads but
/// exports the wrong API version still panics inside ort. Mitigation: the
/// pinned `ort = "=2.0.0-rc.15"` plus a matched ORT runtime tarball makes
/// version mismatches unlikely.
///
/// [pykeio/ort#560]: https://github.com/pykeio/ort/issues/560
/// [`17ed7277`]: https://github.com/pykeio/ort/commit/17ed7277
#[cfg(feature = "provider-onnx-dynamic")]
pub(crate) fn preflight_ort_dylib(alias: &str, provider_label: &str) -> Result<()> {
    let (path_str, source) = match std::env::var("ORT_DYLIB_PATH") {
        Ok(s) if !s.is_empty() => (s, "ORT_DYLIB_PATH env var"),
        _ => (default_dylib_name().to_string(), "platform default"),
    };

    // SAFETY: libloading::Library::new is unsafe because the loaded library
    // may run init code (DT_INIT, DllMain). For ONNX Runtime this is well-defined.
    match unsafe { libloading::Library::new(&path_str) } {
        Ok(lib) => {
            drop(lib);
            Ok(())
        }
        Err(e) => Err(RuntimeError::Config(format!(
            "Alias '{alias}' ({provider_label}): cannot load ONNX Runtime dynamic library `{path_str}` (from {source}): {e}.\n\
             Set ORT_DYLIB_PATH to a downloaded ONNX Runtime release tarball matching your hardware. \
             See `docs/migrations/0.9.0-feature-surface.md` for setup instructions."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod session_config {
        use super::*;
        use ort::session::builder::GraphOptimizationLevel as Level;
        use serde_json::json;

        fn level_of(options: serde_json::Value, default: Level) -> String {
            format!(
                "{:?}",
                SessionConfig::from_options(&options, default)
                    .unwrap()
                    .optimization_level
            )
        }

        #[test]
        fn optimization_level_maps_every_value() {
            let d = Level::Level3;
            for (value, expected) in [
                ("disable", Level::Disable),
                ("basic", Level::Level1),
                ("extended", Level::Level2),
                ("all", Level::All),
            ] {
                assert_eq!(
                    level_of(json!({ "graph_optimization_level": value }), d),
                    format!("{expected:?}"),
                    "{value}"
                );
            }
        }

        #[test]
        fn unset_level_keeps_the_task_default() {
            assert_eq!(
                level_of(json!({}), Level::Level3),
                format!("{:?}", Level::Level3)
            );
            assert_eq!(
                level_of(json!(null), Level::Level1),
                format!("{:?}", Level::Level1)
            );
        }

        #[test]
        fn threads_and_providers_are_read() {
            let cfg = SessionConfig::from_options(
                &json!({
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 2,
                    "execution_providers": ["cpu"],
                }),
                Level::Level3,
            )
            .unwrap();
            assert_eq!(cfg.intra_threads, Some(4));
            assert_eq!(cfg.inter_threads, Some(2));
            assert_eq!(
                cfg.execution_providers,
                Some(vec![OnnxExecutionProvider::Cpu])
            );
            let unset = SessionConfig::from_options(&json!({}), Level::Level3).unwrap();
            assert_eq!(unset.intra_threads, None);
            assert_eq!(unset.execution_providers, None);
        }

        #[test]
        fn cpu_only_keeps_everything_but_the_providers() {
            let cfg = SessionConfig::from_options(
                &json!({ "graph_optimization_level": "basic", "intra_op_num_threads": 3 }),
                Level::Level3,
            )
            .unwrap();
            let cpu = cfg.cpu_only();
            assert_eq!(
                cpu.execution_providers,
                Some(vec![OnnxExecutionProvider::Cpu])
            );
            assert_eq!(cpu.intra_threads, Some(3));
            assert_eq!(
                format!("{:?}", cpu.optimization_level),
                format!("{:?}", Level::Level1)
            );
            assert_eq!(
                cpu.effective_execution_providers(),
                [OnnxExecutionProvider::Cpu]
            );
        }

        #[test]
        fn common_providers_keep_the_first_lists_order() {
            use OnnxExecutionProvider::{CoreMl, Cpu, Cuda};
            assert_eq!(
                common_execution_providers(&[Cuda, CoreMl, Cpu], &[Cpu, Cuda]),
                [Cuda, Cpu]
            );
            assert!(common_execution_providers(&[Cuda], &[Cpu]).is_empty());
        }
    }

    mod registration {
        use super::*;
        use OnnxExecutionProvider::{CoreMl, Cpu, Cuda};
        use std::path::Path;

        fn register(
            plan: &[(OnnxExecutionProvider, bool)],
            failing: &[OnnxExecutionProvider],
        ) -> Result<Registration> {
            register_execution_providers(plan, "a", Path::new("m.onnx"), |ep, _| {
                if failing.contains(&ep) {
                    Err(format!("{} library not found", ep.as_str()))
                } else {
                    Ok(())
                }
            })
        }

        #[test]
        fn all_register() {
            let r = register(&[(Cuda, false), (Cpu, false)], &[]).unwrap();
            assert_eq!(r.attached, [Cuda, Cpu]);
            assert!(r.skipped.is_empty());
        }

        /// The #64 case: CUDA compiled in but unusable reports CPU only.
        #[test]
        fn a_skipped_accelerator_is_not_reported() {
            let r = register(&[(Cuda, false), (Cpu, false)], &[Cuda]).unwrap();
            assert_eq!(r.attached, [Cpu]);
            assert_eq!(r.skipped.len(), 1);
            assert_eq!(r.skipped[0].0, Cuda);
            assert!(r.skipped[0].1.contains("library not found"));
        }

        #[test]
        fn order_is_kept_around_a_skip() {
            let r = register(&[(Cuda, false), (CoreMl, false), (Cpu, false)], &[Cuda]).unwrap();
            assert_eq!(r.attached, [CoreMl, Cpu]);
        }

        #[test]
        fn a_strict_entry_that_fails_is_a_load_error() {
            let err = register(&[(CoreMl, false), (Cuda, true)], &[Cuda]).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("\"cuda\" failed to register"), "{msg}");
            assert!(msg.contains("library not found"), "{msg}");
        }

        #[test]
        fn strictness_is_passed_to_the_registrar() {
            let mut seen = Vec::new();
            register_execution_providers(
                &[(Cuda, false), (Cpu, true)],
                "a",
                Path::new("m.onnx"),
                |ep, strict| {
                    seen.push((ep, strict));
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(seen, [(Cuda, false), (Cpu, true)]);
        }

        #[test]
        fn a_failed_commit_retries_on_cpu_only_with_a_cpu_fallback() {
            let with_cpu = [(Cuda, false), (Cpu, false)];
            assert!(retry_commit_on_cpu(&with_cpu, &[Cuda, Cpu]));
            // Already on CPU alone: retrying would fail the same way.
            assert!(!retry_commit_on_cpu(&with_cpu, &[Cpu]));
            // No cpu entry: the caller asked for the accelerator or nothing.
            assert!(!retry_commit_on_cpu(&[(Cuda, true)], &[Cuda]));
        }

        #[test]
        fn providers_after_cpu_are_not_reported() {
            assert_eq!(reported_providers(vec![Cpu, CoreMl]), [Cpu]);
            assert_eq!(reported_providers(vec![CoreMl, Cpu]), [CoreMl, Cpu]);
            assert_eq!(reported_providers(vec![Cuda]), [Cuda]);
        }

        fn load_failure(cause: &str) -> RuntimeError {
            RuntimeError::OnnxLoadFailure {
                alias: "a".into(),
                path: "m.onnx".into(),
                cause: cause.into(),
            }
        }

        #[test]
        fn a_cpu_retry_that_succeeds_is_used() {
            let got = resolve_cpu_retry("a", &[CoreMl, Cpu], load_failure("coreml"), Ok(7));
            assert_eq!(got.unwrap(), 7);
        }

        /// A model that fails on CPU too (corrupt file) is reported with the
        /// first attempt's error, not the retry's.
        #[test]
        fn a_cpu_retry_that_fails_returns_the_original_error() {
            let got: Result<()> = resolve_cpu_retry(
                "a",
                &[CoreMl, Cpu],
                load_failure("Protobuf parsing failed"),
                Err(load_failure("second attempt")),
            );
            let msg = got.unwrap_err().to_string();
            assert!(msg.contains("Protobuf parsing failed"), "{msg}");
            assert!(!msg.contains("second attempt"), "{msg}");
        }

        #[test]
        fn plan_marks_the_strict_entry() {
            let plan = planned_execution_providers(Some(&[Cpu]), "a", "local/onnx").unwrap();
            assert_eq!(plan, [(Cpu, false)]);
            assert_eq!(
                planned_execution_providers(None, "a", "local/onnx")
                    .unwrap()
                    .iter()
                    .filter(|(_, strict)| *strict)
                    .count(),
                0,
                "the default list is never strict"
            );
        }
    }

    mod probe_fallback {
        use super::*;
        use std::path::Path;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        /// Stand-in for a loaded model: its probe fails until "rebuilt on
        /// CPU" when `accel_broken`, and always when `cpu_broken`.
        #[derive(Default)]
        struct FakeModel {
            accel_broken: bool,
            cpu_broken: bool,
            on_cpu: AtomicBool,
            probes: AtomicUsize,
            rebuilds: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EpProbe for FakeModel {
            async fn probe(&self) -> Result<()> {
                self.probes.fetch_add(1, Ordering::SeqCst);
                let on_cpu = self.on_cpu.load(Ordering::SeqCst);
                if (on_cpu && self.cpu_broken) || (!on_cpu && self.accel_broken) {
                    return Err(RuntimeError::Config("probe exploded".into()));
                }
                Ok(())
            }
        }

        async fn verify(
            model: &FakeModel,
            eps: Vec<OnnxExecutionProvider>,
        ) -> Result<Vec<OnnxExecutionProvider>> {
            verify_execution_providers(model, eps, "embed/test", Path::new("m.onnx"), |m| {
                m.rebuilds.fetch_add(1, Ordering::SeqCst);
                m.on_cpu.store(true, Ordering::SeqCst);
                Ok(())
            })
            .await
        }

        use OnnxExecutionProvider::{CoreMl, Cpu, Cuda};

        #[tokio::test]
        async fn cpu_only_list_skips_the_probe() {
            let model = FakeModel {
                accel_broken: true,
                ..Default::default()
            };
            assert_eq!(verify(&model, vec![Cpu]).await.unwrap(), vec![Cpu]);
            assert_eq!(model.probes.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        async fn working_accelerator_is_kept() {
            let model = FakeModel::default();
            assert_eq!(
                verify(&model, vec![CoreMl, Cpu]).await.unwrap(),
                vec![CoreMl, Cpu]
            );
            assert_eq!(model.probes.load(Ordering::SeqCst), 1);
            assert_eq!(model.rebuilds.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        async fn failing_accelerator_falls_back_to_cpu() {
            let model = FakeModel {
                accel_broken: true,
                ..Default::default()
            };
            assert_eq!(verify(&model, vec![Cuda, Cpu]).await.unwrap(), vec![Cpu]);
            assert_eq!(model.rebuilds.load(Ordering::SeqCst), 1);
            // Once on the accelerator, once more on the rebuilt CPU session.
            assert_eq!(model.probes.load(Ordering::SeqCst), 2);
        }

        #[tokio::test]
        async fn cpu_free_list_fails_the_load_instead_of_falling_back() {
            let model = FakeModel {
                accel_broken: true,
                ..Default::default()
            };
            let err = verify(&model, vec![CoreMl]).await.unwrap_err();
            assert!(
                matches!(&err, RuntimeError::OnnxLoadFailure { cause, .. }
                    if cause.contains("no \"cpu\" entry") && cause.contains("probe exploded")),
                "unexpected error: {err:?}"
            );
            assert_eq!(model.rebuilds.load(Ordering::SeqCst), 0);
        }

        #[tokio::test]
        async fn failure_on_cpu_after_fallback_is_a_load_error() {
            let model = FakeModel {
                accel_broken: true,
                cpu_broken: true,
                ..Default::default()
            };
            let err = verify(&model, vec![CoreMl, Cpu]).await.unwrap_err();
            assert!(
                matches!(&err, RuntimeError::OnnxLoadFailure { cause, .. }
                    if cause.contains("failed on CPU after [coreml] failed")),
                "unexpected error: {err:?}"
            );
        }
    }

    #[test]
    fn from_str_known_values() {
        assert_eq!(
            OnnxExecutionProvider::from_str("cpu"),
            Some(OnnxExecutionProvider::Cpu)
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("cuda"),
            Some(OnnxExecutionProvider::Cuda)
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("coreml"),
            Some(OnnxExecutionProvider::CoreMl)
        );
    }

    #[test]
    fn from_str_unknown_returns_none() {
        assert_eq!(OnnxExecutionProvider::from_str("bogus"), None);
        assert_eq!(OnnxExecutionProvider::from_str(""), None);
    }

    #[test]
    fn parse_array_form() {
        let v = serde_json::json!(["cuda", "cpu"]);
        let parsed = parse_execution_providers_option(Some(&v)).unwrap();
        assert_eq!(
            parsed,
            Some(vec![
                OnnxExecutionProvider::Cuda,
                OnnxExecutionProvider::Cpu
            ])
        );
    }

    #[test]
    fn parse_string_form() {
        let v = serde_json::json!("cuda");
        let parsed = parse_execution_providers_option(Some(&v)).unwrap();
        assert_eq!(parsed, Some(vec![OnnxExecutionProvider::Cuda]));
    }

    #[test]
    fn parse_missing_returns_none() {
        assert!(parse_execution_providers_option(None).unwrap().is_none());
    }

    #[test]
    fn parse_empty_array_returns_none() {
        let v = serde_json::json!([]);
        assert!(
            parse_execution_providers_option(Some(&v))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parse_unknown_entry_errors() {
        let v = serde_json::json!(["cuda", "bogus"]);
        assert!(matches!(
            parse_execution_providers_option(Some(&v)),
            Err(RuntimeError::Config(_))
        ));
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn default_prefers_cuda_when_enabled() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
    #[test]
    fn default_prefers_coreml_when_metal_enabled() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    #[test]
    fn default_is_cpu_only_without_gpu() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(not(feature = "gpu-metal"))]
    #[test]
    fn unsupported_feature_returns_config_error() {
        // CoreML requested without gpu-metal active
        let result = build_execution_providers(
            Some(&[OnnxExecutionProvider::CoreMl]),
            "test/alias",
            "local/test",
        );
        assert!(matches!(result, Err(RuntimeError::Config(_))));
    }

    #[test]
    fn vendor_eps_parse_to_expected_variants() {
        for (s, expected) in [
            ("rocm", OnnxExecutionProvider::Rocm),
            ("directml", OnnxExecutionProvider::DirectMl),
            ("openvino", OnnxExecutionProvider::OpenVino),
            ("qnn", OnnxExecutionProvider::Qnn),
            ("tensorrt", OnnxExecutionProvider::TensorRt),
            ("webgpu", OnnxExecutionProvider::WebGpu),
        ] {
            assert_eq!(OnnxExecutionProvider::from_str(s), Some(expected));
            assert_eq!(expected.as_str(), s);
        }
    }

    #[cfg(not(feature = "provider-onnx-dynamic"))]
    #[test]
    fn vendor_eps_fail_under_bundled_provider() {
        // Each vendor EP must error with a Config message that points the
        // user at provider-onnx-dynamic when the bundled feature is active.
        for ep in [
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            let result = build_execution_providers(Some(&[ep]), "test/alias", "local/test");
            match result {
                Err(RuntimeError::Config(msg)) => {
                    assert!(
                        msg.contains("provider-onnx-dynamic"),
                        "expected message to mention provider-onnx-dynamic, got: {msg}"
                    );
                }
                Ok(_) => panic!("expected Config error for {ep:?}, got a dispatch list"),
                Err(other) => panic!("expected Config error for {ep:?}, got {other:?}"),
            }
        }
    }

    #[cfg(feature = "provider-onnx-dynamic")]
    #[test]
    fn vendor_eps_dispatch_under_dynamic() {
        for ep in [
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            // CPU as fallback so the strict path doesn't kick in. We only
            // assert the dispatch builds — actual EP registration happens
            // at session-build time, which we don't reach here.
            let result = build_execution_providers(
                Some(&[ep, OnnxExecutionProvider::Cpu]),
                "test/alias",
                "local/test",
            );
            assert!(result.is_ok(), "expected dispatch to build for {ep:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Availability filtering: unavailable EPs are dropped when a viable
    // fallback remains, and only a wholly-unavailable list is fatal.
    // -----------------------------------------------------------------------

    /// Regression test: `["cuda", "cpu"]` on a build without `gpu-cuda` used
    /// to fail the entire load, because the per-EP error short-circuited
    /// `collect()` before the `cpu` entry was ever reached.
    #[cfg(not(feature = "gpu-cuda"))]
    #[test]
    fn cuda_with_cpu_fallback_drops_cuda() {
        let result = build_execution_providers(
            Some(&[OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]),
            "test/alias",
            "local/test",
        )
        .expect("an explicit cpu fallback must survive a missing gpu-cuda");
        assert_eq!(result.len(), 1, "only cpu should remain");
    }

    /// CoreML is the exact structural mirror of CUDA, gated on `gpu-metal`.
    #[cfg(not(feature = "gpu-metal"))]
    #[test]
    fn coreml_with_cpu_fallback_drops_coreml() {
        let result = build_execution_providers(
            Some(&[OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]),
            "test/alias",
            "local/test",
        )
        .expect("an explicit cpu fallback must survive a missing gpu-metal");
        assert_eq!(result.len(), 1, "only cpu should remain");
    }

    /// Mirror image of `vendor_eps_fail_under_bundled_provider`: the same
    /// EPs that are fatal alone are merely dropped when cpu backs them up.
    #[cfg(not(feature = "provider-onnx-dynamic"))]
    #[test]
    fn vendor_eps_with_cpu_fallback_are_dropped() {
        for ep in [
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            let result = build_execution_providers(
                Some(&[ep, OnnxExecutionProvider::Cpu]),
                "test/alias",
                "local/test",
            )
            .unwrap_or_else(|e| panic!("expected {ep:?} to be dropped, not fatal: {e}"));
            assert_eq!(result.len(), 1, "only cpu should remain for {ep:?}");
        }
    }

    /// A CPU-free list in which nothing is available stays fatal — that is
    /// the contract the strict path exists to protect. The message still
    /// names the highest-priority entry, and gains a hint about `cpu`.
    #[cfg(all(not(feature = "gpu-cuda"), not(feature = "provider-onnx-dynamic")))]
    #[test]
    fn all_unavailable_cpu_free_list_errors() {
        let err = build_execution_providers(
            Some(&[OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Rocm]),
            "test/alias",
            "local/test",
        )
        .err()
        .expect("a list with no viable entry must fail");
        let msg = err.to_string();
        assert!(msg.contains("CUDA"), "{msg}");
        assert!(msg.contains("gpu-cuda"), "{msg}");
        assert!(msg.contains("cpu"), "should hint at a cpu fallback: {msg}");
    }

    #[test]
    fn cpu_only_list_is_untouched() {
        let result = build_execution_providers(
            Some(&[OnnxExecutionProvider::Cpu]),
            "test/alias",
            "local/test",
        )
        .expect("cpu is always available");
        assert_eq!(result.len(), 1);
    }

    /// The default list is cfg-generated, so filtering it must be a no-op on
    /// every build — this is what lets `strict_index`'s `configured` flag stay
    /// meaningful.
    #[test]
    fn default_list_is_never_filtered() {
        let defaults = default_execution_providers();
        let (kept, dropped) = partition_available(&defaults);
        assert!(dropped.is_empty(), "default named an unavailable EP");
        assert_eq!(kept, defaults);
    }

    // -----------------------------------------------------------------------
    // strict_index — pure, so it is testable under every feature set
    // -----------------------------------------------------------------------

    #[test]
    fn strict_index_marks_last_of_cpu_free_configured_list() {
        assert_eq!(
            strict_index(&[OnnxExecutionProvider::Cuda], true),
            Some(0),
            "a lone configured GPU EP must be strict"
        );
        // The post-filter survivor of ["cuda", "rocm"]: rocm is now last, so
        // strictness shifts onto it.
        assert_eq!(strict_index(&[OnnxExecutionProvider::Rocm], true), Some(0));
    }

    #[test]
    fn strict_index_is_none_when_cpu_can_absorb() {
        assert_eq!(
            strict_index(
                &[OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu],
                true
            ),
            None
        );
        assert_eq!(strict_index(&[OnnxExecutionProvider::Cpu], true), None);
    }

    #[test]
    fn strict_index_never_applies_to_defaults() {
        // `configured == false` means the feature-aware default, which must
        // always be free to chain down to CPU.
        assert_eq!(
            strict_index(
                &[OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu],
                false
            ),
            None
        );
        assert_eq!(strict_index(&[OnnxExecutionProvider::Cuda], false), None);
        assert_eq!(strict_index(&[], true), None);
    }

    // -----------------------------------------------------------------------
    // Consistency between the availability predicate and the dispatch arms
    // -----------------------------------------------------------------------

    /// `is_available` and the `#[cfg]` arms of `execution_provider_dispatch`
    /// encode the same feature rules in two places. Drift where the predicate
    /// is *too permissive* still hard-errors (the backstop); drift where it is
    /// *too strict* would silently drop a usable EP. Pin both directions.
    #[test]
    fn availability_predicate_matches_dispatch() {
        for ep in [
            OnnxExecutionProvider::Cpu,
            OnnxExecutionProvider::Cuda,
            OnnxExecutionProvider::CoreMl,
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            let built = build_execution_providers(Some(&[ep]), "test/alias", "local/test").is_ok();
            assert_eq!(
                ep.is_available(),
                built,
                "is_available() disagrees with dispatch for {ep:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // effective_ep_list — what active_execution_providers() reports
    // -----------------------------------------------------------------------

    #[cfg(not(feature = "gpu-cuda"))]
    #[test]
    fn effective_ep_list_drops_unavailable() {
        assert_eq!(
            effective_ep_list(Some(&[
                OnnxExecutionProvider::Cuda,
                OnnxExecutionProvider::Cpu
            ])),
            vec![OnnxExecutionProvider::Cpu],
            "reporting must not claim an EP that was never compiled in"
        );
    }

    #[test]
    fn effective_ep_list_without_config_is_the_default() {
        assert_eq!(effective_ep_list(None), default_execution_providers());
    }
}
