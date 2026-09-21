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

use ort::execution_providers::ExecutionProviderDispatch;

use ort::ep::CPU;
#[cfg(feature = "gpu-cuda")]
use ort::ep::CUDA;
#[cfg(feature = "gpu-metal")]
use ort::ep::CoreML;
#[cfg(feature = "provider-onnx-dynamic")]
use ort::ep::{DirectML, OpenVINO, QNN, ROCm, TensorRT, WebGPU};

use crate::error::{Result, RuntimeError};

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

/// The EP list a session will *actually* be built with: [`resolve_ep_list`]
/// minus any entry whose backing feature is not compiled in.
///
/// This is what [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers)
/// reports, so a caller can tell that a list resolved to `["cpu"]` only
/// because the build lacks `gpu-cuda` / `gpu-metal`.
///
/// Deliberately infallible. Reporting happens *before* the fail-fast
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

    kept.into_iter()
        .enumerate()
        .map(|(index, provider)| {
            let strict = strict_at == Some(index);
            execution_provider_dispatch(provider, strict, alias, provider_label)
        })
        .collect()
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
/// ort 2.0.0-rc.12 `src/lib.rs:188-194`).
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
/// `ort` 2.0.0-rc.12 has a re-entrant `OnceLock` deadlock in its
/// load-dynamic error path: when `libloading::Library::new()` fails,
/// the failure is wrapped in `ort::Error::new(...)`, whose constructor
/// calls back into `ort::api()` to format the status message — which
/// is exactly the `OnceLock` whose initializer just failed. The thread
/// blocks on the same futex forever and the panic that `setup_api`
/// intends to throw never fires.
///
/// This was reported as [pykeio/ort#560] and fixed by [`17ed7277`] but
/// not yet released as of `rc.12`. Until a new ort release ships, we
/// intercept the missing-dylib case here — *before* any code touches
/// the ort API — and surface a clear `Config` error instead.
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
/// Does **not** catch ABI-version mismatches — those still hit the ort
/// deadlock if they happen. Mitigation: the pinned `ort = "=2.0.0-rc.12"`
/// + matched ORT runtime tarball makes version mismatches unlikely.
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
                other => panic!("expected Config error for {ep:?}, got {other:?}"),
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
