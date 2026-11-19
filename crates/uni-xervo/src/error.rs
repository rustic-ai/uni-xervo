//! Error types for the Uni-Xervo runtime.

use crate::traits::{DimSize, TensorDtype};
use std::path::PathBuf;
use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Unified error type covering configuration, loading, inference, and transport
/// failures.
///
/// Variants are intentionally coarse-grained so that callers can match on error
/// *category* (e.g. retryable vs permanent) rather than on provider-specific
/// details.
///
/// `#[non_exhaustive]`: new failure categories are added without a breaking
/// release, so a downstream `match` needs a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// Invalid or missing configuration (bad alias format, unknown option, etc.).
    #[error("Configuration error: {0}")]
    Config(String),

    /// The requested alias does not exist in the runtime catalog.
    #[error("Alias not found: {alias}")]
    AliasNotFound { alias: String },

    /// The requested provider ID is not registered with the runtime.
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// A model was requested for a task the provider does not support.
    #[error("Capability mismatch: {0}")]
    CapabilityMismatch(String),

    /// The resolved provider does not expose the requested runtime capability.
    #[error(
        "Provider capability missing for alias '{alias}' (provider '{provider_id}'): {capability}"
    )]
    ProviderCapabilityMissing {
        alias: String,
        provider_id: String,
        capability: String,
    },

    /// Model loading or initialization failed (download, weight parsing, etc.).
    #[error("Load error: {0}")]
    Load(String),

    /// An HTTP or transport-level error from a remote provider.
    #[error("API error: {0}")]
    ApiError(String),

    /// An error during model inference (tokenization, forward pass, etc.).
    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("ONNX model not found for alias '{alias}': {path}")]
    OnnxModelNotFound { alias: String, path: PathBuf },

    #[error("ONNX artifact selection failure for alias '{alias}': {cause}")]
    OnnxArtifactSelectionFailure { alias: String, cause: String },

    #[error("ONNX download failure for alias '{alias}': {cause}")]
    OnnxDownloadFailure { alias: String, cause: String },

    #[error("ONNX load failure for alias '{alias}' at '{path}': {cause}")]
    OnnxLoadFailure {
        alias: String,
        path: PathBuf,
        cause: String,
    },

    #[error("ONNX signature introspection failure for alias '{alias}': {cause}")]
    OnnxSignatureIntrospectionFailure { alias: String, cause: String },

    #[error("ONNX input missing for alias '{alias}': required input '{required_input}'")]
    OnnxInputMissing {
        alias: String,
        required_input: String,
    },

    #[error(
        "ONNX input type mismatch for alias '{alias}', input '{input_name}': expected {expected:?}, got {got:?}"
    )]
    OnnxInputTypeMismatch {
        alias: String,
        input_name: String,
        expected: TensorDtype,
        got: TensorDtype,
    },

    #[error(
        "ONNX input shape mismatch for alias '{alias}', input '{input_name}': expected {expected:?}, got {got:?}"
    )]
    OnnxInputShapeMismatch {
        alias: String,
        input_name: String,
        expected: Vec<DimSize>,
        got: Vec<usize>,
    },

    #[error("ONNX invocation failure for alias '{alias}': {cause}")]
    OnnxInvocationFailure { alias: String, cause: String },

    #[error("ONNX batch stacking failure for alias '{alias}': {cause}")]
    OnnxBatchStackingFailure { alias: String, cause: String },

    /// The remote API returned HTTP 429 (too many requests).
    #[error("Rate limited")]
    RateLimited,

    /// The remote API returned HTTP 401/403 (bad or missing credentials).
    #[error("Unauthorized")]
    Unauthorized,

    /// The operation exceeded its configured timeout.
    #[error("Timeout")]
    Timeout,

    /// The service is currently unavailable (HTTP 5xx, circuit breaker open, etc.).
    #[error("Unavailable")]
    Unavailable,
}

impl RuntimeError {
    /// Returns `true` for transient errors that may succeed on retry:
    /// [`RateLimited`](Self::RateLimited), [`Timeout`](Self::Timeout), and
    /// [`Unavailable`](Self::Unavailable).
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited | Self::Timeout | Self::Unavailable)
    }
}
