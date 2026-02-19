//! Error types for the Uni-Xervo runtime.

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Unified error type covering configuration, loading, inference, and transport
/// failures.
///
/// Variants are intentionally coarse-grained so that callers can match on error
/// *category* (e.g. retryable vs permanent) rather than on provider-specific
/// details.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Invalid or missing configuration (bad alias format, unknown option, etc.).
    #[error("Configuration error: {0}")]
    Config(String),

    /// The requested provider ID is not registered with the runtime.
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// A model was requested for a task the provider does not support.
    #[error("Capability mismatch: {0}")]
    CapabilityMismatch(String),

    /// Model loading or initialization failed (download, weight parsing, etc.).
    #[error("Load error: {0}")]
    Load(String),

    /// An HTTP or transport-level error from a remote provider.
    #[error("API error: {0}")]
    ApiError(String),

    /// An error during model inference (tokenization, forward pass, etc.).
    #[error("Inference error: {0}")]
    InferenceError(String),

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
