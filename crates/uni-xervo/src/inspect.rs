// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! What a [`ModelRuntime`](crate::runtime::ModelRuntime) holds, read without
//! loading anything: the aliases in its catalog, each one's load state, the
//! providers registered with it.
//!
//! Built for startup readiness reports and capability listings, so a caller
//! need not keep its own copy of the catalog. Nothing here exposes an alias's
//! `options`: a remote alias shows its endpoint with credentials, query and
//! fragment stripped, and its credential only as the name of the environment
//! variable that holds it.
//!
//! A snapshot is a point-in-time view assembled from several internal maps
//! read one after another, so it can lag a load that is finishing as it is
//! taken.

use std::time::SystemTime;

use crate::api::{ModelTask, WarmupPolicy};

/// One catalog alias, as the runtime sees it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AliasSnapshot {
    pub alias: String,
    pub task: ModelTask,
    pub provider_id: String,
    pub model_id: String,
    pub revision: Option<String>,
    pub warmup: WarmupPolicy,
    pub required: bool,
    pub state: LoadState,
    /// The last [`ModelRuntime::verify`](crate::runtime::ModelRuntime::verify)
    /// of this alias.
    pub verification: VerificationStatus,
    /// The execution providers the loaded model runs on (see
    /// [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers)).
    /// `None` until it is loaded, and for a model that reports none: remote
    /// models, and local backends that cannot say (such as whisper.cpp).
    pub execution_providers: Option<Vec<String>>,
    /// Where a remote alias sends requests, with userinfo, query and fragment
    /// removed. `None` for local models, or when the provider does not say.
    pub endpoint: Option<String>,
    /// The name of the environment variable the alias reads its credential
    /// from, never its value. `None` when it takes none.
    pub credential_env: Option<String>,
}

/// Whether an alias's model is loaded.
///
/// Aliases whose specs resolve to the same model (same provider, model,
/// revision and options) share one load, and so one state.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LoadState {
    /// In the catalog, and no load has been attempted (or one is pending).
    Registered,
    /// A load is in progress.
    Loading,
    /// Loaded and cached; requests are served from memory.
    Loaded,
    /// The last load failed. A later successful load clears this.
    Failed {
        /// The load error's message.
        error: String,
        /// When the load failed.
        at: SystemTime,
    },
}

/// A provider registered with the runtime.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProviderSnapshot {
    pub id: String,
    pub supported_tasks: Vec<ModelTask>,
}

/// What a provider reports about an alias for a snapshot: see
/// [`ModelProvider::describe`](crate::traits::ModelProvider::describe).
///
/// Never secret: an endpoint is redacted again by the runtime, and a
/// credential is named, never read.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ProviderDescription {
    pub endpoint: Option<String>,
    pub credential_env: Option<String>,
}

impl ProviderDescription {
    /// An empty description.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the endpoint the alias sends requests to.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set the name of the environment variable holding the credential.
    #[must_use]
    pub fn with_credential_env(mut self, name: impl Into<String>) -> Self {
        self.credential_env = Some(name.into());
        self
    }
}

/// What [`ModelProvider::verify`](crate::traits::ModelProvider::verify)
/// found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyOutcome {
    /// The endpoint answered and accepted the credential.
    Verified,
    /// Nothing to check: a local provider, or one that does not implement
    /// verification.
    NotApplicable,
}

/// Why a verification failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyErrorKind {
    /// The endpoint rejected the credential (HTTP 401 or 403). A 403 can
    /// also mean a valid credential that lacks a permission.
    Unauthorized,
    /// The endpoint could not be reached (connection refused, DNS failure).
    Unreachable,
    /// The endpoint answered 404: the model or deployment does not exist, or
    /// an OpenAI-compatible server has no `/models` route.
    NotFound,
    /// The endpoint answered 429.
    RateLimited,
    /// No answer within the alias's `load_timeout`.
    Timeout,
    /// The endpoint answered 5xx, or (llama.cpp) is still loading its model.
    Unavailable,
    /// The alias is misconfigured, e.g. its credential's environment
    /// variable is not set.
    Config,
    /// Any other failure.
    Other,
}

impl std::fmt::Display for VerifyErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unauthorized => "unauthorized",
            Self::Unreachable => "unreachable",
            Self::NotFound => "not found",
            Self::RateLimited => "rate limited",
            Self::Timeout => "timed out",
            Self::Unavailable => "unavailable",
            Self::Config => "configuration error",
            Self::Other => "failed",
        })
    }
}

/// A failed verification, as a provider reports it.
///
/// `detail` never holds a credential: providers build it from status codes
/// and redacted URLs only, never from request headers or response bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct VerifyFailure {
    pub kind: VerifyErrorKind,
    pub detail: String,
}

impl VerifyFailure {
    pub fn new(kind: VerifyErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

/// An alias's last verification, for [`AliasSnapshot::verification`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VerificationStatus {
    /// [`ModelRuntime::verify`](crate::runtime::ModelRuntime::verify) has
    /// not been called for this alias.
    NotRequested,
    /// Verification does not apply to this alias's provider.
    NotApplicable { at: SystemTime },
    /// The endpoint was reachable and accepted the credential.
    Verified { at: SystemTime },
    /// The last verification failed.
    Failed {
        kind: VerifyErrorKind,
        detail: String,
        at: SystemTime,
    },
}

/// `url` with any userinfo (`user:pass@`), query (`?…`) and fragment (`#…`)
/// removed, so it can be shown without leaking a credential.
///
/// String-based rather than a full URL parse: it must also clean values that
/// are not valid URLs, and it is applied to whatever a provider reports.
/// Userinfo goes first, through the *last* `@`, because a password may itself
/// contain `?`, `#`, `/` or `@`. That errs toward hiding: an `@` inside a
/// path (rare in a base URL) drops the host before it too, but nothing before
/// an `@` is ever shown.
pub(crate) fn redact_url(url: &str) -> String {
    let (scheme, rest) = match url.find("://") {
        Some(i) => (&url[..i + 3], &url[i + 3..]),
        None => ("", url),
    };
    let rest = rest.rsplit_once('@').map_or(rest, |(_, after)| after);
    let rest = rest.trim_start_matches('/');
    let end = rest.find(['?', '#']).unwrap_or(rest.len());
    format!("{scheme}{}", &rest[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_url_strips_credentials_query_and_fragment() {
        assert_eq!(
            redact_url("https://u:p@h.example/v1?key=x#f"),
            "https://h.example/v1"
        );
        assert_eq!(
            redact_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            redact_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/v1/"
        );
        assert_eq!(
            redact_url("http://[::1]:8080/v1?x=1"),
            "http://[::1]:8080/v1"
        );
        assert_eq!(redact_url("u:p@host/v1"), "host/v1", "no scheme");
        assert_eq!(redact_url("//u:p@h/v1"), "h/v1", "scheme-relative");
        assert_eq!(redact_url("https://h?key=secret"), "https://h");
        assert_eq!(redact_url(""), "");
    }

    /// A password containing `?`, `#`, `/` or `@` must not survive.
    #[test]
    fn redact_url_never_shows_any_part_of_a_password() {
        for (url, shown) in [
            ("https://u:pa?ss@h/v1", "https://h/v1"),
            ("https://u:p#w@h", "https://h"),
            ("https://u:p/w@h/v1", "https://h/v1"),
            ("https://u:p@w@h/v1", "https://h/v1"),
            ("https://u:%40pw@h/v1", "https://h/v1"),
        ] {
            let got = redact_url(url);
            assert_eq!(got, shown, "{url}");
            for part in ["pa", "ss", "p#w", "p/w", "%40pw"] {
                if url.contains(part) {
                    assert!(!got.contains(part), "{url} leaks {part}: {got}");
                }
            }
        }
        // The price of erring toward hiding: an `@` in a path hides the host.
        assert_eq!(redact_url("https://h/p@q"), "https://q");
    }
}
