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
    /// The execution providers the loaded model runs on (see
    /// [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers));
    /// `None` until it is loaded. Empty for a remote model.
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

/// `url` with any userinfo (`user:pass@`), query (`?…`) and fragment (`#…`)
/// removed, so it can be shown without leaking a credential.
///
/// String-based rather than a full URL parse: it must also clean values that
/// are not valid URLs, and it is applied to whatever a provider reports.
pub(crate) fn redact_url(url: &str) -> String {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let url = &url[..end];
    let (scheme, rest) = match url.find("://") {
        Some(i) => (&url[..i + 3], &url[i + 3..]),
        None => ("", url),
    };
    let authority_end = rest.find('/').unwrap_or(rest.len());
    let (authority, path) = rest.split_at(authority_end);
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    format!("{scheme}{host}{path}")
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
        assert_eq!(
            redact_url("https://user@h/p@q"),
            "https://h/p@q",
            "@ in a path is kept"
        );
        assert_eq!(redact_url("u:p@host/v1"), "host/v1", "no scheme");
        assert_eq!(redact_url("https://h?key=secret"), "https://h");
        assert_eq!(redact_url(""), "");
    }
}
