// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Hugging Face Hub access shared by every provider that downloads weights.
//!
//! Four providers (`local/candle`, `local/onnx` in both linking modes, and
//! `local/whisper-cpp`) all want the same three things: a client pointed at
//! this crate's own cache directory, a model repository named by a
//! `owner/name` string, and a revision applied to every file they fetch.
//! [`HfModelRepo`] is that, so the ten-odd download sites differ only in
//! which filenames they ask for and how they report failure.
//!
//! # Why a wrapper rather than `HFRepository` directly
//!
//! `hf-hub` 1.0 takes the revision per download call rather than baking it
//! into the repository handle, which is more flexible than this crate needs:
//! an alias pins one revision for all of its files. Holding it here means a
//! caller cannot fetch two files from a repository at different revisions by
//! accident, and keeps `.maybe_revision(...)` out of every call site.

use std::path::{Path, PathBuf};

use hf_hub::repository::{ModelInfo, RepoTypeModel};
use hf_hub::{HFClientBuilder, HFRepository, HFResult, split_id};

/// One model repository at one revision, cached under `cache_dir`.
pub(crate) struct HfModelRepo {
    repo: HFRepository<RepoTypeModel>,
    revision: Option<String>,
}

impl HfModelRepo {
    /// Open `model_id` (an `owner/name` string, or a bare name for a
    /// canonical repo such as `gpt2`) against `cache_dir`.
    ///
    /// Fails only if the client itself cannot be built; nothing is contacted
    /// until a file is requested.
    pub(crate) fn open(cache_dir: &Path, model_id: &str, revision: Option<&str>) -> HFResult<Self> {
        let client = HFClientBuilder::new()
            .cache_dir(cache_dir.to_path_buf())
            .build()?;
        let (owner, name) = split_id(model_id);
        Ok(Self {
            repo: client.model(owner, name),
            revision: revision.map(str::to_string),
        })
    }

    /// Download one file and return its path in the cache.
    ///
    /// Idempotent: an unchanged cached copy is reused without a network
    /// round trip.
    pub(crate) async fn get(&self, filename: &str) -> HFResult<PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }

    /// Download every file in the repository and return the snapshot
    /// directory holding them.
    ///
    /// Only reached by `local/onnx`'s `raw` task, so it is dead code on a
    /// build without that provider.
    ///
    /// Used where a model's own graph references sibling files by relative
    /// path, so they have to land together and the directory itself is the
    /// useful result.
    #[allow(dead_code)] // Used only by providers that load a whole snapshot.
    pub(crate) async fn snapshot(&self) -> HFResult<PathBuf> {
        self.repo
            .snapshot_download()
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }

    /// Fetch the repository's metadata, including its file list.
    ///
    /// Dead code on the same builds as [`Self::snapshot`]: only a provider
    /// that picks an artifact by name needs the listing.
    #[allow(dead_code)] // Used only by providers that select an artifact.
    pub(crate) async fn info(&self) -> HFResult<ModelInfo> {
        self.repo
            .info()
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }
}
