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
//!
//! # Concurrent loads
//!
//! `hf-hub` 1.0 rewrites `refs/<branch>` on every cached download with a
//! truncate-then-write, and deletes and recreates the snapshot symlink. A
//! second process loading the same repository can read the ref while it is
//! empty and get back `snapshots/<file>` with no commit hash, a path that
//! does not exist; that call also writes the empty ref back, so later loads
//! keep resolving the same bad path (rustic-ai/uni-xervo#67). Three guards
//! keep this crate's downloads correct while that is unfixed upstream:
//!
//! - every download holds an exclusive per-repository lock file in the
//!   cache, so loads from processes using this crate never interleave;
//! - a ref that does not hold a commit hash is removed before downloading,
//!   so `hf-hub` resolves the revision afresh instead of reusing it;
//! - a returned path outside `snapshots/<commit>/` is retried once, which
//!   covers a writer that does not take the lock (another `hf-hub` client).

use std::fs::File;
use std::path::{Component, Path, PathBuf};

use hf_hub::repository::{ModelInfo, RepoTypeModel};
use hf_hub::{HFClientBuilder, HFError, HFRepository, HFResult, split_id};
use tracing::warn;

/// The revision `hf-hub` downloads when none is given.
const DEFAULT_REVISION: &str = "main";

/// One model repository at one revision, cached under `cache_dir`.
pub(crate) struct HfModelRepo {
    repo: HFRepository<RepoTypeModel>,
    revision: Option<String>,
    /// The repository's folder in the cache, `<cache_dir>/models--<owner>--<name>`.
    repo_dir: PathBuf,
    /// Lock file serializing this crate's downloads of the repository.
    lock_path: PathBuf,
}

/// Held for the length of one download; dropping it releases the lock.
///
/// The OS releases the lock when its holder exits, so a crashed process never
/// leaves the repository locked.
struct RepoLock {
    _file: File,
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
        let repo = client.model(owner, name);
        let folder = repo_folder_name(&repo.repo_path());
        Ok(Self {
            repo,
            revision: revision.map(str::to_string),
            repo_dir: cache_dir.join(&folder),
            lock_path: cache_dir
                .join(".locks")
                .join(&folder)
                .join("uni-xervo.lock"),
        })
    }

    /// Take the repository's download lock, waiting for any other holder.
    async fn lock(&self) -> HFResult<RepoLock> {
        let path = self.lock_path.clone();
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::options()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&path)?;
            file.lock()?;
            Ok(RepoLock { _file: file })
        })
        .await
        .map_err(|e| HFError::Other(format!("cache lock task failed: {e}")))?
    }

    /// Remove `refs/<revision>` if it holds anything but a commit hash, so the
    /// next download resolves the revision over the network instead of
    /// reusing a ref a concurrent writer truncated.
    fn discard_invalid_ref(&self) {
        let revision = self.revision.as_deref().unwrap_or(DEFAULT_REVISION);
        if is_commit_hash(revision) {
            return; // pinned: hf-hub never reads or writes a ref for it
        }
        let ref_path = self.repo_dir.join("refs").join(revision);
        let Ok(content) = std::fs::read_to_string(&ref_path) else {
            return;
        };
        if is_commit_hash(content.trim()) {
            return;
        }
        warn!(
            path = %ref_path.display(),
            "Discarding a cached Hugging Face ref that holds no commit hash"
        );
        if let Err(e) = std::fs::remove_file(&ref_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            warn!(path = %ref_path.display(), error = %e, "Could not remove the invalid ref");
        }
    }

    /// Run one download under the repository lock, retrying once when the
    /// returned path is not a real file under `snapshots/<commit>/`.
    async fn locked_download<F, Fut>(
        &self,
        what: &str,
        depth: SnapshotDepth,
        fetch: F,
    ) -> HFResult<PathBuf>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = HFResult<PathBuf>>,
    {
        let _lock = self.lock().await?;
        let mut path = PathBuf::new();
        for attempt in 0..2 {
            self.discard_invalid_ref();
            path = fetch().await?;
            if is_snapshot_path(&self.repo_dir, &path, depth) {
                return Ok(path);
            }
            if attempt == 0 {
                warn!(
                    path = %path.display(),
                    "Hugging Face cache returned a path outside snapshots/<commit>/; retrying"
                );
            }
        }
        Err(HFError::Other(format!(
            "Hugging Face cache resolved {what} to '{}', which is not a file under \
             snapshots/<commit>/; the cache's refs may be corrupt",
            path.display()
        )))
    }

    /// Download one file and return its path in the cache.
    ///
    /// Idempotent: an unchanged cached copy is reused without a network
    /// round trip.
    pub(crate) async fn get(&self, filename: &str) -> HFResult<PathBuf> {
        self.locked_download(&format!("'{filename}'"), SnapshotDepth::File, || {
            self.repo
                .download_file()
                .filename(filename)
                .maybe_revision(self.revision.clone())
                .send()
        })
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
        self.locked_download("the snapshot", SnapshotDepth::Dir, || {
            self.repo
                .snapshot_download()
                .maybe_revision(self.revision.clone())
                .send()
        })
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

/// What a download returns: a file inside a snapshot, or the snapshot itself.
#[derive(Clone, Copy)]
enum SnapshotDepth {
    File,
    Dir,
}

/// The cache folder name `hf-hub` uses for a model repository:
/// `models--<owner>--<name>`.
fn repo_folder_name(repo_path: &str) -> String {
    std::iter::once("models")
        .chain(repo_path.split('/'))
        .collect::<Vec<_>>()
        .join("--")
}

/// A full 40-character hex commit hash.
fn is_commit_hash(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether `path` exists and sits under `<repo_dir>/snapshots/<commit>/`:
/// strictly inside it for a file, the commit folder itself for a snapshot.
fn is_snapshot_path(repo_dir: &Path, path: &Path, depth: SnapshotDepth) -> bool {
    let Ok(rest) = path.strip_prefix(repo_dir.join("snapshots")) else {
        return false;
    };
    let mut parts = rest.components();
    let commit_ok = matches!(
        parts.next(),
        Some(Component::Normal(c)) if c.to_str().is_some_and(is_commit_hash)
    );
    let depth_ok = match depth {
        SnapshotDepth::File => parts.next().is_some(),
        SnapshotDepth::Dir => parts.next().is_none(),
    };
    commit_ok && depth_ok && path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "0123456789abcdef0123456789abcdef01234567";

    fn repo(cache: &Path, revision: Option<&str>) -> HfModelRepo {
        HfModelRepo::open(cache, "owner/name", revision).expect("open repo")
    }

    #[test]
    fn repo_folder_matches_hf_hub_layout() {
        assert_eq!(repo_folder_name("owner/name"), "models--owner--name");
        assert_eq!(repo_folder_name("gpt2"), "models--gpt2");
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), None);
        assert_eq!(r.repo_dir, dir.path().join("models--owner--name"));
    }

    #[test]
    fn invalid_ref_is_discarded_and_valid_ref_kept() {
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), None);
        let refs = r.repo_dir.join("refs");
        std::fs::create_dir_all(&refs).unwrap();

        // A ref truncated by a concurrent writer.
        std::fs::write(refs.join("main"), "").unwrap();
        r.discard_invalid_ref();
        assert!(!refs.join("main").exists(), "empty ref removed");

        std::fs::write(refs.join("main"), format!("{HASH}\n")).unwrap();
        r.discard_invalid_ref();
        assert_eq!(
            std::fs::read_to_string(refs.join("main")).unwrap().trim(),
            HASH,
            "a ref holding a commit hash is left alone"
        );

        // A missing ref is not an error.
        std::fs::remove_file(refs.join("main")).unwrap();
        r.discard_invalid_ref();
    }

    #[test]
    fn pinned_revision_never_touches_refs() {
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), Some(HASH));
        let refs = r.repo_dir.join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), "").unwrap();
        r.discard_invalid_ref();
        assert!(
            refs.join("main").exists(),
            "only the alias's revision is checked"
        );
    }

    #[test]
    fn snapshot_paths_must_carry_a_commit_hash_and_exist() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir.path().join("models--owner--name");
        let snap = repo_dir.join("snapshots").join(HASH);
        std::fs::create_dir_all(snap.join("onnx")).unwrap();
        std::fs::write(snap.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(snap.join("onnx/model.onnx"), "").unwrap();
        // What an empty ref resolves to: no commit folder.
        std::fs::write(repo_dir.join("snapshots/tokenizer.json"), "{}").unwrap();

        let file = SnapshotDepth::File;
        assert!(is_snapshot_path(
            &repo_dir,
            &snap.join("tokenizer.json"),
            file
        ));
        assert!(is_snapshot_path(
            &repo_dir,
            &snap.join("onnx/model.onnx"),
            file
        ));
        assert!(
            !is_snapshot_path(&repo_dir, &repo_dir.join("snapshots/tokenizer.json"), file),
            "hashless path rejected even though it exists"
        );
        assert!(
            !is_snapshot_path(&repo_dir, &snap.join("missing.json"), file),
            "missing file rejected"
        );
        assert!(
            !is_snapshot_path(&repo_dir, &snap, file),
            "a folder is not a file"
        );

        assert!(is_snapshot_path(&repo_dir, &snap, SnapshotDepth::Dir));
        assert!(!is_snapshot_path(
            &repo_dir,
            &repo_dir.join("snapshots"),
            SnapshotDepth::Dir
        ));
    }

    #[tokio::test]
    async fn lock_is_exclusive_until_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), None);
        let held = r.lock().await.unwrap();

        // A second handle (as another process would open) cannot take it.
        let other = File::options().write(true).open(&r.lock_path).unwrap();
        assert!(other.try_lock().is_err(), "lock held");
        drop(held);
        assert!(other.try_lock().is_ok(), "lock released on drop");
    }

    /// Several processes loading the same cached files at once all get real
    /// `snapshots/<commit>/` paths (rustic-ai/uni-xervo#67).
    ///
    /// Re-runs this test binary as child processes that share one cache dir.
    /// Needs network (HEAD requests to the Hub), so it is opt-in:
    /// `EXPENSIVE_TESTS=1 cargo test --lib hf::tests::concurrent -- --ignored`.
    #[test]
    #[ignore]
    fn concurrent_processes_resolve_hashed_paths() {
        if std::env::var("EXPENSIVE_TESTS").is_err() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
        const CHILDREN: usize = 12;
        let dir = tempfile::tempdir().unwrap();
        // Warm the cache once so the children all take the cached path.
        run_stress_child(dir.path(), 1);

        // A cache a past race left broken recovers on the next load: an empty
        // ref plus the hashless `snapshots/<file>` link the racing call
        // created. Left alone, hf-hub matches that link's ETag, gets a 304
        // and returns the hashless path on every later load.
        let repo_dir = dir.path().join("models--Xenova--bge-small-en-v1.5");
        let ref_path = repo_dir.join("refs/main");
        let commit = std::fs::read_to_string(&ref_path).unwrap();
        let good = repo_dir
            .join("snapshots")
            .join(commit.trim())
            .join("tokenizer.json");
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            std::fs::read_link(&good).unwrap(),
            repo_dir.join("snapshots/tokenizer.json"),
        )
        .unwrap();
        std::fs::write(&ref_path, "").unwrap();
        run_stress_child(dir.path(), 1);
        let restored = std::fs::read_to_string(&ref_path).unwrap();
        assert!(
            is_commit_hash(restored.trim()),
            "ref restored, got {restored:?}"
        );

        let exe = std::env::current_exe().unwrap();
        let children: Vec<_> = (0..CHILDREN)
            .map(|_| {
                std::process::Command::new(&exe)
                    .args([
                        "hf::tests::stress_child",
                        "--exact",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("UNI_XERVO_HF_STRESS_CACHE", dir.path())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let failures: Vec<String> = children
            .into_iter()
            .map(|c| c.wait_with_output().unwrap())
            .filter(|o| !o.status.success())
            .map(|o| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                )
            })
            .collect();
        assert!(
            failures.is_empty(),
            "{} child(ren) failed:\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }

    /// Child half of [`concurrent_processes_resolve_hashed_paths`]; a no-op
    /// unless that test spawned it.
    #[test]
    #[ignore]
    fn stress_child() {
        if let Some(cache) = std::env::var_os("UNI_XERVO_HF_STRESS_CACHE") {
            run_stress_child(Path::new(&cache), 20);
        }
    }

    fn run_stress_child(cache: &Path, rounds: usize) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let r = HfModelRepo::open(cache, "Xenova/bge-small-en-v1.5", None).unwrap();
            for _ in 0..rounds {
                for file in ["tokenizer.json", "config.json"] {
                    let path = r
                        .get(file)
                        .await
                        .unwrap_or_else(|e| panic!("get {file}: {e}"));
                    assert!(
                        is_snapshot_path(&r.repo_dir, &path, SnapshotDepth::File),
                        "bad path for {file}: {}",
                        path.display()
                    );
                }
            }
        });
    }
}
