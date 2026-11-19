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
//!   cache, so loads from processes using this crate never interleave. The
//!   wait is bounded, and a cache where the lock cannot be taken (read-only,
//!   or a filesystem without file locks) is used unlocked, as before; a
//!   pinned commit whose file is already cached needs no lock at all;
//! - a ref that does not hold a commit hash is removed before downloading,
//!   so `hf-hub` resolves the revision afresh instead of reusing it;
//! - a returned path outside `snapshots/<commit>/` is retried once, which
//!   covers a writer that does not take the lock (another `hf-hub` client).

use std::fs::{File, TryLockError};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use hf_hub::repository::{ModelInfo, RepoTypeModel};
use hf_hub::{HFClientBuilder, HFError, HFRepository, HFResult, split_id};
use tracing::warn;

/// The revision `hf-hub` downloads when none is given.
const DEFAULT_REVISION: &str = "main";

/// How long a download waits for another process's lock before going ahead
/// without it. Long enough for a cached load or a modest download to finish;
/// a holder stuck mid-download no longer stalls everyone else indefinitely.
const LOCK_WAIT: Duration = Duration::from_secs(60);

/// One model repository at one revision, cached under `cache_dir`.
pub(crate) struct HfModelRepo {
    repo: HFRepository<RepoTypeModel>,
    revision: Option<String>,
    /// The repository's folder in the cache, `<cache_dir>/models--<owner>--<name>`.
    repo_dir: PathBuf,
    /// Lock file serializing this crate's downloads of the repository.
    lock_path: PathBuf,
    /// How long to wait for another holder; [`LOCK_WAIT`] outside tests.
    lock_wait: Duration,
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
        if let Some(rev) = revision
            && !is_safe_revision(rev)
        {
            return Err(HFError::InvalidParameter(format!(
                "revision '{rev}' must be a branch, tag or commit, not a path"
            )));
        }
        let (owner, name) = split_id(model_id);
        let repo = client.model(owner, name);
        let folder = repo_folder_name(&repo.repo_path());
        Ok(Self {
            repo,
            revision: revision.map(str::to_string),
            repo_dir: cache_dir.join(&folder),
            lock_path: repo_lock_path(cache_dir, &folder),
            lock_wait: LOCK_WAIT,
        })
    }

    /// Take the repository's download lock, waiting up to [`LOCK_WAIT`] for
    /// another holder.
    ///
    /// `None` means go ahead unlocked, as this crate did before it locked at
    /// all: the cache is read-only, the filesystem has no file locks, or the
    /// holder did not finish in time. The other guards still apply. Polls
    /// with `try_lock` rather than blocking a thread in `lock`, so a load
    /// cancelled by its timeout leaves nothing waiting behind it.
    async fn lock(&self) -> Option<RepoLock> {
        let file = match self.open_lock_file() {
            Ok(file) => file,
            Err(e) => {
                warn!(
                    path = %self.lock_path.display(),
                    error = %e,
                    "Cannot create the Hugging Face cache lock; downloading unlocked"
                );
                return None;
            }
        };
        let started = Instant::now();
        let mut pause = Duration::from_millis(20);
        loop {
            match file.try_lock() {
                Ok(()) => return Some(RepoLock { _file: file }),
                Err(TryLockError::WouldBlock) if started.elapsed() < self.lock_wait => {
                    tokio::time::sleep(pause).await;
                    pause = (pause * 2).min(Duration::from_millis(500));
                }
                Err(TryLockError::WouldBlock) => {
                    warn!(
                        path = %self.lock_path.display(),
                        waited_secs = self.lock_wait.as_secs(),
                        "Another process still holds the Hugging Face cache lock; downloading unlocked"
                    );
                    return None;
                }
                Err(TryLockError::Error(e)) => {
                    warn!(
                        path = %self.lock_path.display(),
                        error = %e,
                        "Cannot take the Hugging Face cache lock; downloading unlocked"
                    );
                    return None;
                }
            }
        }
    }

    fn open_lock_file(&self) -> std::io::Result<File> {
        if let Some(parent) = self.lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock_path)
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
        let _lock = self.lock().await;
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
        let cause = if is_snapshot_path_unchecked(&self.repo_dir, &path, depth) {
            "which does not exist: its blob is missing, so delete that snapshot \
             entry (or the repository's cache folder) to download it again"
        } else {
            "which is not under snapshots/<commit>/: the cache's refs may be corrupt"
        };
        Err(HFError::Other(format!(
            "Hugging Face cache resolved {what} to '{}', {cause}",
            path.display()
        )))
    }

    /// Download one file and return its path in the cache.
    ///
    /// Idempotent: an unchanged cached copy is reused without a network
    /// round trip.
    pub(crate) async fn get(&self, filename: &str) -> HFResult<PathBuf> {
        // A pinned commit already in the cache is read as is: hf-hub returns
        // it without touching the network or writing, so it needs no lock
        // and works on a read-only cache.
        if let Some(rev) = self.revision.as_deref()
            && is_commit_hash(rev)
        {
            let cached = self.repo_dir.join("snapshots").join(rev).join(filename);
            if is_snapshot_path(&self.repo_dir, &cached, SnapshotDepth::File) {
                return Ok(cached);
            }
        }
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

/// The lock file for the repository cached at `<cache_dir>/<folder>`.
fn repo_lock_path(cache_dir: &Path, folder: &str) -> PathBuf {
    cache_dir.join(".locks").join(folder).join("uni-xervo.lock")
}

/// Hold a **shared** lock on the repository a cached file belongs to, while
/// a reader opens it and the files beside it.
///
/// A cached download through `hf-hub` deletes and recreates the file's
/// snapshot symlinks. ONNX Runtime, building a session from
/// `snapshots/<commit>/model.onnx`, resolves its external-data sidecar
/// (`model.onnx.data`) through such a link; if another load of the same repo
/// recreates the link at that moment, the sidecar briefly resolves to a path
/// outside the model's directory and the session fails with "External data
/// path escapes model directory". Downloads take this lock exclusively, so
/// holding it shared keeps them out while the reader runs, in this process
/// or any other.
///
/// `None`, meaning read unlocked, when `path` is not inside a Hugging Face
/// cache (a local model file), the lock cannot be taken (read-only cache, no
/// file locks), or a download holds it past [`LOCK_WAIT`].
pub(crate) fn lock_cached_repo_for_reading(path: &Path) -> Option<File> {
    // `<cache_dir>/models--<owner>--<name>/snapshots/<commit>/<file…>`
    let repo_dir = path.ancestors().find(|dir| {
        dir.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("models--"))
            && path.starts_with(dir.join("snapshots"))
    })?;
    let folder = repo_dir.file_name()?.to_str()?;
    let lock_path = repo_lock_path(repo_dir.parent()?, folder);
    std::fs::create_dir_all(lock_path.parent()?).ok()?;
    let file = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .ok()?;
    let started = Instant::now();
    let mut pause = Duration::from_millis(20);
    loop {
        match file.try_lock_shared() {
            Ok(()) => return Some(file),
            Err(TryLockError::WouldBlock) if started.elapsed() < LOCK_WAIT => {
                // Called from synchronous session building, which already
                // blocks its thread for far longer than this wait.
                std::thread::sleep(pause);
                pause = (pause * 2).min(Duration::from_millis(500));
            }
            Err(_) => {
                warn!(
                    path = %lock_path.display(),
                    "Reading a cached Hugging Face model without its repository lock"
                );
                return None;
            }
        }
    }
}

/// Whether a revision is safe to join onto `refs/`: no absolute path, `..`
/// or empty component, which no branch, tag or commit contains.
fn is_safe_revision(revision: &str) -> bool {
    !revision.is_empty()
        && Path::new(revision)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// Whether `path` exists and sits under `<repo_dir>/snapshots/<commit>/`:
/// strictly inside it for a file, the commit folder itself for a snapshot.
fn is_snapshot_path(repo_dir: &Path, path: &Path, depth: SnapshotDepth) -> bool {
    is_snapshot_path_unchecked(repo_dir, path, depth) && path.exists()
}

/// [`is_snapshot_path`] without the existence check.
fn is_snapshot_path_unchecked(repo_dir: &Path, path: &Path, depth: SnapshotDepth) -> bool {
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
    commit_ok && depth_ok
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

    #[test]
    fn revisions_that_are_paths_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["../../etc", "/abs", "a/../b", "./main", ""] {
            assert!(
                HfModelRepo::open(dir.path(), "owner/name", Some(bad)).is_err(),
                "{bad:?} accepted"
            );
        }
        for good in ["main", "v1.0", "refs/pr/1", HASH] {
            assert!(HfModelRepo::open(dir.path(), "owner/name", Some(good)).is_ok());
        }
    }

    /// A pinned commit already cached loads from a read-only cache: no lock
    /// file is created and nothing is written.
    #[cfg(unix)]
    #[tokio::test]
    async fn pinned_cached_file_loads_from_read_only_cache() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let snap = cache.join("models--owner--name/snapshots").join(HASH);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("tokenizer.json"), "{}").unwrap();
        let read_only = |mode| {
            for d in [
                cache.clone(),
                cache.join("models--owner--name"),
                cache.join("models--owner--name/snapshots"),
                snap.clone(),
            ] {
                std::fs::set_permissions(&d, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        read_only(0o555);

        let r = repo(&cache, Some(HASH));
        let got = r.get("tokenizer.json").await;
        read_only(0o755); // let tempdir clean up
        assert_eq!(got.unwrap(), snap.join("tokenizer.json"));
        assert!(!cache.join(".locks").exists(), "no lock taken");
    }

    /// A cache where the lock file cannot be created is used unlocked.
    #[cfg(unix)]
    #[tokio::test]
    async fn unlockable_cache_downloads_unlocked() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), None);
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        let lock = r.lock().await;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(lock.is_none());
    }

    /// A holder that never lets go stalls others only for `lock_wait`.
    #[tokio::test]
    async fn lock_wait_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = repo(dir.path(), None);
        r.lock_wait = Duration::from_millis(200);
        let _held = r.lock().await.expect("first lock");
        let started = Instant::now();
        assert!(r.lock().await.is_none(), "second waiter gives up");
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(200), "waited {waited:?}");
        assert!(waited < Duration::from_secs(5), "waited {waited:?}");
    }

    #[test]
    fn dangling_snapshot_entry_is_told_apart_from_a_hashless_path() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir.path().join("models--owner--name");
        let file = SnapshotDepth::File;
        let dangling = repo_dir.join("snapshots").join(HASH).join("gone.json");
        assert!(is_snapshot_path_unchecked(&repo_dir, &dangling, file));
        assert!(!is_snapshot_path(&repo_dir, &dangling, file));
        let hashless = repo_dir.join("snapshots/gone.json");
        assert!(!is_snapshot_path_unchecked(&repo_dir, &hashless, file));
    }

    #[tokio::test]
    async fn readers_and_downloads_exclude_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path(), None);
        let file = r.repo_dir.join("snapshots").join(HASH).join("model.onnx");

        // Two readers share the lock; a download cannot take it meanwhile.
        let first = lock_cached_repo_for_reading(&file).expect("shared lock");
        let second = lock_cached_repo_for_reading(&file).expect("shared again");
        let probe = File::options().write(true).open(&r.lock_path).unwrap();
        assert!(probe.try_lock().is_err(), "download blocked by readers");
        drop((first, second));
        assert!(
            probe.try_lock().is_ok(),
            "download proceeds once they finish"
        );
        probe.unlock().unwrap();

        // A model file outside any Hugging Face cache needs no lock.
        assert!(lock_cached_repo_for_reading(&dir.path().join("local.onnx")).is_none());
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
