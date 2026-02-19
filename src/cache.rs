//! Model and weight cache directory resolution.
//!
//! Local providers download model weights to a per-provider, per-model directory.
//! This module determines where that directory lives based on (in priority order):
//!
//! 1. A per-model `cache_dir` option in the spec's JSON options.
//! 2. The `UNI_CACHE_DIR` environment variable (global root override).
//! 3. A default `.uni_cache/` directory relative to the working directory.

use serde_json::Value;
use std::path::PathBuf;

/// Replace `/` with `--` and strip characters that are unsafe in directory names.
pub fn sanitize_model_name(model_id: &str) -> String {
    model_id
        .replace('/', "--")
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect()
}

/// The environment variable used to override the root cache directory.
pub const CACHE_ROOT_ENV: &str = "UNI_CACHE_DIR";

/// Default root cache directory name (relative to CWD).
const DEFAULT_CACHE_ROOT: &str = ".uni_cache";

/// Return the cache root directory, respecting the `UNI_CACHE_DIR` env var.
fn cache_root() -> PathBuf {
    std::env::var(CACHE_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CACHE_ROOT))
}

/// Resolve the root cache directory for a provider (no model sub-directory).
///
/// Used when setting a process-global cache env var (e.g. `HF_HOME` for mistralrs)
/// before the first model load.
///
/// Priority:
/// 1. `UNI_CACHE_DIR` env var -- resolves to `$UNI_CACHE_DIR/<provider>`
/// 2. `.uni_cache/<provider>` -- default
pub fn resolve_provider_cache_root(provider: &str) -> PathBuf {
    cache_root().join(provider)
}

/// Resolve the cache directory for a given provider and model.
///
/// Priority (highest first):
/// 1. `options["cache_dir"]` -- per-model override
/// 2. `UNI_CACHE_DIR` env var -- global root override; resolves to `$UNI_CACHE_DIR/<provider>/<model>`
/// 3. `.uni_cache/<provider>/<model>` -- default
pub fn resolve_cache_dir(provider: &str, model_id: &str, options: &Value) -> PathBuf {
    if let Some(dir) = options.get("cache_dir").and_then(|v| v.as_str()) {
        return PathBuf::from(dir);
    }
    cache_root()
        .join(provider)
        .join(sanitize_model_name(model_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Serialise all tests that read or write UNI_CACHE_DIR to avoid races
    // between parallel test threads (env vars are process-global).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn sanitize_slash_replaced_with_double_dash() {
        assert_eq!(
            sanitize_model_name("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers--all-MiniLM-L6-v2"
        );
    }

    #[test]
    fn sanitize_strips_unsafe_chars() {
        assert_eq!(sanitize_model_name("foo:bar@baz"), "foobarbaz");
    }

    #[test]
    fn sanitize_keeps_safe_chars() {
        assert_eq!(
            sanitize_model_name("BAAI--bge-small-en-v1.5"),
            "BAAI--bge-small-en-v1.5"
        );
    }

    #[test]
    fn resolve_default_path() {
        let _lock = ENV_LOCK.lock().unwrap();
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var(CACHE_ROOT_ENV) };
        let path = resolve_cache_dir("fastembed", "BAAI/bge-small-en-v1.5", &json!({}));
        assert_eq!(
            path,
            PathBuf::from(".uni_cache/fastembed/BAAI--bge-small-en-v1.5")
        );
    }

    #[test]
    fn resolve_env_var_root() {
        let _lock = ENV_LOCK.lock().unwrap();
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var(CACHE_ROOT_ENV, "/data/models") };
        let path = resolve_cache_dir("fastembed", "BAAI/bge-small-en-v1.5", &json!({}));
        unsafe { std::env::remove_var(CACHE_ROOT_ENV) };
        assert_eq!(
            path,
            PathBuf::from("/data/models/fastembed/BAAI--bge-small-en-v1.5")
        );
    }

    #[test]
    fn resolve_options_cache_dir_takes_priority_over_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::set_var(CACHE_ROOT_ENV, "/data/models") };
        let opts = json!({ "cache_dir": "/tmp/my_cache" });
        let path = resolve_cache_dir("fastembed", "some-model", &opts);
        unsafe { std::env::remove_var(CACHE_ROOT_ENV) };
        assert_eq!(path, PathBuf::from("/tmp/my_cache"));
    }

    #[test]
    fn resolve_user_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var(CACHE_ROOT_ENV) };
        let opts = json!({ "cache_dir": "/tmp/my_cache" });
        let path = resolve_cache_dir("fastembed", "some-model", &opts);
        assert_eq!(path, PathBuf::from("/tmp/my_cache"));
    }

    #[test]
    fn resolve_candle_path() {
        let _lock = ENV_LOCK.lock().unwrap();
        // SAFETY: protected by ENV_LOCK
        unsafe { std::env::remove_var(CACHE_ROOT_ENV) };
        let path = resolve_cache_dir(
            "candle",
            "sentence-transformers/all-MiniLM-L6-v2",
            &json!({}),
        );
        assert_eq!(
            path,
            PathBuf::from(".uni_cache/candle/sentence-transformers--all-MiniLM-L6-v2")
        );
    }
}
