# Release Guide

This repository uses tag-driven crate publishing.

## Release Checklist

1. Ensure local branch is clean and up to date with `main`.
2. Set release version in `Cargo.toml` (example: `0.1.0`).
3. Run local verification:
   - `./scripts/test.sh`
   - `cargo publish --locked --dry-run`
4. Update `CHANGELOG.md` with the new release entry and date.
5. Commit the release prep changes.
6. Create and push a matching tag:
   - `git tag -a v0.1.0 -m "uni-xervo 0.1.0"`
   - `git push origin v0.1.0`

## Notes

- If `cargo publish --dry-run` fails with "failed to retrieve git status ...
  Failed to update the excludes stack", a model cache sits inside a crate
  directory: running the real-model tests with plain `cargo test` caches into
  `crates/<crate>/.uni_cache`, and its Hugging Face snapshots hold
  `.gitattributes` as a symlink, which cargo's git status check rejects even
  in an ignored directory. Move the cache out (`mv crates/uni-xervo/.uni_cache
  .uni_cache`, then use `UNI_CACHE_DIR=$PWD/.uni_cache`) or delete it.
  `scripts/acceptance.sh` already caches at the repository root.
- `publish.yml` validates that the pushed tag matches `Cargo.toml` version.
- Publishing uses crates.io trusted publishing via GitHub OIDC (no long-lived token secret).
- Configure this repository as a trusted publisher on crates.io before tagging a release.
