# CLI Reference

Uni-Xervo ships with `uni-prefetch` for pre-downloading local model artifacts referenced by a catalog.

## Command

```bash
uni-prefetch <catalog.json> [--cache-dir <path>] [--dry-run]
```

## What it does

- Loads a catalog JSON file.
- Splits aliases into local vs remote providers.
- Skips remote providers because they have no local artifact cache step.
- Registers available local providers compiled into the binary.
- Forces remaining local aliases to eager warmup and builds a runtime to cache them.

## Options

- `--cache-dir <path>`: override cache root (`UNI_CACHE_DIR` also supported).
- `--dry-run`: print what would be fetched without downloading.
- `--help`: print usage.

## Exit behavior

- Exits with code `0` on success.
- Exits non-zero and prints `error: ...` on argument, validation, provider registration, or load failures.

## Examples

```bash
# Show planned local downloads
uni-prefetch catalog.json --dry-run

# Pre-cache local models in a custom path
uni-prefetch catalog.json --cache-dir /opt/model-cache
```

## Notes

- Remote aliases (`remote/*`) are always skipped.
- If a local provider feature is not compiled in, that provider is warned and skipped.
