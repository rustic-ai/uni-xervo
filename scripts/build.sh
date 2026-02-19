#!/usr/bin/env bash
set -euo pipefail

# Default build is debug. Pass --release for optimized builds.
if [[ "${1:-}" == "--release" ]]; then
  shift
  cargo build --locked --release "$@"
else
  cargo build --locked "$@"
fi
