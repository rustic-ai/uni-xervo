#!/usr/bin/env bash
set -euo pipefail

echo "==> Formatting check"
cargo fmt --all -- --check

echo "==> Compile check"
cargo check --locked

echo "==> Running tests"
cargo test --locked
