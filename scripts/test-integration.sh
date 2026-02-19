#!/usr/bin/env bash
set -euo pipefail

# Default to running expensive tests when this script is used.
export EXPENSIVE_TESTS="${EXPENSIVE_TESTS:-1}"

echo "==========================================================="
echo "  uni-xervo integration test runner"
echo "==========================================================="

missing_keys=0
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Warning: OPENAI_API_KEY is not set."
  missing_keys=1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "Warning: GEMINI_API_KEY is not set."
  missing_keys=1
fi

if [[ "${missing_keys}" -eq 1 ]]; then
  echo
  echo "Remote provider tests may be skipped without API keys."
  echo "Set the keys before running full remote integration tests:"
  echo "  export OPENAI_API_KEY='sk-...'"
  echo "  export GEMINI_API_KEY='...'"
  echo
fi

echo "Running ignored integration tests with all provider features..."
if command -v cargo-nextest >/dev/null 2>&1; then
  echo "Using cargo-nextest."
  cargo nextest run --all-features --test real_providers_test --run-ignored all
else
  echo "cargo-nextest not found. Falling back to cargo test."
  cargo test --all-features --test real_providers_test -- --ignored
fi

echo
echo "Integration tests finished."
