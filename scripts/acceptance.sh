#!/usr/bin/env bash
# Deployment acceptance matrix for uni-xervo's local backends (#63).
#
#   scripts/acceptance.sh [cpu|coreml|cuda] [--only SUITE[,SUITE...]]
#
# Runs every opt-in real-model test suite for one platform target, one cargo
# invocation per suite, and writes a report of what ran, what passed and
# what each suite cost, plus a manifest of the model revisions the run used:
#
#   target/acceptance/<target>/report.md
#
# Targets:
#   cpu     default features (every build can run this)
#   coreml  local/onnx with `gpu-metal`, plus the CoreML inference tests (macOS)
#   cuda    local/onnx with `gpu-cuda`, plus the CUDA inference tests (needs nvcc)
#
# Needs network the first time (models download into the uni-xervo cache,
# `crates/uni-xervo/.uni_cache` unless UNI_CACHE_DIR is set) and no API keys:
# remote providers are not part of this matrix, see real_providers_test.
#
# Optional:
#   KNIV_V7_MODEL_DIR  kniv v7 release for the v7 suite. Without it the suite
#                      runs its model-agnostic tests on kniv-v7-tiny-en; the
#                      golden tests need an xlarge release (see that test).
#   ORT_DYLIB_PATH     an ONNX Runtime library, enabling the provider-onnx-
#                      dynamic execution-provider skip test.
#   ACCEPTANCE_SKIP    comma-separated suite names to leave out.
#
# Exit status is non-zero if any suite failed. A suite that cannot run on
# this machine (missing toolchain or library) is reported as skipped.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-cpu}"
shift || true
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$TARGET" in
  cpu) ACCEL_FEATURES="" ;;
  coreml) ACCEL_FEATURES="gpu-metal" ;;
  cuda) ACCEL_FEATURES="gpu-cuda" ;;
  *) echo "target must be cpu, coreml or cuda (got '$TARGET')" >&2; exit 2 ;;
esac

OUT="$ROOT/target/acceptance/$TARGET"
mkdir -p "$OUT"
REPORT="$OUT/report.md"
export EXPENSIVE_TESTS=1

# Peak RSS of a command and its children, in MiB, via /usr/bin/time when
# it exists (reported as "?" otherwise).
if /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -l)   # BSD / macOS: bytes
  TIME_FLAG="-l"
elif /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -v)   # GNU: kilobytes
  TIME_FLAG="-v"
else
  TIME_CMD=()
  TIME_FLAG=""
fi

peak_rss_mib() {
  local file="$1"
  if [[ "$TIME_FLAG" == "-l" ]]; then
    awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576}' "$file"
  elif [[ "$TIME_FLAG" == "-v" ]]; then
    awk -F: '/Maximum resident set size/ {printf "%.0f", $2 / 1024}' "$file"
  fi
}

# name | package | features | test target | args after `--` | requirement
#
# features: comma-separated; `+accel` adds the target's accelerator feature.
# requirement: `-` always, `env:VAR` when VAR is set, `target:NAME` only for
# that target.
SUITES=$(cat <<'EOF'
onnx-models|uni-xervo|provider-onnx,+accel|onnx_models_expensive_test|--ignored|-
onnx-hf-download|uni-xervo|provider-onnx,+accel|local_onnx_hf_e2e_test|--ignored|-
rerank|uni-xervo|provider-onnx,+accel|local_onnx_reranker_hf_e2e_test||-
sparse|uni-xervo|provider-onnx,+accel|local_onnx_embed_sparse_expensive_test|--ignored|-
multi-vector|uni-xervo|provider-onnx,+accel|local_onnx_embed_multivector_expensive_test|--ignored|-
hybrid-bge-m3|uni-xervo|provider-onnx,+accel|local_onnx_embed_hybrid_expensive_test|--ignored|-
ocr|uni-xervo|provider-onnx,+accel|local_onnx_ocr_detection_expensive_test|--ignored|-
pdf-to-ocr|uni-xervo-pdf||pdf_to_ocr_expensive_test|--ignored|-
nlp-kniv-v7|uni-xervo|provider-onnx,+accel|local_onnx_nlp_v7_golden_expensive_test|@KNIV|-
ep-reporting|uni-xervo|provider-onnx,+accel|local_onnx_ep_reporting_test||-
ep-skip-dynamic|uni-xervo|provider-onnx-dynamic|onnx_ep_attached_dynamic_test||env:ORT_DYLIB_PATH
coreml-inference|uni-xervo|provider-onnx,gpu-metal|gpu_metal_inference_test|--ignored|target:coreml
cuda-inference|uni-xervo|provider-onnx,gpu-cuda|gpu_cuda_inference_test|--ignored|target:cuda
whisper|uni-xervo|provider-whisper-cpp|local_whisper_cpp_test|--ignored|-
EOF
)

listed() { # listed NAME LIST
  [[ ",$2," == *",$1,"* ]]
}

git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
git_dirty=""
git diff --quiet 2>/dev/null || git_dirty=" (with uncommitted changes)"
ort_version="$(awk '/^name = "ort"$/ {getline; gsub(/version = |"/, ""); print; exit}' Cargo.lock)"
cache_root="${UNI_CACHE_DIR:-$ROOT/crates/uni-xervo/.uni_cache}"

rows=""
failed=0
while IFS='|' read -r name package features test args requirement; do
  [[ -z "$name" ]] && continue
  if [[ -n "$ONLY" ]] && ! listed "$name" "$ONLY"; then continue; fi
  if [[ -n "${ACCEPTANCE_SKIP:-}" ]] && listed "$name" "$ACCEPTANCE_SKIP"; then continue; fi

  skip_reason=""
  case "$requirement" in
    env:*) var="${requirement#env:}"; [[ -z "${!var:-}" ]] && skip_reason="$var not set" ;;
    target:*) [[ "${requirement#target:}" != "$TARGET" ]] && continue ;;
  esac

  feats="${features//+accel/$ACCEL_FEATURES}"
  feats="$(echo "$feats" | sed -E 's/,+/,/g; s/^,//; s/,$//')"
  feature_args=()
  if [[ -n "$feats" ]]; then
    if [[ "$feats" == *provider-* ]] || [[ "$feats" == *gpu-* ]]; then
      feature_args=(--no-default-features --features "$feats")
    fi
  fi
  # The PDF crate and default-feature suites keep their default features.
  [[ "$package" == "uni-xervo-pdf" ]] && feature_args=()

  test_args=(--nocapture)
  extra_env=()
  case "$args" in
    --ignored) test_args=(--ignored --nocapture) ;;
    @KNIV)
      if [[ -n "${KNIV_V7_MODEL_DIR:-}" ]]; then
        test_args=(--nocapture)
      else
        # Model-agnostic tests only; goldens need an xlarge release.
        extra_env=("KNIV_V7_MODEL_DIR=dragonscale-ai/kniv-v7-tiny-en")
        test_args=(--nocapture narrow_task reports_execution)
      fi
      ;;
  esac

  cmd=(cargo test --release -p "$package" ${feature_args[@]+"${feature_args[@]}"} --test "$test" -- ${test_args[@]+"${test_args[@]}"})
  pretty="${extra_env[*]+${extra_env[*]} }EXPENSIVE_TESTS=1 ${cmd[*]}"
  log="$OUT/$name.log"

  if [[ -n "$skip_reason" ]]; then
    echo "--- $name: skipped ($skip_reason)"
    rows+="| \`$name\` | skipped: $skip_reason | | | | |"$'\n'
    continue
  fi

  echo "--- $name: building"
  if ! env ${extra_env[@]+"${extra_env[@]}"} cargo test --release -p "$package" ${feature_args[@]+"${feature_args[@]}"} --test "$test" --no-run >"$log" 2>&1; then
    echo "    build failed (see $log)"
    rows+="| \`$name\` | skipped: does not build here (see \`$name.log\`) | | | | \`$pretty\` |"$'\n'
    continue
  fi

  echo "--- $name: running"
  start=$(date +%s)
  env ${extra_env[@]+"${extra_env[@]}"} ${TIME_CMD[@]+"${TIME_CMD[@]}"} "${cmd[@]}" >"$log" 2>"$log.time"
  status=$?
  secs=$(( $(date +%s) - start ))
  cat "$log.time" >>"$log"
  rss="$(peak_rss_mib "$log.time")"
  rm -f "$log.time"

  summary="$(grep -E '^test result:' "$log" | tail -1)"
  passed="$(echo "$summary" | sed -nE 's/.* ([0-9]+) passed.*/\1/p')"
  fails="$(echo "$summary" | sed -nE 's/.* ([0-9]+) failed.*/\1/p')"
  skipped="$(grep -cE '^Skipping' "$log" || true)"
  if [[ $status -eq 0 ]]; then
    result="pass"
  else
    result="**FAIL**"
    failed=1
  fi
  echo "    $result: ${passed:-0} passed, ${fails:-0} failed, ${skipped} skipped, ${secs}s, ${rss:-?} MiB"
  rows+="| \`$name\` | $result | ${passed:-0} / ${fails:-0} / ${skipped} | ${secs} s | ${rss:-?} MiB | \`$pretty\` |"$'\n'
done <<<"$SUITES"

# Models the run used: every Hugging Face repo in the cache, with the commit
# its `main` (or other) ref points at and the size of its snapshot.
manifest=""
if [[ -d "$cache_root" ]]; then
  while IFS= read -r ref; do
    repo_dir="$(dirname "$(dirname "$ref")")"
    repo="$(basename "$repo_dir" | sed -E 's/^models--//; s/--/\//')"
    rev="$(basename "$ref")"
    commit="$(tr -d '[:space:]' <"$ref")"
    size="$(du -sLh "$repo_dir/snapshots/$commit" 2>/dev/null | cut -f1)"
    manifest+="| \`$repo\` | \`$rev\` | \`$commit\` | ${size:-?} |"$'\n'
  done < <(find "$cache_root" -path '*/models--*/refs/*' -type f 2>/dev/null | sort)
fi

{
  echo "# Acceptance run: \`$TARGET\`"
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| Date | $(date -u +%Y-%m-%dT%H:%MZ) |"
  echo "| Commit | \`$git_rev\`$git_dirty |"
  echo "| Platform | $(uname -s) $(uname -m) |"
  echo "| Rust | $(rustc --version) |"
  echo "| ONNX Runtime binding | \`ort\` $ort_version |"
  echo "| Accelerator feature | ${ACCEL_FEATURES:-none} |"
  echo
  echo "## Suites"
  echo
  echo "Passed / failed / skipped counts tests; a test that returns early because"
  echo "its model or key is unavailable prints \`Skipping\` and counts as skipped."
  echo "Time and peak memory cover the test run (the build is done first, unmeasured)."
  echo
  echo "| Suite | Result | Passed / failed / skipped | Time | Peak RSS | Command |"
  echo "|---|---|---|---|---|---|"
  printf '%s' "$rows"
  echo
  echo "## Models"
  echo
  echo "Every Hugging Face repository in the cache (\`$cache_root\`), with the"
  echo "commit its ref resolved to. Pin a suite to these with each alias's"
  echo "\`revision\` to reproduce the run exactly."
  echo
  echo "| Repository | Ref | Commit | Snapshot size |"
  echo "|---|---|---|---|"
  printf '%s' "$manifest"
} >"$REPORT"

echo
echo "Report: $REPORT"
exit $failed
