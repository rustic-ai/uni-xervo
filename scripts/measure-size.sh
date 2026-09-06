#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/uni-xervo-size-harness}"
BASELINE_DIR="$WORK_DIR/baseline-app"
HARNESS_DIR="$WORK_DIR/xervo-app"
CARGO_HOME_DIR="$WORK_DIR/cargo-home"

feature_label() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        printf '%s' "core"
    else
        printf '%s' "$raw"
    fi
}

feature_table() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        printf '%s' "(none)"
    else
        printf '%s' "$raw"
    fi
}

create_baseline() {
    mkdir -p "$BASELINE_DIR/src"
    cat >"$BASELINE_DIR/Cargo.toml" <<'EOF'
[package]
name = "ux-size-baseline"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
EOF

    cat >"$BASELINE_DIR/src/main.rs" <<'EOF'
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    Ok(())
}
EOF
}

create_harness() {
    mkdir -p "$HARNESS_DIR/src"
    cat >"$HARNESS_DIR/Cargo.toml" <<EOF
[package]
name = "ux-size-xervo"
version = "0.1.0"
edition = "2024"

[features]
default = []
provider-candle = ["uni-xervo/provider-candle"]
provider-onnx = ["uni-xervo/provider-onnx"]
provider-openai = ["uni-xervo/provider-openai"]
provider-llamacpp = ["uni-xervo/provider-llamacpp"]
provider-mistralrs = ["uni-xervo/provider-mistralrs"]
gpu-cuda = ["uni-xervo/gpu-cuda"]

[dependencies]
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
uni-xervo = { path = "$ROOT_DIR/crates/uni-xervo", default-features = false }
EOF

    cat >"$HARNESS_DIR/src/main.rs" <<'EOF'
use anyhow::Result;
use uni_xervo::api::ModelAliasSpec;
use uni_xervo::runtime::ModelRuntime;

#[tokio::main]
async fn main() -> Result<()> {
    let builder = ModelRuntime::builder();

    #[cfg(feature = "provider-candle")]
    let builder = builder.register_provider(uni_xervo::provider::candle::LocalCandleProvider::new());

    #[cfg(feature = "provider-onnx")]
    let builder = builder.register_provider(uni_xervo::provider::local_onnx::LocalOnnxProvider::new());

    #[cfg(feature = "provider-openai")]
    let builder = builder.register_provider(uni_xervo::provider::RemoteOpenAIProvider::new());

    #[cfg(feature = "provider-llamacpp")]
    let builder = builder.register_provider(uni_xervo::provider::RemoteLlamaCppProvider::new());

    #[cfg(feature = "provider-mistralrs")]
    let builder = builder.register_provider(uni_xervo::provider::mistralrs::LocalMistralRsProvider::new());

    let _ = builder.catalog(Vec::<ModelAliasSpec>::new());

    Ok(())
}
EOF
}

measure_binary() {
    local bin_path="$1"
    local tmp_copy="$2"

    cp "$bin_path" "$tmp_copy"
    if command -v strip >/dev/null 2>&1; then
        strip "$tmp_copy"
    fi
    stat -c '%s' "$tmp_copy"
}

run_build() {
    local manifest="$1"
    local target_dir="$2"
    shift 2
    CARGO_HOME="$CARGO_HOME_DIR" CARGO_TARGET_DIR="$target_dir" cargo build --release --manifest-path "$manifest" "$@" >/dev/null
}

create_baseline
create_harness
mkdir -p "$CARGO_HOME_DIR"

run_build "$BASELINE_DIR/Cargo.toml" "$WORK_DIR/target-baseline"
baseline_size="$(measure_binary "$WORK_DIR/target-baseline/release/ux-size-baseline" "$WORK_DIR/ux-size-baseline.stripped")"

declare -a feature_sets=(
    ""
    "provider-candle"
    "provider-onnx"
    "provider-openai"
    "provider-llamacpp"
    "provider-mistralrs"
)

printf '| Scenario | Features | Stripped release bytes | Delta vs baseline |\n'
printf '| --- | --- | ---: | ---: |\n'
printf '| baseline | n/a | %s | %s |\n' "$baseline_size" "0"

for feature_set in "${feature_sets[@]}"; do
    label="$(feature_label "$feature_set")"
    target_dir="$WORK_DIR/target-${label}"

    if [[ -z "$feature_set" ]]; then
        run_build "$HARNESS_DIR/Cargo.toml" "$target_dir" --no-default-features
    else
        run_build "$HARNESS_DIR/Cargo.toml" "$target_dir" --no-default-features --features "$feature_set"
    fi

    bin_size="$(measure_binary "$target_dir/release/ux-size-xervo" "$WORK_DIR/ux-size-xervo-${label}.stripped")"
    delta="$((bin_size - baseline_size))"

    printf '| %s | %s | %s | %s |\n' \
        "$label" \
        "$(feature_table "$feature_set")" \
        "$bin_size" \
        "$delta"
done
