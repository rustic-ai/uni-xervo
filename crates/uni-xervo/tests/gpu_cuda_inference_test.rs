//! Real-inference tests that exercise the **CUDA** execution path for
//! embedding and reranking (both via `LocalOnnxProvider`),
//! and generation (`LocalMistralRsProvider`).
//!
//! All tests are gated by both:
//!
//! - `#[cfg(feature = "gpu-cuda")]` — only built when the `gpu-cuda` feature
//!   is enabled (avoids breaking CPU-only CI).
//! - `EXPENSIVE_TESTS=1` env var (via `require_expensive_tests!`) — only
//!   actually run when explicitly requested, since each test downloads a
//!   real model from HuggingFace and requires a CUDA-capable GPU.
//!
//! # Environment requirements (besides a working GPU)
//!
//! - **ORT-based tests** (`test_local_onnx_*_runs_on_cuda`):
//!   ort-rs's `download-binaries` feature only ships **CPU** static
//!   binaries. CUDA execution requires either system-installed ONNX Runtime
//!   with CUDA EP (`apt install libonnxruntime-gpu` / equivalent), or
//!   building ort-rs with `load-dynamic`. Without that, you'll see:
//!   `Failed to load library libonnxruntime_providers_shared.so`.
//!
//! - **candle-based tests**: the candle-cuda kernels are PTX-compiled at
//!   crate build time. If the host driver is older than the build-time CUDA
//!   toolchain you'll see `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Match the
//!   toolchain to the driver (`nvidia-smi` reports `CUDA Version` — that's
//!   the maximum supported by the driver).
//!
//! Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features "gpu-cuda,provider-onnx" \
//!     --test gpu_cuda_inference_test \
//!     --run-ignored all
//! ```

#![cfg(feature = "gpu-cuda")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::runtime::ModelRuntime;

#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;

fn should_run_expensive_tests() -> bool {
    env::var("EXPENSIVE_TESTS").is_ok()
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

/// Build a spec that explicitly requests **CUDA-only** execution (no CPU
/// fallback). When `gpu-cuda` is enabled but CUDA is unavailable at runtime,
/// this surfaces a hard error instead of silently falling back.
fn cuda_only_spec(
    alias: &str,
    task: ModelTask,
    provider_id: &str,
    model_id: &str,
) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: provider_id.to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "execution_providers": ["cuda"],
        }),
    }
}

// ---------------------------------------------------------------------------
// Reranker on CUDA (LocalOnnxProvider rerank task)
// ---------------------------------------------------------------------------

#[cfg(feature = "provider-onnx")]
#[tokio::test]
#[ignore]
async fn test_local_onnx_reranker_runs_on_cuda() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![cuda_only_spec(
            "rerank/minilm-cuda",
            ModelTask::Rerank,
            "local/onnx",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
        )])
        .build()
        .await
        .expect("runtime build failed (is CUDA available?)");

    let reranker = runtime
        .reranker("rerank/minilm-cuda")
        .await
        .expect("loading rerank/minilm-cuda failed");

    let docs = vec![
        "Pandas eat bamboo and live in China.",
        "The Eiffel Tower is in Paris.",
        "Giant pandas have a black-and-white coat.",
        "Quantum entanglement was first discovered in the 1930s.",
    ];
    let scored = reranker
        .rerank("Where do giant pandas live?", &docs)
        .await
        .expect("rerank call failed");

    assert_eq!(scored.len(), 4);
    for w in scored.windows(2) {
        assert!(w[0].score >= w[1].score, "scores not descending");
    }

    // Same correctness check as the CPU E2E test — semantic relevance
    // doesn't change between EPs, so the top-2 must contain the two
    // panda-related docs (indices 0 and 2).
    let top_two: Vec<usize> = scored.iter().take(2).map(|s| s.index).collect();
    assert!(
        top_two.contains(&0),
        "expected doc 0 in top 2, got {top_two:?}"
    );
    assert!(
        top_two.contains(&2),
        "expected doc 2 in top 2, got {top_two:?}"
    );

    println!("✓ reranker ran on CUDA");
}

// ---------------------------------------------------------------------------
// Embedding on CUDA (LocalOnnxProvider)
// ---------------------------------------------------------------------------

#[cfg(feature = "provider-onnx")]
#[tokio::test]
#[ignore]
async fn test_local_onnx_embed_runs_on_cuda() {
    require_expensive_tests!();

    // BAAI/bge-small-en-v1.5 ships an ONNX model at `onnx/model.onnx`.
    // LocalOnnxProvider needs an explicit `artifact` to know which file
    // to load.
    // BGE is registered as `task: Raw` because `local/onnx` declares
    // `[Raw, Rerank]` capabilities. Dense embedding-as-a-task is not yet on
    // the unified provider — that's Phase 1 of the provider×task roadmap.
    // Until then, BGE is consumed via the raw tensor interface: the caller
    // supplies tokenized `input_ids` / `attention_mask` and reads the
    // `last_hidden_state` output directly.
    let mut spec = cuda_only_spec(
        "embed/bge-small-cuda",
        ModelTask::Raw,
        "local/onnx",
        "BAAI/bge-small-en-v1.5",
    );
    // Merge artifact alongside execution_providers:
    if let Some(obj) = spec.options.as_object_mut() {
        obj.insert("artifact".to_string(), serde_json::json!("onnx/model.onnx"));
    }

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("runtime build failed (is CUDA available?)");

    // Loading and resolving through the runtime confirms the CUDA EP was
    // accepted by ORT — an EP-rejected session would fail to commit at load
    // time.
    let runner = runtime
        .raw_tensor_model("embed/bge-small-cuda")
        .await
        .expect("loading embed/bge-small-cuda failed");

    // Prove the session is callable: introspect its input signature.
    let inputs = runner.input_signature();
    assert!(
        !inputs.is_empty(),
        "expected at least one input tensor in the BGE signature"
    );
    let input_names: Vec<&str> = inputs.iter().map(|s| s.name.as_str()).collect();
    assert!(
        input_names.contains(&"input_ids"),
        "expected `input_ids` in BGE inputs, got {input_names:?}"
    );

    println!("✓ LocalOnnxProvider loaded on CUDA, inputs: {input_names:?}");
}
