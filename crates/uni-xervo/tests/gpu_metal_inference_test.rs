//! Real-inference tests that exercise the **CoreML** execution path for
//! embedding and reranking via `LocalOnnxProvider`.
//!
//! The Apple counterpart to `gpu_cuda_inference_test.rs`. Note the EP is
//! called *CoreML*, not "Metal": ONNX Runtime has no Metal execution
//! provider, and CoreML is how ORT reaches the Apple GPU and the Neural
//! Engine. The `gpu-metal` feature switches on `ort/coreml` alongside the
//! Metal kernels in candle and mistral.rs.
//!
//! All tests are gated by both:
//!
//! - `#[cfg(feature = "gpu-metal")]` — only built when the `gpu-metal`
//!   feature is enabled (avoids breaking CPU-only CI). Note `gpu-metal` is
//!   *not* a default feature, so an ordinary `cargo test` compiles this file
//!   out entirely.
//! - `EXPENSIVE_TESTS=1` env var (via `require_expensive_tests!`) — only
//!   actually run when explicitly requested, since each test downloads a
//!   real model from HuggingFace.
//!
//! # Environment requirements
//!
//! - An Apple-silicon or CoreML-capable macOS host. There is no equivalent
//!   of the CUDA driver/PTX mismatch here: CoreML ships with the OS.
//! - Unlike the CUDA tests, no special ORT distribution is needed — pyke's
//!   bundled binaries carry the CoreML EP when `ort/coreml` is enabled.
//! - CoreML silently declines operators it cannot handle, falling back to CPU
//!   per-op. These tests therefore assert that the session *builds and runs*
//!   with CoreML requested, not that every kernel executed on the ANE.
//!
//! mistral.rs generation on Metal is deliberately not covered here: it
//! auto-selects the device from the feature flag and exposes no
//! `execution_providers` knob, so there is nothing EP-specific to assert.
//!
//! Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features "gpu-metal,provider-onnx" \
//!     --test gpu_metal_inference_test \
//!     --run-ignored all
//! ```

#![cfg(feature = "gpu-metal")]

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

/// Build a spec that explicitly requests **CoreML-only** execution (no CPU
/// fallback entry). With `gpu-metal` enabled but CoreML unavailable at
/// runtime, this surfaces a hard error instead of silently falling back —
/// the same strictness contract `cuda_only_spec` relies on.
///
/// A list *with* a `cpu` entry would instead degrade quietly, which is the
/// behaviour covered by the unit tests in `provider::onnx_ep`.
fn coreml_only_spec(
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
            "execution_providers": ["coreml"],
        }),
    }
}

// ---------------------------------------------------------------------------
// Reranker on CoreML (LocalOnnxProvider rerank task)
// ---------------------------------------------------------------------------

#[cfg(feature = "provider-onnx")]
#[tokio::test]
#[ignore]
async fn test_local_onnx_reranker_runs_on_coreml() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![coreml_only_spec(
            "rerank/minilm-coreml",
            ModelTask::Rerank,
            "local/onnx",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
        )])
        .build()
        .await
        .expect("runtime build failed (is CoreML available?)");

    let reranker = runtime
        .reranker("rerank/minilm-coreml")
        .await
        .expect("loading rerank/minilm-coreml failed");

    assert!(
        reranker
            .active_execution_providers()
            .iter()
            .any(|ep| ep == "coreml"),
        "coreml was requested and gpu-metal is on, so it must survive EP filtering; got {:?}",
        reranker.active_execution_providers()
    );

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

    // Semantic relevance doesn't change between EPs, so the top-2 must hold
    // the two panda documents (indices 0 and 2) exactly as on CPU and CUDA.
    let top_two: Vec<usize> = scored.iter().take(2).map(|s| s.index).collect();
    assert!(
        top_two.contains(&0),
        "expected doc 0 in top 2, got {top_two:?}"
    );
    assert!(
        top_two.contains(&2),
        "expected doc 2 in top 2, got {top_two:?}"
    );

    println!("✓ reranker ran on CoreML");
}

// ---------------------------------------------------------------------------
// Dense embedding on CoreML (LocalOnnxProvider embed task)
// ---------------------------------------------------------------------------

#[cfg(feature = "provider-onnx")]
#[tokio::test]
#[ignore]
async fn test_local_onnx_embed_runs_on_coreml() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![coreml_only_spec(
            "embed/bge-small-coreml",
            ModelTask::Embed,
            "local/onnx",
            "BGESmallENV15",
        )])
        .build()
        .await
        .expect("runtime build failed (is CoreML available?)");

    let model = runtime
        .embedding("embed/bge-small-coreml")
        .await
        .expect("loading embed/bge-small-coreml failed");

    assert!(
        model
            .active_execution_providers()
            .iter()
            .any(|ep| ep == "coreml"),
        "coreml was requested and gpu-metal is on, so it must survive EP filtering; got {:?}",
        model.active_execution_providers()
    );

    let texts = vec!["CoreML embedding smoke test", "A second input"];
    let embeddings = model.embed(&texts).await.expect("embedding failed");

    assert_eq!(embeddings.vectors.len(), 2);
    for vector in &embeddings.vectors {
        assert_eq!(vector.len() as u32, model.dimensions());
        assert!(
            vector.iter().all(|v| v.is_finite()),
            "embedding contains non-finite values"
        );
        // The BGE presets normalize, so every vector should be unit-norm
        // regardless of which EP produced it.
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
    }

    println!(
        "✓ LocalOnnxProvider embedded on CoreML, dim {}",
        model.dimensions()
    );
}
