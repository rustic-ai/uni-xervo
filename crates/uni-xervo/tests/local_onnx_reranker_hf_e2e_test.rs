#![cfg(feature = "provider-onnx")]
//! End-to-end test for the rerank task on `LocalOnnxProvider` against a real
//! HuggingFace cross-encoder model.
//!
//! Gated by `EXPENSIVE_TESTS=1` (matches `local_onnx_hf_e2e_test.rs`)
//! because it downloads ~90MB and runs ORT inference. Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features provider-onnx \
//!     --test local_onnx_reranker_hf_e2e_test
//! ```

#[path = "common/accelerator.rs"]
mod accelerator;

use std::env;
use std::sync::Arc;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

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

fn rerank_spec(alias: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

async fn runtime_with_minilm() -> Arc<ModelRuntime> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![rerank_spec(
            "rerank/minilm",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
        )])
        .build()
        .await
        .expect("runtime build failed")
}

#[tokio::test]
async fn rerank_orders_relevant_doc_first() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let runtime = runtime_with_minilm().await;
    let reranker = runtime
        .reranker("rerank/minilm")
        .await
        .expect("loading rerank/minilm failed");
    accelerator::expect_accelerator("rerank/minilm", &*reranker);

    let docs = vec![
        "Pandas eat bamboo and live in China.", // 0 — clearly relevant
        "The Eiffel Tower is in Paris.",        // 1 — irrelevant
        "Giant pandas have a black-and-white coat.", // 2 — relevant
        "Quantum entanglement was first discovered in the 1930s.", // 3 — irrelevant
    ];
    // Default spec (no `execution_providers` set) — request list should be
    // the feature-aware default: `[cpu]` on a CPU-only build, `[cuda, cpu]`
    // when the `gpu-cuda` hint is on. Either way, must be non-empty.
    let requested = reranker.active_execution_providers();
    assert!(
        !requested.is_empty(),
        "active_execution_providers() should return at least one EP after load"
    );
    assert!(
        requested.iter().any(|ep| ep == "cpu"),
        "default EP list should always include cpu, got {requested:?}"
    );

    let scored = reranker
        .rerank("Where do giant pandas live?", &docs)
        .await
        .expect("rerank call failed");

    assert_eq!(scored.len(), 4);

    // Sorted descending by score (the provider does this internally).
    for w in scored.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores not descending: {} then {}",
            w[0].score,
            w[1].score
        );
    }

    // The two clearly-relevant docs (0 and 2) should rank above the two
    // irrelevant ones (1 and 3).
    let top_two_indices: Vec<usize> = scored.iter().take(2).map(|s| s.index).collect();
    assert!(
        top_two_indices.contains(&0),
        "expected doc 0 in top 2, got {top_two_indices:?}"
    );
    assert!(
        top_two_indices.contains(&2),
        "expected doc 2 in top 2, got {top_two_indices:?}"
    );
}

#[tokio::test]
async fn rerank_empty_docs_returns_empty() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let runtime = runtime_with_minilm().await;
    let reranker = runtime
        .reranker("rerank/minilm")
        .await
        .expect("loading rerank/minilm failed");
    accelerator::expect_accelerator("rerank/minilm", &*reranker);

    let scored = reranker
        .rerank("anything", &[])
        .await
        .expect("rerank call failed");
    assert!(scored.is_empty());
}
