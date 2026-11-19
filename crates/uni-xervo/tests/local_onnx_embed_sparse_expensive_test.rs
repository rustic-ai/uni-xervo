//! Real-model test for the `embed_sparse` task on `local/onnx`.
//!
//! Downloads SPLADE++ (`prithivida/Splade_PP_en_v1`, ~500 MB) from HuggingFace
//! and asserts the sparse output is well-formed and lexically sensible.
//!
//! Gated twice so it never runs in normal CI:
//! - `#![cfg(feature = "provider-onnx")]`
//! - the `EXPENSIVE_TESTS=1` env var (plus `#[ignore]`).
//!
//! Run with (`--test-threads=1` avoids a HF download-lock race when both
//! tests fetch the same model concurrently):
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test -p uni-xervo --features provider-onnx \
//!     --test local_onnx_embed_sparse_expensive_test -- --ignored --test-threads=1
//! ```

#![cfg(feature = "provider-onnx")]

#[path = "common/accelerator.rs"]
mod accelerator;

use std::env;
use std::sync::Arc;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn should_run_expensive_tests() -> bool {
    match env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping expensive test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

async fn splade_runtime() -> Arc<ModelRuntime> {
    let spec = ModelAliasSpec {
        alias: "embed_sparse/splade".to_string(),
        task: ModelTask::EmbedSparse,
        provider_id: "local/onnx".to_string(),
        model_id: "prithivida/Splade_PP_en_v1".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("build runtime")
}

#[tokio::test]
#[ignore]
async fn splade_produces_nonzero_sparse_terms() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let runtime = splade_runtime().await;
    let model = runtime
        .sparse_embedder("embed_sparse/splade")
        .await
        .expect("load sparse model");
    accelerator::expect_accelerator("embed_sparse/splade", &*model);

    let result = model
        .embed(&["how to bake sourdough bread", "the moon orbits the earth"])
        .await
        .expect("embed");

    assert_eq!(result.vectors.len(), 2);
    for vector in &result.vectors {
        // SPLADE always activates several vocab terms (incl. expansion).
        assert!(
            vector.len() > 3,
            "expected multiple non-zero terms, got {}",
            vector.len()
        );
        // All weights are positive (post log(1+ReLU)) and term ids in range.
        for &(term, weight) in vector {
            assert!(weight > 0.0, "weight must be positive, got {weight}");
            assert!(term < model.vocab_size(), "term id out of vocab range");
        }
    }
}

#[tokio::test]
#[ignore]
async fn splade_top_k_caps_term_count() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let spec = ModelAliasSpec {
        alias: "embed_sparse/splade-topk".to_string(),
        task: ModelTask::EmbedSparse,
        provider_id: "local/onnx".to_string(),
        model_id: "prithivida/Splade_PP_en_v1".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "top_k": 8 }),
    };
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("build runtime");

    let model = runtime
        .sparse_embedder("embed_sparse/splade-topk")
        .await
        .expect("load");
    accelerator::expect_accelerator("embed_sparse/splade-topk", &*model);
    let result = model
        .embed(&["a fairly long sentence with many distinct content words"])
        .await
        .expect("embed");

    assert!(result.vectors[0].len() <= 8, "top_k=8 must cap term count");
}

// ── Additional preset coverage (Splade v2, BGE-M3 sparse) ───────────────

async fn sparse_runtime_for(alias: &str, model_id: &str) -> Arc<ModelRuntime> {
    let spec = ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::EmbedSparse,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("build runtime")
}

#[tokio::test]
#[ignore]
async fn splade_v2_produces_nonzero_sparse_terms() {
    accelerator::log_warnings();
    require_expensive_tests!();
    let runtime = sparse_runtime_for("embed_sparse/splade2", "prithivida/Splade_PP_en_v2").await;
    let model = runtime
        .sparse_embedder("embed_sparse/splade2")
        .await
        .expect("load SPLADE++ v2");
    accelerator::expect_accelerator("embed_sparse/splade2", &*model);
    let result = model
        .embed(&["how do tides work", "the capital of france is paris"])
        .await
        .expect("embed");
    assert_eq!(result.vectors.len(), 2);
    for vector in &result.vectors {
        assert!(
            vector.len() > 3,
            "expected several terms, got {}",
            vector.len()
        );
        for &(term, weight) in vector {
            assert!(weight > 0.0);
            assert!(term < model.vocab_size());
        }
    }
}

#[tokio::test]
#[ignore]
async fn bge_m3_sparse_produces_lexical_terms() {
    accelerator::log_warnings();
    require_expensive_tests!();
    let runtime = sparse_runtime_for("embed_sparse/bgem3", "aapot/bge-m3-onnx").await;
    let model = runtime
        .sparse_embedder("embed_sparse/bgem3")
        .await
        .expect("load BGE-M3 sparse");
    accelerator::expect_accelerator("embed_sparse/bgem3", &*model);
    let result = model
        .embed(&["climate change and renewable energy"])
        .await
        .expect("embed");
    assert_eq!(result.vectors.len(), 1);
    // Lexical head: keys are input token ids; expect a handful of weighted terms.
    let v = &result.vectors[0];
    assert!(!v.is_empty(), "expected at least one lexical term");
    for &(_, weight) in v {
        assert!(weight > 0.0, "lexical weights are post-ReLU positive");
    }
}
