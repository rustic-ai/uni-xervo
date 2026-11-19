//! Real-model test for the `embed_multi_vector` task on `local/onnx`.
//!
//! Downloads `answerdotai/answerai-colbert-small-v1` (~130 MB) and asserts the
//! per-token output shape, dimensionality, and that late-interaction MaxSim
//! ranks a relevant document above an irrelevant one.
//!
//! Gated twice so it never runs in normal CI:
//! - `#![cfg(feature = "provider-onnx")]`
//! - the `EXPENSIVE_TESTS=1` env var (plus `#[ignore]`).
//!
//! Run with (`--test-threads=1` avoids a HF download-lock race when both
//! tests fetch the same model concurrently):
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test -p uni-xervo --features provider-onnx \
//!     --test local_onnx_embed_multivector_expensive_test -- --ignored --test-threads=1
//! ```

#![cfg(feature = "provider-onnx")]

#[path = "common/accelerator.rs"]
mod accelerator;

use std::env;
use std::sync::Arc;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::score::max_sim;

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

async fn colbert_runtime() -> Arc<ModelRuntime> {
    let spec = ModelAliasSpec {
        alias: "embed_mv/colbert".to_string(),
        task: ModelTask::EmbedMultiVector,
        provider_id: "local/onnx".to_string(),
        model_id: "answerdotai/answerai-colbert-small-v1".to_string(),
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
async fn colbert_emits_per_token_vectors() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let runtime = colbert_runtime().await;
    let model = runtime
        .multi_vector_embedder("embed_mv/colbert")
        .await
        .expect("load multi-vector model");
    accelerator::expect_accelerator("embed_mv/colbert", &*model);

    let result = model.embed(&["how do tides work"]).await.expect("embed");

    assert_eq!(result.vectors.len(), 1);
    let tokens = &result.vectors[0];
    // More than one token vector, each of the configured dimension (96).
    assert!(tokens.len() > 1, "expected several token vectors");
    for tok in tokens {
        assert_eq!(tok.len() as u32, model.dimensions());
        // Vectors are L2-normalized by default: norm ~= 1.
        let norm = tok.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-2,
            "token vector should be unit norm, got {norm}"
        );
    }
}

#[tokio::test]
#[ignore]
async fn colbert_maxsim_ranks_relevant_above_irrelevant() {
    accelerator::log_warnings();
    require_expensive_tests!();

    let runtime = colbert_runtime().await;
    let model = runtime
        .multi_vector_embedder("embed_mv/colbert")
        .await
        .expect("load");
    accelerator::expect_accelerator("embed_mv/colbert", &*model);

    let result = model
        .embed(&[
            "what causes ocean tides",
            "ocean tides are caused by the gravitational pull of the moon and sun",
            "sourdough bread is leavened by a fermented flour and water starter",
        ])
        .await
        .expect("embed");

    let query = &result.vectors[0];
    let relevant = &result.vectors[1];
    let irrelevant = &result.vectors[2];

    let s_rel = max_sim(query, relevant);
    let s_irr = max_sim(query, irrelevant);
    assert!(
        s_rel > s_irr,
        "relevant doc should score higher: rel={s_rel} irr={s_irr}"
    );
}

// ── Additional preset coverage (GTE-ModernColBERT, mxbai, BGE-M3) ────────
//
// These exercise the in-graph-projection assumption for each export. If a
// model's `model.onnx` emits un-projected hidden states instead of ColBERT
// vectors, the load/embed fails loudly with a dimension mismatch — a real
// finding, not a silent wrong answer.

async fn mv_runtime_for(alias: &str, model_id: &str) -> Arc<ModelRuntime> {
    let spec = ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::EmbedMultiVector,
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

/// Embed a query + relevant + irrelevant doc; assert per-token dim and that
/// MaxSim ranks the relevant doc higher.
async fn assert_colbert_model(alias: &str, model_id: &str, expected_dim: u32) {
    let runtime = mv_runtime_for(alias, model_id).await;
    let model = runtime
        .multi_vector_embedder(alias)
        .await
        .unwrap_or_else(|e| panic!("load {model_id}: {e:?}"));
    accelerator::expect_accelerator(alias, &*model);
    assert_eq!(model.dimensions(), expected_dim);

    let result = model
        .embed(&[
            "what causes ocean tides",
            "ocean tides are caused by the gravitational pull of the moon and sun",
            "sourdough bread is leavened by a fermented flour and water starter",
        ])
        .await
        .unwrap_or_else(|e| panic!("embed {model_id}: {e:?}"));

    for tokens in &result.vectors {
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0].len() as u32, expected_dim, "per-token dim");
    }
    let s_rel = max_sim(&result.vectors[0], &result.vectors[1]);
    let s_irr = max_sim(&result.vectors[0], &result.vectors[2]);
    assert!(
        s_rel > s_irr,
        "{model_id}: relevant should rank higher (rel={s_rel} irr={s_irr})"
    );
}

#[tokio::test]
#[ignore]
async fn gte_moderncolbert_works() {
    accelerator::log_warnings();
    require_expensive_tests!();
    assert_colbert_model("embed_mv/gte", "lightonai/GTE-ModernColBERT-v1", 128).await;
}

#[tokio::test]
#[ignore]
async fn mxbai_edge_colbert_17m_works() {
    accelerator::log_warnings();
    require_expensive_tests!();
    assert_colbert_model(
        "embed_mv/mxbai17",
        "mixedbread-ai/mxbai-edge-colbert-v0-17m",
        48,
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn mxbai_edge_colbert_32m_works() {
    accelerator::log_warnings();
    require_expensive_tests!();
    assert_colbert_model(
        "embed_mv/mxbai32",
        "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        64,
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn bge_m3_colbert_works() {
    accelerator::log_warnings();
    require_expensive_tests!();
    assert_colbert_model("embed_mv/bgem3", "aapot/bge-m3-onnx", 1024).await;
}
