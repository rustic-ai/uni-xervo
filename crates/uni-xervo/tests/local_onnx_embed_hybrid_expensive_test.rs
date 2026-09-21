//! End-to-end test for the single-pass hybrid embedder against the live
//! `aapot/bge-m3-onnx` multi-output export.
//!
//! Gated on `EXPENSIVE_TESTS=1` (downloads ~2.1 GB of BGE-M3 weights, shared
//! with the per-task BGE-M3 expensive tests) and `#[ignore]` so it never runs
//! in the default suite. Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test -p uni-xervo --features provider-onnx \
//!   --test local_onnx_embed_hybrid_expensive_test -- --ignored --test-threads=1
//! ```
//!
//! The headline assertion is **parity**: the fused hybrid output must equal the
//! output of the three trusted per-task handles (dense / sparse / multi-vector)
//! on identical inputs. Shape checks alone would pass even if the fusion read
//! the wrong head — parity is what proves correctness.

#![cfg(feature = "provider-onnx")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{HeadSet, SparseVector};

fn should_run_expensive_tests() -> bool {
    match env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

/// Build a `local/onnx` spec for `task` pointing at a BGE-M3 preset alias.
fn spec(alias: &str, task: ModelTask, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    }
}

/// The three short sentences used across every assertion in this file.
const TEXTS: [&str; 3] = [
    "a cat sat on the warm windowsill",
    "the kitten napped by the sunny window",
    "quarterly bond yields fell after the rate decision",
];

const FLOAT_TOL: f32 = 1e-4;

fn assert_dense_eq(hybrid: &[Vec<f32>], per_task: &[Vec<f32>]) {
    assert_eq!(hybrid.len(), per_task.len(), "dense row count");
    for (i, (h, p)) in hybrid.iter().zip(per_task).enumerate() {
        assert_eq!(h.len(), p.len(), "dense row {i} width");
        for (j, (a, b)) in h.iter().zip(p).enumerate() {
            assert!(
                (a - b).abs() < FLOAT_TOL,
                "dense[{i}][{j}]: hybrid {a} vs per-task {b}"
            );
        }
    }
}

fn assert_sparse_eq(hybrid: &[SparseVector], per_task: &[SparseVector]) {
    assert_eq!(hybrid.len(), per_task.len(), "sparse row count");
    // Lexical sparse vectors are an unordered (term_id, weight) set — the
    // post-processor collects from a HashMap, so iteration order is not stable.
    // Compare as sets by sorting on term id first.
    let sorted = |v: &SparseVector| {
        let mut s = v.clone();
        s.sort_by_key(|&(id, _)| id);
        s
    };
    for (i, (h, p)) in hybrid.iter().zip(per_task).enumerate() {
        let (h, p) = (sorted(h), sorted(p));
        assert_eq!(h.len(), p.len(), "sparse row {i} term count");
        for ((ht, hw), (pt, pw)) in h.iter().zip(&p) {
            assert_eq!(ht, pt, "sparse row {i}: term id");
            assert!(
                (hw - pw).abs() < FLOAT_TOL,
                "sparse row {i} term {ht}: hybrid {hw} vs per-task {pw}"
            );
        }
    }
}

fn assert_multi_vector_eq(hybrid: &[Vec<Vec<f32>>], per_task: &[Vec<Vec<f32>>]) {
    assert_eq!(hybrid.len(), per_task.len(), "multi-vector row count");
    for (i, (h, p)) in hybrid.iter().zip(per_task).enumerate() {
        assert_eq!(h.len(), p.len(), "multi-vector row {i} token count");
        for (t, (ht, pt)) in h.iter().zip(p).enumerate() {
            assert_eq!(ht.len(), pt.len(), "multi-vector row {i} token {t} width");
            for (a, b) in ht.iter().zip(pt) {
                assert!(
                    (a - b).abs() < FLOAT_TOL,
                    "multi-vector[{i}][{t}]: hybrid {a} vs per-task {b}"
                );
            }
        }
    }
}

/// Parity: one hybrid forward pass reproduces all three per-task paths exactly,
/// plus shape sanity on the fused output.
#[tokio::test]
#[ignore]
async fn hybrid_matches_per_task_paths() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![
            spec("hybrid/bgem3", ModelTask::EmbedHybrid, "BGEM3Hybrid"),
            spec("dense/bgem3", ModelTask::Embed, "BGEM3Dense"),
            spec("sparse/bgem3", ModelTask::EmbedSparse, "BGEM3Sparse"),
            spec("colbert/bgem3", ModelTask::EmbedMultiVector, "BGEM3Colbert"),
        ])
        .build()
        .await
        .expect("runtime build failed");

    let texts: Vec<&str> = TEXTS.to_vec();

    // The single fused pass.
    let hybrid = runtime
        .hybrid_embedder("hybrid/bgem3")
        .await
        .expect("resolve hybrid embedder");
    assert_eq!(
        hybrid.available_heads(),
        HeadSet::ALL,
        "BGE-M3 exposes all heads"
    );

    let fused = hybrid
        .embed(&texts, HeadSet::ALL)
        .await
        .expect("hybrid embed failed");

    let dense = fused.dense.as_ref().expect("dense head present");
    let sparse = fused.sparse.as_ref().expect("sparse head present");
    let multi_vector = fused
        .multi_vector
        .as_ref()
        .expect("multi-vector head present");

    // Shape sanity.
    assert_eq!(dense.len(), 3);
    for (i, row) in dense.iter().enumerate() {
        assert_eq!(row.len(), 1024, "dense row {i} is 1024-d");
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-2,
            "dense row {i} unit norm, got {norm}"
        );
    }
    for (i, row) in sparse.iter().enumerate() {
        assert!(!row.is_empty(), "sparse row {i} has terms");
        assert!(
            row.iter().all(|&(_, w)| w > 0.0),
            "sparse row {i} positive weights"
        );
    }
    for (i, doc) in multi_vector.iter().enumerate() {
        assert!(!doc.is_empty(), "multi-vector row {i} has tokens");
        assert!(
            doc.iter().all(|tok| tok.len() == 1024),
            "row {i} per-token 1024-d"
        );
    }

    // Parity against the trusted per-task handles on identical inputs.
    let dense_ref = runtime
        .embedding("dense/bgem3")
        .await
        .expect("resolve dense")
        .embed(&texts)
        .await
        .expect("dense embed");
    assert_dense_eq(dense, &dense_ref.vectors);

    let sparse_ref = runtime
        .sparse_embedder("sparse/bgem3")
        .await
        .expect("resolve sparse")
        .embed(&texts)
        .await
        .expect("sparse embed");
    assert_sparse_eq(sparse, &sparse_ref.vectors);

    let mv_ref = runtime
        .multi_vector_embedder("colbert/bgem3")
        .await
        .expect("resolve multi-vector")
        .embed(&texts)
        .await
        .expect("multi-vector embed");
    assert_multi_vector_eq(multi_vector, &mv_ref.vectors);
}

/// Selection semantics: only requested heads are materialized; an empty batch
/// still yields `Some(empty)` for each requested-and-available head.
#[tokio::test]
#[ignore]
async fn hybrid_respects_head_selection() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec(
            "hybrid/bgem3",
            ModelTask::EmbedHybrid,
            "BGEM3Hybrid",
        )])
        .build()
        .await
        .expect("runtime build failed");

    let hybrid = runtime
        .hybrid_embedder("hybrid/bgem3")
        .await
        .expect("resolve hybrid embedder");

    // Subset request: multi-vector is left out.
    let texts: Vec<&str> = TEXTS.to_vec();
    let subset = hybrid
        .embed(&texts, HeadSet::DENSE | HeadSet::SPARSE)
        .await
        .expect("subset embed failed");
    assert!(subset.dense.is_some(), "dense requested");
    assert!(subset.sparse.is_some(), "sparse requested");
    assert!(subset.multi_vector.is_none(), "multi-vector not requested");

    // Empty batch: each requested-and-available head is Some(empty), not None.
    let empty = hybrid
        .embed(&[], HeadSet::ALL)
        .await
        .expect("empty embed failed");
    assert_eq!(empty.dense.as_deref(), Some(&[][..]));
    assert_eq!(empty.sparse.as_deref(), Some(&[][..]));
    assert_eq!(empty.multi_vector.as_deref(), Some(&[][..]));
}

/// A single `EmbedHybrid` alias must satisfy the single-head accessors
/// (issue #49). Before the fix these returned `ProviderCapabilityMissing`, so
/// a hybrid retrieval pipeline had to register three aliases for one graph —
/// and any caller that didn't hit the failure only at *query* time.
///
/// Parity with the per-task handles is what proves the adapter reads the right
/// head; shape checks alone would pass on a mis-wired one.
#[tokio::test]
#[ignore]
async fn hybrid_alias_serves_single_head_accessors() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![
            // One alias, as the issue's repro registers it.
            spec("embed/hybrid", ModelTask::EmbedHybrid, "BGEM3Hybrid"),
            // Per-task handles to compare against.
            spec("dense/bgem3", ModelTask::Embed, "BGEM3Dense"),
            spec("sparse/bgem3", ModelTask::EmbedSparse, "BGEM3Sparse"),
            spec("colbert/bgem3", ModelTask::EmbedMultiVector, "BGEM3Colbert"),
        ])
        .build()
        .await
        .expect("runtime build failed");

    let texts: Vec<&str> = TEXTS.to_vec();

    // --- sparse, resolved from the hybrid alias ---------------------------
    let sparse = runtime
        .sparse_embedder("embed/hybrid")
        .await
        .expect("hybrid alias must serve a sparse query");
    assert_eq!(
        sparse.vocab_size(),
        runtime
            .sparse_embedder("sparse/bgem3")
            .await
            .unwrap()
            .vocab_size(),
        "vocab size must match the per-task sparse handle"
    );
    let via_hybrid = sparse.embed(&texts).await.unwrap();
    let via_task = runtime
        .sparse_embedder("sparse/bgem3")
        .await
        .unwrap()
        .embed(&texts)
        .await
        .unwrap();
    assert_eq!(via_hybrid.vectors.len(), texts.len());
    for (a, b) in via_hybrid.vectors.iter().zip(via_task.vectors.iter()) {
        assert_eq!(a.len(), b.len(), "sparse term count differs from per-task");
    }

    // --- multi-vector, resolved from the hybrid alias ---------------------
    let mv = runtime
        .multi_vector_embedder("embed/hybrid")
        .await
        .expect("hybrid alias must serve a multi-vector query");
    let mv_out = mv.embed(&texts).await.unwrap();
    assert_eq!(mv_out.vectors.len(), texts.len());
    assert_eq!(mv.dimensions(), 1024);
    for per_input in &mv_out.vectors {
        assert!(!per_input.is_empty(), "expected per-token vectors");
        assert!(per_input.iter().all(|v| v.len() == 1024));
    }

    // --- dense, resolved from the hybrid alias ----------------------------
    let dense = runtime
        .embedding("embed/hybrid")
        .await
        .expect("hybrid alias must serve a dense query");
    assert_eq!(dense.dimensions(), 1024);
    let dense_out = dense.embed(&texts).await.unwrap();
    assert_eq!(dense_out.vectors.len(), texts.len());
    assert!(dense_out.vectors.iter().all(|v| v.len() == 1024));
}
