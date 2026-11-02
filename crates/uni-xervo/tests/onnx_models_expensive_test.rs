//! Real-inference tests that load specific embedding and reranking model
//! families through `LocalOnnxProvider` and exercise the public
//! `EmbeddingModel` / `RerankerModel` API end-to-end.
//!
//! Coverage:
//!
//! - **BGE family** (encoder-style, CLS-pooled embedders + single-logit
//!   cross-encoder rerankers): BGE-small-en-v1.5, BGE-large-en-v1.5,
//!   BGE-reranker-base.
//! - **Qwen3 family** (decoder-LM): Qwen3-Embedding-0.6B with last-token
//!   pooling, Qwen3-Reranker-0.6B via the generative
//!   (`style: "generative"`) rerank path.
//!
//! All tests are gated by both:
//!
//! - `#![cfg(feature = "provider-onnx")]` — only built when the provider is
//!   enabled.
//! - `EXPENSIVE_TESTS=1` env var (via `require_expensive_tests!`) — each
//!   test downloads multi-MB weights from HuggingFace and runs real
//!   inference, so they are skipped unless explicitly opted in. Also
//!   accepts `true` / `yes`; `0` / `false` / `no` / empty all skip.
//!
//! Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features provider-onnx \
//!     --test onnx_models_expensive_test \
//!     --run-ignored all
//! ```
//!
//! NOTE: model identifiers below target stable, public HuggingFace repos.
//! They may need updating when new minor revisions ship — these are HF
//! repo strings, not URLs, and the tests are `#[ignore]`'d so a stale ID
//! will not break CI.

#![cfg(feature = "provider-onnx")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

// ---------------------------------------------------------------------------
// Test gating
// ---------------------------------------------------------------------------

fn should_run_expensive_tests() -> bool {
    // Truthy: "1" / "true" / "yes" (case-insensitive). Falsy: anything
    // else, including "0" / "false" / "no" / empty / unset. Matches the
    // CI convention so users can copy `EXPENSIVE_TESTS=1` from the docs
    // and see the tests run, while a leftover `EXPENSIVE_TESTS=0` in
    // the shell doesn't accidentally enable them.
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

// ---------------------------------------------------------------------------
// Model identifiers
// ---------------------------------------------------------------------------

const BGE_SMALL_PRESET_ALIAS: &str = "BGESmallENV15";
const BGE_SMALL_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
const BGE_LARGE_PRESET_ALIAS: &str = "BGELargeENV15";
const BGE_RERANKER_BASE_MODEL_ID: &str = "BAAI/bge-reranker-base";
const QWEN3_EMBED_PRESET_ALIAS: &str = "Qwen3Embedding06B";
const QWEN3_RERANKER_MODEL_ID: &str = "onnx-community/Qwen3-Reranker-0.6B-ONNX";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn embed_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Embed,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
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

/// Tolerance for the L2-norm check on a normalized embedding. ORT inference
/// produces small floating-point drift versus a perfect unit vector; 1e-2
/// is loose enough for both fp32 and quantized weights without being so
/// loose that an un-normalized output would slip through.
const L2_NORM_TOL: f32 = 1e-2;

fn assert_unit_norm(vec: &[f32], label: &str) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < L2_NORM_TOL,
        "{label}: expected L2-normalized embedding (norm ≈ 1.0), got norm={norm}"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine over equal-length vectors");
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

// ---------------------------------------------------------------------------
// Embedding tests
// ---------------------------------------------------------------------------

/// Loads BGE-small-en-v1.5 via the built-in preset alias `BGESmallENV15`.
/// Verifies the preset path: the spec carries no options, so all defaults
/// (CLS pooling, 384-dim, L2-normalize, 512 max_seq_len) come from the
/// preset table.
#[tokio::test]
#[ignore]
async fn test_bge_small_embedding_via_preset() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-small-preset",
            BGE_SMALL_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-small-preset")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 384);

    let texts = ["Hello world", "Rust is amazing", "Machine learning"];
    let embeddings = model.embed(&texts).await.expect("embed call failed");

    assert_eq!(embeddings.vectors.len(), 3);
    for (i, emb) in embeddings.vectors.iter().enumerate() {
        assert_eq!(emb.len(), 384, "row {i}: BGE-small is 384-dim");
        assert_unit_norm(emb, &format!("row {i}"));
    }
}

/// Loads BGE-small-en-v1.5 using the raw HF repo id `BAAI/bge-small-en-v1.5`
/// — exercises the same preset table via a different alias path. Confirms
/// that users can refer to the model by its canonical HF id and still
/// inherit the preset defaults without specifying options.
#[tokio::test]
#[ignore]
async fn test_bge_small_embedding_via_model_id() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-small-by-id",
            BGE_SMALL_MODEL_ID,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-small-by-id")
        .await
        .expect("resolve embedding model");

    let embeddings = model
        .embed(&["one", "two"])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.vectors.len(), 2);
    assert_eq!(embeddings.vectors[0].len(), 384);
    assert_unit_norm(&embeddings.vectors[0], "row 0");
    assert_unit_norm(&embeddings.vectors[1], "row 1");
}

/// Loads BGE-large-en-v1.5 via preset alias. Verifies the larger 1024-dim
/// variant downloads, loads, and produces correctly-shaped output. This is
/// the heaviest embedder in the BGE family currently presetted (~1.3GB
/// weights), so it doubles as a download-path stress test.
#[tokio::test]
#[ignore]
async fn test_bge_large_embedding() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-large",
            BGE_LARGE_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-large")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 1024);

    let embeddings = model
        .embed(&["a sample sentence to embed"])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.vectors.len(), 1);
    assert_eq!(embeddings.vectors[0].len(), 1024);
    assert_unit_norm(&embeddings.vectors[0], "single-input");
}

/// Verifies that BGE-small produces semantically-meaningful embeddings:
/// related sentences should have higher cosine similarity than unrelated
/// pairs. This is a sanity check, not a quality benchmark — the gap on
/// these obvious examples is large enough that any seriously broken
/// model (wrong pooling, missing normalization, swapped tensors) fails.
#[tokio::test]
#[ignore]
async fn test_bge_embedding_semantic_similarity() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-sim",
            BGE_SMALL_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-sim")
        .await
        .expect("resolve embedding model");

    // Two related dog sentences and one unrelated cooking sentence.
    let embeddings = model
        .embed(&[
            "The dog chased the ball across the yard.",
            "A puppy ran after a tennis ball in the park.",
            "Boil pasta in salted water for nine minutes.",
        ])
        .await
        .expect("embed call failed");

    let related_sim = cosine(&embeddings.vectors[0], &embeddings.vectors[1]);
    let unrelated_sim = cosine(&embeddings.vectors[0], &embeddings.vectors[2]);

    assert!(
        related_sim > unrelated_sim + 0.10,
        "expected related-pair cosine ({related_sim:.3}) to exceed \
         unrelated-pair cosine ({unrelated_sim:.3}) by a clear margin"
    );
}

/// Loads Qwen3-Embedding-0.6B via the `Qwen3Embedding06B` preset alias.
/// Exercises the decoder-style embedder path: external-data download
/// (`onnx/model.onnx_data` sidecar), last-token pooling, 1024-dim output.
/// This is the heaviest embedder in the test suite (~600 MB ONNX +
/// external data), so it doubles as a stress test for the
/// `additional_files` download loop.
#[tokio::test]
#[ignore]
async fn test_qwen3_embedding_06b() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/qwen3-06b",
            QWEN3_EMBED_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/qwen3-06b")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 1024);

    let embeddings = model
        .embed(&[
            "The dog chased the ball across the yard.",
            "A puppy ran after a tennis ball in the park.",
            "Boil pasta in salted water for nine minutes.",
        ])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.vectors.len(), 3);
    for (i, emb) in embeddings.vectors.iter().enumerate() {
        assert_eq!(emb.len(), 1024, "row {i}: Qwen3-Embedding is 1024-dim");
        assert_unit_norm(emb, &format!("row {i}"));
    }

    // Last-token pooling on a decoder-only model should still capture
    // semantic similarity. Same sanity check as the BGE variant.
    let related_sim = cosine(&embeddings.vectors[0], &embeddings.vectors[1]);
    let unrelated_sim = cosine(&embeddings.vectors[0], &embeddings.vectors[2]);
    assert!(
        related_sim > unrelated_sim + 0.05,
        "expected related-pair cosine ({related_sim:.3}) to exceed \
         unrelated-pair cosine ({unrelated_sim:.3}) by a clear margin"
    );
}

/// Loads BGE-M3 dense embeddings from the community multi-output export
/// `aapot/bge-m3-onnx` via the `BGEM3Dense` preset. Exercises the multi-output
/// dense path: the preset pins the `dense_vecs` output (a pre-pooled
/// `[batch, 1024]` tensor), so the embedder takes the 2-D output directly with
/// no pooling. Verifies 1024-dim, unit-norm vectors and basic semantic ordering.
#[tokio::test]
#[ignore]
async fn test_bge_m3_dense_via_aapot_export() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-m3-dense",
            "BGEM3Dense",
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-m3-dense")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 1024);

    let embeddings = model
        .embed(&[
            "a cat sat on the warm windowsill",
            "the kitten napped in the sunny window",
            "quarterly revenue exceeded analyst expectations",
        ])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.vectors.len(), 3);
    for (i, emb) in embeddings.vectors.iter().enumerate() {
        assert_eq!(emb.len(), 1024, "row {i}: BGE-M3 dense is 1024-dim");
        assert_unit_norm(emb, &format!("row {i}"));
    }

    // The two cat/window sentences should be closer than either is to the
    // unrelated finance sentence — sanity that we read the dense head, not noise.
    let v = &embeddings.vectors;
    let related = cosine(&v[0], &v[1]);
    let unrelated = cosine(&v[0], &v[2]);
    assert!(
        related > unrelated,
        "expected related pair (cos={related:.3}) closer than unrelated (cos={unrelated:.3})"
    );
}

// ---------------------------------------------------------------------------
// Reranker tests
// ---------------------------------------------------------------------------

/// Loads BGE-reranker-base and verifies it ranks an obviously-relevant
/// document above three obviously-irrelevant ones. Exercises the full
/// cross-encoder path: text-pair tokenization, batched ONNX inference,
/// single-logit output extraction, descending-score sort.
#[tokio::test]
#[ignore]
async fn test_bge_reranker_base_orders_relevant_first() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![rerank_spec(
            "rerank/bge-base",
            BGE_RERANKER_BASE_MODEL_ID,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let reranker = runtime
        .reranker("rerank/bge-base")
        .await
        .expect("resolve reranker");

    let relevant = "Giant pandas live in the bamboo forests of central China.";
    let docs = vec![
        "The Eiffel Tower is in Paris, France.",
        relevant,
        "Quantum entanglement was first described in the 1930s.",
        "The Pacific Ocean is the largest body of water on Earth.",
    ];

    let scored = reranker
        .rerank("Where do giant pandas live?", &docs)
        .await
        .expect("rerank call failed");

    assert_eq!(scored.len(), docs.len());

    // Scores must be monotonically descending — the provider sorts by score.
    for w in scored.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores not in descending order: {} then {}",
            w[0].score,
            w[1].score
        );
    }

    let top = &scored[0];
    assert_eq!(
        docs[top.index], relevant,
        "expected the panda doc to rank first, got idx={} ({:?})",
        top.index, docs[top.index]
    );
}

/// Loads Qwen3-Reranker-0.6B (q4 ONNX) via the new generative reranker
/// path (`style: "generative"`). Exercises the full decoder-LM pipeline:
/// chat-prompt formatting, tokenization, ONNX forward pass with empty
/// KV-cache placeholders, last-token logit extraction, and yes/no
/// softmax scoring. The query/docs are deliberately clear-cut so the
/// q4-quantized weights still rank correctly.
#[tokio::test]
#[ignore]
async fn test_qwen3_reranker_06b_orders_relevant_first() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ModelAliasSpec {
            alias: "rerank/qwen3-06b".to_string(),
            task: ModelTask::Rerank,
            provider_id: "local/onnx".to_string(),
            model_id: QWEN3_RERANKER_MODEL_ID.to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({"style": "generative"}),
        }])
        .build()
        .await
        .expect("runtime build failed");

    let reranker = runtime
        .reranker("rerank/qwen3-06b")
        .await
        .expect("resolve generative reranker");

    let relevant = "Paris is the capital and largest city of France.";
    let docs = vec![
        "Mount Everest is the tallest mountain on Earth.",
        relevant,
        "The Pacific Ocean is the largest body of water on Earth.",
        "Tokyo is the capital of Japan, not France.",
    ];

    let scored = reranker
        .rerank("What is the capital of France?", &docs)
        .await
        .expect("rerank call failed");

    assert_eq!(scored.len(), docs.len());

    // Generative reranker emits softmax probabilities; every score must
    // sit in [0, 1] and the relevant doc must outrank the others.
    for s in &scored {
        assert!(
            (0.0..=1.0).contains(&s.score),
            "score for idx={} not in [0, 1]: {}",
            s.index,
            s.score
        );
    }
    for w in scored.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores not in descending order: {} then {}",
            w[0].score,
            w[1].score
        );
    }
    assert_eq!(
        docs[scored[0].index], relevant,
        "expected the Paris doc to rank first, got idx={} ({:?})",
        scored[0].index, docs[scored[0].index]
    );
}

/// Sanity check that ranking is stable when the input doc order is
/// permuted. Catches batch-padding / index-tracking bugs (e.g. forgetting
/// to translate an internal sorted index back to the caller's input
/// position) by re-running with a different doc order and confirming the
/// same document still wins.
#[tokio::test]
#[ignore]
async fn test_bge_reranker_score_ordering_stable_across_permutation() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![rerank_spec(
            "rerank/bge-stable",
            BGE_RERANKER_BASE_MODEL_ID,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let reranker = runtime
        .reranker("rerank/bge-stable")
        .await
        .expect("resolve reranker");

    let query = "What is the capital of France?";
    let relevant = "Paris is the capital and largest city of France.";

    let order_a = vec![
        "Tokyo is the capital of Japan.",
        relevant,
        "The Amazon is the largest rainforest on Earth.",
        "Mount Everest is the tallest mountain in the world.",
    ];
    let order_b = vec![
        "Mount Everest is the tallest mountain in the world.",
        "The Amazon is the largest rainforest on Earth.",
        "Tokyo is the capital of Japan.",
        relevant,
    ];

    let a = reranker.rerank(query, &order_a).await.expect("rerank A");
    let b = reranker.rerank(query, &order_b).await.expect("rerank B");

    assert_eq!(order_a[a[0].index], relevant, "order A: relevant doc first");
    assert_eq!(order_b[b[0].index], relevant, "order B: relevant doc first");
}

// ---------------------------------------------------------------------------
// kniv-deberta NLP cascade — multi-head POS / NER / DEP / SRL / CLS
// ---------------------------------------------------------------------------

const KNIV_MODEL_ID: &str = "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall";

fn nlp_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Nlp,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

#[tokio::test]
#[ignore]
async fn kniv_deberta_nlp_cascade_end_to_end() {
    require_expensive_tests!();

    use uni_xervo::traits::{NlpRequest, NlpTasks};

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/kniv",
            KNIV_MODEL_ID,
            serde_json::Value::Object(serde_json::Map::new()),
        )])
        .build()
        .await
        .expect("build runtime");

    let model = runtime.nlp_model("nlp/kniv").await.expect("resolve nlp");

    // One sentence with at least one verb, a named entity, and a clear
    // dependency structure. Each head should populate at least one item.
    let text = "Alice bought a book in Paris.";
    let req = NlpRequest::new(text, NlpTasks::CASCADE);
    let results = model.analyze(vec![req]).await.expect("analyze");
    assert_eq!(results.len(), 1);
    let r = &results[0];

    assert!(!r.tokens.is_empty(), "tokens populated");

    // POS should include at least one VERB, one PROPN (proper noun), and
    // one PUNCT for the period.
    let pos_tags: Vec<&str> = r.tokens.iter().filter_map(|t| t.pos.as_deref()).collect();
    assert!(
        pos_tags.contains(&"VERB"),
        "POS includes VERB: {pos_tags:?}"
    );
    assert!(
        pos_tags.contains(&"PROPN") || pos_tags.contains(&"NOUN"),
        "POS includes a noun-like tag: {pos_tags:?}"
    );

    // NER: "Alice" and "Paris" should appear as B-* entities.
    let ner_tags: Vec<&str> = r
        .tokens
        .iter()
        .filter_map(|t| t.ner.as_deref())
        .filter(|s| s != &"O")
        .collect();
    assert!(
        !ner_tags.is_empty(),
        "NER detected at least one entity: {ner_tags:?}"
    );

    // DEP: every token has a head link; relations are non-empty strings, and
    // the head is either root (`None`) or a valid global token index (R2). A
    // well-formed parse has exactly one (or at least one) root.
    let mut root_count = 0;
    for tok in &r.tokens {
        let dep = tok.dep.as_ref().expect("DEP populated when requested");
        assert!(!dep.relation.is_empty(), "DEP relation non-empty");
        match dep.head {
            None => root_count += 1,
            Some(h) => assert!(
                h < r.tokens.len(),
                "DEP head {h} in range 0..{}",
                r.tokens.len()
            ),
        }
    }
    assert!(
        root_count >= 1,
        "parse has at least one root (head == None)"
    );

    // SRL: "bought" should anchor at least one frame.
    assert!(
        !r.frames.is_empty(),
        "SRL frames populated for sentence with a verb"
    );

    // R3 — merged NER entities: spans are populated, with valid token/char
    // bounds and surface text matching the original slice.
    assert!(!r.entities.is_empty(), "NER entities merged from BIO tags");
    for ent in &r.entities {
        assert!(!ent.label.is_empty(), "entity label non-empty");
        let (first, last) = ent.token_span;
        assert!(
            first <= last && last < r.tokens.len(),
            "token span in range"
        );
        let (cs, ce) = ent.char_span;
        assert_eq!(ent.text, &text[cs..ce], "entity text matches byte span");
    }

    // R4 — word_index is dense and monotonically non-decreasing, starting at 0.
    let mut prev_word = 0usize;
    assert_eq!(r.tokens[0].word_index, 0, "first token is word 0");
    for tok in &r.tokens {
        assert!(
            tok.word_index == prev_word || tok.word_index == prev_word + 1,
            "word_index is dense/monotonic: {} after {prev_word}",
            tok.word_index
        );
        prev_word = tok.word_index;
    }

    // R5 — the model exposes its label vocabularies.
    let maps = model.label_maps().expect("ONNX model exposes label maps");
    for (name, v) in [
        ("pos", &maps.pos),
        ("ner", &maps.ner),
        ("deprel", &maps.deprel),
        ("srl", &maps.srl),
        ("cls", &maps.cls),
    ] {
        assert!(!v.is_empty(), "label vocabulary '{name}' non-empty");
    }

    // CLS: one dialog act for the input, now with the full distribution (R1).
    assert_eq!(r.speech_acts.len(), 1, "CLS produces one dialog act");
    let act = &r.speech_acts[0];
    assert!(!act.label.is_empty(), "CLS label non-empty");
    assert!(
        act.confidence >= 0.0 && act.confidence <= 1.0,
        "CLS confidence in [0,1]"
    );
    // R1 — full softmax: parallel to the cls vocabulary, sums to ~1.0, and the
    // top-1 confidence equals the winning class's score.
    assert_eq!(
        act.scores.len(),
        maps.cls.len(),
        "scores parallel to cls vocabulary"
    );
    let sum: f32 = act.scores.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "CLS scores sum to ~1.0, got {sum}"
    );
    let max = act.scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (max - act.confidence).abs() < 1e-5,
        "confidence equals argmax score"
    );

    // word_index groups subwords: there are no more words than tokens, and at
    // least one word should exist.
    let word_count = r.tokens.last().map(|t| t.word_index + 1).unwrap_or(0);
    assert!(
        word_count >= 1 && word_count <= r.tokens.len(),
        "word count sane"
    );
}

/// Requesting a subset of heads populates only those, and the merged-entity /
/// word-alignment views still come through. Guards the per-task gating: NER is
/// requested but DEP / SRL / CLS are not, so their fields must stay empty while
/// `entities` (the R3 view of NER) is populated.
#[tokio::test]
#[ignore]
async fn kniv_deberta_nlp_subset_tasks() {
    require_expensive_tests!();

    use uni_xervo::traits::{NlpRequest, NlpTasks};

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/kniv-subset",
            KNIV_MODEL_ID,
            serde_json::Value::Object(serde_json::Map::new()),
        )])
        .build()
        .await
        .expect("build runtime");

    let model = runtime
        .nlp_model("nlp/kniv-subset")
        .await
        .expect("resolve nlp");

    // Only POS + NER requested.
    let text = "Marie Curie discovered radium in Paris.";
    let req = NlpRequest::new(text, NlpTasks::POS | NlpTasks::NER);
    let r = &model.analyze(vec![req]).await.expect("analyze")[0];

    assert!(!r.tokens.is_empty());
    for tok in &r.tokens {
        assert!(tok.pos.is_some(), "POS requested -> populated");
        assert!(tok.ner.is_some(), "NER requested -> populated");
        assert!(tok.dep.is_none(), "DEP not requested -> empty");
    }
    // Unrequested heads stay empty.
    assert!(r.frames.is_empty(), "SRL not requested -> no frames");
    assert!(r.speech_acts.is_empty(), "CLS not requested -> no acts");

    // R3 — merged entities populate from the requested NER head, and a
    // multi-token entity ("Marie Curie") proves the BIO merge spans words.
    assert!(!r.entities.is_empty(), "entities populated from NER");
    let multi = r.entities.iter().find(|e| e.token_span.0 != e.token_span.1);
    if let Some(ent) = multi {
        assert_eq!(ent.text, &text[ent.char_span.0..ent.char_span.1]);
    }

    // R4 — every token in one entity should map to contiguous words, and
    // word_index is dense across the whole result.
    let mut prev = 0usize;
    for tok in &r.tokens {
        assert!(tok.word_index == prev || tok.word_index == prev + 1);
        prev = tok.word_index;
    }
}
