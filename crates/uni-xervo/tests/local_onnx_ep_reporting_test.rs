//! Every `local/onnx` model reports the execution providers it runs on
//! (rustic-ai/uni-xervo#65).
//!
//! Runs offline: each alias loads tiny generated graphs from a cache
//! pre-filled at a pinned commit, which the Hugging Face client reads without
//! touching the network. Every model must report the build's default list
//! (`["cpu"]` without a GPU feature); before #65 the NLP, OCR, image, sparse,
//! multi-vector and hybrid models reported nothing.

#![cfg(feature = "provider-onnx")]

use std::path::Path;

use serde_json::{Value, json};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/onnx")
        .join(name)
}

/// A cache holding `files` (repo path → contents source) for `repo` at
/// [`COMMIT`], laid out as the Hugging Face client expects.
fn cache_with(repo: &str, files: &[(&str, FileSource)]) -> tempfile::TempDir {
    let cache = tempfile::tempdir().unwrap();
    let snapshot = cache
        .path()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(COMMIT);
    for (path, source) in files {
        let dest = snapshot.join(path);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        match source {
            FileSource::Fixture(name) => {
                std::fs::copy(fixture(name), &dest).unwrap();
            }
            FileSource::Text(text) => std::fs::write(&dest, text).unwrap(),
        }
    }
    cache
}

enum FileSource {
    Fixture(&'static str),
    Text(&'static str),
}
use FileSource::{Fixture, Text};

async fn runtime(
    task: ModelTask,
    model_id: &str,
    cache: &tempfile::TempDir,
    mut options: Value,
) -> std::sync::Arc<ModelRuntime> {
    options["cache_dir"] = json!(cache.path().to_str().unwrap());
    let spec = ModelAliasSpec {
        alias: "test/model".to_string(),
        task,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: Some(COMMIT.to_string()),
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    };
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap()
}

/// What a model reports with no `execution_providers` option: the build's
/// default list, which leads with the accelerator a GPU feature enables.
#[cfg(feature = "gpu-cuda")]
const DEFAULT_EPS: [&str; 2] = ["cuda", "cpu"];
#[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
const DEFAULT_EPS: [&str; 2] = ["coreml", "cpu"];
#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
const DEFAULT_EPS: [&str; 1] = ["cpu"];

#[tokio::test]
async fn ocr_reports_cpu_with_and_without_a_detector() {
    let cache = cache_with(
        "fixture/ocr",
        &[
            ("rec.onnx", Fixture("ctc_fixed_batch_1.onnx")),
            ("det.onnx", Fixture("tiny_detector.onnx")),
            ("dict.txt", Text("a\nb\n")),
        ],
    );
    let base = json!({ "onnx_path": "rec.onnx", "char_dict_path": "dict.txt" });
    let rt = runtime(ModelTask::Ocr, "fixture/ocr", &cache, base.clone()).await;
    let ocr = rt.ocr_model("test/model").await.unwrap();
    assert_eq!(
        ocr.active_execution_providers(),
        DEFAULT_EPS,
        "recognizer only"
    );

    // A detector-shaped graph: the load-time check on an accelerator runs it.
    let mut with_det = base;
    with_det["det_onnx_path"] = json!("det.onnx");
    let rt = runtime(ModelTask::Ocr, "fixture/ocr", &cache, with_det).await;
    let ocr = rt.ocr_model("test/model").await.unwrap();
    assert_eq!(
        ocr.active_execution_providers(),
        DEFAULT_EPS,
        "with a detector"
    );
}

#[tokio::test]
async fn image_embedder_reports_cpu() {
    let cache = cache_with(
        "fixture/image",
        &[("model.onnx", Fixture("tiny_image_encoder.onnx"))],
    );
    let rt = runtime(
        ModelTask::EmbedImage,
        "fixture/image",
        &cache,
        json!({ "onnx_path": "model.onnx", "image_size": 8, "dimensions": 3 }),
    )
    .await;
    let model = rt.image_embedder("test/model").await.unwrap();
    assert_eq!(model.active_execution_providers(), DEFAULT_EPS);
}

#[tokio::test]
async fn sparse_embedder_reports_cpu() {
    let cache = cache_with(
        "fixture/encoder",
        &[
            ("model.onnx", Fixture("tiny_encoder.onnx")),
            ("tokenizer.json", Fixture("tiny_tokenizer.json")),
        ],
    );
    let rt = runtime(
        ModelTask::EmbedSparse,
        "fixture/encoder",
        &cache,
        json!({ "artifact": "model.onnx", "sparse_method": "lexical" }),
    )
    .await;
    let model = rt.sparse_embedder("test/model").await.unwrap();
    assert_eq!(model.active_execution_providers(), DEFAULT_EPS);
}

#[tokio::test]
async fn multi_vector_embedder_reports_cpu() {
    let cache = cache_with(
        "fixture/encoder",
        &[
            ("model.onnx", Fixture("tiny_encoder.onnx")),
            ("tokenizer.json", Fixture("tiny_tokenizer.json")),
        ],
    );
    let rt = runtime(
        ModelTask::EmbedMultiVector,
        "fixture/encoder",
        &cache,
        json!({ "artifact": "model.onnx", "dimensions": 4 }),
    )
    .await;
    let model = rt.multi_vector_embedder("test/model").await.unwrap();
    assert_eq!(model.active_execution_providers(), DEFAULT_EPS);
}

#[tokio::test]
async fn hybrid_embedder_and_its_views_report_cpu() {
    // Hybrid is preset-driven, so the cache holds the BGE-M3 preset's layout.
    let cache = cache_with(
        "aapot/bge-m3-onnx",
        &[
            ("model.onnx", Fixture("tiny_encoder.onnx")),
            ("tokenizer.json", Fixture("tiny_tokenizer.json")),
            ("model.onnx.data", Text("")),
            ("Constant_685_attr__value", Text("")),
        ],
    );
    let rt = runtime(ModelTask::EmbedHybrid, "BGEM3Hybrid", &cache, json!({})).await;
    let hybrid = rt.hybrid_embedder("test/model").await.unwrap();
    assert_eq!(hybrid.active_execution_providers(), DEFAULT_EPS);
    // The single-head views of a hybrid alias forward the same list.
    let dense = rt.embedding("test/model").await.unwrap();
    assert_eq!(dense.active_execution_providers(), DEFAULT_EPS);
    let sparse = rt.sparse_embedder("test/model").await.unwrap();
    assert_eq!(sparse.active_execution_providers(), DEFAULT_EPS);
}

#[tokio::test]
async fn nlp_v1_reports_cpu() {
    let cache = cache_with(
        "fixture/nlp",
        &[
            ("onnx/cascade.onnx", Fixture("tiny_encoder.onnx")),
            ("tokenizer.json", Fixture("tiny_tokenizer.json")),
            (
                "label_maps.json",
                Text(
                    r#"{"pos": ["X"], "ner": ["O"], "srl": ["O"], "cls": ["statement"], "deprel": ["dep"]}"#,
                ),
            ),
        ],
    );
    let rt = runtime(
        ModelTask::Nlp,
        "fixture/nlp",
        &cache,
        json!({ "generation": "v1" }),
    )
    .await;
    let model = rt.nlp_model("test/model").await.unwrap();
    assert_eq!(model.active_execution_providers(), DEFAULT_EPS);
}

#[tokio::test]
async fn raw_model_reports_cpu() {
    // Raw takes a local path directly; its snapshot download always asks the
    // Hub, so it is not served from the pinned cache like the others.
    let spec = ModelAliasSpec {
        alias: "test/model".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: fixture("dynamic_batch_linear.onnx")
            .to_str()
            .unwrap()
            .to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: Value::Null,
    };
    let rt = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();
    let model = rt.raw_tensor_model("test/model").await.unwrap();
    assert_eq!(model.active_execution_providers(), DEFAULT_EPS);
}

/// A raw alias on a local model file with the given provider list.
#[cfg(feature = "gpu-metal")]
async fn raw_with(path: &Path, eps: Value) -> uni_xervo::error::Result<Vec<String>> {
    let spec = ModelAliasSpec {
        alias: "test/model".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: path.to_str().unwrap().to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: json!({ "execution_providers": eps }),
    };
    let rt = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;
    Ok(rt
        .raw_tensor_model("test/model")
        .await?
        .active_execution_providers())
}

/// Nothing listed after `cpu` is reported: CPU claims every node first.
#[cfg(feature = "gpu-metal")]
#[tokio::test]
async fn providers_after_cpu_are_not_reported() {
    let model = fixture("dynamic_batch_linear.onnx");
    assert_eq!(
        raw_with(&model, json!(["cpu", "coreml"])).await.unwrap(),
        ["cpu"]
    );
    assert_eq!(
        raw_with(&model, json!(["coreml", "cpu"])).await.unwrap(),
        ["coreml", "cpu"]
    );
}

/// A model that cannot load on any provider fails with its own error, not a
/// CPU-retry's.
#[cfg(feature = "gpu-metal")]
#[tokio::test]
async fn a_broken_model_fails_with_its_own_error() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("garbage.onnx");
    std::fs::write(&model, b"not an onnx model").unwrap();
    let err = raw_with(&model, json!(["coreml", "cpu"]))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("protobuf"), "{err}");
}
