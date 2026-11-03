#![cfg(feature = "provider-onnx")]

use hf_hub::{HFClientBuilder, split_id};
use ndarray::{Array2, Ix2, Ix3};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use tokenizers::{Encoding, Tokenizer};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::cache::resolve_cache_dir;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{RawTensorModel, TensorBatch, TensorDtype, TensorValue};

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

#[derive(Debug, Deserialize)]
struct HfConfig {
    id2label: HashMap<String, String>,
}

fn make_spec(alias: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "artifact": "model.onnx"
        }),
    }
}

async fn runtime_for(spec: ModelAliasSpec) -> std::sync::Arc<ModelRuntime> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap()
}

/// Locate a file the provider has already pulled into the cache.
///
/// `local_files_only` keeps this a pure cache lookup — the provider's own load
/// did the downloading, and a second network fetch here would make the test
/// pass even if the provider had failed to materialize the snapshot.
async fn snapshot_file(spec: &ModelAliasSpec, filename: &str) -> std::path::PathBuf {
    let cache_dir = resolve_cache_dir("onnx", &spec.model_id, &spec.options);
    let client = HFClientBuilder::new()
        .cache_dir(cache_dir)
        .build()
        .expect("failed to build HF client");
    let (owner, name) = split_id(&spec.model_id);
    client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .maybe_revision(spec.revision.clone())
        .local_files_only(true)
        .send()
        .await
        .unwrap_or_else(|e| {
            panic!(
                "missing snapshot file '{filename}' for {}: {e}",
                spec.model_id
            )
        })
}

async fn tokenizer_for_spec(spec: &ModelAliasSpec) -> Tokenizer {
    let tokenizer_path = snapshot_file(spec, "tokenizer.json").await;
    Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer")
}

async fn config_for_spec(spec: &ModelAliasSpec) -> HfConfig {
    let config_path = snapshot_file(spec, "config.json").await;
    let config = std::fs::read_to_string(config_path).expect("failed to read config");
    serde_json::from_str(&config).expect("failed to parse config.json")
}

fn build_inputs(runner: &dyn RawTensorModel, encoding: &Encoding) -> TensorBatch {
    let ids = encoding
        .get_ids()
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    let mask = encoding
        .get_attention_mask()
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    let type_ids = encoding
        .get_type_ids()
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    let seq_len = ids.len();

    let mut batch = TensorBatch::new();
    for spec in runner.input_signature() {
        let source = match spec.name.as_str() {
            "input_ids" => &ids,
            "attention_mask" => &mask,
            "token_type_ids" => &type_ids,
            other => panic!("unexpected text-model input tensor '{other}'"),
        };
        batch.insert(
            spec.name.clone(),
            tensor_from_i64(spec.dtype, source, seq_len),
        );
    }
    batch
}

fn tensor_from_i64(dtype: TensorDtype, values: &[i64], seq_len: usize) -> TensorValue {
    match dtype {
        TensorDtype::I64 => TensorValue::I64(
            Array2::from_shape_vec((1, seq_len), values.to_vec())
                .unwrap()
                .into_dyn(),
        ),
        TensorDtype::I32 => TensorValue::I32(
            Array2::from_shape_vec((1, seq_len), values.iter().map(|&v| v as i32).collect())
                .unwrap()
                .into_dyn(),
        ),
        other => panic!("unexpected input dtype for tokenized text model: {other:?}"),
    }
}

fn first_f32_output(batch: &TensorBatch) -> ndarray::ArrayD<f32> {
    batch
        .tensors
        .values()
        .find_map(|value| match value {
            TensorValue::F32(array) => Some(array.clone()),
            _ => None,
        })
        .expect("expected an f32 logits tensor")
}

fn argmax(values: impl IntoIterator<Item = f32>) -> usize {
    values
        .into_iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
        .map(|(idx, _)| idx)
        .unwrap()
}

#[tokio::test]
#[ignore]
async fn test_local_onnx_hf_sequence_classification_e2e() {
    require_expensive_tests!();

    let spec = make_spec(
        "raw/sequence-classify",
        "smokxy/sequence_classification_onnx",
    );
    let runtime = runtime_for(spec.clone()).await;
    let runner = runtime
        .raw_tensor_model("raw/sequence-classify")
        .await
        .expect("failed to resolve onnx runner");
    let tokenizer = tokenizer_for_spec(&spec).await;
    let config = config_for_spec(&spec).await;

    let positive = tokenizer
        .encode("I love this movie.", true)
        .expect("failed to encode positive text");
    let positive_inputs = build_inputs(runner.as_ref(), &positive);
    let positive_output = runner
        .run(&positive_inputs)
        .await
        .expect("positive run failed");
    let positive_logits = first_f32_output(&positive_output)
        .into_dimensionality::<Ix2>()
        .expect("expected [1, num_labels] logits");
    let positive_label = config
        .id2label
        .get(&argmax(positive_logits.row(0).iter().copied()).to_string())
        .expect("missing positive label mapping");
    assert_eq!(positive_label, "POSITIVE");

    let negative = tokenizer
        .encode("I hate this movie.", true)
        .expect("failed to encode negative text");
    let negative_inputs = build_inputs(runner.as_ref(), &negative);
    let negative_output = runner
        .run(&negative_inputs)
        .await
        .expect("negative run failed");
    let negative_logits = first_f32_output(&negative_output)
        .into_dimensionality::<Ix2>()
        .expect("expected [1, num_labels] logits");
    let negative_label = config
        .id2label
        .get(&argmax(negative_logits.row(0).iter().copied()).to_string())
        .expect("missing negative label mapping");
    assert_eq!(negative_label, "NEGATIVE");
}

#[tokio::test]
#[ignore]
async fn test_local_onnx_hf_ner_e2e() {
    require_expensive_tests!();

    let spec = make_spec("raw/ner", "protectai/bert-base-NER-onnx");
    let runtime = runtime_for(spec.clone()).await;
    let runner = runtime
        .raw_tensor_model("raw/ner")
        .await
        .expect("failed to resolve onnx runner");
    let tokenizer = tokenizer_for_spec(&spec).await;
    let config = config_for_spec(&spec).await;

    let text = "My name is Sarah and I live in London";
    let encoding = tokenizer
        .encode(text, true)
        .expect("failed to encode ner text");
    let inputs = build_inputs(runner.as_ref(), &encoding);
    let output = runner.run(&inputs).await.expect("ner run failed");
    let logits = first_f32_output(&output)
        .into_dimensionality::<Ix3>()
        .expect("expected [1, seq_len, num_labels] logits");

    let tokens = encoding.get_tokens();
    let special_mask = encoding.get_special_tokens_mask();
    let predictions = tokens
        .iter()
        .enumerate()
        .filter(|(idx, _)| special_mask[*idx] == 0)
        .map(|(idx, token)| {
            let label_idx = argmax(
                logits
                    .index_axis(ndarray::Axis(0), 0)
                    .index_axis(ndarray::Axis(0), idx)
                    .iter()
                    .copied(),
            );
            let label = config
                .id2label
                .get(&label_idx.to_string())
                .cloned()
                .unwrap_or_else(|| panic!("missing label for index {label_idx}"));
            (token.clone(), label)
        })
        .collect::<Vec<_>>();

    assert!(
        predictions
            .iter()
            .any(|(token, label)| token == "Sarah" && label == "B-PER"),
        "expected Sarah to be tagged as B-PER, got {predictions:?}"
    );
    assert!(
        predictions
            .iter()
            .any(|(token, label)| token == "London" && label == "B-LOC"),
        "expected London to be tagged as B-LOC, got {predictions:?}"
    );
}
