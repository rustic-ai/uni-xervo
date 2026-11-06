// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Structured NLP task for `local/onnx`.
//!
//! Implements [`NlpModel`] against the kniv-deberta cascade family
//! (canonical default: `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`).
//! One DeBERTa-v3 encoder forward pass produces all five heads in parallel:
//! POS / NER / DEP arc + label / SRL / dialog-act CLS.
//!
//! Per the model card, SRL is per-predicate: `predicate_idx` selects one
//! verb to score. To populate every frame in a sentence the provider runs
//! the forward once with `predicate_idx = 0` (sentinel "no SRL") to extract
//! POS / NER / DEP / CLS, then identifies verbs from the POS output and
//! re-runs once per verb with the matching index. This multi-pass dance is
//! invisible to callers; the `analyze` contract just asks for `NlpTasks`
//! and returns populated heads.
//!
//! Inputs longer than `max_seq_len` (default 128 — the model's training
//! cap) are chunked into non-overlapping token windows. Each chunk is
//! decoded independently; per-token byte offsets stay correct because the
//! tokenizer is run once over the full text with `Tokenizer::encode` and
//! the resulting offsets are absolute. Dependency arcs do not cross chunk
//! boundaries — each chunk gets its own local dependency tree.

use crate::hf::HfModelRepo;
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, new_session_builder,
    parse_execution_providers_option,
};
use crate::traits::{
    DepLink, NerEntity, NlpLabelMaps, NlpModel, NlpRequest, NlpResult, NlpSentence, NlpTasks,
    NlpToken, SpeechAct, SrlFrame, SrlRole,
};

// Defaults — match the canonical kniv-deberta repo layout.
const DEFAULT_ONNX_PATH: &str = "onnx/cascade.onnx";
const DEFAULT_TOKENIZER_PATH: &str = "tokenizer.json";
const DEFAULT_LABEL_MAPS_PATH: &str = "label_maps.json";
const DEFAULT_MAX_SEQ_LEN: usize = 128;

/// Entry point called from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::Nlp`.
pub(super) async fn load_nlp(spec: &ModelAliasSpec) -> Result<Arc<dyn NlpModel>> {
    let model = OnnxNlpModel::load(spec).await?;
    Ok(Arc::new(model) as Arc<dyn NlpModel>)
}

/// Resolved configuration after option parsing.
struct NlpConfig {
    hf_repo: String,
    onnx_path: String,
    tokenizer_path: String,
    label_maps_path: String,
    max_seq_len: usize,
}

impl NlpConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let opts = &spec.options;
        let get_str = |key: &str| -> Option<String> {
            opts.get(key).and_then(Value::as_str).map(str::to_string)
        };
        Ok(Self {
            hf_repo: spec.model_id.clone(),
            onnx_path: get_str("onnx_path").unwrap_or_else(|| DEFAULT_ONNX_PATH.to_string()),
            tokenizer_path: get_str("tokenizer_path")
                .unwrap_or_else(|| DEFAULT_TOKENIZER_PATH.to_string()),
            label_maps_path: get_str("label_maps_path")
                .unwrap_or_else(|| DEFAULT_LABEL_MAPS_PATH.to_string()),
            max_seq_len: opts
                .get("max_seq_len")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_MAX_SEQ_LEN),
        })
    }
}

/// Parse the kniv `label_maps.json` into the public [`NlpLabelMaps`].
///
/// Required keys: `pos`, `ner`, `srl`, `cls`, `deprel`. Each maps to an array
/// of strings whose position is the model's class index. The result is the
/// same vocabulary the model decodes against, exposed verbatim via
/// [`NlpModel::label_maps`] so consumers need not embed a parallel copy.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if the file cannot be read or parsed, a
/// required key is missing, or an entry is not a string.
fn parse_label_maps(path: &Path) -> Result<NlpLabelMaps> {
    let source = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| {
        RuntimeError::Config(format!("Failed to read NLP label maps at {source}: {e}"))
    })?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|e| {
        RuntimeError::Config(format!("Failed to parse NLP label maps at {source}: {e}"))
    })?;
    label_maps_from_value(&value, &source)
}

/// Build [`NlpLabelMaps`] from an already-parsed `label_maps.json` value.
///
/// `source` names the origin (a path) for error messages. Split out from file
/// IO so the required-key / non-string failure paths are unit-testable.
///
/// # Errors
/// Returns [`RuntimeError::Config`] if a required key (`pos`, `ner`, `srl`,
/// `cls`, `deprel`) is absent, is not an array, or holds a non-string entry.
fn label_maps_from_value(value: &Value, source: &str) -> Result<NlpLabelMaps> {
    let take = |key: &str| -> Result<Vec<String>> {
        value
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| {
                RuntimeError::Config(format!(
                    "NLP label maps at {source} is missing required array '{key}'",
                ))
            })?
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "NLP label maps at {source}: '{key}' contains non-string entries",
                    ))
                })
            })
            .collect()
    };
    Ok(NlpLabelMaps {
        pos: take("pos")?,
        ner: take("ner")?,
        deprel: take("deprel")?,
        srl: take("srl")?,
        cls: take("cls")?,
        ..Default::default()
    })
}

/// ONNX-backed multi-head structured NLP model.
struct OnnxNlpModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    labels: NlpLabelMaps,
    alias: String,
    model_id: String,
    max_seq_len: usize,
}

impl OnnxNlpModel {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = NlpConfig::resolve(spec)?;
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;

        // Validate EP list FIRST so misconfigurations fail with a precise error.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-nlp", &cfg.hf_repo, &spec.options);
        let (model_path, tokenizer_path, label_maps_path) = download_nlp_artifacts(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
            &cfg.tokenizer_path,
            &cfg.label_maps_path,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            model_path = %model_path.display(),
            max_seq_len = cfg.max_seq_len,
            "Loading ONNX NLP cascade"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let labels = parse_label_maps(&label_maps_path)?;
        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            labels,
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            max_seq_len: cfg.max_seq_len,
        })
    }

    /// Run the cascade for one request, handling chunking and SRL multi-pass.
    fn analyze_one(&self, request: &NlpRequest<'_>) -> Result<NlpResult> {
        // Single tokenization over the whole text — offsets returned are
        // absolute UTF-8 byte offsets in `request.text`, which keeps
        // NlpToken::{start,end} stable across any chunk slicing below.
        let encoding = self.tokenizer.encode(request.text, true).map_err(|e| {
            RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Tokenization failed: {e}"),
            }
        })?;
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets();
        let special_mask = encoding.get_special_tokens_mask();
        // Per-token word id from the tokenizer (None for special tokens). Drives
        // `NlpToken::word_index`; the byte-gap fallback below covers tokenizers
        // that leave word ids unset.
        let word_ids = encoding.get_word_ids();
        let total = ids.len();

        // Walk the token sequence in non-overlapping windows of size
        // `max_seq_len`. The last window may be shorter than the cap; the
        // ONNX graph declares dynamic seq so the smaller batch is fine.
        let mut all_tokens: Vec<NlpToken> = Vec::with_capacity(total);
        let mut all_frames: Vec<SrlFrame> = Vec::new();
        let mut cls_logits_first: Option<Array1<f32>> = None;

        // Running state for assigning a dense, monotonic `word_index` across all
        // chunks. A token opens a new word when its tokenizer word id differs
        // from the previous token's (or, lacking word ids, when a byte gap
        // separates it from the previous token).
        let mut next_word_index: usize = 0;
        let mut pushed_any_token = false;
        let mut prev_word_id: Option<u32> = None;
        let mut prev_token_end: usize = 0;

        let mut start = 0;
        while start < total {
            let end = (start + self.max_seq_len).min(total);
            let chunk_ids: Vec<i64> = ids[start..end].iter().map(|&v| v as i64).collect();
            let chunk_mask: Vec<i64> = mask[start..end].iter().map(|&v| v as i64).collect();
            let chunk_offsets = &offsets[start..end];
            let chunk_special = &special_mask[start..end];

            // Pass 1 — predicate_idx = 0 — produces POS / NER / DEP / CLS
            // for this chunk plus the CLS-level dialog act for the first
            // chunk only (subsequent chunks' CLS is dropped since dialog
            // act is a whole-utterance signal).
            let outputs = self.forward(&chunk_ids, &chunk_mask, 0)?;

            let pos_indices = argmax_last_axis(&outputs.pos_logits)?;
            let ner_indices = argmax_last_axis(&outputs.ner_logits)?;
            // `arc_scores` is `[seq, seq]` (token rows x candidate-head
            // columns); the best head per token is a per-row argmax.
            let dep_heads = argmax_last_axis(&outputs.arc_scores)?;
            // For each (token, head), look up the relation argmax over the
            // 53 deprel classes.
            let dep_relations = dep_relation_per_token(
                &outputs.label_scores,
                &dep_heads,
                self.labels.deprel.len(),
            )?;

            // First pass: map each chunk-local position to its global index
            // into `all_tokens` (None for special tokens / padding). DEP heads
            // and SRL spans are remapped through this so every emitted index is
            // global, never chunk- or model-tokenization-local.
            let mut chunk_token_global_indices: Vec<Option<usize>> = vec![None; chunk_ids.len()];
            let mut next_global = all_tokens.len();
            for (i, (&off, &is_special)) in chunk_offsets.iter().zip(chunk_special).enumerate() {
                if is_special != 0 || off == (0, 0) {
                    continue;
                }
                chunk_token_global_indices[i] = Some(next_global);
                next_global += 1;
            }

            // Second pass: build NlpTokens, now able to remap DEP heads through
            // the fully-populated global-index map (a head may point forward).
            for (i, (&off, &is_special)) in chunk_offsets.iter().zip(chunk_special).enumerate() {
                if is_special != 0 || off == (0, 0) {
                    continue;
                }
                let pos = request
                    .tasks
                    .contains(NlpTasks::POS)
                    .then(|| label_at(&self.labels.pos, pos_indices[i]));
                let ner = request
                    .tasks
                    .contains(NlpTasks::NER)
                    .then(|| label_at(&self.labels.ner, ner_indices[i]));
                let dep = request.tasks.contains(NlpTasks::DEP).then(|| DepLink {
                    head: remap_dep_head(dep_heads[i], i, &chunk_token_global_indices),
                    relation: label_at(&self.labels.deprel, dep_relations[i]),
                });

                // Assign `word_index`: new word when the tokenizer word id
                // changes, else (no word ids) when a byte gap precedes us.
                let raw_word_id = word_ids.get(start + i).copied().flatten();
                let starts_new_word = pushed_any_token
                    && is_word_boundary(prev_word_id, raw_word_id, prev_token_end, off.0);
                if starts_new_word {
                    next_word_index += 1;
                }
                pushed_any_token = true;
                prev_word_id = raw_word_id;
                prev_token_end = off.1;

                // Surface form from the absolute byte range.
                let text = request.text.get(off.0..off.1).unwrap_or("").to_string();
                all_tokens.push(NlpToken {
                    text,
                    start: off.0,
                    end: off.1,
                    pos,
                    ner,
                    dep,
                    word_index: next_word_index,
                });
            }

            // Keep CLS logits from the very first chunk only.
            if start == 0 {
                cls_logits_first = Some(outputs.cls_logits);
            }

            // SRL multi-pass: identify chunk-local verb tokens via POS
            // argmax and re-run forward once per verb with the matching
            // predicate_idx. Frames are appended to the running list.
            if request.tasks.contains(NlpTasks::SRL) {
                let verb_id = self
                    .labels
                    .pos
                    .iter()
                    .position(|tag| tag == "VERB")
                    .unwrap_or(usize::MAX);
                for (local_i, &is_special) in chunk_special.iter().enumerate() {
                    if is_special != 0 {
                        continue;
                    }
                    // predicate_idx == 0 is the sentinel "no SRL"; the
                    // model card guarantees position 0 is always a special
                    // token ([CLS]). Skip it explicitly.
                    if local_i == 0 {
                        continue;
                    }
                    if pos_indices[local_i] != verb_id {
                        continue;
                    }
                    let pred_outputs = self.forward(&chunk_ids, &chunk_mask, local_i as i64)?;
                    let srl_indices = argmax_last_axis(&pred_outputs.srl_logits)?;
                    let frame = decode_srl_frame(
                        local_i,
                        &srl_indices,
                        &chunk_token_global_indices,
                        chunk_offsets,
                        chunk_special,
                        &self.labels.srl,
                    );
                    if let Some(frame) = frame {
                        all_frames.push(frame);
                    }
                }
            }

            start = end;
        }

        let mut sentences = Vec::new();
        if !all_tokens.is_empty() {
            sentences.push(NlpSentence {
                token_range: (0, all_tokens.len() - 1),
                start: all_tokens.first().map(|t| t.start).unwrap_or(0),
                end: all_tokens
                    .last()
                    .map(|t| t.end)
                    .unwrap_or(request.text.len()),
            });
        }

        let speech_acts = if request.tasks.contains(NlpTasks::CLS) {
            cls_logits_first
                .as_ref()
                .map(|logits| {
                    // Expose the whole distribution (R1); `confidence` is just
                    // its argmax probability, so consumers can threshold any
                    // class instead of only the top-1 label.
                    let scores = softmax_full(logits);
                    let id = argmax_1d(logits);
                    vec![SpeechAct {
                        sentence_index: 0,
                        label: label_at(&self.labels.cls, id),
                        confidence: scores.get(id).copied().unwrap_or(0.0),
                        scores,
                    }]
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // BIO-merge the per-token NER tags into entity spans (R3).
        let entities = if request.tasks.contains(NlpTasks::NER) {
            merge_ner_spans(&all_tokens, request.text)
        } else {
            Vec::new()
        };

        Ok(NlpResult {
            tokens: all_tokens,
            sentences,
            frames: all_frames,
            speech_acts,
            entities,
            ..Default::default()
        })
    }

    /// One ONNX forward pass on a single chunk.
    fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        predicate_idx: i64,
    ) -> Result<CascadeOutputs> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: "Empty input chunk".to_string(),
            });
        }

        let input_ids_arr = Array2::<i64>::from_shape_vec((1, seq_len), input_ids.to_vec())
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("input_ids shape error: {e}"),
            })?;
        let attention_mask_arr =
            Array2::<i64>::from_shape_vec((1, seq_len), attention_mask.to_vec()).map_err(|e| {
                RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("attention_mask shape error: {e}"),
                }
            })?;
        let predicate_idx_arr = Array1::<i64>::from_vec(vec![predicate_idx]);

        let mut session = self
            .session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Session lock poisoned: {e}"),
            })?;

        let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let ids_tensor = ort::value::Tensor::from_array(input_ids_arr.into_dyn())
            .map_err(|e| map_err(e, "input_ids tensor"))?;
        let mask_tensor = ort::value::Tensor::from_array(attention_mask_arr.into_dyn())
            .map_err(|e| map_err(e, "attention_mask tensor"))?;
        let pred_tensor = ort::value::Tensor::from_array(predicate_idx_arr.into_dyn())
            .map_err(|e| map_err(e, "predicate_idx tensor"))?;

        let inputs: Vec<(String, ort::value::DynTensor)> = vec![
            ("input_ids".to_string(), ids_tensor.upcast()),
            ("attention_mask".to_string(), mask_tensor.upcast()),
            ("predicate_idx".to_string(), pred_tensor.upcast()),
        ];

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "ONNX inference"))?;

        let pos_logits = extract_3d_first_batch(&outputs, "pos_logits", &self.alias)?;
        let ner_logits = extract_3d_first_batch(&outputs, "ner_logits", &self.alias)?;
        let arc_scores = extract_3d_first_batch(&outputs, "arc_scores", &self.alias)?;
        let label_scores = extract_4d_first_batch(&outputs, "label_scores", &self.alias)?;
        let srl_logits = extract_3d_first_batch(&outputs, "srl_logits", &self.alias)?;
        let cls_logits = extract_2d_first_batch_1d(&outputs, "cls_logits", &self.alias)?;

        Ok(CascadeOutputs {
            pos_logits,
            ner_logits,
            arc_scores,
            label_scores,
            srl_logits,
            cls_logits,
        })
    }
}

/// Per-chunk ONNX outputs (batch dim stripped — batch is always 1).
struct CascadeOutputs {
    pos_logits: Array2<f32>,   // [seq, 17]
    ner_logits: Array2<f32>,   // [seq, 37]
    arc_scores: Array2<f32>,   // [seq, seq]
    label_scores: Array3<f32>, // [seq, seq, 53]
    srl_logits: Array2<f32>,   // [seq, 42]
    cls_logits: Array1<f32>,   // [8]
}

impl crate::traits::ModelInfo for OnnxNlpModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl NlpModel for OnnxNlpModel {
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>> {
        // The kniv graph accepts dynamic batch in principle, but our SRL
        // multi-pass orchestration is per-sentence — running each request
        // serially keeps the post-processing straightforward and matches
        // typical caller batch sizes (1–8).
        let mut results = Vec::with_capacity(requests.len());
        for req in &requests {
            results.push(self.analyze_one(req)?);
        }
        Ok(results)
    }

    fn supported_tasks(&self) -> NlpTasks {
        // The v1 cascade has five heads; the v7 additions (morphology,
        // lemmas, sentiment, coreference, entities, facts) need the v7
        // provider path in `nlp_v7`.
        NlpTasks::CASCADE
    }

    fn label_maps(&self) -> Option<&NlpLabelMaps> {
        Some(&self.labels)
    }
}

// ---------------------------------------------------------------------------
// Output decoding helpers
// ---------------------------------------------------------------------------

fn label_at(labels: &[String], id: usize) -> String {
    labels
        .get(id)
        .cloned()
        .unwrap_or_else(|| format!("UNK_{id}"))
}

/// Argmax over the last axis of a 2-D array `[seq, classes]`.
fn argmax_last_axis(arr: &Array2<f32>) -> Result<Vec<usize>> {
    Ok((0..arr.shape()[0])
        .map(|i| {
            let row = arr.row(i);
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect())
}

/// Argmax over a 1-D array.
fn argmax_1d(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Remap a model-space dependency head to a global `NlpResult::tokens` index.
///
/// `head` and `token` index the model's tokenized sequence (special tokens
/// included); `chunk_token_global_indices` maps those positions to global token
/// indices, with `None` for special tokens. Returns `None` (the sentence root)
/// in the two ways the cascade marks a root: a self-loop (`head == token`) or a
/// head landing on a special token such as `[CLS]`. Otherwise returns the head's
/// global token index.
fn remap_dep_head(
    head: usize,
    token: usize,
    chunk_token_global_indices: &[Option<usize>],
) -> Option<usize> {
    if head == token {
        // Self-loop is the cascade's root marker (verified against the kniv
        // model: roots point at themselves, not at `[CLS]`).
        return None;
    }
    chunk_token_global_indices.get(head).copied().flatten()
}

/// Whether `token` opens a new word relative to the previous emitted token.
///
/// Prefers the tokenizer's word ids: a change in word id starts a new word. When
/// either id is absent (the tokenizer left it unset), falls back to a byte gap —
/// a `token_start` past the previous token's end means whitespace separated them.
fn is_word_boundary(
    prev_word_id: Option<u32>,
    word_id: Option<u32>,
    prev_token_end: usize,
    token_start: usize,
) -> bool {
    match (prev_word_id, word_id) {
        (Some(prev), Some(cur)) => cur != prev,
        _ => token_start > prev_token_end,
    }
}

/// For each token, look up the deprel label given its already-decoded head.
///
/// `label_scores` is `[seq, seq, n_rel]`; for token `i` with head `h`,
/// `label_scores[[i, h, k]]` is the score for relation class `k`.
fn dep_relation_per_token(
    label_scores: &Array3<f32>,
    dep_heads: &[usize],
    n_rel: usize,
) -> Result<Vec<usize>> {
    let seq = label_scores.shape()[0];
    if dep_heads.len() != seq {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: "nlp".to_string(),
            cause: format!(
                "DEP shape mismatch: arc_scores seq {seq} vs heads len {}",
                dep_heads.len()
            ),
        });
    }
    if label_scores.shape()[2] != n_rel {
        warn!(
            label_scores_last = label_scores.shape()[2],
            expected = n_rel,
            "DEP label_scores last dim does not match deprel label map size; \
             will still decode but UNK_<id> may appear in output"
        );
    }
    Ok((0..seq)
        .map(|i| {
            let h = dep_heads[i].min(seq.saturating_sub(1));
            let slice = label_scores.slice(ndarray::s![i, h, ..]);
            slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect())
}

/// Numerically stable softmax over a 1-D logit vector.
///
/// Subtracts the max before exponentiating to avoid overflow. Returns a vector
/// parallel to `logits` that sums to ~1.0, or all-zeros if the inputs degenerate
/// (e.g. an empty vector).
fn softmax_full(logits: &Array1<f32>) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    } else {
        exps.iter_mut().for_each(|e| *e = 0.0);
    }
    exps
}

/// BIO-merge per-token NER tags into [`NerEntity`] spans (R3).
///
/// Walks the already-assembled `tokens` and collapses their [`NlpToken::ner`]
/// BIO tags: a `B-` (or orphan/label-changing `I-`) tag opens a span, a matching
/// `I-` tag extends it, and an `O` tag (or missing tag) closes it. Token indices
/// in the returned spans are global positions in `tokens`; `char_span` and
/// `text` are taken from the tokens' absolute byte offsets into `text`.
fn merge_ner_spans(tokens: &[NlpToken], text: &str) -> Vec<NerEntity> {
    let mut entities: Vec<NerEntity> = Vec::new();
    // (label without BIO prefix, first token index, last token index)
    let mut current: Option<(String, usize, usize)> = None;

    let flush = |cur: &Option<(String, usize, usize)>, out: &mut Vec<NerEntity>| {
        if let Some((label, first, last)) = cur {
            let char_start = tokens[*first].start;
            let char_end = tokens[*last].end;
            out.push(NerEntity {
                text: text.get(char_start..char_end).unwrap_or("").to_string(),
                label: label.clone(),
                token_span: (*first, *last),
                char_span: (char_start, char_end),
            });
        }
    };

    for (i, tok) in tokens.iter().enumerate() {
        let tag = tok.ner.as_deref().unwrap_or("O");
        if tag == "O" || tag.is_empty() {
            flush(&current, &mut entities);
            current = None;
            continue;
        }
        // Tags without a prefix are treated as span openers, mirroring
        // `decode_srl_frame`.
        let (prefix, base) = tag.split_once('-').unwrap_or(("B", tag));
        match current {
            Some((ref label, _, ref mut last)) if prefix == "I" && label.as_str() == base => {
                *last = i;
            }
            _ => {
                flush(&current, &mut entities);
                current = Some((base.to_string(), i, i));
            }
        }
    }
    flush(&current, &mut entities);
    entities
}

/// Decode a BIO-tagged SRL span sequence into one [`SrlFrame`].
///
/// Returns `None` if no role span was detected. Token spans use **global**
/// NlpResult token indices via `chunk_token_global_indices`.
fn decode_srl_frame(
    predicate_local: usize,
    srl_indices: &[usize],
    chunk_token_global_indices: &[Option<usize>],
    chunk_offsets: &[(usize, usize)],
    chunk_special: &[u32],
    srl_labels: &[String],
) -> Option<SrlFrame> {
    let predicate_global = *chunk_token_global_indices.get(predicate_local)?.as_ref()?;
    let mut roles: Vec<SrlRole> = Vec::new();
    let mut current: Option<(String, usize, usize)> = None; // (label, start_global, end_global)

    let close = |cur: &Option<(String, usize, usize)>, roles: &mut Vec<SrlRole>| {
        if let Some((label, s, e)) = cur {
            roles.push(SrlRole {
                span: (*s, *e),
                label: label.clone(),
            });
        }
    };

    for (local_i, &class_id) in srl_indices.iter().enumerate() {
        if local_i >= chunk_special.len() || chunk_special[local_i] != 0 {
            close(&current, &mut roles);
            current = None;
            continue;
        }
        let global = chunk_token_global_indices.get(local_i).and_then(|x| *x);
        if global.is_none() || chunk_offsets[local_i] == (0, 0) {
            close(&current, &mut roles);
            current = None;
            continue;
        }
        let global = global.unwrap();
        let raw_label = srl_labels
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("UNK_{class_id}"));

        if raw_label == "O" || raw_label == "V" {
            close(&current, &mut roles);
            current = None;
            continue;
        }

        // BIO scheme: "B-FOO" starts a new span; "I-FOO" continues if the
        // previous tag was matching "B-FOO" / "I-FOO".
        let (prefix, base) = raw_label
            .split_once('-')
            .map(|(a, b)| (a, b.to_string()))
            .unwrap_or(("B", raw_label.clone()));

        match prefix {
            "B" => {
                close(&current, &mut roles);
                current = Some((base, global, global));
            }
            "I" => {
                if let Some((ref existing_label, _, ref mut end)) = current {
                    if existing_label == &base {
                        *end = global;
                    } else {
                        // Mismatched I- tag — treat as a fresh span.
                        close(&current, &mut roles);
                        current = Some((base, global, global));
                    }
                } else {
                    current = Some((base, global, global));
                }
            }
            _ => {
                close(&current, &mut roles);
                current = None;
            }
        }
    }
    close(&current, &mut roles);

    if roles.is_empty() {
        None
    } else {
        Some(SrlFrame {
            predicate_token: predicate_global,
            predicate_sense: None,
            roles,
        })
    }
}

// ---------------------------------------------------------------------------
// HF download + session build (parallel structure to embedding.rs)
// ---------------------------------------------------------------------------

async fn download_nlp_artifacts(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
    tokenizer_path: &str,
    label_maps_path: &str,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        }
    })?;
    let model_file =
        api_repo
            .get(onnx_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download ONNX model '{onnx_path}': {e}"),
            })?;
    let tokenizer_file =
        api_repo
            .get(tokenizer_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download tokenizer '{tokenizer_path}': {e}"),
            })?;
    let label_file =
        api_repo
            .get(label_maps_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download label maps '{label_maps_path}': {e}"),
            })?;
    Ok((model_file, tokenizer_file, label_file))
}

fn build_session(
    path: &Path,
    spec: &ModelAliasSpec,
    execution_providers: Option<&[OnnxExecutionProvider]>,
) -> Result<Session> {
    let builder = new_session_builder()?.map_err(|e| RuntimeError::OnnxLoadFailure {
        alias: spec.alias.clone(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    })?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;
    let dispatch = build_execution_providers(execution_providers, &spec.alias, "local/onnx")?;
    builder =
        builder
            .with_execution_providers(dispatch)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
    builder
        .commit_from_file(path)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Output-tensor extraction helpers — strip the always-1 batch dim.
// ---------------------------------------------------------------------------

fn extract_3d_first_batch(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array2<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr3 = view.into_dimensionality::<ndarray::Ix3>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 3: {e}"),
        }
    })?;
    let (b, s, c) = (arr3.shape()[0], arr3.shape()[1], arr3.shape()[2]);
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    // Take batch 0 — always 1 in our serial-per-request orchestration.
    let mut out = Array2::<f32>::zeros((s, c));
    for i in 0..s {
        for j in 0..c {
            out[[i, j]] = arr3[[0, i, j]];
        }
    }
    Ok(out)
}

fn extract_4d_first_batch(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array3<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr4: Array4<f32> = view
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 4: {e}"),
        })?
        .to_owned();
    let (b, s1, s2, c) = (
        arr4.shape()[0],
        arr4.shape()[1],
        arr4.shape()[2],
        arr4.shape()[3],
    );
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    let mut out = Array3::<f32>::zeros((s1, s2, c));
    for i in 0..s1 {
        for j in 0..s2 {
            for k in 0..c {
                out[[i, j, k]] = arr4[[0, i, j, k]];
            }
        }
    }
    Ok(out)
}

fn extract_2d_first_batch_1d(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array1<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr2 = view.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 2: {e}"),
        }
    })?;
    let (b, c) = (arr2.shape()[0], arr2.shape()[1]);
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    let mut out = Array1::<f32>::zeros(c);
    for j in 0..c {
        out[j] = arr2[[0, j]];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Decode-path contract tests.
    //!
    //! These pin the pure tensor-shape helpers that turn the cascade's raw
    //! score arrays into NLP labels. They need no model download or ONNX
    //! runtime, so unlike the `#[ignore]`d e2e test they run on every
    //! `provider-onnx` build. The original `argmax_axis` bug shipped because
    //! this whole family of functions had only model-gated coverage; each
    //! test below guards one axis/shape contract that a "leftover 3-D mental
    //! model" could silently break.
    use super::*;
    use ndarray::{Array3, array};

    /// DEP head decode picks the best head per token via per-row argmax.
    ///
    /// `arc_scores` is `[seq, seq]` (token rows x candidate-head columns);
    /// regression guard for the bug where the call site asked for a
    /// nonexistent axis on the 2-D score matrix and failed every input.
    #[test]
    fn dep_decode_uses_per_row_argmax_over_candidate_heads() {
        // Row 0's max is column 2; row 1's max is column 0.
        let arc_scores = array![[0.1_f32, 0.2, 0.9], [0.7, 0.3, 0.1]];
        assert_eq!(argmax_last_axis(&arc_scores).unwrap(), vec![2, 0]);
    }

    /// DEP relation decode indexes `label_scores[[token, head, class]]`.
    ///
    /// `label_scores` is `[seq, seq, n_rel]`; for each token it must read the
    /// slice at its already-decoded head and argmax over the relation axis.
    /// A transposed axis order here would not error — it would silently emit
    /// wrong relations — so this pins the exact `[i, h, ..]` contract.
    #[test]
    fn dep_relation_decode_reads_slice_at_decoded_head() {
        // seq=2, n_rel=3. Token 0's head is 1, token 1's head is 0.
        // label_scores[[0, 1, ..]] -> class 1; label_scores[[1, 0, ..]] -> class 2.
        #[rustfmt::skip]
        let data = vec![
            0.0_f32, 0.0, 0.0, /* [0,0,..] unused */
            0.1, 0.9, 0.2,     /* [0,1,..] head of token 0 -> class 1 */
            0.5, 0.1, 0.8,     /* [1,0,..] head of token 1 -> class 2 */
            0.0, 0.0, 0.0,     /* [1,1,..] unused */
        ];
        let label_scores = Array3::from_shape_vec((2, 2, 3), data).unwrap();
        let dep_heads = vec![1usize, 0];
        assert_eq!(
            dep_relation_per_token(&label_scores, &dep_heads, 3).unwrap(),
            vec![1, 2]
        );
    }

    /// DEP relation decode rejects a head vector whose length != seq.
    #[test]
    fn dep_relation_decode_errors_on_seq_mismatch() {
        let label_scores = Array3::<f32>::zeros((2, 2, 3));
        let dep_heads = vec![0usize]; // len 1, seq 2
        assert!(dep_relation_per_token(&label_scores, &dep_heads, 3).is_err());
    }

    /// CLS (dialog-act) decode is a plain argmax over a 1-D logit vector.
    #[test]
    fn cls_decode_is_argmax_over_1d_logits() {
        assert_eq!(argmax_1d(&array![0.2_f32, 0.7, 0.1]), 1);
    }

    /// CLS softmax returns a full distribution that sums to ~1.0 (R1).
    #[test]
    fn softmax_full_returns_normalized_distribution() {
        // Two equal logits -> 0.5 each.
        let dist = softmax_full(&array![1.0_f32, 1.0]);
        assert_eq!(dist.len(), 2);
        assert!(
            (dist[0] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            dist[0]
        );
        assert!(
            (dist.iter().sum::<f32>() - 1.0).abs() < 1e-6,
            "must sum to 1.0"
        );
        // The argmax probability equals the winning class's score.
        let logits = array![0.1_f32, 2.0, 0.3];
        let dist = softmax_full(&logits);
        assert_eq!(argmax_1d(&logits), 1);
        assert!(dist[1] > dist[0] && dist[1] > dist[2]);
    }

    /// SRL BIO decode collapses `B-`/`I-` runs into one span at global indices.
    #[test]
    fn srl_decode_collapses_bio_run_into_one_span() {
        // Chunk: [CLS], tok1, tok2, verb. Labels: O / B-ARG0 / I-ARG0 / V.
        let srl_labels = ["O", "B-ARG0", "I-ARG0", "V"].map(String::from);
        let srl_indices = [0usize, 1, 2, 3];
        let chunk_special = [1u32, 0, 0, 0];
        let chunk_offsets = [(0usize, 0usize), (0, 3), (4, 7), (8, 10)];
        let chunk_token_global_indices = [None, Some(0usize), Some(1), Some(2)];

        let frame = decode_srl_frame(
            3, // predicate is the verb token
            &srl_indices,
            &chunk_token_global_indices,
            &chunk_offsets,
            &chunk_special,
            &srl_labels,
        )
        .expect("one role span");

        assert_eq!(frame.predicate_token, 2);
        assert_eq!(frame.roles.len(), 1);
        assert_eq!(frame.roles[0].label, "ARG0");
        assert_eq!(frame.roles[0].span, (0, 1));
    }

    /// DEP heads remap from model token space to global indices, with both root
    /// conventions (self-loop and head-on-special) mapping to `None` (R2).
    #[test]
    fn dep_head_remaps_to_global_index_with_root_as_none() {
        // Model sequence: [CLS], tok0, tok1. Special at 0; tok0->global 0,
        // tok1->global 1.
        let map = [None, Some(0usize), Some(1usize)];
        // Self-loop is the cascade's root marker: tok0 (model idx 1) heads itself.
        assert_eq!(remap_dep_head(1, 1, &map), None);
        // A head landing on `[CLS]` (a special) is also root.
        assert_eq!(remap_dep_head(0, 2, &map), None);
        // tok1 (model idx 2) heads tok0 (model idx 1) -> global token 0.
        assert_eq!(remap_dep_head(1, 2, &map), Some(0));
        // tok0 (model idx 1) heads tok1 (model idx 2) -> global token 1.
        assert_eq!(remap_dep_head(2, 1, &map), Some(1));
        // Out-of-range head (defensive) is treated as root, not a panic.
        assert_eq!(remap_dep_head(99, 1, &map), None);
    }

    /// DEP head indices are global across chunks, not chunk-local (R2).
    ///
    /// Simulates a second chunk whose first real token is already global #5;
    /// remapping must yield those global indices, never restart at 0.
    #[test]
    fn dep_head_remap_is_global_across_chunks() {
        // Second chunk: [CLS], tokA, tokB with globals 5 and 6.
        let map = [None, Some(5usize), Some(6usize)];
        assert_eq!(remap_dep_head(2, 1, &map), Some(6)); // tokA heads tokB
        assert_eq!(remap_dep_head(1, 2, &map), Some(5)); // tokB heads tokA
        assert_eq!(remap_dep_head(1, 1, &map), None); // tokA is root (self-loop)
    }

    /// NER BIO tags collapse into entity spans, handling orphan-`I-` and label
    /// changes, while `O` closes the current span (R3).
    #[test]
    fn merge_ner_spans_collapses_bio_runs() {
        let text = "Ada Lovelace met IBM today";
        //          0         1         2
        //          0123456789012345678901234567
        let tags = [
            ("Ada", "B-PERSON"),
            ("Lovelace", "I-PERSON"),
            ("met", "O"),
            ("IBM", "B-ORG"),
            ("today", "O"),
        ];
        let mut tokens = Vec::new();
        for (surface, tag) in tags {
            let start = text.find(surface).unwrap();
            tokens.push(NlpToken {
                text: surface.to_string(),
                start,
                end: start + surface.len(),
                ner: Some(tag.to_string()),
                ..Default::default()
            });
        }

        let entities = merge_ner_spans(&tokens, text);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].label, "PERSON");
        assert_eq!(entities[0].text, "Ada Lovelace");
        assert_eq!(entities[0].token_span, (0, 1));
        assert_eq!(entities[0].char_span, (0, "Ada Lovelace".len()));
        assert_eq!(entities[1].label, "ORG");
        assert_eq!(entities[1].text, "IBM");
        assert_eq!(entities[1].token_span, (3, 3));

        // An orphan `I-` tag opens a fresh span rather than being dropped.
        let orphan = NlpToken {
            text: "Paris".to_string(),
            end: 5,
            ner: Some("I-GPE".to_string()),
            ..Default::default()
        };
        let entities = merge_ner_spans(std::slice::from_ref(&orphan), "Paris");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].label, "GPE");
    }

    // ---- softmax_full: degenerate and numeric-stability variations (R1) ----

    /// Softmax of a single logit is the point mass `[1.0]`.
    #[test]
    fn softmax_full_single_element_is_unit() {
        assert_eq!(softmax_full(&array![42.0_f32]), vec![1.0]);
    }

    /// Softmax of an empty logit vector is empty, not a panic.
    #[test]
    fn softmax_full_empty_is_empty() {
        assert!(softmax_full(&Array1::<f32>::zeros(0)).is_empty());
    }

    /// Large logits do not overflow to NaN/Inf thanks to max-subtraction.
    #[test]
    fn softmax_full_is_numerically_stable_for_large_logits() {
        let dist = softmax_full(&array![1000.0_f32, 1001.0, 999.0]);
        assert!(dist.iter().all(|p| p.is_finite()), "no NaN/Inf: {dist:?}");
        assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax_1d(&array![1000.0_f32, 1001.0, 999.0]), 1);
        assert!(dist[1] > dist[0] && dist[1] > dist[2]);
    }

    /// Negative logits still produce a valid distribution.
    #[test]
    fn softmax_full_handles_negative_logits() {
        let dist = softmax_full(&array![-5.0_f32, -1.0, -3.0]);
        assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(dist[1] > dist[0] && dist[1] > dist[2]);
    }

    // ---- is_word_boundary: every branch (R4) ----

    /// Same tokenizer word id => same word; different id => new word.
    #[test]
    fn word_boundary_follows_tokenizer_word_ids() {
        // Same id -> continuation (byte gap is ignored when ids agree).
        assert!(!is_word_boundary(Some(3), Some(3), 4, 10));
        // Different id -> new word (even when bytes are contiguous).
        assert!(is_word_boundary(Some(3), Some(4), 4, 4));
    }

    /// With no word ids, a byte gap (whitespace) opens a new word; contiguity
    /// continues the current word.
    #[test]
    fn word_boundary_falls_back_to_byte_gap() {
        // Contiguous: token starts exactly where the previous ended.
        assert!(!is_word_boundary(None, None, 5, 5));
        // Gap: a space separated the tokens.
        assert!(is_word_boundary(None, None, 5, 6));
        // A mixed Some/None pair also falls back to the byte-gap rule.
        assert!(is_word_boundary(Some(1), None, 5, 7));
        assert!(!is_word_boundary(None, Some(1), 5, 5));
    }

    // ---- merge_ner_spans: empties, label changes, trailing spans (R3) ----

    fn ner_token(text: &str, start: usize, tag: Option<&str>) -> NlpToken {
        NlpToken {
            text: text.to_string(),
            start,
            end: start + text.len(),
            ner: tag.map(str::to_string),
            ..Default::default()
        }
    }

    /// No tokens and all-`O` tokens both yield zero entities.
    #[test]
    fn merge_ner_spans_yields_nothing_without_entities() {
        assert!(merge_ner_spans(&[], "").is_empty());
        let toks = [ner_token("a", 0, Some("O")), ner_token("b", 2, Some("O"))];
        assert!(merge_ner_spans(&toks, "a b").is_empty());
    }

    /// A missing (`None`) tag behaves like `O` and closes the current span.
    #[test]
    fn merge_ner_spans_treats_missing_tag_as_outside() {
        // "B-PERSON", then None, then "I-PERSON": the None breaks the run, so
        // the trailing I- opens a *separate* one-token span.
        let toks = [
            ner_token("Al", 0, Some("B-PERSON")),
            ner_token("gap", 3, None),
            ner_token("Bo", 7, Some("I-PERSON")),
        ];
        let ents = merge_ner_spans(&toks, "Al gap Bo");
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].token_span, (0, 0));
        assert_eq!(ents[1].token_span, (2, 2));
    }

    /// Adjacent spans without an `O` between them split on label change and on
    /// a fresh `B-`, and a span open at the final token is still flushed.
    #[test]
    fn merge_ner_spans_splits_on_label_change_and_flushes_trailing() {
        // B-PERSON, I-ORG (label change -> new span), B-ORG (new span),
        // I-ORG (extends the last one, which is open at the final token).
        let toks = [
            ner_token("A", 0, Some("B-PERSON")),
            ner_token("B", 2, Some("I-ORG")),
            ner_token("C", 4, Some("B-ORG")),
            ner_token("D", 6, Some("I-ORG")),
        ];
        let ents = merge_ner_spans(&toks, "A B C D");
        assert_eq!(ents.len(), 3);
        assert_eq!(
            (ents[0].label.as_str(), ents[0].token_span),
            ("PERSON", (0, 0))
        );
        assert_eq!(
            (ents[1].label.as_str(), ents[1].token_span),
            ("ORG", (1, 1))
        );
        // Trailing ORG span spans the last two tokens.
        assert_eq!(
            (ents[2].label.as_str(), ents[2].token_span),
            ("ORG", (2, 3))
        );
    }

    /// Two consecutive `B-` tags of the same type are two separate entities.
    #[test]
    fn merge_ner_spans_consecutive_b_tags_are_separate() {
        let toks = [
            ner_token("Sun", 0, Some("B-ORG")),
            ner_token("Moon", 4, Some("B-ORG")),
        ];
        let ents = merge_ner_spans(&toks, "Sun Moon");
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "Sun");
        assert_eq!(ents[1].text, "Moon");
    }

    /// A bare tag with no BIO prefix (e.g. `"PERSON"`) is treated as an opener.
    #[test]
    fn merge_ner_spans_treats_prefixless_tag_as_opener() {
        let toks = [ner_token("Ada", 0, Some("PERSON"))];
        let ents = merge_ner_spans(&toks, "Ada");
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].label, "PERSON");
        assert_eq!(ents[0].token_span, (0, 0));
    }

    // ---- decode_srl_frame: empties, multi-arg, mismatched-I (R6) ----

    /// All-`O` (plus the predicate `V`) yields no frame.
    #[test]
    fn srl_decode_returns_none_without_roles() {
        let srl_labels = ["O", "V"].map(String::from);
        // [CLS], tok, verb -> O, O, V. No argument spans.
        let frame = decode_srl_frame(
            2,
            &[0usize, 0, 1],
            &[None, Some(0usize), Some(1)],
            &[(0usize, 0usize), (0, 3), (4, 8)],
            &[1u32, 0, 0],
            &srl_labels,
        );
        assert!(frame.is_none());
    }

    /// Two `O`-separated argument runs decode into two distinct role spans.
    #[test]
    fn srl_decode_splits_two_arguments() {
        // [CLS] tokA tokB tokC verb : O B-ARG0 O B-ARG1 V
        let srl_labels = ["O", "B-ARG0", "B-ARG1", "V"].map(String::from);
        let frame = decode_srl_frame(
            4,
            &[0usize, 1, 0, 2, 3],
            &[None, Some(0usize), Some(1), Some(2), Some(3)],
            &[(0usize, 0usize), (0, 1), (2, 3), (4, 5), (6, 9)],
            &[1u32, 0, 0, 0, 0],
            &srl_labels,
        )
        .expect("two role spans");
        assert_eq!(frame.predicate_token, 3);
        assert_eq!(frame.roles.len(), 2);
        assert_eq!(
            (frame.roles[0].label.as_str(), frame.roles[0].span),
            ("ARG0", (0, 0))
        );
        assert_eq!(
            (frame.roles[1].label.as_str(), frame.roles[1].span),
            ("ARG1", (2, 2))
        );
    }

    /// A mismatched `I-` (different base than the open span) starts a fresh span.
    #[test]
    fn srl_decode_mismatched_i_starts_new_span() {
        // [CLS] tokA tokB verb : O B-ARG0 I-ARG1 V
        let srl_labels = ["O", "B-ARG0", "I-ARG1", "V"].map(String::from);
        let frame = decode_srl_frame(
            3,
            &[0usize, 1, 2, 3],
            &[None, Some(0usize), Some(1), Some(2)],
            &[(0usize, 0usize), (0, 1), (2, 3), (4, 7)],
            &[1u32, 0, 0, 0],
            &srl_labels,
        )
        .expect("two role spans");
        assert_eq!(frame.roles.len(), 2);
        assert_eq!(frame.roles[0].label, "ARG0");
        assert_eq!(frame.roles[1].label, "ARG1");
    }

    /// A predicate that maps to no global token (special slot) yields no frame.
    #[test]
    fn srl_decode_none_when_predicate_not_a_token() {
        let srl_labels = ["O", "B-ARG0", "V"].map(String::from);
        // Predicate local index 0 is `[CLS]` -> global None -> no frame.
        let frame = decode_srl_frame(
            0,
            &[2usize, 1, 0],
            &[None, Some(0usize), Some(1)],
            &[(0usize, 0usize), (0, 3), (4, 8)],
            &[1u32, 0, 0],
            &srl_labels,
        );
        assert!(frame.is_none());
    }

    // ---- label_maps_from_value: parsing and failure paths (R5) ----

    fn full_label_maps_json() -> Value {
        serde_json::json!({
            "pos": ["NOUN", "VERB"],
            "ner": ["O", "B-PERSON"],
            "deprel": ["root", "nsubj"],
            "srl": ["O", "B-ARG0", "V"],
            "cls": ["statement", "question"],
        })
    }

    /// A well-formed value parses into the five vocabularies in order.
    #[test]
    fn label_maps_from_value_parses_all_heads() {
        let maps = label_maps_from_value(&full_label_maps_json(), "test").expect("parse");
        assert_eq!(maps.pos, ["NOUN", "VERB"]);
        assert_eq!(maps.ner, ["O", "B-PERSON"]);
        assert_eq!(maps.deprel, ["root", "nsubj"]);
        assert_eq!(maps.srl, ["O", "B-ARG0", "V"]);
        assert_eq!(maps.cls, ["statement", "question"]);
    }

    /// A missing required key is a `Config` error naming the key and source.
    #[test]
    fn label_maps_from_value_errors_on_missing_key() {
        let mut value = full_label_maps_json();
        value.as_object_mut().unwrap().remove("cls");
        let err = label_maps_from_value(&value, "src.json").expect_err("missing cls");
        let msg = err.to_string();
        assert!(msg.contains("cls"), "names the key: {msg}");
        assert!(msg.contains("src.json"), "names the source: {msg}");
    }

    /// A key whose value is not an array is rejected.
    #[test]
    fn label_maps_from_value_errors_on_non_array_key() {
        let value = serde_json::json!({
            "pos": "NOUN", "ner": [], "deprel": [], "srl": [], "cls": [],
        });
        assert!(label_maps_from_value(&value, "src").is_err());
    }

    /// A non-string entry inside a vocabulary is rejected.
    #[test]
    fn label_maps_from_value_errors_on_non_string_entry() {
        let value = serde_json::json!({
            "pos": ["NOUN", 7], "ner": [], "deprel": [], "srl": [], "cls": [],
        });
        let err = label_maps_from_value(&value, "src").expect_err("non-string");
        assert!(err.to_string().contains("non-string"), "{err}");
    }

    /// `parse_label_maps` surfaces a read error for a missing file.
    #[test]
    fn parse_label_maps_errors_on_missing_file() {
        let err =
            parse_label_maps(Path::new("/no/such/label_maps.json")).expect_err("missing file");
        assert!(err.to_string().contains("Failed to read"), "{err}");
    }
}
