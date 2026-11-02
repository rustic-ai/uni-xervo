// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! kniv v7 structured NLP for `local/onnx`.
//!
//! Implements [`NlpModel`] against the kniv v7 family —
//! `dragonscale-ai/kniv-v7-xlarge-u-en` (UltraBERT-380M, 1,024 subwords) and
//! `dragonscale-ai/kniv-v7-xlarge-d-en` (DeBERTa-v3-large, 512 subwords).
//! The v1 cascade keeps its own path in [`super::nlp`]; nothing here changes
//! it.
//!
//! A v7 release is not one graph but eight, and the host code around them is
//! most of the model's behaviour. The split across this module's children
//! follows the reference runtime so the two can be read side by side:
//!
//! | module | reference |
//! |---|---|
//! | [`text`] | `windows.py` — units, sentences, words, windows |
//! | [`align`] | `align`, `word_ends`, `Tok.labels`, `marker_input` |
//! | [`decode`] | Viterbi, Chu-Liu-Edmonds, span and cluster decoding |
//! | [`entities`] | `entities.py` — mention grouping |
//! | [`meta`] | `onnx_meta.json`, `calibration.json`, `label_bank.json` |
//! | [`graphs`] | `OnnxBackend` — variant fallback and sessions |
//! | [`pipeline`] | `Kniv.window` and the methods it calls |
//!
//! # Byte offsets
//!
//! [`NlpResult::tokens`] carries UTF-8 byte offsets into the request's
//! input, as the v1 path does. For a [`NlpRequest::turns`] request there is
//! no single input string, so offsets are relative to the turn each word
//! came from; [`WindowSentence::unit`] says which turn that is.

mod align;
mod decode;
mod entities;
mod graphs;
mod meta;
mod pipeline;
mod text;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{build_execution_providers, parse_execution_providers_option};
use crate::traits::{
    DepLink, NerEntity, NlpCalibration, NlpLabelMaps, NlpModel, NlpRequest, NlpResult, NlpSentence,
    NlpTasks, NlpToken, NlpWindow, SpeechAct, SrlFrame, SrlRole,
};

use graphs::{GRAPH_NAMES, Graphs, variant_candidates, variant_chain};
use meta::OnnxMeta;
use pipeline::{LabelCache, Runner};
use text::{Unit, Word};

/// Where `onnx_meta.json` lives in a release, which is also how a v7 repo is
/// told apart from a v1 one.
pub(super) const META_PATH: &str = "onnx/onnx_meta.json";
const DEFAULT_TOKENIZER_PATH: &str = "tokenizer.json";
const DEFAULT_CALIBRATION_PATH: &str = "calibration.json";
const DEFAULT_LABEL_BANK_PATH: &str = "label_bank.json";

/// Which generation of the kniv NLP family a repo holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Generation {
    /// The 5-head cascade: one graph, 128 subwords.
    V1,
    /// kniv v7: eight graphs, word-level windows, schema extraction.
    V7,
}

/// Option keys that only mean something for the v1 cascade.
const V1_ONLY_KEYS: [&str; 3] = ["onnx_path", "label_maps_path", "max_seq_len"];
/// Option keys that only mean something for v7.
pub(super) const V7_ONLY_KEYS: [&str; 3] = ["variant", "calibration_path", "label_bank_path"];

/// A local release directory, if `model_id` names one.
///
/// Mirrors `raw::classify_model_source`: an absolute path, an explicitly
/// relative one, or a relative one that exists is a directory on disk;
/// anything else is a Hugging Face repo id.
fn local_release_dir(model_id: &str) -> Option<PathBuf> {
    let path = PathBuf::from(model_id);
    if path.is_absolute() || model_id.starts_with("./") || model_id.starts_with("../") {
        return path.is_dir().then_some(path);
    }
    path.is_dir().then_some(path)
}

/// Decide which generation `spec` names, without building any session.
///
/// Priority: an explicit `generation` option; then the v7-only option keys;
/// then the repo's own contents — `onnx/onnx_meta.json` for v7 against
/// `onnx/cascade.onnx` for v1. The probe costs one cached HTTP request for a
/// v1 alias and nothing at all for a local directory.
pub(super) async fn detect_generation(spec: &ModelAliasSpec) -> Result<Generation> {
    match spec.options.get("generation").and_then(Value::as_str) {
        Some("v1") => return Ok(Generation::V1),
        Some("v7") => return Ok(Generation::V7),
        Some(other) => {
            return Err(RuntimeError::Config(format!(
                "Option 'generation' for alias '{}' must be 'v1' or 'v7', got '{other}'",
                spec.alias
            )));
        }
        None => {}
    }
    if V7_ONLY_KEYS.iter().any(|k| spec.options.get(*k).is_some()) {
        return Ok(Generation::V7);
    }
    if let Some(dir) = local_release_dir(&spec.model_id) {
        return Ok(if dir.join(META_PATH).is_file() {
            Generation::V7
        } else {
            Generation::V1
        });
    }

    // A repo that serves `onnx/onnx_meta.json` is v7. Any failure here —
    // a 404, or no network — falls back to the v1 path, which then reports
    // its own download error naming the file it wanted.
    let cache_dir = resolve_cache_dir("onnx-nlp", &spec.model_id, &spec.options);
    match fetch(spec, &cache_dir, META_PATH).await {
        Ok(_) => Ok(Generation::V7),
        Err(e) => {
            debug!(
                alias = %spec.alias,
                model_id = %spec.model_id,
                cause = %e,
                "No kniv v7 metadata in repo; using the v1 cascade path"
            );
            Ok(Generation::V1)
        }
    }
}

/// Entry point from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// for a v7 repo.
pub(super) async fn load_nlp_v7(spec: &ModelAliasSpec) -> Result<Arc<dyn NlpModel>> {
    Ok(Arc::new(KnivV7Model::load(spec).await?) as Arc<dyn NlpModel>)
}

/// Fetch one file from the model repo.
async fn fetch(spec: &ModelAliasSpec, cache_dir: &Path, path: &str) -> Result<PathBuf> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: spec.alias.clone(),
            cause: e.to_string(),
        })?;
    let repo = match spec.revision.as_deref() {
        Some(rev) => Repo::with_revision(spec.model_id.clone(), RepoType::Model, rev.to_string()),
        None => Repo::model(spec.model_id.clone()),
    };
    api.repo(repo)
        .get(path)
        .await
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: spec.alias.clone(),
            cause: format!("Could not download '{path}': {e}"),
        })
}

/// Resolved artifact paths for one release.
struct Artifacts {
    meta: PathBuf,
    tokenizer: PathBuf,
    calibration: Option<PathBuf>,
    label_bank: Option<PathBuf>,
    graphs: graphs::GraphPaths,
}

/// Resolve every file the configured variant needs, downloading when the
/// model is a Hugging Face repo.
///
/// Only the variant's own graph files are fetched, so an INT8 configuration
/// never pulls the 1.5 GB fp32 encoders.
async fn resolve_artifacts(spec: &ModelAliasSpec, chain: &[String]) -> Result<Artifacts> {
    let opts = &spec.options;
    let get_str = |key: &str, default: &str| -> String {
        opts.get(key)
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_string()
    };
    let tokenizer_path = get_str("tokenizer_path", DEFAULT_TOKENIZER_PATH);
    let calibration_path = get_str("calibration_path", DEFAULT_CALIBRATION_PATH);
    let label_bank_path = get_str("label_bank_path", DEFAULT_LABEL_BANK_PATH);

    if let Some(dir) = local_release_dir(&spec.model_id) {
        let meta = dir.join(META_PATH);
        if !meta.is_file() {
            return Err(RuntimeError::OnnxModelNotFound {
                alias: spec.alias.clone(),
                path: meta,
            });
        }
        return Ok(Artifacts {
            meta,
            tokenizer: dir.join(&tokenizer_path),
            calibration: Some(dir.join(&calibration_path)).filter(|p| p.is_file()),
            label_bank: Some(dir.join(&label_bank_path)).filter(|p| p.is_file()),
            graphs: graphs::resolve_local_graphs(&dir.join("onnx"), chain, &spec.alias)?,
        });
    }

    let cache_dir = resolve_cache_dir("onnx-nlp", &spec.model_id, &spec.options);
    let meta = fetch(spec, &cache_dir, META_PATH).await?;
    let tokenizer = fetch(spec, &cache_dir, &tokenizer_path).await?;
    // Both are optional: a release without them decodes with the
    // reference's own defaults.
    let calibration = fetch(spec, &cache_dir, &calibration_path).await.ok();
    let label_bank = fetch(spec, &cache_dir, &label_bank_path).await.ok();

    let mut graph_paths = graphs::GraphPaths::new();
    for graph in GRAPH_NAMES {
        let candidates = variant_candidates(graph, chain);
        let last = candidates.len() - 1;
        let mut resolved = None;
        for (i, candidate) in candidates.iter().enumerate() {
            let remote = format!("onnx/{candidate}");
            match fetch(spec, &cache_dir, &remote).await {
                Ok(path) => {
                    resolved = Some(path);
                    break;
                }
                // Not the last candidate: this variant simply is not
                // published for this graph, which is the documented
                // fallback. The last candidate's failure is a real error.
                Err(e) if i < last => {
                    debug!(alias = %spec.alias, file = %remote, cause = %e, "variant not published");
                }
                Err(e) => return Err(e),
            }
        }
        let Some(path) = resolved else {
            return Err(RuntimeError::OnnxDownloadFailure {
                alias: spec.alias.clone(),
                cause: format!("No file found for kniv v7 graph '{graph}'"),
            });
        };
        graph_paths.insert(graph.to_string(), path);
    }

    Ok(Artifacts {
        meta,
        tokenizer,
        calibration,
        label_bank,
        graphs: graph_paths,
    })
}

/// A loaded kniv v7 release.
struct KnivV7Model {
    meta: OnnxMeta,
    tokenizer: tokenizers::Tokenizer,
    graphs: Graphs,
    calibration: NlpCalibration,
    label_cache: LabelCache,
    alias: String,
    model_id: String,
}

impl KnivV7Model {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        for key in V1_ONLY_KEYS {
            if spec.options.get(key).is_some() {
                return Err(RuntimeError::Config(format!(
                    "Option '{key}' does not apply to the kniv v7 model '{}' on alias '{}'; \
                     v7 reads its graph set and window size from {META_PATH}",
                    spec.model_id, spec.alias
                )));
            }
        }

        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        // Validate the EP list before anything expensive happens.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let variant = spec.options.get("variant").and_then(Value::as_str);
        let chain = variant_chain(variant);
        let artifacts = resolve_artifacts(spec, &chain).await?;

        let meta = OnnxMeta::load(&artifacts.meta)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&artifacts.tokenizer).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: artifacts.tokenizer.clone(),
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;
        let calibration = meta::load_calibration(
            artifacts.calibration.as_deref(),
            artifacts.label_bank.as_deref(),
        )?;
        if artifacts.calibration.is_none() {
            warn!(
                alias = %spec.alias,
                model_id = %spec.model_id,
                "No calibration.json in the kniv v7 release; decoding with \
                 reference defaults, which are not the published thresholds"
            );
        }

        // Fail now, not mid-request, if a release's `plain` graph no
        // longer declares the tensors the pipeline reads.
        let declared = meta.outputs_of("plain")?;
        for name in ["pos", "ner", "arc", "srl_pred", "cls", "ent", "span_words"] {
            if !declared.iter().any(|d| d == name) {
                return Err(RuntimeError::Config(format!(
                    "kniv v7 release '{}' on alias '{}' declares no 'plain' output \
                     '{name}'; this provider cannot serve it",
                    meta.run, spec.alias
                )));
            }
        }
        if !meta.declares_input("plain", "first_subword") {
            return Err(RuntimeError::Config(format!(
                "kniv v7 release '{}' on alias '{}' declares no 'first_subword' \
                 input on its 'plain' graph",
                meta.run, spec.alias
            )));
        }

        let graphs = Graphs::new(artifacts.graphs, spec, execution_providers);
        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            run = %meta.run,
            variant = variant.unwrap_or("fp32"),
            max_subwords = meta.max_subwords,
            plain = ?graphs.path_of("plain").and_then(Path::file_name),
            "Loading kniv v7 NLP model"
        );

        Ok(Self {
            meta,
            tokenizer,
            graphs,
            calibration,
            label_cache: LabelCache::default(),
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
        })
    }

    fn analyze_one(&self, request: &NlpRequest<'_>) -> Result<NlpResult> {
        let units: Vec<Unit<'_>> = match request.turns {
            Some(turns) => text::turns_as_units(turns),
            None => text::paragraphs(request.text),
        };
        let windows = text::build_windows(&units);

        let calibration = match request.calibration {
            Some(o) => self.calibration.with_overrides(o),
            None => self.calibration.clone(),
        };
        let runner = Runner {
            meta: &self.meta,
            tokenizer: &self.tokenizer,
            graphs: &self.graphs,
            calibration: &calibration,
            labels: &self.meta.labels,
            label_cache: &self.label_cache,
            alias: &self.alias,
        };

        let mut analysed = Vec::with_capacity(windows.len());
        for window in &windows {
            analysed.push(runner.window(window, request.tasks, request.schema)?);
        }
        Ok(flatten(&windows, &analysed, &units))
    }
}

impl crate::traits::ModelInfo for KnivV7Model {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl NlpModel for KnivV7Model {
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>> {
        // The graphs are batch-1 by construction, so a batch is served one
        // request at a time; parallelism belongs across requests, where each
        // graph's own `Mutex` lets a `marker` pass overlap another request's
        // `plain` pass.
        let mut results = Vec::with_capacity(requests.len());
        for req in &requests {
            results.push(self.analyze_one(req)?);
        }
        Ok(results)
    }

    fn supported_tasks(&self) -> NlpTasks {
        NlpTasks::ALL
    }

    fn label_maps(&self) -> Option<&NlpLabelMaps> {
        Some(&self.meta.labels)
    }

    fn calibration(&self) -> Option<&NlpCalibration> {
        Some(&self.calibration)
    }

    async fn warmup(&self) -> Result<()> {
        // One short pass builds the plain session and the heads a caller is
        // most likely to want; schema label vectors are cached on their
        // first real request.
        let request = NlpRequest::new(
            "Ada Lovelace was born in London in 1815.",
            NlpTasks::LINGUISTIC,
        );
        self.analyze_one(&request).map(|_| ())
    }
}

/// Project per-window results onto the v1-shaped [`NlpResult`] fields.
///
/// Callers that only read `tokens` / `sentences` / `frames` / `entities` keep
/// working against a v7 model; everything the projection cannot express —
/// coreference, typed entities, facts and their attributes — is in
/// [`NlpResult::windows`].
fn flatten(windows: &[text::Window], analysed: &[NlpWindow], units: &[Unit<'_>]) -> NlpResult {
    let mut out = NlpResult {
        windows: analysed.to_vec(),
        ..Default::default()
    };
    let mut token_base = 0usize;

    for (window, result) in windows.iter().zip(analysed) {
        // Per-word BIO tags, rebuilt from the window's entity spans so the
        // per-token `ner` field means what it does on the v1 path.
        let mut ner_tags: Vec<Option<String>> = vec![None; window.words.len()];
        for span in &result.ner {
            for i in span.start..span.end.min(ner_tags.len()) {
                let prefix = if i == span.start { "B" } else { "I" };
                ner_tags[i] = Some(format!("{prefix}-{}", span.label));
            }
        }

        for (i, word) in window.words.iter().enumerate() {
            let analysis = result.words.get(i);
            out.tokens.push(NlpToken {
                text: word.text.clone(),
                start: word.start,
                end: word.end,
                pos: analysis.and_then(|a| a.pos.clone()),
                ner: ner_tags[i].clone(),
                dep: analysis.and_then(|a| {
                    a.deprel.clone().map(|relation| DepLink {
                        // The window reports `0` for a root and otherwise a
                        // 1-based word index; the trait wants `None` and a
                        // global token index.
                        head: a.head.filter(|&h| h > 0).map(|h| token_base + h - 1),
                        relation,
                    })
                }),
                word_index: token_base + i,
            });
        }

        for sentence in &result.sentences {
            if sentence.end == sentence.start {
                continue;
            }
            let first = &window.words[sentence.start];
            let last = &window.words[sentence.end - 1];
            out.sentences.push(NlpSentence {
                token_range: (token_base + sentence.start, token_base + sentence.end - 1),
                start: first.start,
                end: last.end,
            });
            // v7 dialogue acts are multi-label sigmoids, so `confidence` is
            // the top probability and `scores` carries all six for callers
            // that want to threshold a class of their own.
            if !sentence.cls_scores.is_empty() {
                let top = decode::argmax(&sentence.cls_scores);
                out.speech_acts.push(SpeechAct {
                    sentence_index: out.sentences.len() - 1,
                    label: sentence
                        .cls
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "NONE".to_string()),
                    confidence: sentence.cls_scores[top],
                    scores: sentence.cls_scores.clone(),
                });
            }
        }

        for frame in &result.srl {
            out.frames.push(SrlFrame {
                predicate_token: token_base + frame.predicate,
                predicate_sense: None,
                roles: frame
                    .args
                    .iter()
                    .map(|arg| SrlRole {
                        span: (token_base + arg.start, token_base + arg.end - 1),
                        label: arg.label.clone(),
                    })
                    .collect(),
            });
        }

        for span in &result.ner {
            let unit = window
                .sentences
                .iter()
                .find(|s| s.start <= span.start && span.start < s.end)
                .map(|s| s.unit);
            out.entities.push(NerEntity {
                text: surface(units, unit, &window.words[span.start..span.end]),
                label: span.label.clone(),
                token_span: (token_base + span.start, token_base + span.end - 1),
                char_span: (
                    window.words[span.start].start,
                    window.words[span.end - 1].end,
                ),
            });
        }

        token_base += window.words.len();
    }
    out
}

/// The input's own bytes for a word span, falling back to the words joined
/// with single spaces when the span's unit cannot be sliced.
fn surface(units: &[Unit<'_>], unit: Option<usize>, words: &[Word]) -> String {
    if let (Some(idx), Some(first), Some(last)) = (unit, words.first(), words.last())
        && let Some(u) = units.get(idx)
        && let Some(start) = first.start.checked_sub(u.offset)
        && let Some(end) = last.end.checked_sub(u.offset)
        && let Some(slice) = u.text.get(start..end)
    {
        return slice.to_string();
    }
    words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{WindowSentence, WordAnalysis, WordSpan};

    fn spec(model_id: &str, options: Value) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: "nlp/test".to_string(),
            task: crate::api::ModelTask::Nlp,
            provider_id: "local/onnx".to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: crate::api::WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options,
        }
    }

    #[tokio::test]
    async fn generation_comes_from_the_explicit_option_first() {
        let s = spec("whatever/repo", serde_json::json!({"generation": "v7"}));
        assert_eq!(detect_generation(&s).await.unwrap(), Generation::V7);
        let s = spec(
            "whatever/repo",
            serde_json::json!({"generation": "v1", "variant": "int8"}),
        );
        assert_eq!(detect_generation(&s).await.unwrap(), Generation::V1);
        let s = spec("whatever/repo", serde_json::json!({"generation": "v9"}));
        assert!(detect_generation(&s).await.is_err());
    }

    #[tokio::test]
    async fn a_v7_only_option_implies_v7() {
        for key in V7_ONLY_KEYS {
            let s = spec("whatever/repo", serde_json::json!({key: "int8,fp16"}));
            assert_eq!(
                detect_generation(&s).await.unwrap(),
                Generation::V7,
                "{key}"
            );
        }
    }

    #[tokio::test]
    async fn a_local_directory_is_classified_by_its_contents() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = spec(&dir.path().display().to_string(), serde_json::json!({}));
        assert_eq!(detect_generation(&s).await.unwrap(), Generation::V1);

        std::fs::create_dir_all(dir.path().join("onnx")).expect("mkdir");
        std::fs::write(dir.path().join(META_PATH), b"{}").expect("write");
        assert_eq!(detect_generation(&s).await.unwrap(), Generation::V7);
    }

    #[tokio::test]
    async fn v1_only_options_are_rejected_for_a_v7_repo() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("onnx")).expect("mkdir");
        std::fs::write(dir.path().join(META_PATH), b"{}").expect("write");
        for key in V1_ONLY_KEYS {
            let s = spec(
                &dir.path().display().to_string(),
                serde_json::json!({key: if key == "max_seq_len" { Value::from(128) } else { Value::from("x") }}),
            );
            let err = match KnivV7Model::load(&s).await {
                Ok(_) => panic!("{key} should have been rejected"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    /// Two one-sentence windows, so the projection's global indexing is
    /// exercised rather than assumed.
    const FIXTURE_TEXT: &str = "Ada moved. She stayed.";

    fn fixture() -> (Vec<text::Window>, Vec<NlpWindow>, Vec<Unit<'static>>) {
        let units = vec![Unit {
            text: FIXTURE_TEXT,
            offset: 0,
            speaker: None,
        }];
        // Both sentences fit one window, so the two-window case — the one
        // where the projection's global indexing can actually be wrong — is
        // built by splitting it at the sentence boundary.
        let packed = text::build_windows(&units);
        let split_at = packed[0].sentences[1].start;
        let total = packed[0].words.len();
        let windows = vec![
            text::Window {
                words: packed[0].words[..split_at].to_vec(),
                sentences: vec![text::SentenceSpan {
                    start: 0,
                    end: split_at,
                    unit: 0,
                    speaker: None,
                }],
            },
            text::Window {
                words: packed[0].words[split_at..].to_vec(),
                sentences: vec![text::SentenceSpan {
                    start: 0,
                    end: total - split_at,
                    unit: 0,
                    speaker: None,
                }],
            },
        ];
        let analysed: Vec<NlpWindow> = windows
            .iter()
            .enumerate()
            .map(|(wi, w)| NlpWindow {
                tokens: w.words.iter().map(|x| x.text.clone()).collect(),
                analysed_words: w.words.len(),
                sentences: w
                    .sentences
                    .iter()
                    .map(|s| WindowSentence {
                        start: s.start,
                        end: s.end,
                        unit: s.unit,
                        cls_scores: vec![0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
                        cls: vec!["Inform".to_string()],
                        whole: true,
                        ..Default::default()
                    })
                    .collect(),
                words: w
                    .words
                    .iter()
                    .enumerate()
                    .map(|(i, _)| WordAnalysis {
                        pos: Some("X".to_string()),
                        // Word 0 roots; the rest attach to word 1 (1-based).
                        head: Some(if i == 0 { 0 } else { 1 }),
                        deprel: Some("dep".to_string()),
                        ..Default::default()
                    })
                    .collect(),
                ner: if wi == 0 {
                    vec![WordSpan {
                        label: "PERSON".to_string(),
                        start: 0,
                        end: 1,
                    }]
                } else {
                    Vec::new()
                },
                ..Default::default()
            })
            .collect();
        (windows, analysed, units)
    }

    #[test]
    fn projection_indexes_tokens_globally_across_windows() {
        let (windows, analysed, units) = fixture();
        assert_eq!(windows.len(), 2, "two paragraphs, two windows");
        let r = flatten(&windows, &analysed, &units);

        let texts: Vec<&str> = r.tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["Ada", "moved", ".", "She", "stayed", "."]);
        // `word_index` is dense and global, not window-local.
        assert_eq!(
            r.tokens.iter().map(|t| t.word_index).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        // The second window's sentence is reported at global token indices.
        assert_eq!(r.sentences.len(), 2);
        assert_eq!(r.sentences[1].token_range, (3, 5));
        assert_eq!(r.windows.len(), 2, "the full analysis is kept too");
    }

    #[test]
    fn projection_maps_roots_to_none_and_heads_to_global_indices() {
        let (windows, analysed, units) = fixture();
        let r = flatten(&windows, &analysed, &units);
        assert_eq!(r.tokens[0].dep.as_ref().unwrap().head, None, "a root");
        assert_eq!(r.tokens[1].dep.as_ref().unwrap().head, Some(0));
        // Window 2, word 1: head 1 (1-based) is that window's word 0, which
        // is global token 3.
        assert_eq!(r.tokens[4].dep.as_ref().unwrap().head, Some(3));
    }

    #[test]
    fn projection_rebuilds_bio_tags_and_byte_offsets() {
        let (windows, analysed, units) = fixture();
        let r = flatten(&windows, &analysed, &units);
        assert_eq!(r.tokens[0].ner.as_deref(), Some("B-PERSON"));
        assert_eq!(r.tokens[1].ner, None);
        assert_eq!(r.entities.len(), 1);
        assert_eq!(r.entities[0].text, "Ada");
        assert_eq!(r.entities[0].char_span, (0, 3));
        // The second paragraph's offsets are absolute in the input text.
        let she = &r.tokens[3];
        assert_eq!(FIXTURE_TEXT.get(she.start..she.end), Some("She"));
    }

    #[test]
    fn projection_reports_multi_label_dialogue_acts_with_all_scores() {
        let (windows, analysed, units) = fixture();
        let r = flatten(&windows, &analysed, &units);
        assert_eq!(r.speech_acts.len(), 2);
        assert_eq!(r.speech_acts[0].label, "Inform");
        assert_eq!(r.speech_acts[0].scores.len(), 6);
        assert!((r.speech_acts[0].confidence - 0.9).abs() < 1e-6);
        assert_eq!(r.speech_acts[1].sentence_index, 1);
    }
}
