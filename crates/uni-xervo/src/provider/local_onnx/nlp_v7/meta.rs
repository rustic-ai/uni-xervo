// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! The three JSON side-files a kniv v7 release ships.
//!
//! `onnx/onnx_meta.json` is the contract: it names every graph's inputs and
//! outputs, the label vocabularies, the subword budget and the word-pooling
//! mode. Reading it is what lets one provider serve both released encoders,
//! whose budgets (1,024 vs 512 subwords) and pooling (`mean` vs `first`)
//! differ. `calibration.json` carries the dev-tuned decoding thresholds, and
//! `label_bank.json` says which relations those thresholds were tuned for.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::Value;

use crate::error::{Result, RuntimeError};
use crate::traits::{NlpCalibration, NlpLabelMaps};

/// How the encoder pools a word's subwords.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WordPool {
    /// Mean over the word's subwords: the graph takes a `word_end` input.
    Mean,
    /// The word's first subword only: no `word_end` input.
    First,
}

/// `onnx/onnx_meta.json`.
#[derive(Debug, Clone)]
pub(crate) struct OnnxMeta {
    /// The release identifier, e.g. `ultrabert-380m-v7b8ev`.
    pub run: String,
    /// Subwords the encoder accepts per window; words past the cut are not
    /// analysed.
    pub max_subwords: usize,
    /// Subword pooling mode.
    pub word_pool: WordPool,
    /// Padding token id, for the label graph's batches.
    pub pad_token_id: i64,
    /// Separator token id, appended when a window is cut.
    pub sep_token_id: i64,
    /// Each graph's declared output names, in order — the graphs return
    /// positional tensors, so this is how outputs get their names.
    pub graph_outputs: HashMap<String, Vec<String>>,
    /// Each graph's declared input names. Used to feed a session only what
    /// it declares, which is how one code path serves both encoders.
    pub graph_inputs: HashMap<String, Vec<String>>,
    /// The label vocabularies, as [`NlpLabelMaps`].
    pub labels: NlpLabelMaps,
}

fn cfg(msg: impl Into<String>) -> RuntimeError {
    RuntimeError::Config(msg.into())
}

fn string_list(value: &Value, key: &str, source: &str) -> Result<Vec<String>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| cfg(format!("{source} is missing required array '{key}'")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| cfg(format!("{source}: '{key}' has a non-string entry")))
        })
        .collect()
}

/// Optional vocabulary: absent keys give an empty list rather than an error,
/// so a release that drops a head still loads.
fn optional_string_list(value: &Value, key: &str, source: &str) -> Result<Vec<String>> {
    if value.get(key).is_none() {
        return Ok(Vec::new());
    }
    string_list(value, key, source)
}

impl OnnxMeta {
    /// Parse `onnx/onnx_meta.json`.
    pub fn load(path: &Path) -> Result<Self> {
        let source = path.display().to_string();
        let bytes = std::fs::read(path)
            .map_err(|e| cfg(format!("Failed to read kniv v7 metadata at {source}: {e}")))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| cfg(format!("Failed to parse kniv v7 metadata at {source}: {e}")))?;
        Self::from_value(&value, &source)
    }

    /// Parse an already-decoded `onnx_meta.json`. Split out from file IO so
    /// the failure paths are unit-testable.
    pub fn from_value(value: &Value, source: &str) -> Result<Self> {
        let u64_at = |key: &str| -> Result<u64> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| cfg(format!("{source} is missing integer '{key}'")))
        };
        let word_pool = match value.get("word_pool").and_then(Value::as_str) {
            Some("mean") => WordPool::Mean,
            Some("first") => WordPool::First,
            Some(other) => {
                return Err(cfg(format!(
                    "{source}: unknown word_pool '{other}'; expected 'mean' or 'first'"
                )));
            }
            None => return Err(cfg(format!("{source} is missing string 'word_pool'"))),
        };

        let graphs = value
            .get("graphs")
            .and_then(Value::as_object)
            .ok_or_else(|| cfg(format!("{source} is missing object 'graphs'")))?;
        let mut graph_outputs = HashMap::new();
        let mut graph_inputs = HashMap::new();
        for (name, spec) in graphs {
            let where_ = format!("{source}: graphs.{name}");
            graph_inputs.insert(name.clone(), string_list(spec, "inputs", &where_)?);
            graph_outputs.insert(name.clone(), string_list(spec, "outputs", &where_)?);
        }

        let labels_value = value
            .get("labels")
            .ok_or_else(|| cfg(format!("{source} is missing object 'labels'")))?;
        let where_ = format!("{source}: labels");
        let labels = NlpLabelMaps {
            pos: string_list(labels_value, "pos", &where_)?,
            ner: string_list(labels_value, "ner", &where_)?,
            deprel: string_list(labels_value, "dep", &where_)?,
            srl: string_list(labels_value, "srl", &where_)?,
            cls: string_list(labels_value, "cls", &where_)?,
            morph: optional_string_list(labels_value, "morph", &where_)?,
            lemma: optional_string_list(labels_value, "lemma", &where_)?,
            sentiment: optional_string_list(labels_value, "sentiment", &where_)?,
            ent_type: optional_string_list(labels_value, "ent_type", &where_)?,
            ..Default::default()
        };

        Ok(Self {
            run: value
                .get("run")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            max_subwords: u64_at("max_subwords")? as usize,
            word_pool,
            pad_token_id: u64_at("pad_token_id")? as i64,
            sep_token_id: u64_at("sep_token_id")? as i64,
            graph_outputs,
            graph_inputs,
            labels,
        })
    }

    /// Output names of `graph`, in positional order.
    pub fn outputs_of(&self, graph: &str) -> Result<&[String]> {
        self.graph_outputs
            .get(graph)
            .map(Vec::as_slice)
            .ok_or_else(|| cfg(format!("kniv v7 metadata declares no graph '{graph}'")))
    }

    /// Whether `graph` declares input `name`.
    pub fn declares_input(&self, graph: &str, name: &str) -> bool {
        self.graph_inputs
            .get(graph)
            .is_some_and(|v| v.iter().any(|i| i == name))
    }
}

/// Parse `calibration.json`, folding in the relations `label_bank.json`
/// marks as calibrated.
///
/// Relations tuned one by one get their own dev-tuned offset; the rest of
/// the calibrated set gets the global `rel_offset`; everything else gets
/// `-unseen_offset`. The two never stack.
pub(crate) fn load_calibration(
    calibration_path: Option<&Path>,
    label_bank_path: Option<&Path>,
) -> Result<NlpCalibration> {
    let cal: Value = match calibration_path {
        Some(p) if p.exists() => {
            let source = p.display().to_string();
            let bytes = std::fs::read(p)
                .map_err(|e| cfg(format!("Failed to read calibration at {source}: {e}")))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| cfg(format!("Failed to parse calibration at {source}: {e}")))?
        }
        _ => Value::Object(Default::default()),
    };
    let banked = match label_bank_path {
        Some(p) if p.exists() => calibrated_bank_relations(p)?,
        _ => HashSet::new(),
    };
    Ok(calibration_from_value(&cal, &banked))
}

/// The relation names `label_bank.json` marks `"calibrated": true`.
fn calibrated_bank_relations(path: &Path) -> Result<HashSet<String>> {
    let source = path.display().to_string();
    let bytes = std::fs::read(path)
        .map_err(|e| cfg(format!("Failed to read label bank at {source}: {e}")))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| cfg(format!("Failed to parse label bank at {source}: {e}")))?;
    Ok(value
        .get("relations")
        .and_then(Value::as_array)
        .map(|rels| {
            rels.iter()
                .filter(|r| r.get("calibrated").and_then(Value::as_bool) == Some(true))
                .filter_map(|r| r.get("name").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default())
}

/// Build the calibration from a decoded `calibration.json` and the set of
/// bank-calibrated relation names.
pub(crate) fn calibration_from_value(cal: &Value, banked: &HashSet<String>) -> NlpCalibration {
    let f32_at = |key: &str, default: f32| -> f32 {
        cal.get(key)
            .and_then(Value::as_f64)
            .map_or(default, |v| v as f32)
    };

    let mut rel_offsets: HashMap<String, f32> = HashMap::new();
    if let Some(global) = cal.get("rel_offset").and_then(Value::as_f64) {
        for name in banked {
            rel_offsets.insert(name.clone(), global as f32);
        }
    }
    if let Some(map) = cal.get("rel_offsets").and_then(Value::as_object) {
        for (name, value) in map {
            if let Some(v) = value.as_f64() {
                rel_offsets.insert(name.clone(), v as f32);
            }
        }
    }

    NlpCalibration {
        rel_offsets,
        unseen_offset: f32_at("unseen_offset", 0.0),
        offset: f32_at("offset", 0.0),
        rel_top: cal
            .get("rel_top")
            .and_then(Value::as_u64)
            .map_or(4, |v| v as usize),
        span_offset: f32_at("span_offset", 0.0),
        coref_link_bias: f32_at("coref_link_bias", 0.0),
        ent_o_bias: f32_at("ent_o_bias", 0.0),
        entity_coref: cal
            .get("entity_coref")
            .and_then(Value::as_str)
            .unwrap_or("all")
            .to_string(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta_value() -> Value {
        json!({
            "run": "ultrabert-380m-v7b8ev",
            "max_subwords": 1024,
            "word_pool": "mean",
            "pad_token_id": 0,
            "sep_token_id": 15,
            "graphs": {
                "plain": {"inputs": ["input_ids", "word_end"], "outputs": ["pos", "ner"]},
                "attr": {"inputs": ["pair"], "outputs": ["polarity"]}
            },
            "labels": {
                "pos": ["NOUN", "VERB"],
                "ner": ["O", "B-PER"],
                "dep": ["root", "nsubj"],
                "srl": ["O", "V"],
                "cls": ["Inform"],
                "morph": ["_"],
                "lemma": ["|0|"],
                "sentiment": ["positive", "negative", "neutral"],
                "ent_type": ["LOC"]
            }
        })
    }

    #[test]
    fn meta_parses_graphs_and_vocabularies() {
        let m = OnnxMeta::from_value(&meta_value(), "test").expect("parse");
        assert_eq!(m.max_subwords, 1024);
        assert_eq!(m.word_pool, WordPool::Mean);
        assert_eq!(m.sep_token_id, 15);
        assert_eq!(m.outputs_of("plain").unwrap(), ["pos", "ner"]);
        assert!(m.declares_input("plain", "word_end"));
        assert!(!m.declares_input("attr", "word_end"));
        assert_eq!(m.labels.sentiment, ["positive", "negative", "neutral"]);
        assert!(m.outputs_of("nope").is_err());
    }

    #[test]
    fn deberta_style_meta_declares_no_word_end() {
        let mut v = meta_value();
        v["word_pool"] = json!("first");
        v["max_subwords"] = json!(512);
        v["graphs"]["plain"]["inputs"] = json!(["input_ids"]);
        let m = OnnxMeta::from_value(&v, "test").expect("parse");
        assert_eq!(m.word_pool, WordPool::First);
        assert_eq!(m.max_subwords, 512);
        assert!(!m.declares_input("plain", "word_end"));
    }

    #[test]
    fn meta_rejects_a_missing_vocabulary_or_unknown_pooling() {
        let mut v = meta_value();
        v["labels"].as_object_mut().unwrap().remove("pos");
        assert!(OnnxMeta::from_value(&v, "test").is_err());

        let mut v = meta_value();
        v["word_pool"] = json!("max");
        let err = OnnxMeta::from_value(&v, "test").unwrap_err().to_string();
        assert!(err.contains("word_pool"), "{err}");
    }

    #[test]
    fn calibration_layers_bank_defaults_under_per_relation_offsets() {
        let cal = json!({
            "entity_coref": "none",
            "coref_link_bias": 2.0,
            "unseen_offset": -1.5,
            "rel_offset": 1.0,
            "span_offset": -2.0,
            "rel_offsets": {"capital": -1.0}
        });
        let banked: HashSet<String> = ["capital", "employer"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let c = calibration_from_value(&cal, &banked);
        // The bank default applies, then the explicit entry wins.
        assert_eq!(c.rel_offsets["employer"], 1.0);
        assert_eq!(c.rel_offsets["capital"], -1.0);
        assert_eq!(c.unseen_offset, -1.5);
        assert_eq!(c.span_offset, -2.0);
        assert_eq!(c.coref_link_bias, 2.0);
        assert_eq!(c.entity_coref, "none");
        // Unset keys take the reference's defaults, not zero for everything.
        assert_eq!(c.rel_top, 4);
        assert_eq!(c.offset, 0.0);
    }

    #[test]
    fn an_absent_calibration_still_yields_reference_defaults() {
        let c = calibration_from_value(&json!({}), &HashSet::new());
        assert_eq!(c.rel_top, 4);
        assert_eq!(c.entity_coref, "all");
        assert!(c.rel_offsets.is_empty());
    }
}
