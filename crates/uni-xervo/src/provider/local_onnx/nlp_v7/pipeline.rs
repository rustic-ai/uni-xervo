// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! The host pipeline: one window in, one full analysis out.
//!
//! A port of `Kniv.window` and the methods it calls. The order is fixed by
//! the graphs' data dependencies: the `plain` pass produces the per-word
//! representations every other graph consumes, dependency labels need
//! already-decoded heads, SRL roles need already-decoded predicates, entity
//! grouping needs coreference clusters, the relation pass needs the grouped
//! entities, and the attribute pass needs the kept facts' rows.
//!
//! Per window that is: `plain` once, then at most one call each to
//! `dep_rel`, `srl_roles`, `coref_spans`, `span`, `marker` and `attr`, plus
//! `label` once per schema never seen before. `marker` dominates when a
//! window has many entities, because it scores every one of the `E(E-1)`
//! ordered pairs.

use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::{Array2, Array3, ArrayD, IxDyn};

use crate::error::{Result, RuntimeError};
use crate::traits::{
    ExtractionSchema, FactModality, FactPolarity, LabelSpec, NlpCalibration, NlpEntity, NlpFact,
    NlpLabelMaps, NlpTasks, NlpWindow, WindowFrame, WindowSentence, WordAnalysis, WordSpan,
};

use super::align::{
    Alignment, SPAN_MAX_WIDTH, align, encode_labels, marker_input, relation_text, speaker_turns,
    type_text,
};
use super::decode::{
    NEG, apply_script, argmax, argmax_rows, bio_constraints, bio_spans, decode_spans, decode_trees,
    link_clusters, log_softmax, sigmoid, span_candidates, viterbi,
};
use super::entities::{CorefMode, Mention, build_entities, default_entity_tags};
use super::graphs::{GraphInput, Graphs, Outputs};
use super::meta::{OnnxMeta, WordPool};
use super::text::Window;

/// Schemas held in the label-embedding cache before it is cleared, as the
/// reference does.
const LABEL_CACHE_CAPACITY: usize = 256;

/// The names [`FactPolarity`] is decoded against, in head order.
const POLARITIES: [FactPolarity; 2] = [FactPolarity::Positive, FactPolarity::Negative];
/// The names [`FactModality`] is decoded against, in head order.
const MODALITIES: [FactModality; 6] = [
    FactModality::Asserted,
    FactModality::Hypothetical,
    FactModality::Planned,
    FactModality::Desired,
    FactModality::Questioned,
    FactModality::Reported,
];

/// Schema label embeddings, cached by the exact list of label texts.
///
/// A stable schema therefore costs one `label` pass for the lifetime of the
/// model; a schema that changes per request costs one pass per request,
/// which is why the adoption guidance is to keep schemas stable.
#[derive(Default)]
pub(crate) struct LabelCache {
    entries: Mutex<HashMap<Vec<String>, Array2<f32>>>,
}

/// Everything one window's analysis needs, borrowed from the loaded model.
pub(crate) struct Runner<'a> {
    /// Parsed `onnx_meta.json`.
    pub meta: &'a OnnxMeta,
    /// The encoder's tokenizer.
    pub tokenizer: &'a tokenizers::Tokenizer,
    /// The graph sessions.
    pub graphs: &'a Graphs,
    /// Decoding thresholds, already merged with any request overrides.
    pub calibration: &'a NlpCalibration,
    /// Label vocabularies.
    pub labels: &'a NlpLabelMaps,
    /// Shared label-embedding cache.
    pub label_cache: &'a LabelCache,
    /// The catalog alias, for error messages.
    pub alias: &'a str,
}

/// Which graph outputs a request's tasks need. Requesting only what is used
/// keeps the big per-word matrices (`coref_words`, `span_words`,
/// `srl_words`, each `W x 1024` floats) out of memory when nothing reads
/// them.
fn plain_outputs(tasks: NlpTasks, want_coref: bool) -> Vec<String> {
    let mut out: Vec<&str> = Vec::new();
    if tasks.contains(NlpTasks::POS) {
        out.push("pos");
    }
    if tasks.contains(NlpTasks::MORPH) {
        out.push("morph");
    }
    if tasks.contains(NlpTasks::LEMMA) {
        out.push("lemma");
    }
    if tasks.contains(NlpTasks::NER) {
        out.push("ner");
    }
    if tasks.contains(NlpTasks::DEP) {
        out.extend(["arc", "rel_dep", "rel_head"]);
    }
    if tasks.contains(NlpTasks::SRL) {
        out.extend(["srl_pred", "srl_words"]);
    }
    if tasks.contains(NlpTasks::CLS) {
        out.push("cls");
    }
    if tasks.contains(NlpTasks::SENTIMENT) {
        out.push("sentiment");
    }
    if want_coref {
        out.extend(["coref_words", "coref_scores", "coref_cand"]);
    }
    if tasks.contains(NlpTasks::ENTITIES) {
        out.extend(["ent", "span_words"]);
    }
    out.iter().map(|s| (*s).to_string()).collect()
}

impl Runner<'_> {
    /// Analyse one window.
    pub fn window(
        &self,
        w: &Window,
        tasks: NlpTasks,
        schema: Option<&ExtractionSchema>,
    ) -> Result<NlpWindow> {
        let tasks = tasks.expanded();
        let tokens: Vec<String> = w.words.iter().map(|x| x.text.clone()).collect();
        let spans: Vec<(usize, usize)> = w.sentences.iter().map(|s| (s.start, s.end)).collect();

        let alignment = align(
            &tokens,
            self.tokenizer,
            self.meta.max_subwords,
            self.meta.sep_token_id,
            self.alias,
        )?;
        let n_words = alignment.limit;
        let word_mask = alignment.word_mask();

        let mut sentence_of_word = vec![-1i64; n_words];
        for (j, &(s, e)) in spans.iter().enumerate() {
            for slot in &mut sentence_of_word[s.min(n_words)..e.min(n_words)] {
                *slot = j as i64;
            }
        }

        let mut out = NlpWindow {
            tokens: tokens.clone(),
            analysed_words: n_words,
            sentences: w
                .sentences
                .iter()
                .map(|s| WindowSentence {
                    start: s.start,
                    end: s.end,
                    unit: s.unit,
                    speaker: s.speaker.clone(),
                    whole: s.end <= n_words,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        if n_words == 0 {
            // A window whose words all fell past the subword budget: an
            // empty analysis, not an error.
            return Ok(out);
        }

        // Entity grouping only reads coreference in the modes that use it,
        // and both released models ship `entity_coref: "none"`.
        let coref_mode = CorefMode::parse(&self.calibration.entity_coref);
        let want_coref = tasks.contains(NlpTasks::COREF)
            || (tasks.contains(NlpTasks::ENTITIES) && coref_mode != CorefMode::None);

        let wanted = plain_outputs(tasks, want_coref);
        if wanted.is_empty() {
            return Ok(out);
        }
        let plain = self.run_plain(
            &alignment,
            &word_mask,
            &sentence_of_word,
            spans.len(),
            &wanted,
        )?;

        let coref = if want_coref {
            self.coref(&plain, &sentence_of_word, n_words)?
        } else {
            Vec::new()
        };

        if tasks.intersects(NlpTasks::LINGUISTIC) {
            self.linguistic(
                &plain,
                &tokens,
                &spans,
                &sentence_of_word,
                &word_mask,
                n_words,
                tasks,
                &mut out,
            )?;
        }
        if tasks.contains(NlpTasks::COREF) {
            out.coref = coref.clone();
        }

        if tasks.contains(NlpTasks::ENTITIES) {
            let grouped = self.entities(
                &plain,
                &tokens,
                &spans,
                &sentence_of_word,
                &word_mask,
                n_words,
                &coref,
                coref_mode,
                schema.and_then(|s| s.entity_types.as_deref()),
            )?;
            let (entities, facts) = self.relations(
                w,
                &tokens,
                &spans,
                &alignment,
                n_words,
                &grouped,
                schema.filter(|_| tasks.contains(NlpTasks::FACTS)),
            )?;
            out.entities = entities;
            out.facts = facts;
        }
        Ok(out)
    }

    fn run_plain(
        &self,
        alignment: &Alignment,
        word_mask: &[bool],
        sentence_of_word: &[i64],
        n_sentences: usize,
        wanted: &[String],
    ) -> Result<Outputs> {
        let n_subwords = alignment.ids.len();
        let declared = self
            .meta
            .graph_inputs
            .get("plain")
            .cloned()
            .unwrap_or_default();
        self.graphs.run(
            "plain",
            vec![
                ("input_ids", GraphInput::row_i64(alignment.ids.clone())),
                ("attention_mask", GraphInput::row_i64(vec![1; n_subwords])),
                (
                    "first_subword",
                    GraphInput::row_i64(alignment.first_subword()),
                ),
                // Mean pooling needs each word's subword span; the
                // first-subword encoder declares no such input, so the
                // backwards walk is skipped rather than computed and
                // dropped.
                (
                    "word_end",
                    GraphInput::row_i64(if self.meta.word_pool == WordPool::Mean {
                        alignment.word_ends()
                    } else {
                        Vec::new()
                    }),
                ),
                ("word_mask", GraphInput::row_bool(word_mask.to_vec())),
                (
                    "sent_of_word",
                    GraphInput::row_i64(sentence_of_word.to_vec()),
                ),
                (
                    "sent_ids",
                    GraphInput::vec_i64((0..n_sentences as i64).collect()),
                ),
            ],
            &declared,
            wanted,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linguistic(
        &self,
        plain: &Outputs,
        tokens: &[String],
        spans: &[(usize, usize)],
        sentence_of_word: &[i64],
        word_mask: &[bool],
        n_words: usize,
        tasks: NlpTasks,
        out: &mut NlpWindow,
    ) -> Result<()> {
        let pos = if tasks.contains(NlpTasks::POS) {
            Some(argmax_rows(plain.f32_batch0("pos")?.view()))
        } else {
            None
        };
        let morph = if tasks.contains(NlpTasks::MORPH) {
            Some(argmax_rows(plain.f32_batch0("morph")?.view()))
        } else {
            None
        };
        let lemma = if tasks.contains(NlpTasks::LEMMA) {
            Some(argmax_rows(plain.f32_batch0("lemma")?.view()))
        } else {
            None
        };

        let dep = if tasks.contains(NlpTasks::DEP) {
            let arc = plain.f32_batch0("arc")?;
            let heads = decode_trees(arc.view(), sentence_of_word, word_mask);
            let declared = self
                .meta
                .graph_inputs
                .get("dep_rel")
                .cloned()
                .unwrap_or_default();
            let rel = self.graphs.run(
                "dep_rel",
                vec![
                    ("rel_dep", GraphInput::f32(plain.passthrough("rel_dep")?)),
                    ("rel_head", GraphInput::f32(plain.passthrough("rel_head")?)),
                    ("head", GraphInput::row_i64(heads.clone())),
                ],
                &declared,
                &["dep_rel".to_string()],
            )?;
            let labels = argmax_rows(rel.f32_batch0("dep_rel")?.view());
            Some((heads, labels))
        } else {
            None
        };

        if pos.is_some() || morph.is_some() || lemma.is_some() || dep.is_some() {
            out.words = (0..n_words)
                .map(|i| WordAnalysis {
                    pos: pos.as_ref().map(|v| label_at(&self.labels.pos, v[i])),
                    morph: morph.as_ref().map(|v| label_at(&self.labels.morph, v[i])),
                    lemma: lemma
                        .as_ref()
                        .map(|v| apply_script(&tokens[i], &label_at(&self.labels.lemma, v[i]))),
                    head: dep.as_ref().map(|(h, _)| h[i].max(0) as usize),
                    deprel: dep
                        .as_ref()
                        .map(|(_, r)| label_at(&self.labels.deprel, r[i])),
                })
                .collect();
        }

        if tasks.contains(NlpTasks::NER) {
            let ner = plain.f32_batch0("ner")?;
            let (allowed, start) = bio_constraints(&self.labels.ner);
            for &(s, e) in spans.iter().filter(|&&(s, _)| s < n_words) {
                let slice = ner.slice(ndarray::s![s..e.min(n_words), ..]);
                let path = viterbi(log_softmax(slice).view(), &allowed, &start, None);
                let tags: Vec<String> = path
                    .iter()
                    .map(|&k| label_at(&self.labels.ner, k))
                    .collect();
                for (label, a, b) in bio_spans(&tags) {
                    out.ner.push(WordSpan {
                        label,
                        start: s + a,
                        end: s + b,
                    });
                }
            }
        }

        if tasks.contains(NlpTasks::SRL) {
            out.srl = self.srl(plain, sentence_of_word, word_mask, n_words)?;
        }

        if tasks.contains(NlpTasks::CLS) {
            let cls = plain.f32_batch0("cls")?;
            for (j, sentence) in out.sentences.iter_mut().enumerate() {
                if j >= cls.nrows() {
                    break;
                }
                // Multi-label sigmoids, not a softmax: a sentence can be
                // both an answer and a question.
                let scores: Vec<f32> = cls.row(j).iter().map(|&v| sigmoid(v)).collect();
                sentence.cls = scores
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p > 0.5)
                    .map(|(k, _)| label_at(&self.labels.cls, k))
                    .collect();
                sentence.cls_scores = scores;
            }
        }
        if tasks.contains(NlpTasks::SENTIMENT) {
            let sentiment = plain.f32_batch0("sentiment")?;
            for (j, sentence) in out.sentences.iter_mut().enumerate() {
                if j >= sentiment.nrows() {
                    break;
                }
                let row: Vec<f32> = sentiment.row(j).to_vec();
                sentence.sentiment = Some(label_at(&self.labels.sentiment, argmax(&row)));
            }
        }
        Ok(())
    }

    /// Semantic roles: one frame per predicate, scored against its own
    /// sentence only.
    fn srl(
        &self,
        plain: &Outputs,
        sentence_of_word: &[i64],
        word_mask: &[bool],
        n_words: usize,
    ) -> Result<Vec<WindowFrame>> {
        let pred_logits = plain.f32_batch0("srl_pred")?;
        let predicates: Vec<usize> = (0..n_words)
            .filter(|&i| word_mask[i] && argmax(&pred_logits.row(i).to_vec()) == 1)
            .collect();
        if predicates.is_empty() {
            return Ok(Vec::new());
        }
        let words = plain.f32_batch0("srl_words")?;
        let hidden = words.ncols();

        // Each predicate is scored over a contiguous slice covering its
        // sentence; words of other sentences are masked out afterwards.
        let same: Vec<Vec<bool>> = predicates
            .iter()
            .map(|&p| {
                (0..n_words)
                    .map(|i| {
                        word_mask[i]
                            && sentence_of_word[p] >= 0
                            && sentence_of_word[i] == sentence_of_word[p]
                    })
                    .collect()
            })
            .collect();
        let starts: Vec<usize> = same
            .iter()
            .map(|m| m.iter().position(|&x| x).unwrap_or(0))
            .collect();
        let lengths: Vec<usize> = same
            .iter()
            .zip(&starts)
            .map(|(m, &s)| {
                let end = m.iter().rposition(|&x| x).unwrap_or(0);
                end.saturating_sub(s) + 1
            })
            .collect();
        let max_len = lengths.iter().copied().max().unwrap_or(1);

        let f = predicates.len();
        let mut batch = Array3::<f32>::zeros((f, max_len, hidden));
        let mut mask = vec![vec![false; max_len]; f];
        for i in 0..f {
            for j in 0..lengths[i] {
                for h in 0..hidden {
                    batch[[i, j, h]] = words[[starts[i] + j, h]];
                }
                mask[i][j] = true;
            }
        }
        let declared = self
            .meta
            .graph_inputs
            .get("srl_roles")
            .cloned()
            .unwrap_or_default();
        let roles = self.graphs.run(
            "srl_roles",
            vec![
                ("words", GraphInput::cube_f32(batch)),
                (
                    "pred",
                    GraphInput::vec_i64(
                        predicates
                            .iter()
                            .zip(&starts)
                            .map(|(&p, &s)| (p - s) as i64)
                            .collect(),
                    ),
                ),
                ("mask", GraphInput::mat_bool(mask)),
            ],
            &declared,
            &["roles".to_string()],
        )?;
        let role_logits = roles.f32("roles")?;

        let tags = &self.labels.srl;
        let (allowed, start_ok) = bio_constraints(tags);
        let v = tags.iter().position(|t| t == "V").ok_or_else(|| {
            RuntimeError::Config(format!(
                "kniv v7 alias '{}': the SRL vocabulary has no 'V' tag",
                self.alias
            ))
        })?;

        let mut frames = Vec::with_capacity(f);
        for i in 0..f {
            let p = predicates[i];
            let mut role = Array2::<f32>::from_elem((n_words, tags.len()), NEG);
            for j in 0..lengths[i] {
                let w = starts[i] + j;
                for k in 0..tags.len() {
                    role[[w, k]] = role_logits[IxDyn(&[i, j, k])];
                }
            }
            for w in 0..n_words {
                if !same[i][w] {
                    for k in 0..tags.len() {
                        role[[w, k]] = NEG;
                    }
                }
            }
            let lp = log_softmax(role.view());
            // `V` belongs to the predicate and to nothing else, which also
            // keeps the predicate out of its own argument list.
            let mut keep = Array2::from_elem((n_words, tags.len()), true);
            for w in 0..n_words {
                keep[[w, v]] = false;
            }
            for k in 0..tags.len() {
                keep[[p, k]] = false;
            }
            keep[[p, v]] = true;

            let path = viterbi(lp.view(), &allowed, &start_ok, Some(&keep));
            let decoded: Vec<String> = path.iter().map(|&k| label_at(tags, k)).collect();
            frames.push(WindowFrame {
                predicate: p,
                args: bio_spans(&decoded)
                    .into_iter()
                    .filter(|(label, _, _)| label != "V")
                    .map(|(label, a, b)| WordSpan {
                        label,
                        start: a,
                        end: b,
                    })
                    .collect(),
            });
        }
        Ok(frames)
    }

    /// Coreference clusters as inclusive word spans.
    fn coref(
        &self,
        plain: &Outputs,
        sentence_of_word: &[i64],
        n_words: usize,
    ) -> Result<Vec<Vec<(usize, usize)>>> {
        let mut scores = plain.f32_batch0("coref_scores")?;
        // Column 0 is "no antecedent"; the bias is how a release trades
        // coreference recall for precision.
        for r in 0..scores.nrows() {
            scores[[r, 0]] += self.calibration.coref_link_bias;
        }
        let candidates = plain.i64_2d("coref_cand")?;
        let best = argmax_rows(scores.view());
        let mut antecedents: Vec<(usize, usize)> = Vec::new();
        for w in 0..n_words.min(best.len()) {
            if best[w] > 0 {
                let col = best[w] - 1;
                if col < candidates.ncols() {
                    let a = candidates[[w, col]];
                    if a >= 0 {
                        antecedents.push((w, a as usize));
                    }
                }
            }
        }
        let clusters = link_clusters(&antecedents);
        let heads: Vec<usize> = clusters.iter().flatten().copied().collect();
        if heads.is_empty() {
            return Ok(Vec::new());
        }

        let declared = self
            .meta
            .graph_inputs
            .get("coref_spans")
            .cloned()
            .unwrap_or_default();
        let spans = self.graphs.run(
            "coref_spans",
            vec![
                (
                    "coref_words",
                    GraphInput::f32(plain.passthrough("coref_words")?),
                ),
                (
                    "heads",
                    GraphInput::vec_i64(heads.iter().map(|&h| h as i64).collect()),
                ),
                (
                    "sent_of_word",
                    GraphInput::row_i64(sentence_of_word.to_vec()),
                ),
            ],
            &declared,
            &["span_logits".to_string(), "span_pos".to_string()],
        )?;
        let logits = spans.f32("span_logits")?;
        let positions = spans.i64_2d("span_pos")?;
        let k = positions.ncols();

        let mut boundary: HashMap<usize, (usize, usize)> = HashMap::new();
        for (i, &h) in heads.iter().enumerate() {
            let h = h as i64;
            // The start candidate must be at or before the head, the end at
            // or after it.
            let start_scores: Vec<f32> = (0..k)
                .map(|j| {
                    if positions[[i, j]] > h {
                        -1e9
                    } else {
                        logits[IxDyn(&[i, j, 0])]
                    }
                })
                .collect();
            let end_scores: Vec<f32> = (0..k)
                .map(|j| {
                    if positions[[i, j]] < h {
                        -1e9
                    } else {
                        logits[IxDyn(&[i, j, 1])]
                    }
                })
                .collect();
            let s = positions[[i, argmax(&start_scores)]].max(0) as usize;
            let e = positions[[i, argmax(&end_scores)]].max(0) as usize;
            boundary.insert(h as usize, (s, e));
        }
        Ok(clusters
            .iter()
            .map(|c| {
                c.iter()
                    .filter_map(|w| boundary.get(w).copied())
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    /// The label graph, cached by the exact list of label texts.
    fn label_embeddings(&self, texts: &[String]) -> Result<Array2<f32>> {
        if let Ok(cache) = self.label_cache.entries.lock()
            && let Some(hit) = cache.get(texts)
        {
            return Ok(hit.clone());
        }
        let (ids, mask) = encode_labels(texts, self.tokenizer, self.meta.pad_token_id, self.alias)?;
        let declared = self
            .meta
            .graph_inputs
            .get("label")
            .cloned()
            .unwrap_or_default();
        let out = self.graphs.run(
            "label",
            vec![
                ("label_ids", GraphInput::mat_i64(ids)),
                ("label_mask", GraphInput::mat_i64(mask)),
            ],
            &declared,
            &["label_emb".to_string()],
        )?;
        let emb = out.f32_2d("label_emb")?;
        if let Ok(mut cache) = self.label_cache.entries.lock() {
            if cache.len() >= LABEL_CACHE_CAPACITY {
                cache.clear();
            }
            cache.insert(texts.to_vec(), emb.clone());
        }
        Ok(emb)
    }

    /// Typed mention spans, grouped into entities.
    #[allow(clippy::too_many_arguments)]
    fn entities(
        &self,
        plain: &Outputs,
        tokens: &[String],
        spans: &[(usize, usize)],
        sentence_of_word: &[i64],
        word_mask: &[bool],
        n_words: usize,
        coref: &[Vec<(usize, usize)>],
        coref_mode: CorefMode,
        entity_types: Option<&[LabelSpec]>,
    ) -> Result<Vec<(Option<String>, Vec<Mention>)>> {
        let mut span_type: HashMap<(usize, usize), String> = HashMap::new();
        let tags_by_sentence: Vec<Option<Vec<String>>> = match entity_types {
            Some(types) if !types.is_empty() => {
                let candidates = span_candidates(sentence_of_word, word_mask, SPAN_MAX_WIDTH);
                let predicted = if candidates.is_empty() {
                    Vec::new()
                } else {
                    let texts: Vec<String> = types.iter().map(type_text).collect();
                    let type_emb = self.label_embeddings(&texts)?;
                    let declared = self
                        .meta
                        .graph_inputs
                        .get("span")
                        .cloned()
                        .unwrap_or_default();
                    let out = self.graphs.run(
                        "span",
                        vec![
                            (
                                "span_words",
                                GraphInput::f32(plain.passthrough("span_words")?),
                            ),
                            (
                                "start",
                                GraphInput::vec_i64(
                                    candidates.iter().map(|&(s, _)| s as i64).collect(),
                                ),
                            ),
                            (
                                "end",
                                GraphInput::vec_i64(
                                    candidates.iter().map(|&(_, e)| e as i64).collect(),
                                ),
                            ),
                            ("type_emb", GraphInput::f32(type_emb.into_dyn())),
                        ],
                        &declared,
                        &["span_logits".to_string()],
                    )?;
                    let logits = out.f32_2d("span_logits")?;
                    decode_spans(logits.view(), &candidates, self.calibration.span_offset)
                };
                // The grouping stage works on BIO tags, so the kept spans
                // are re-emitted under a placeholder type and their real
                // type is carried alongside.
                let mut by_sentence = Vec::with_capacity(spans.len());
                for &(s, e) in spans {
                    let mut tags = vec!["O".to_string(); e.saturating_sub(s)];
                    for &(a, b, t) in &predicted {
                        if s <= a && b <= e {
                            if let Some(name) = types.get(t) {
                                span_type.insert((a, b), name.name.clone());
                            }
                            tags[a - s] = "B-MISC".to_string();
                            for i in (a + 1)..b {
                                tags[i - s] = "I-MISC".to_string();
                            }
                        }
                    }
                    by_sentence.push(if s < n_words { Some(tags) } else { None });
                }
                by_sentence
            }
            _ => {
                let ent_tags = default_entity_tags();
                let (allowed, start) = bio_constraints(&ent_tags);
                let mut lp = log_softmax(plain.f32_batch0("ent")?.view());
                for r in 0..lp.nrows() {
                    lp[[r, 0]] += self.calibration.ent_o_bias;
                }
                spans
                    .iter()
                    .map(|&(s, e)| {
                        if s >= n_words {
                            return None;
                        }
                        let slice = lp.slice(ndarray::s![s..e.min(n_words), ..]);
                        Some(
                            viterbi(slice, &allowed, &start, None)
                                .iter()
                                .map(|&k| label_at(&ent_tags, k))
                                .collect(),
                        )
                    })
                    .collect()
            }
        };

        let grouped = build_entities(tokens, spans, &tags_by_sentence, coref, coref_mode);
        Ok(grouped
            .into_iter()
            .map(|mentions| {
                // An entity's type is the most frequent type among its typed
                // spans, ties to the first mention that carries one.
                let mut counts: Vec<(String, usize)> = Vec::new();
                for m in &mentions {
                    if let Some(name) = span_type.get(&m.span) {
                        match counts.iter_mut().find(|(n, _)| n == name) {
                            Some((_, c)) => *c += 1,
                            None => counts.push((name.clone(), 1)),
                        }
                    }
                }
                let ty = counts
                    .iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(n, _)| n.clone());
                (ty, mentions)
            })
            .collect())
    }

    /// Entities (speakers first) and facts with their attributes.
    #[allow(clippy::too_many_arguments)]
    fn relations(
        &self,
        w: &Window,
        tokens: &[String],
        spans: &[(usize, usize)],
        alignment: &Alignment,
        n_words: usize,
        grouped: &[(Option<String>, Vec<Mention>)],
        schema: Option<&ExtractionSchema>,
    ) -> Result<(Vec<NlpEntity>, Vec<NlpFact>)> {
        let analysed = alignment.word_mask();
        let participants = w.participants();
        let sentence_speakers: Vec<Option<String>> =
            w.sentences.iter().map(|s| s.speaker.clone()).collect();
        let sentence_starts: Vec<usize> = spans.iter().map(|&(s, _)| s).collect();
        let turns: Vec<(usize, usize)> =
            speaker_turns(&sentence_speakers, &sentence_starts, &participants)
                .into_iter()
                .filter(|&(s, _)| s < n_words && analysed.get(s).copied().unwrap_or(false))
                .collect();

        let mut nodes: Vec<NlpEntity> = Vec::new();
        let mut mention_spans: Vec<Vec<(usize, usize)>> = Vec::new();
        // In a conversation the participants are entities in their own
        // right, so "I live in Lisbon" can be stored about the speaker.
        for (p, who) in participants.iter().enumerate() {
            let ms: Vec<(usize, usize)> = turns
                .iter()
                .filter(|&&(_, q)| q == p)
                .map(|&(s, _)| (s, s + 1))
                .collect();
            if ms.is_empty() {
                continue;
            }
            nodes.push(NlpEntity {
                ty: "speaker".to_string(),
                text: who.clone(),
                speaker: true,
                ..Default::default()
            });
            mention_spans.push(ms);
        }
        let n_participants = nodes.len();

        for (ty, mentions) in grouped {
            let mut ms: Vec<(usize, usize)> = mentions
                .iter()
                .map(|m| m.span)
                .filter(|&(a, b)| b <= n_words && analysed.get(a).copied().unwrap_or(false))
                .collect();
            ms.sort_unstable();
            ms.dedup();
            if ms.is_empty() {
                continue;
            }
            let fallback = mentions.first().map(|m| m.ty.clone()).unwrap_or_default();
            nodes.push(NlpEntity {
                ty: ty.clone().unwrap_or(fallback),
                text: tokens[ms[0].0..ms[0].1].join(" "),
                speaker: false,
                mentions: ms.iter().map(|&(a, b)| tokens[a..b].join(" ")).collect(),
                spans: ms.clone(),
                ..Default::default()
            });
            mention_spans.push(ms);
        }

        let relations: &[LabelSpec] = match schema {
            Some(s) if !s.relations.is_empty() => &s.relations,
            _ => return Ok((nodes, Vec::new())),
        };
        if mention_spans.len() < 2 {
            return Ok((nodes, Vec::new()));
        }

        let participant_turns: Vec<(usize, usize)> = turns
            .iter()
            .copied()
            .filter(|&(_, p)| p < n_participants)
            .collect();
        let (marker_ids, mention_positions) = marker_input(
            tokens,
            &mention_spans[n_participants..],
            &participant_turns,
            &participants[..n_participants],
            self.tokenizer,
            self.meta.sep_token_id,
            self.alias,
        )?;

        let n_entities = mention_spans.len();
        let mut men_pos: Vec<i64> = Vec::new();
        let mut men_ent: Vec<i64> = Vec::new();
        for (k, ms) in mention_spans.iter().enumerate() {
            let positions = mention_positions.get(k).cloned().unwrap_or_default();
            for (j, _) in ms.iter().enumerate() {
                men_pos.push(positions.get(j).copied().unwrap_or(-1));
                men_ent.push(k as i64);
            }
        }
        let pairs: Vec<(usize, usize)> = (0..n_entities)
            .flat_map(|h| (0..n_entities).map(move |t| (h, t)))
            .filter(|&(h, t)| h != t)
            .collect();

        let texts: Vec<String> = relations.iter().map(relation_text).collect();
        let label_emb = self.label_embeddings(&texts)?;
        let declared = self
            .meta
            .graph_inputs
            .get("marker")
            .cloned()
            .unwrap_or_default();
        let n_marker_subwords = marker_ids.len();
        let marker = self.graphs.run(
            "marker",
            vec![
                ("rm_input_ids", GraphInput::row_i64(marker_ids)),
                (
                    "rm_attention_mask",
                    GraphInput::row_i64(vec![1; n_marker_subwords]),
                ),
                ("men_pos", GraphInput::vec_i64(men_pos)),
                ("men_ent", GraphInput::vec_i64(men_ent)),
                (
                    "ent_ids",
                    GraphInput::vec_i64((0..n_entities as i64).collect()),
                ),
                (
                    "pair_h",
                    GraphInput::vec_i64(pairs.iter().map(|&(h, _)| h as i64).collect()),
                ),
                (
                    "pair_t",
                    GraphInput::vec_i64(pairs.iter().map(|&(_, t)| t as i64).collect()),
                ),
                ("label_emb", GraphInput::f32(label_emb.into_dyn())),
            ],
            &declared,
            &[
                "logits".to_string(),
                "pair".to_string(),
                "labels".to_string(),
                "ent".to_string(),
            ],
        )?;
        let logits = marker.f32_2d("logits")?;

        // A relation the model was calibrated for uses its own dev-tuned
        // offset; every other relation uses `-unseen_offset`. The two never
        // stack.
        let offsets: Vec<f32> = relations
            .iter()
            .map(|r| {
                self.calibration
                    .rel_offsets
                    .get(&r.name)
                    .copied()
                    .unwrap_or(-self.calibration.unseen_offset)
            })
            .collect();
        let top = schema
            .and_then(|s| s.top_per_pair)
            .unwrap_or(self.calibration.rel_top);

        let mut facts: Vec<NlpFact> = Vec::new();
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for (pi, &(h, t)) in pairs.iter().enumerate() {
            if pi >= logits.nrows() {
                break;
            }
            let threshold = logits[[pi, 0]] + self.calibration.offset;
            let margins: Vec<f32> = (0..relations.len())
                .map(|k| logits[[pi, k + 1]] - threshold + offsets[k])
                .collect();
            let mut order: Vec<usize> = (0..relations.len()).collect();
            order.sort_by(|&a, &b| {
                margins[b]
                    .partial_cmp(&margins[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            for &k in order.iter().take(top) {
                if margins[k] > 0.0 {
                    facts.push(NlpFact {
                        head: h,
                        tail: t,
                        relation: relations[k].name.clone(),
                        // The reference rounds this to three decimals for
                        // its JSON; an f32 field has no reason to throw the
                        // precision away, and callers map it to a
                        // confidence (`sigmoid(margin)`).
                        margin: margins[k],
                        ..Default::default()
                    });
                    kept.push((pi, k));
                }
            }
        }

        if !facts.is_empty() {
            let pair = marker.f32_2d("pair")?;
            let label_rows = marker.f32_2d("labels")?;
            let ent = marker.passthrough("ent")?;
            let declared = self
                .meta
                .graph_inputs
                .get("attr")
                .cloned()
                .unwrap_or_default();
            let attr = self.graphs.run(
                "attr",
                vec![
                    (
                        "pair",
                        GraphInput::f32(select_rows(&pair, kept.iter().map(|&(p, _)| p))),
                    ),
                    (
                        "labels",
                        GraphInput::f32(select_rows(&label_rows, kept.iter().map(|&(_, k)| k))),
                    ),
                    ("ent", GraphInput::f32(ent)),
                ],
                &declared,
                &[
                    "polarity".to_string(),
                    "modality".to_string(),
                    "holder".to_string(),
                    "time".to_string(),
                ],
            )?;
            let polarity = attr.f32_2d("polarity")?;
            let modality = attr.f32_2d("modality")?;
            let holder = attr.f32_2d("holder")?;
            let time = attr.f32_2d("time")?;
            for (i, fact) in facts.iter_mut().enumerate() {
                fact.polarity = POLARITIES
                    .get(argmax(&polarity.row(i).to_vec()))
                    .copied()
                    .unwrap_or_default();
                fact.modality = MODALITIES
                    .get(argmax(&modality.row(i).to_vec()))
                    .copied()
                    .unwrap_or_default();
                // Column 0 of both heads is "none"; the rest index entities.
                let h = argmax(&holder.row(i).to_vec());
                fact.holder = (h > 0).then(|| h - 1);
                let t = argmax(&time.row(i).to_vec());
                fact.time = (t > 0).then(|| t - 1);
            }
        }
        Ok((nodes, facts))
    }
}

/// Gather `rows` of `source` into a fresh `[n, cols]` tensor.
fn select_rows(source: &Array2<f32>, rows: impl Iterator<Item = usize>) -> ArrayD<f32> {
    let indices: Vec<usize> = rows.collect();
    let cols = source.ncols();
    let mut out = Array2::<f32>::zeros((indices.len(), cols));
    for (i, &r) in indices.iter().enumerate() {
        if r < source.nrows() {
            for c in 0..cols {
                out[[i, c]] = source[[r, c]];
            }
        }
    }
    out.into_dyn()
}

fn label_at(labels: &[String], id: usize) -> String {
    labels
        .get(id)
        .cloned()
        .unwrap_or_else(|| format!("UNK_{id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_outputs_only_asks_for_what_is_read() {
        let only_ner = plain_outputs(NlpTasks::NER, false);
        assert_eq!(only_ner, ["ner"]);

        let dep = plain_outputs(NlpTasks::DEP, false);
        assert_eq!(dep, ["arc", "rel_dep", "rel_head"]);

        // The big per-word matrices stay out unless something reads them.
        let linguistic = plain_outputs(NlpTasks::LINGUISTIC, false);
        assert!(!linguistic.contains(&"coref_words".to_string()));
        assert!(!linguistic.contains(&"span_words".to_string()));

        let all = plain_outputs(NlpTasks::ALL.expanded(), true);
        for name in [
            "coref_words",
            "coref_scores",
            "coref_cand",
            "ent",
            "span_words",
        ] {
            assert!(all.contains(&name.to_string()), "{name} missing");
        }
    }

    #[test]
    fn select_rows_gathers_in_order() {
        let src = ndarray::array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let got = select_rows(&src, [2usize, 0, 2].into_iter());
        assert_eq!(got.shape(), [3, 2]);
        assert_eq!(got[IxDyn(&[0, 0])], 5.0);
        assert_eq!(got[IxDyn(&[1, 1])], 2.0);
        assert_eq!(got[IxDyn(&[2, 0])], 5.0);
    }

    #[test]
    fn attribute_vocabularies_match_the_head_order() {
        assert_eq!(POLARITIES[0].as_str(), "positive");
        assert_eq!(POLARITIES[1].as_str(), "negative");
        let names: Vec<&str> = MODALITIES.iter().map(|m| m.as_str()).collect();
        assert_eq!(
            names,
            [
                "asserted",
                "hypothetical",
                "planned",
                "desired",
                "questioned",
                "reported"
            ]
        );
    }
}
