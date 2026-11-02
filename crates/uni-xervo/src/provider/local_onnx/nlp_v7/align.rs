// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Words to subwords: alignment, the encoder's word-level index tensors, the
//! label batch, and the marker text.
//!
//! A port of `align`, `word_ends`, `Tok.labels`, `speaker_turns` and
//! `marker_input` from `kniv_v7/runtime.py`.
//!
//! The reference computes word offsets in Python characters; this works in
//! UTF-8 bytes. Both index the same joined string monotonically and word
//! boundaries land on character boundaries either way, so the word each
//! subword maps to is identical.

use crate::error::{Result, RuntimeError};

/// Subwords a label text is cut to, keeping its final special token.
pub(crate) const LABEL_MAX_SUBWORDS: usize = 48;
/// Subwords the marker pass accepts, independent of the plain pass's budget.
pub(crate) const MARKER_MAX_SUBWORDS: usize = 1024;
/// The marker word wrapped around every mention and speaker name.
pub(crate) const MARKER: &str = "*";
/// Widest entity-type candidate span, in words.
pub(crate) const SPAN_MAX_WIDTH: usize = 12;

/// The subword view of one window of words.
#[derive(Debug, Clone)]
pub(crate) struct Alignment {
    /// Subword ids, including special tokens, cut to the budget.
    pub ids: Vec<i64>,
    /// Each analysed word's first subword index. `-1` when no subword maps
    /// to the word at all, which the word mask then hides.
    pub first: Vec<i64>,
    /// How many leading words fit in the budget.
    pub limit: usize,
}

impl Alignment {
    /// The word mask the encoder takes: which analysed words have a subword.
    pub fn word_mask(&self) -> Vec<bool> {
        self.first.iter().map(|&f| f >= 0).collect()
    }

    /// `first_subword`, with the `-1` sentinel clamped to `0` as the graph
    /// expects (the mask is what hides those words).
    pub fn first_subword(&self) -> Vec<i64> {
        self.first.iter().map(|&f| f.max(0)).collect()
    }

    /// Each word's exclusive subword end, for `mean` pooling.
    ///
    /// Walks backwards so a word's span runs up to the next word's first
    /// subword; a word with no subword of its own gets a one-wide span.
    pub fn word_ends(&self) -> Vec<i64> {
        let n = self.ids.len() as i64;
        let mut end = vec![1i64; self.first.len()];
        let mut next = n - 1;
        for i in (0..self.first.len()).rev() {
            let f = self.first[i];
            if f >= 0 && f < next {
                end[i] = next;
                next = f;
            } else {
                end[i] = (f.max(0) + 1).min(n);
            }
        }
        end
    }
}

/// Align `words` to subwords with `tokenizer`.
///
/// Words are joined with single spaces and encoded once with special tokens.
/// A word's first subword is the first token whose byte range ends inside
/// the word. When the encoding overruns `max_subwords` it is cut to
/// `max_subwords - 1` tokens plus `sep_token_id`, and every word whose first
/// subword fell at or after the cut is reported as not analysed.
pub(crate) fn align(
    words: &[String],
    tokenizer: &tokenizers::Tokenizer,
    max_subwords: usize,
    sep_token_id: i64,
    alias: &str,
) -> Result<Alignment> {
    let mut starts: Vec<usize> = Vec::with_capacity(words.len());
    let mut pos = 0usize;
    for w in words {
        starts.push(pos);
        pos += w.len() + 1;
    }
    let joined = words.join(" ");
    let encoding = tokenizer.encode(joined.as_str(), true).map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("kniv v7 tokenization failed: {e}"),
        }
    })?;

    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
    let mut first: Vec<i64> = vec![-1; words.len()];
    for (ti, &(s, e)) in encoding.get_offsets().iter().enumerate() {
        if e <= s {
            // A special token: offsets collapse to an empty range.
            continue;
        }
        // The word whose start is the last one before `e`, i.e. the word
        // the token's final byte falls inside.
        let wi = starts.partition_point(|&st| st < e).saturating_sub(1);
        if first[wi] < 0 {
            first[wi] = ti as i64;
        }
    }

    let mut limit = words.len();
    if ids.len() > max_subwords {
        let cut = max_subwords - 1;
        ids.truncate(cut);
        ids.push(sep_token_id);
        if let Some(i) = first.iter().position(|&f| f >= cut as i64) {
            limit = i;
        }
    }
    first.truncate(limit);
    Ok(Alignment { ids, first, limit })
}

/// The label graph's padded batch: `(ids, mask)`, each `[K, L]`.
pub(crate) type LabelBatch = (Vec<Vec<i64>>, Vec<Vec<i64>>);

/// Encode schema label texts into the label graph's padded batch:
/// `(ids, mask)` of shape `[K, L]`.
///
/// Each text is cut to [`LABEL_MAX_SUBWORDS`] keeping its final special
/// token, then the batch is padded to the longest row.
pub(crate) fn encode_labels(
    texts: &[String],
    tokenizer: &tokenizers::Tokenizer,
    pad_token_id: i64,
    alias: &str,
) -> Result<LabelBatch> {
    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(texts.len());
    for t in texts {
        let encoding = tokenizer.encode(t.as_str(), true).map_err(|e| {
            RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("kniv v7 label tokenization failed: {e}"),
            }
        })?;
        let mut row: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        if row.len() > LABEL_MAX_SUBWORDS {
            let last = row[row.len() - 1];
            row.truncate(LABEL_MAX_SUBWORDS - 1);
            row.push(last);
        }
        rows.push(row);
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut ids = Vec::with_capacity(rows.len());
    let mut mask = Vec::with_capacity(rows.len());
    for row in rows {
        let mut id_row = vec![pad_token_id; width];
        let mut mask_row = vec![0i64; width];
        for (i, v) in row.iter().enumerate() {
            id_row[i] = *v;
            mask_row[i] = 1;
        }
        ids.push(id_row);
        mask.push(mask_row);
    }
    Ok((ids, mask))
}

/// The label text the span head was trained on.
pub(crate) fn type_text(spec: &crate::traits::LabelSpec) -> String {
    match spec.description.as_deref().filter(|d| !d.is_empty()) {
        Some(d) => format!("entity type: {}. {d}", spec.name),
        None => format!("entity type: {}", spec.name),
    }
}

/// The label text the relation head was trained on.
pub(crate) fn relation_text(spec: &crate::traits::LabelSpec) -> String {
    match spec.description.as_deref().filter(|d| !d.is_empty()) {
        Some(d) => format!("relation: {}. {d}", spec.name),
        None => format!("relation: {}", spec.name),
    }
}

/// Where each participant starts a turn: `(word index, participant index)`.
///
/// Only the first sentence of a run by one speaker counts, so a speaker with
/// three consecutive sentences gets one turn-start, not three.
pub(crate) fn speaker_turns(
    sentence_speakers: &[Option<String>],
    sentence_starts: &[usize],
    participants: &[String],
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut last: Option<&str> = None;
    for (j, &start) in sentence_starts.iter().enumerate() {
        let who = sentence_speakers.get(j).and_then(Option::as_deref);
        if let Some(who) = who
            && Some(who) != last
            && let Some(p) = participants.iter().position(|x| x == who)
        {
            out.push((start, p));
        }
        last = who;
    }
    out
}

/// The marker pass's input: the window rewritten with `*` around every
/// mention and `* speaker * :` opening each turn.
///
/// Returns the subword ids and, per entity, each mention's **start-marker**
/// subword position (`-1` past the subword budget). Speakers come first, in
/// participant order, and their positions are the markers opening their
/// turns.
pub(crate) fn marker_input(
    tokens: &[String],
    mention_spans: &[Vec<(usize, usize)>],
    turns: &[(usize, usize)],
    participants: &[String],
    tokenizer: &tokenizers::Tokenizer,
    sep_token_id: i64,
    alias: &str,
) -> Result<(Vec<i64>, Vec<Vec<i64>>)> {
    use std::collections::{HashMap, HashSet};

    let mut starts: HashSet<usize> = HashSet::new();
    let mut ends: HashSet<usize> = HashSet::new();
    for ms in mention_spans {
        for &(a, b) in ms {
            starts.insert(a);
            ends.insert(b.saturating_sub(1));
        }
    }
    let turn_at: HashMap<usize, usize> = turns.iter().copied().collect();

    let mut words: Vec<String> = Vec::with_capacity(tokens.len() * 2);
    let mut marker_of: HashMap<usize, usize> = HashMap::new();
    let mut speaker_words: Vec<Vec<usize>> = vec![Vec::new(); participants.len()];
    for (i, w) in tokens.iter().enumerate() {
        if let Some(&p) = turn_at.get(&i)
            && let Some(name) = participants.get(p)
        {
            speaker_words[p].push(words.len());
            words.push(MARKER.to_string());
            words.push(name.clone());
            words.push(MARKER.to_string());
            words.push(":".to_string());
        }
        if starts.contains(&i) {
            marker_of.insert(i, words.len());
            words.push(MARKER.to_string());
        }
        words.push(w.clone());
        if ends.contains(&i) {
            words.push(MARKER.to_string());
        }
    }

    let a = align(&words, tokenizer, MARKER_MAX_SUBWORDS, sep_token_id, alias)?;
    let at = |wi: usize| -> i64 { if wi < a.limit { a.first[wi] } else { -1 } };

    let mut positions: Vec<Vec<i64>> = speaker_words
        .iter()
        .map(|ws| ws.iter().map(|&w| at(w)).collect())
        .collect();
    positions.extend(mention_spans.iter().map(|ms| {
        ms.iter()
            .map(|&(start, _)| marker_of.get(&start).copied().map_or(-1, at))
            .collect()
    }));
    Ok((a.ids, positions))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_ends_span_up_to_the_next_word() {
        // Four words over eight subwords: [CLS] a b1 b2 c d [SEP] = 7.
        let a = Alignment {
            ids: vec![0; 7],
            first: vec![1, 2, 4, 5],
            limit: 4,
        };
        assert_eq!(a.word_ends(), [2, 4, 5, 6]);
        assert_eq!(a.first_subword(), [1, 2, 4, 5]);
        assert_eq!(a.word_mask(), [true, true, true, true]);
    }

    #[test]
    fn a_word_with_no_subword_gets_a_one_wide_span_and_is_masked() {
        let a = Alignment {
            ids: vec![0; 5],
            first: vec![1, -1, 3],
            limit: 3,
        };
        assert_eq!(a.word_mask(), [true, false, true]);
        assert_eq!(a.first_subword(), [1, 0, 3]);
        let ends = a.word_ends();
        // The masked word falls back to first+1 clamped to the length.
        assert_eq!(ends[1], 1);
        assert_eq!(ends[2], 4);
    }

    #[test]
    fn label_texts_follow_the_trained_templates() {
        use crate::traits::LabelSpec;
        assert_eq!(
            relation_text(&LabelSpec::described("lives in", "where the head resides")),
            "relation: lives in. where the head resides"
        );
        assert_eq!(
            relation_text(&LabelSpec::name("employer")),
            "relation: employer"
        );
        assert_eq!(
            type_text(&LabelSpec::name("location")),
            "entity type: location"
        );
    }

    #[test]
    fn speaker_turns_mark_only_the_first_sentence_of_a_run() {
        let speakers = [
            Some("user".to_string()),
            Some("user".to_string()),
            Some("assistant".to_string()),
            Some("user".to_string()),
        ];
        let starts = [0usize, 5, 9, 14];
        let parts = ["user".to_string(), "assistant".to_string()];
        assert_eq!(
            speaker_turns(&speakers, &starts, &parts),
            [(0, 0), (9, 1), (14, 0)]
        );
    }

    #[test]
    fn speaker_turns_ignore_speakers_outside_the_participant_list() {
        let speakers = [Some("tool".to_string()), Some("user".to_string())];
        let starts = [0usize, 3];
        let parts = ["user".to_string()];
        assert_eq!(speaker_turns(&speakers, &starts, &parts), [(3, 0)]);
    }
}
