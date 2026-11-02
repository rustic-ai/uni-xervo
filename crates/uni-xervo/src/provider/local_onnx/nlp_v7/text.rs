// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Text to windows: units, sentences, words.
//!
//! A faithful port of `kniv_v7/windows.py`. The model is specified in terms
//! of *words*, not subwords: every index it emits — dependency heads, entity
//! mentions, role spans — is a word index into a window produced here, so a
//! divergence in word segmentation is a divergence in the model's output.
//!
//! Unlike the Python reference, each [`Word`] also carries the UTF-8 byte
//! range it came from. The reference never needs it; the provider does, to
//! fill the byte offsets the [`NlpResult`](crate::traits::NlpResult) contract
//! promises.
//!
//! One deliberate divergence: Python's `\w` admits `\p{Nl}` / `\p{No}`
//! (roman numerals, vulgar fractions) where Rust's admits only `\p{Nd}`.
//! The models are English-only and the training data was tokenized with the
//! Python class, so the difference can only show up on exotic numerals.

use std::sync::LazyLock;

use regex::Regex;

/// Words per window. Set by the model's training, not by the encoder's
/// subword budget — see [`crate::provider::local_onnx::nlp_v7::align`] for
/// the subword cut, which is separate.
pub(crate) const MAX_WORDS_PER_WINDOW: usize = 512;

/// Clitics split off the end of a word, in the reference's alternation
/// order. Matched case-insensitively.
const CLITICS: [&str; 7] = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"];

/// A whitespace-separated chunk matching this in full stays one word.
static URLISH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:https?://|www\.)\S+|\S+@\S+\.\w+)$").expect("URLISH regex")
});

/// Numbers (with percents and ordinals), words (hyphenated, internal
/// apostrophe), ellipsis, single punctuation — in preference order, which
/// both `re` and `regex` honour (leftmost-first).
static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?|\w+(?:[-']\w+)*|\.\.\.|[^\w\s]")
        .expect("TOKEN regex")
});

/// One word of a window, with the byte range it was taken from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Word {
    /// Surface form, always a contiguous slice of the input.
    pub text: String,
    /// Byte offset of the word's first byte, in the coordinates of the unit
    /// text it came from.
    pub start: usize,
    /// Byte offset one past the word's last byte.
    pub end: usize,
}

/// One input unit: a paragraph of prose, or one turn of a conversation.
#[derive(Debug, Clone)]
pub(crate) struct Unit<'a> {
    /// The unit's text.
    pub text: &'a str,
    /// Byte offset of `text` within the caller's input. Always `0` for a
    /// conversation turn, whose text is its own string.
    pub offset: usize,
    /// Who said it, for a conversation.
    pub speaker: Option<&'a str>,
}

/// One sentence of a window, as word indices into [`Window::words`].
#[derive(Debug, Clone)]
pub(crate) struct SentenceSpan {
    /// First word index, inclusive.
    pub start: usize,
    /// Last word index, exclusive.
    pub end: usize,
    /// Index into the request's units.
    pub unit: usize,
    /// The unit's speaker.
    pub speaker: Option<String>,
}

/// A window: whole sentences packed up to [`MAX_WORDS_PER_WINDOW`] words.
#[derive(Debug, Clone, Default)]
pub(crate) struct Window {
    /// The window's words, in order.
    pub words: Vec<Word>,
    /// Its sentences.
    pub sentences: Vec<SentenceSpan>,
}

impl Window {
    /// The distinct speakers of this window, in order of first appearance —
    /// the model's conversation participants.
    pub fn participants(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for s in &self.sentences {
            if let Some(who) = &s.speaker
                && !out.iter().any(|p| p == who)
            {
                out.push(who.clone());
            }
        }
        out
    }
}

/// Prose split into paragraphs at blank lines, with their byte offsets.
///
/// Mirrors the reference's `text.split("\n\n")` plus `Document`'s drop of
/// blank units, while keeping each paragraph's position in `text`.
pub(crate) fn paragraphs(text: &str) -> Vec<Unit<'_>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for part in text.split("\n\n") {
        if !part.trim().is_empty() {
            out.push(Unit {
                text: part,
                offset,
                speaker: None,
            });
        }
        offset += part.len() + 2;
    }
    out
}

/// Conversation turns as units, dropping blank ones along with their
/// speakers (`Document.__post_init__`).
pub(crate) fn turns_as_units<'a>(turns: &'a [crate::traits::Turn<'a>]) -> Vec<Unit<'a>> {
    turns
        .iter()
        .filter(|t| !t.text.trim().is_empty())
        .map(|t| Unit {
            text: t.text,
            offset: 0,
            speaker: Some(t.speaker),
        })
        .collect()
}

/// Whether `c` can open a sentence, per the reference's lookahead
/// `[A-Z0-9"'(\[]`.
fn opens_sentence(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, '"' | '\'' | '(' | '[')
}

/// Split a unit into sentences, as `(byte offset, text)` pairs.
///
/// Ports `_SENT_END = (?<=[.!?])["')\]]*\s+(?=[A-Z0-9"'(\[])` used with
/// `re.split`. The Rust `regex` crate has no look-behind, so this is the
/// equivalent scan: a boundary sits right after `.`, `!` or `?`, skips any
/// closing quotes or brackets, consumes whitespace, and must land on a
/// character that can open a sentence. The quote run is greedy with
/// backtracking, which is why `k` counts down.
pub(crate) fn split_sentences(text: &str) -> Vec<(usize, &str)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces: Vec<(usize, usize)> = Vec::new(); // byte ranges, untrimmed
    let mut piece_start = 0usize;
    let mut i = 1usize;
    while i < chars.len() {
        if !matches!(chars[i - 1].1, '.' | '!' | '?') {
            i += 1;
            continue;
        }
        // The greedy `["')\]]*` run starting at i.
        let mut quotes = 0usize;
        while i + quotes < chars.len() && matches!(chars[i + quotes].1, '"' | '\'' | ')' | ']') {
            quotes += 1;
        }
        let mut matched = None;
        for k in (0..=quotes).rev() {
            let ws_start = i + k;
            if ws_start >= chars.len() || !chars[ws_start].1.is_whitespace() {
                continue;
            }
            let mut ws_end = ws_start;
            while ws_end < chars.len() && chars[ws_end].1.is_whitespace() {
                ws_end += 1;
            }
            if ws_end < chars.len() && opens_sentence(chars[ws_end].1) {
                matched = Some(ws_end);
                break;
            }
        }
        match matched {
            // `re.split` cuts at the match start and resumes at its end.
            Some(ws_end) => {
                pieces.push((piece_start, chars[i].0));
                piece_start = chars[ws_end].0;
                i = ws_end;
            }
            None => i += 1,
        }
    }
    pieces.push((piece_start, text.len()));

    let mut out: Vec<(usize, &str)> = Vec::new();
    for (a, b) in pieces {
        let raw = &text[a..b];
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            out.push((a + (raw.len() - raw.trim_start().len()), trimmed));
        }
    }
    if out.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            out.push((text.len() - text.trim_start().len(), trimmed));
        }
    }
    out
}

/// Split a trailing clitic off `tok`, as the reference's
/// `(?i)(.+?)(n't|'s|'re|'ve|'ll|'d|'m)$` does.
///
/// The lazy `.+?` makes Python prefer the *shortest* stem, so this looks for
/// the earliest split point whose remainder is exactly one clitic.
fn split_clitic(tok: &str) -> Option<(usize, usize)> {
    let mut idx = tok.char_indices().map(|(i, _)| i).collect::<Vec<_>>();
    idx.push(tok.len());
    // `.+?` needs at least one character, so the stem starts at idx[0] and
    // the earliest possible break is idx[1].
    for &cut in idx.iter().skip(1) {
        if cut == 0 || cut == tok.len() {
            continue;
        }
        let tail = &tok[cut..];
        if CLITICS.iter().any(|c| tail.eq_ignore_ascii_case(c)) {
            return Some((cut, tok.len()));
        }
    }
    None
}

/// Tokenize one sentence into words with byte offsets.
///
/// `base` is added to every offset, so callers can report positions in the
/// coordinates of the whole input. Ports `windows.tokenize`.
pub(crate) fn tokenize(text: &str, base: usize) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    for (chunk_off, chunk) in whitespace_chunks(text) {
        if URLISH.is_match(chunk) {
            out.push(Word {
                text: chunk.to_string(),
                start: base + chunk_off,
                end: base + chunk_off + chunk.len(),
            });
            continue;
        }
        for m in TOKEN.find_iter(chunk) {
            let tok = m.as_str();
            let tok_off = chunk_off + m.start();
            match split_clitic(tok) {
                Some((cut, end)) => {
                    out.push(Word {
                        text: tok[..cut].to_string(),
                        start: base + tok_off,
                        end: base + tok_off + cut,
                    });
                    out.push(Word {
                        text: tok[cut..end].to_string(),
                        start: base + tok_off + cut,
                        end: base + tok_off + end,
                    });
                }
                None => out.push(Word {
                    text: tok.to_string(),
                    start: base + tok_off,
                    end: base + tok_off + tok.len(),
                }),
            }
        }
    }
    out
}

/// `str::split_whitespace`, but keeping each chunk's byte offset.
fn whitespace_chunks(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                out.push((s, &text[s..i]));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, &text[s..]));
    }
    out
}

/// Pack a document's units into windows of at most
/// [`MAX_WORDS_PER_WINDOW`] words.
///
/// A sentence never crosses a window and a window never crosses a document,
/// so results are independent per window. Ports `windows.build_windows`
/// with `min_tokens = 1`: a window is emitted whenever it holds any word.
pub(crate) fn build_windows(units: &[Unit<'_>]) -> Vec<Window> {
    let mut windows: Vec<Window> = Vec::new();
    let mut cur = Window::default();

    for (unit_idx, unit) in units.iter().enumerate() {
        for (sent_off, sent) in split_sentences(unit.text) {
            let mut words = tokenize(sent, unit.offset + sent_off);
            if words.is_empty() {
                continue;
            }
            // A single sentence past the budget is cut, not dropped.
            words.truncate(MAX_WORDS_PER_WINDOW);
            if cur.words.len() + words.len() > MAX_WORDS_PER_WINDOW && !cur.words.is_empty() {
                windows.push(std::mem::take(&mut cur));
            }
            let start = cur.words.len();
            cur.words.extend(words);
            cur.sentences.push(SentenceSpan {
                start,
                end: cur.words.len(),
                unit: unit_idx,
                speaker: unit.speaker.map(str::to_string),
            });
        }
    }
    if !cur.words.is_empty() {
        windows.push(cur);
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        tokenize(text, 0).into_iter().map(|w| w.text).collect()
    }

    #[test]
    fn clitics_split_at_the_shortest_stem() {
        assert_eq!(words("don't"), ["do", "n't"]);
        assert_eq!(words("isn't"), ["is", "n't"]);
        assert_eq!(words("Ada's"), ["Ada", "'s"]);
        assert_eq!(words("they're"), ["they", "'re"]);
        assert_eq!(words("I'd've"), ["I'd", "'ve"]);
        assert_eq!(words("I'm"), ["I", "'m"]);
        // A bare clitic has no stem, so the apostrophe falls out as
        // punctuation instead — matching the reference exactly.
        assert_eq!(words("'s"), ["'", "s"]);
    }

    #[test]
    fn urls_emails_numbers_and_ellipses_stay_whole() {
        assert_eq!(
            words("See https://example.com/a?b=1 now"),
            ["See", "https://example.com/a?b=1", "now"]
        );
        assert_eq!(words("ada@example.com"), ["ada@example.com"]);
        assert_eq!(words("www.example.com"), ["www.example.com"]);
        assert_eq!(
            words("3.14 and 1,200 and 50%"),
            ["3.14", "and", "1,200", "and", "50%"]
        );
        assert_eq!(words("the 3rd of 21st"), ["the", "3rd", "of", "21st"]);
        assert_eq!(
            words("wait... really?!"),
            ["wait", "...", "really", "?", "!"]
        );
        assert_eq!(words("state-of-the-art"), ["state-of-the-art"]);
    }

    #[test]
    fn word_offsets_point_back_at_the_input() {
        let text = "Ada  moved to Lisbon.";
        for w in tokenize(text, 0) {
            assert_eq!(&text[w.start..w.end], w.text, "offset of {:?}", w.text);
        }
        let shifted = tokenize(text, 100);
        assert_eq!(shifted[0].start, 100);
    }

    #[test]
    fn sentences_split_after_terminals_and_closing_quotes() {
        let got: Vec<&str> = split_sentences("One. Two! Three?")
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        assert_eq!(got, ["One.", "Two!", "Three?"]);

        // The closing quote is part of the separator, so `re.split`
        // discards it — the reference does the same.
        let got: Vec<&str> = split_sentences("He said \"stop.\" Then he left.")
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        assert_eq!(got, ["He said \"stop.", "Then he left."]);

        // No whitespace after the terminal: not a boundary.
        let got: Vec<&str> = split_sentences("v1.2 shipped")
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        assert_eq!(got, ["v1.2 shipped"]);

        // Lowercase after the terminal: not a boundary.
        let got: Vec<&str> = split_sentences("e.g. this one")
            .into_iter()
            .map(|(_, s)| s)
            .collect();
        assert_eq!(got, ["e.g. this one"]);
    }

    #[test]
    fn sentence_offsets_point_back_at_the_input() {
        let text = "One. Two! Three?";
        for (off, s) in split_sentences(text) {
            assert_eq!(&text[off..off + s.len()], s);
        }
    }

    #[test]
    fn paragraphs_drop_blanks_and_keep_offsets() {
        let text = "First para.\n\n\n\nSecond para.";
        let units = paragraphs(text);
        assert_eq!(units.len(), 2);
        for u in &units {
            assert_eq!(&text[u.offset..u.offset + u.text.len()], u.text);
        }
    }

    #[test]
    fn units_of_one_document_share_a_window() {
        // Windows never cross a *document*; paragraphs and turns of the
        // same document pack together while they fit the budget.
        let units = paragraphs("Ada moved.\n\nShe stayed.");
        let windows = build_windows(&units);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].sentences.len(), 2);
        assert_eq!(windows[0].sentences[1].unit, 1);
    }

    #[test]
    fn a_sentence_never_crosses_a_window() {
        let long = format!("{} end.", "word ".repeat(400));
        let second = format!("{} fin.", "other ".repeat(300));
        let text = format!("{long}\n\n{second}");
        let units = paragraphs(&text);
        let windows = build_windows(&units);
        assert!(windows.len() >= 2);
        for w in &windows {
            assert!(w.words.len() <= MAX_WORDS_PER_WINDOW);
            for s in &w.sentences {
                assert!(s.end <= w.words.len());
            }
            // Sentences tile the window with no gap.
            assert_eq!(w.sentences.first().map(|s| s.start), Some(0));
            assert_eq!(w.sentences.last().map(|s| s.end), Some(w.words.len()));
        }
    }

    #[test]
    fn turns_become_units_with_speakers() {
        let turns = [
            crate::traits::Turn::new("user", "I moved to Lisbon last year."),
            crate::traits::Turn::new("assistant", "   "),
            crate::traits::Turn::new("assistant", "Congratulations!"),
        ];
        let units = turns_as_units(&turns);
        assert_eq!(units.len(), 2, "blank turns are dropped with their speaker");
        let windows = build_windows(&units);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].participants(), ["user", "assistant"]);
    }
}
