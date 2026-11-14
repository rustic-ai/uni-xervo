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

/// Abbreviations a sentence does not end after — titles, months, `No.`,
/// `vs.` — besides single initials (`D.`) and dotted acronyms (`U.S.`).
/// Ambiguous ones that often *do* end a sentence (`Inc.`, `etc.`, the state
/// names) are deliberately absent, and so is `May`, which is an ordinary
/// word. Ports `_NO_SPLIT`.
const NO_SPLIT: [&str; 45] = [
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Gen", "Col", "Lt", "Sgt", "Capt", "Cmdr", "Adm", "Maj",
    "Rep", "Sen", "Gov", "Pres", "Rev", "Hon", "St", "Mt", "Ft", "Sr", "Jr", "Jan", "Feb", "Mar",
    "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "No", "Nos", "vs", "Fig",
    "approx", "Vol", "pp", "Ave", "Blvd",
];

/// The rest of the reference's `_ABBREV`: company suffixes, `etc.` and the
/// US state abbreviations. These keep their period as part of the token (the
/// Penn Treebank and CoNLL convention) but are no bar to a sentence break.
const ABBREV_EXTRA: [&str; 39] = [
    "Inc", "Corp", "Co", "Ltd", "Bros", "etc", "Ala", "Ariz", "Ark", "Calif", "Colo", "Conn",
    "Del", "Fla", "Ga", "Ill", "Ind", "Kan", "Ky", "La", "Md", "Mass", "Mich", "Minn", "Miss",
    "Mo", "Mont", "Neb", "Nev", "Okla", "Ore", "Pa", "Tenn", "Tex", "Va", "Vt", "Wash", "Wis",
    "Wyo",
];

/// `_ABBREV` — [`NO_SPLIT`] plus [`ABBREV_EXTRA`] — longest first, which is
/// how the reference's `"|".join(sorted(_ABBREV, key=len, reverse=True))`
/// alternation comes to prefer `Sept` over `Sep`.
static ABBREV_BY_LEN: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut all: Vec<&'static str> = NO_SPLIT
        .iter()
        .chain(ABBREV_EXTRA.iter())
        .copied()
        .collect();
    all.sort_by_key(|s| std::cmp::Reverse(s.len()));
    all
});

/// Dotted acronyms kept whole: `U.S.`, `a.m.`, `N.C.` — the reference's
/// `(?:[A-Za-z]\.){2,}`, the first alternative of `_TOKEN`.
static ACRONYM_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[A-Za-z]\.){2,}").expect("ACRONYM_TOKEN regex"));

/// Numbers (with percents and ordinals), words (hyphenated, internal
/// apostrophe), ellipsis, single punctuation — the last alternatives of
/// `_TOKEN`, in preference order, which both `re` and `regex` honour
/// (leftmost-first). Anchored, because [`next_token`] walks the chunk a
/// position at a time the way `re.findall` does.
static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?|\w+(?:[-']\w+)*|\.\.\.|[^\w\s])")
        .expect("TOKEN regex")
});

/// A word that is itself a dotted acronym shorn of its final period —
/// `U.S` out of `U.S.`. The reference's
/// `re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", w)`.
static ACRONYM_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[A-Za-z]\.)+[A-Za-z]$").expect("ACRONYM_WORD regex"));

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

/// Whether the text up to a candidate break ends in something that is not a
/// sentence end after all: a title or month (`Sen.`, `Dec.`), a single
/// capital initial (`F.`), or a dotted acronym (`U.S.`).
///
/// Ports `_no_split_after`, whose `_LAST_WORD = (\S+)\.$` picks the last
/// whitespace-free run before the terminal period — so only a `.` can be
/// guarded, and a sentence ending in `!` or `?` always breaks.
fn no_split_after(part: &str) -> bool {
    let part = part.trim_end_matches(['"', '\'', ')', ']']);
    // `(\S+)\.$`: a period, preceded by at least one non-space character.
    let Some(stem) = part.strip_suffix('.') else {
        return false;
    };
    let word = match stem.rsplit_once(char::is_whitespace) {
        Some((_, last)) => last,
        None => stem,
    };
    if word.is_empty() {
        return false;
    }
    let word = word.trim_start_matches(['"', '\'', '(', '[']);
    NO_SPLIT.contains(&word)
        || (word.chars().count() == 1 && word.chars().all(char::is_uppercase))
        || ACRONYM_WORD.is_match(word)
}

/// Split a unit into sentences, as `(byte offset, text)` pairs.
///
/// Ports `_SENT_END = (?<=[.!?])["')\]]*\s+(?=[A-Z0-9"'(\[])` used with
/// `re.split`, together with the `_no_split_after` guard the reference
/// applies at every candidate break. The Rust `regex` crate has no
/// look-behind, so this is the equivalent scan: a boundary sits right after `.`, `!` or `?`, skips any
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
            // A guarded break is no break: the piece simply runs on, which
            // for a byte range means leaving `piece_start` where it is.
            Some(ws_end) => {
                if !no_split_after(&text[piece_start..chars[i].0]) {
                    pieces.push((piece_start, chars[i].0));
                    piece_start = chars[ws_end].0;
                }
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

/// Byte length of the token starting at `rest`, or `None` if nothing matches
/// there.
///
/// `_TOKEN`'s alternatives in the reference's order, two of which end in a
/// look-ahead the `regex` crate cannot express, so they are checked by hand:
/// an abbreviation keeps its period only before the end of the chunk or
/// `,;:)`, and a single capital initial only before the end of the chunk.
/// Within a chunk `\s` is unreachable, since `tokenize` splits on
/// whitespace first; it is accepted here for the sake of the invariant.
fn next_token(rest: &str) -> Option<usize> {
    if let Some(m) = ACRONYM_TOKEN.find(rest) {
        return Some(m.end());
    }
    let first = rest.as_bytes().first().copied().unwrap_or(0);
    for abbrev in ABBREV_BY_LEN.iter().filter(|a| a.as_bytes()[0] == first) {
        let Some(after_dot) = rest
            .strip_prefix(*abbrev)
            .and_then(|tail| tail.strip_prefix('.'))
        else {
            continue;
        };
        let stops = after_dot
            .chars()
            .next()
            .is_none_or(|c| c.is_whitespace() || matches!(c, ',' | ';' | ':' | ')'));
        if stops {
            return Some(abbrev.len() + 1);
        }
    }
    let mut initial = rest.chars();
    if let (Some(c), Some('.')) = (initial.next(), initial.next())
        && c.is_ascii_uppercase()
        && initial.next().is_none_or(char::is_whitespace)
    {
        return Some(2);
    }
    TOKEN.find(rest).map(|m| m.end())
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
        // `re.findall`'s own scan: try the alternatives at each position in
        // turn, and step over a character that starts none of them.
        let mut pos = 0usize;
        while pos < chunk.len() {
            // `next_token` cannot return 0 — every alternative needs at
            // least one character — but the loop advances on its own rather
            // than trusting that, since a zero-length match would hang it.
            let Some(len) = next_token(&chunk[pos..]).filter(|n| *n > 0) else {
                pos += chunk[pos..].chars().next().map(char::len_utf8).unwrap_or(1);
                continue;
            };
            let tok = &chunk[pos..pos + len];
            let tok_off = chunk_off + pos;
            pos += len;
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
    fn offsets_survive_abbreviations_and_multibyte_text() {
        // Every new token shape — acronym, abbreviation, initial — plus a
        // multibyte neighbour, since `next_token` does its own byte
        // arithmetic over the chunk.
        for text in [
            "Sen. Robert F. Kennedy met Acme Inc. in the U.S. at 9 a.m.",
            "Jos\u{e9} saw Dr. Garc\u{ed}a — Lt. Col. \u{d8}sterg\u{e5}rd — on Dec. 13.",
            "caf\u{e9}s, na\u{ef}ve r\u{e9}sum\u{e9}s... really?!",
        ] {
            let got = tokenize(text, 0);
            assert!(!got.is_empty(), "{text:?}");
            for w in &got {
                assert_eq!(text.get(w.start..w.end), Some(w.text.as_str()), "{text:?}");
            }
            // The words tile the input in order, with only whitespace and
            // unmatched characters between them.
            let mut prev_end = 0usize;
            for w in &got {
                assert!(w.start >= prev_end, "{text:?} overlap at {:?}", w.text);
                prev_end = w.end;
            }
        }
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

    fn sentences(text: &str) -> Vec<&str> {
        split_sentences(text).into_iter().map(|(_, s)| s).collect()
    }

    #[test]
    fn titles_initials_and_acronyms_do_not_end_a_sentence() {
        // Each of these is one sentence, not several (issue #54).
        assert_eq!(
            sentences("Sen. Robert F. Kennedy spoke today."),
            ["Sen. Robert F. Kennedy spoke today."]
        );
        assert_eq!(
            sentences("Dr. Smith met J. K. Rowling."),
            ["Dr. Smith met J. K. Rowling."]
        );
        assert_eq!(
            sentences("He joined the U.S. Army in 1990."),
            ["He joined the U.S. Army in 1990."]
        );
        assert_eq!(
            sentences("It was Dec. 13 when Lt. Col. Smith arrived."),
            ["It was Dec. 13 when Lt. Col. Smith arrived."]
        );
        assert_eq!(
            sentences("Mr. and Mrs. Lee met Prof. Chan at St. Mary hospital."),
            ["Mr. and Mrs. Lee met Prof. Chan at St. Mary hospital."]
        );
        assert_eq!(
            sentences("Gov. Brown signed it on Jan. 5 in Washington, D.C. after a vote."),
            ["Gov. Brown signed it on Jan. 5 in Washington, D.C. after a vote."]
        );
        assert_eq!(
            sentences("The case was Roe vs. Wade."),
            ["The case was Roe vs. Wade."]
        );
        assert_eq!(
            sentences("Michael D. Papagiannis wrote No. 7."),
            ["Michael D. Papagiannis wrote No. 7."]
        );
    }

    #[test]
    fn real_sentence_ends_still_split() {
        assert_eq!(
            sentences("The vote was 5 to 4. Justice Kennedy wrote it."),
            ["The vote was 5 to 4.", "Justice Kennedy wrote it."]
        );
        // `!` and `?` are never guarded: `_LAST_WORD` only looks at a period.
        assert_eq!(
            sentences("Is it done? Yes! It is."),
            ["Is it done?", "Yes!", "It is."]
        );
        // An ordinary word that happens to end in a period is no abbreviation.
        assert_eq!(
            sentences("She left. He stayed."),
            ["She left.", "He stayed."]
        );
        // `May` is a word before it is a month, so it is not in `NO_SPLIT`.
        assert_eq!(
            sentences("It happened in May. Then it stopped."),
            ["It happened in May.", "Then it stopped."]
        );
    }

    #[test]
    fn guarded_sentences_keep_their_offsets() {
        let text = "Sen. Robert F. Kennedy spoke. Dr. Smith agreed.";
        let got = split_sentences(text);
        assert_eq!(got.len(), 2);
        for (off, s) in got {
            assert_eq!(&text[off..off + s.len()], s);
        }
    }

    #[test]
    fn abbreviations_initials_and_acronyms_tokenize_whole() {
        assert_eq!(
            words("Sen. Robert F. Kennedy"),
            ["Sen.", "Robert", "F.", "Kennedy"]
        );
        assert_eq!(words("the U.S. Army"), ["the", "U.S.", "Army"]);
        assert_eq!(words("Dec. 13"), ["Dec.", "13"]);
        assert_eq!(words("Acme Inc. filed"), ["Acme", "Inc.", "filed"]);
        assert_eq!(words("e.g. this"), ["e.g.", "this"]);
        assert_eq!(words("at 9 a.m. sharp"), ["at", "9", "a.m.", "sharp"]);
        // The longest abbreviation wins, as the length-sorted alternation does.
        assert_eq!(words("Sept. 1"), ["Sept.", "1"]);
        // An abbreviation's period survives a following comma or paren.
        assert_eq!(words("Acme Corp., Ltd."), ["Acme", "Corp.", ",", "Ltd."]);
        // Not followed by the end of the chunk or `,;:)`: the period splits off.
        assert_eq!(words("Dec.13"), ["Dec", ".", "13"]);
        // A lone capital is an initial only at the end of its chunk.
        assert_eq!(words("F.Kennedy"), ["F", ".", "Kennedy"]);
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
