// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Entities from typed mention spans and coreference chains.
//!
//! A port of `kniv_v7/entities.py`. Mentions become entities in two stages:
//! a coreference chain (in the configured mode) absorbs the spans it
//! overlaps, then exact normalised surface forms are merged for names.
//! Numbers and times are never merged by surface form — two occurrences of
//! `1` or `Monday` are not the same entity.
//!
//! The `partial_names`, `link` and `link_merge` options of the reference
//! function are not ported: `Kniv.entities` never passes them, so no
//! released model's output depends on them.

use std::collections::HashMap;

use super::decode::bio_spans;

/// Fine OntoNotes types to the six coarse types.
///
/// Unreachable from the shipped pipeline — the entity head already emits
/// coarse tags and the span head emits a placeholder — but kept so a future
/// fine-grained tag source decodes the same way the reference would.
const TYPE_MAP: &[(&str, &str)] = &[
    ("PERSON", "PER"),
    ("ORG", "ORG"),
    ("GPE", "LOC"),
    ("LOC", "LOC"),
    ("FAC", "LOC"),
    ("DATE", "TIME"),
    ("TIME", "TIME"),
    ("PERCENT", "NUM"),
    ("MONEY", "NUM"),
    ("QUANTITY", "NUM"),
    ("ORDINAL", "NUM"),
    ("CARDINAL", "NUM"),
    ("NORP", "MISC"),
    ("PRODUCT", "MISC"),
    ("EVENT", "MISC"),
    ("WORK_OF_ART", "MISC"),
    ("LAW", "MISC"),
    ("LANGUAGE", "MISC"),
];

/// Types whose mentions merge on an equal normalised surface form.
const MERGEABLE: [&str; 4] = ["PER", "ORG", "LOC", "MISC"];
/// The six coarse types, which pass through [`TYPE_MAP`] untouched.
const COARSE_TYPES: [&str; 6] = ["LOC", "MISC", "NUM", "ORG", "PER", "TIME"];
/// Counts and ordinals are not entities for relation extraction.
const DROP_TYPES: [&str; 2] = ["CARDINAL", "ORDINAL"];

/// The coarse entity types of the default entity head, and the BIO tag
/// vocabulary built from them — `["O", "B-LOC", "I-LOC", ...]`, in the
/// reference's order.
pub(crate) fn default_entity_tags() -> Vec<String> {
    let mut out = vec!["O".to_string()];
    for t in COARSE_TYPES {
        out.push(format!("B-{t}"));
        out.push(format!("I-{t}"));
    }
    out
}

/// Function words ignored when comparing two names.
const FUNCTION: [&str; 22] = [
    "the", "a", "an", "of", "and", "&", "'s", "’s", ",", ".", "-", "inc", "inc.", "ltd", "ltd.",
    "co", "co.", "corp", "corp.", "mr", "mr.", "mrs",
];
/// The reference's `_FUNCTION` set also carries these; kept separate only
/// because Rust array lengths are part of the type.
const FUNCTION_EXTRA: [&str; 3] = ["mrs.", "dr", "dr."];

fn is_function_word(w: &str) -> bool {
    FUNCTION.contains(&w) || FUNCTION_EXTRA.contains(&w)
}

/// The content words of a mention, lowercased.
fn name_tokens(tokens: &[String], a: usize, b: usize) -> Vec<String> {
    tokens[a..b]
        .iter()
        .map(|t| t.to_lowercase())
        .filter(|t| !is_function_word(t))
        .collect()
}

/// Whether `short` is an acronym of `long_raw`, e.g. `UN` of
/// `United Nations`.
fn is_acronym(raw_short: &[String], long_raw: &[String]) -> bool {
    let s: String = raw_short.concat().replace('.', "");
    let caps: Vec<&String> = long_raw
        .iter()
        .filter(|t| {
            t.chars().next().is_some_and(char::is_uppercase) && !is_function_word(&t.to_lowercase())
        })
        .collect();
    let initials: String = caps
        .iter()
        .filter_map(|t| t.chars().next())
        .collect::<String>();
    s.chars().count() >= 2
        && !s.is_empty()
        && s.chars().all(|c| !c.is_lowercase())
        && s.chars().any(char::is_uppercase)
        && caps.len() >= 2
        && s == initials
}

/// Could two mentions name the same thing: equal names, one name's content
/// words inside the other's (`Obama` / `Barack Obama`), or an acronym of it.
pub(crate) fn names_compatible(tokens: &[String], m1: (usize, usize), m2: (usize, usize)) -> bool {
    let n1 = name_tokens(tokens, m1.0, m1.1);
    let n2 = name_tokens(tokens, m2.0, m2.1);
    if n1.is_empty() || n2.is_empty() {
        return false;
    }
    if n1 == n2 {
        return true;
    }
    let (short, long_, rs, rl) = if n1.len() <= n2.len() {
        (&n1, &n2, m1, m2)
    } else {
        (&n2, &n1, m2, m1)
    };
    if short.iter().all(|t| long_.contains(t)) {
        return true;
    }
    is_acronym(&tokens[rs.0..rs.1], &tokens[rl.0..rl.1])
}

/// One mention of a grouped entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mention {
    /// Surface text, words joined with single spaces.
    pub name: String,
    /// Half-open word span, window-absolute.
    pub span: (usize, usize),
    /// Which sentence the mention sits in.
    pub sentence: usize,
    /// Coarse type from the tag source.
    pub ty: String,
}

/// How coreference chains group entity mentions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CorefMode {
    /// Chains are not used — both released kniv v7 models ship this.
    None,
    /// Every chain groups the mentions it overlaps, split by type.
    All,
    /// Within a chain, only mentions whose names are compatible.
    Compatible,
}

impl CorefMode {
    /// Parse a `calibration.json` `entity_coref` value. Unknown values fall
    /// back to the reference's own default, `"all"`.
    pub fn parse(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "compatible" => Self::Compatible,
            _ => Self::All,
        }
    }
}

/// Group typed mention spans into entities.
///
/// `ner_by_sentence[i]` holds sentence `i`'s BIO tags, or `None` when the
/// sentence was not analysed. `coref_chains` are clusters of **inclusive**
/// word spans. Returns an empty vector when fewer than two entities survive:
/// a relation needs a pair, and a window with one entity is not evidence of
/// anything.
pub(crate) fn build_entities(
    tokens: &[String],
    sentence_spans: &[(usize, usize)],
    ner_by_sentence: &[Option<Vec<String>>],
    coref_chains: &[Vec<(usize, usize)>],
    mode: CorefMode,
) -> Vec<Vec<Mention>> {
    // 1. Typed spans, window-absolute.
    let mut spans: Vec<(String, usize, usize)> = Vec::new();
    for (si, &(s, _e)) in sentence_spans.iter().enumerate() {
        let Some(Some(tags)) = ner_by_sentence.get(si) else {
            continue;
        };
        for (ty, a, b) in bio_spans(tags) {
            if DROP_TYPES.contains(&ty.as_str()) {
                continue;
            }
            let coarse = if COARSE_TYPES.contains(&ty.as_str()) {
                ty
            } else {
                TYPE_MAP
                    .iter()
                    .find(|(k, _)| *k == ty)
                    .map_or("MISC", |(_, v)| v)
                    .to_string()
            };
            spans.push((coarse, s + a, s + b));
        }
    }
    if spans.len() < 2 {
        return Vec::new();
    }

    // 2. Coreference grouping. A chain is not evidence that two mentions are
    //    the same entity when the tag source calls them different things, so
    //    members are split by type before grouping.
    let mut used: Vec<bool> = vec![false; spans.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    if mode != CorefMode::None {
        for chain in coref_chains {
            let mut members: Vec<usize> = Vec::new();
            for &(cs, ce) in chain {
                for (i, (_, a, b)) in spans.iter().enumerate() {
                    if !used[i] && *a <= ce && cs <= b.saturating_sub(1) {
                        members.push(i);
                        used[i] = true;
                    }
                }
            }
            // Insertion-ordered buckets, as Python dicts are.
            let mut order: Vec<String> = Vec::new();
            let mut by_type: HashMap<String, Vec<usize>> = HashMap::new();
            for i in members {
                let ty = spans[i].0.clone();
                if !by_type.contains_key(&ty) {
                    order.push(ty.clone());
                }
                by_type.entry(ty).or_default().push(i);
            }
            for ty in order {
                let ms = by_type.remove(&ty).unwrap_or_default();
                if mode == CorefMode::Compatible {
                    groups.extend(split_by_compatible_names(tokens, &spans, &ms));
                } else {
                    groups.push(ms);
                }
            }
        }
    }
    for (i, &taken) in used.iter().enumerate() {
        if !taken {
            groups.push(vec![i]);
        }
    }

    // 3. Merge names: equal normalised surface forms are one entity.
    let mut by_name: HashMap<(String, String), usize> = HashMap::new();
    let mut merged: Vec<Vec<usize>> = Vec::new();
    for g in groups {
        let (ref ty, a, b) = spans[g[0]];
        let key = (ty.clone(), tokens[a..b].join(" ").to_lowercase());
        let mergeable = MERGEABLE.contains(&ty.as_str());
        if mergeable && key.1.chars().count() > 2 && by_name.contains_key(&key) {
            let at = by_name[&key];
            merged[at].extend(g);
        } else {
            if mergeable {
                by_name.insert(key, merged.len());
            }
            merged.push(g);
        }
    }

    // 4. Mentions, dropping any that cross a sentence boundary.
    let offsets: Vec<usize> = sentence_spans.iter().map(|&(s, _)| s).collect();
    let sentence_of = |pos: usize| -> usize {
        offsets
            .iter()
            .enumerate()
            .filter(|&(_, &o)| o <= pos)
            .map(|(i, _)| i)
            .next_back()
            .unwrap_or(0)
    };
    let mut out: Vec<Vec<Mention>> = Vec::new();
    for g in merged {
        let mut mentions: Vec<Mention> = Vec::new();
        for i in g {
            let (ty, a, b) = &spans[i];
            let si = sentence_of(*a);
            let (_, se) = sentence_spans[si];
            // A mention that runs past its own sentence has no reading.
            if *b > se {
                continue;
            }
            mentions.push(Mention {
                name: tokens[*a..*b].join(" "),
                span: (*a, *b),
                sentence: si,
                ty: ty.clone(),
            });
        }
        if !mentions.is_empty() {
            out.push(mentions);
        }
    }
    if out.len() < 2 { Vec::new() } else { out }
}

/// Within one chain-and-type bucket, split into connected components of
/// name-compatible mentions.
fn split_by_compatible_names(
    tokens: &[String],
    spans: &[(String, usize, usize)],
    ms: &[usize],
) -> Vec<Vec<usize>> {
    let mut parent: HashMap<usize, usize> = ms.iter().map(|&i| (i, i)).collect();
    fn find(parent: &mut HashMap<usize, usize>, mut i: usize) -> usize {
        while parent[&i] != i {
            let p = parent[&i];
            let pp = parent[&p];
            parent.insert(i, pp);
            i = pp;
        }
        i
    }
    for x in 0..ms.len() {
        for y in (x + 1)..ms.len() {
            let (ix, iy) = (ms[x], ms[y]);
            let m1 = (spans[ix].1, spans[ix].2);
            let m2 = (spans[iy].1, spans[iy].2);
            if names_compatible(tokens, m1, m2) {
                let rx = find(&mut parent, ix);
                let ry = find(&mut parent, iy);
                parent.insert(rx, ry);
            }
        }
    }
    let mut order: Vec<usize> = Vec::new();
    let mut comp: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in ms {
        let r = find(&mut parent, i);
        if !comp.contains_key(&r) {
            order.push(r);
        }
        comp.entry(r).or_default().push(i);
    }
    order.into_iter().filter_map(|r| comp.remove(&r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn tags(v: &[&str]) -> Option<Vec<String>> {
        Some(v.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn default_entity_tags_match_the_reference_order() {
        let t = default_entity_tags();
        assert_eq!(t.len(), 13);
        assert_eq!(&t[..5], ["O", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]);
        assert_eq!(&t[11..], ["B-TIME", "I-TIME"]);
    }

    #[test]
    fn names_compatible_covers_subset_and_acronym() {
        let t = toks(&[
            "Barack", "Obama", "Obama", "United", "Nations", "UN", "Mary", "Smith",
        ]);
        assert!(names_compatible(&t, (0, 2), (2, 3)), "Obama ⊂ Barack Obama");
        assert!(
            names_compatible(&t, (3, 5), (5, 6)),
            "UN acronym of United Nations"
        );
        assert!(!names_compatible(&t, (0, 2), (6, 8)), "different people");
    }

    #[test]
    fn same_surface_form_merges_but_numbers_never_do() {
        let tokens = toks(&["Acme", "grew", ".", "Acme", "hired", "3", "and", "3"]);
        let sentences = [(0usize, 3usize), (3, 8)];
        let ner = [
            tags(&["B-ORG", "O", "O"]),
            tags(&["B-ORG", "O", "B-NUM", "O", "B-NUM"]),
        ];
        let ents = build_entities(&tokens, &sentences, &ner, &[], CorefMode::None);
        let org: Vec<&Vec<Mention>> = ents.iter().filter(|e| e[0].ty == "ORG").collect();
        assert_eq!(org.len(), 1, "both Acme mentions are one entity");
        assert_eq!(org[0].len(), 2);
        let num = ents.iter().filter(|e| e[0].ty == "NUM").count();
        assert_eq!(num, 2, "two 3s are two entities");
    }

    #[test]
    fn fewer_than_two_entities_is_no_entities() {
        let tokens = toks(&["Acme", "grew"]);
        let ner = [tags(&["B-ORG", "O"])];
        assert!(build_entities(&tokens, &[(0, 2)], &ner, &[], CorefMode::None).is_empty());
    }

    #[test]
    fn coref_chains_group_only_in_the_modes_that_use_them() {
        // "Acme ... it ... Globex": a chain over Acme and it.
        let tokens = toks(&["Acme", "grew", ".", "It", "hired", "Globex"]);
        let sentences = [(0usize, 3usize), (3, 6)];
        let ner = [tags(&["B-ORG", "O", "O"]), tags(&["B-ORG", "O", "B-ORG"])];
        let chain = vec![vec![(0usize, 0usize), (3, 3)]];

        let none = build_entities(&tokens, &sentences, &ner, &chain, CorefMode::None);
        assert_eq!(none.len(), 3, "three separate ORGs: {none:?}");

        let all = build_entities(&tokens, &sentences, &ner, &chain, CorefMode::All);
        assert_eq!(all.len(), 2, "Acme and It merge: {all:?}");
        assert!(all.iter().any(|e| e.len() == 2));

        // "Acme" and "It" have no compatible names, so Compatible keeps them
        // apart even though the chain links them.
        let compat = build_entities(&tokens, &sentences, &ner, &chain, CorefMode::Compatible);
        assert_eq!(compat.len(), 3, "{compat:?}");
    }

    #[test]
    fn cardinals_and_ordinals_are_dropped() {
        let tokens = toks(&["Acme", "is", "first", "of", "3", "firms", "like", "Globex"]);
        let ner = [tags(&[
            "B-ORG",
            "O",
            "B-ORDINAL",
            "O",
            "B-CARDINAL",
            "O",
            "O",
            "B-ORG",
        ])];
        let ents = build_entities(&tokens, &[(0, 8)], &ner, &[], CorefMode::None);
        assert_eq!(ents.len(), 2);
        assert!(ents.iter().all(|e| e[0].ty == "ORG"));
    }
}
