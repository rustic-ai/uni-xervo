// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Decoding: the host-side half of the model.
//!
//! A port of the decoding section of `kniv_v7/runtime.py`. None of this is an
//! implementation choice — the graphs emit scores that are *defined* to be
//! decoded this way, and the published accuracy figures are figures for
//! these algorithms. Replacing the Viterbi pass with per-token argmax, or
//! Chu-Liu-Edmonds with a per-row argmax, changes the model's answers.
//!
//! Every function here is pure, so each is unit-tested against small
//! hand-built matrices rather than against a 1.5 GB graph.

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

/// The reference's masking constant for "this cell is not available".
pub(crate) const NEG: f32 = -1e4;
/// The reference's Viterbi-internal "forbidden" score. Distinct from
/// [`NEG`]: a path may legitimately score below `NEG`.
const NEG_VITERBI: f64 = -1e9;

/// Row-wise `log_softmax`, computed in `f64` as the reference does before
/// casting back.
pub(crate) fn log_softmax(x: ArrayView2<f32>) -> Array2<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = Array2::<f32>::zeros((rows, cols));
    for r in 0..rows {
        let row = x.row(r);
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        if !max.is_finite() {
            // An all-`-inf` row: the reference would produce NaN here, but
            // such a row is never read (its word is masked out).
            continue;
        }
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        let log_sum = sum.ln();
        for c in 0..cols {
            out[[r, c]] = (row[c] as f64 - max - log_sum) as f32;
        }
    }
    out
}

/// `sigmoid`, in `f64` as the reference does.
pub(crate) fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

/// Index of the largest element, ties to the lowest index.
pub(crate) fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// [`argmax`] in `f64`, so the dependency decoder never loses a tie to a
/// narrowing cast.
fn argmax_f64(row: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Row-wise [`argmax`] over `[rows, classes]`.
pub(crate) fn argmax_rows(x: ArrayView2<f32>) -> Vec<usize> {
    (0..x.nrows())
        .map(|r| {
            let row = x.row(r);
            let slice: Vec<f32> = row.to_vec();
            argmax(&slice)
        })
        .collect()
}

/// BIO transition constraints: `(allowed, can_start)`.
///
/// An `I-X` tag may follow only `B-X` or `I-X`, and may not open a sequence.
/// Without this the heads routinely emit `O I-PER`, which has no span
/// reading at all.
pub(crate) fn bio_constraints(tags: &[String]) -> (Array2<bool>, Vec<bool>) {
    let c = tags.len();
    let mut allowed = Array2::from_elem((c, c), true);
    let mut start = vec![true; c];
    for (j, t) in tags.iter().enumerate() {
        if let Some(base) = t.strip_prefix("I-") {
            start[j] = false;
            let b = format!("B-{base}");
            for (i, s) in tags.iter().enumerate() {
                allowed[[i, j]] = s == &b || s == t;
            }
        }
    }
    (allowed, start)
}

/// Constrained Viterbi over `[T, C]` log-probabilities.
///
/// `keep`, when given, masks out (tag, position) pairs the caller forbids —
/// SRL uses it to pin `V` to the predicate and nowhere else.
pub(crate) fn viterbi(
    logp: ArrayView2<f32>,
    allowed: &Array2<bool>,
    start: &[bool],
    keep: Option<&Array2<bool>>,
) -> Vec<usize> {
    let (t_len, c) = (logp.nrows(), logp.ncols());
    if t_len == 0 {
        return Vec::new();
    }
    let mut lp = Array2::<f64>::zeros((t_len, c));
    for t in 0..t_len {
        for k in 0..c {
            let keep_it = keep.is_none_or(|m| m[[t, k]]);
            lp[[t, k]] = if keep_it {
                logp[[t, k]] as f64
            } else {
                NEG_VITERBI
            };
        }
    }

    let mut score: Vec<f64> = (0..c)
        .map(|k| if start[k] { lp[[0, k]] } else { NEG_VITERBI })
        .collect();
    let mut back = Array2::<usize>::zeros((t_len, c));
    for t in 1..t_len {
        let mut next = vec![0.0f64; c];
        for j in 0..c {
            // argmax over the previous tag, ties to the lowest index — which
            // is what numpy's `argmax(0)` gives.
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for i in 0..c {
                let v = score[i] + if allowed[[i, j]] { 0.0 } else { NEG_VITERBI };
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            back[[t, j]] = best;
            next[j] = best_v + lp[[t, j]];
        }
        score = next;
    }

    let mut path = vec![0usize; t_len];
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (k, &v) in score.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = k;
        }
    }
    path[t_len - 1] = best;
    for t in (1..t_len).rev() {
        path[t - 1] = back[[t, path[t]]];
    }
    path
}

/// `(label, start, end_exclusive)` spans from BIO tags.
///
/// Ports `entities.bio_spans` literally, including its tolerance of an
/// `I-Y` continuing a `B-X` span — the constrained Viterbi above never
/// produces that, but a caller decoding without constraints might.
pub(crate) fn bio_spans(tags: &[String]) -> Vec<(String, usize, usize)> {
    let mut spans = Vec::new();
    let mut cur: Option<String> = None;
    let mut start = 0usize;
    for i in 0..=tags.len() {
        let t: &str = if i < tags.len() { &tags[i] } else { "O" };
        match t.strip_prefix("I-") {
            None => {
                if let Some(label) = cur.take() {
                    spans.push((label, start, i));
                }
                cur = t.strip_prefix("B-").map(str::to_string);
                start = i;
            }
            Some(base) => {
                if cur.is_none() {
                    cur = Some(base.to_string());
                    start = i;
                }
            }
        }
    }
    spans
}

/// Apply a lemmatiser edit script `"case|strip|append"`.
///
/// An empty `case` field keeps the surface case; otherwise the word is
/// lowercased first. `strip` characters come off the end, then `append` goes
/// on. Returns the word unchanged when the script is malformed.
pub(crate) fn apply_script(word: &str, script: &str) -> String {
    let mut parts = script.splitn(3, '|');
    let (case, strip, append) = match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return word.to_string(),
    };
    let w = if case.is_empty() {
        word.to_string()
    } else {
        word.to_lowercase()
    };
    let strip: usize = strip.parse().unwrap_or(0);
    // The reference slices by Python characters, so count characters here
    // rather than bytes.
    let keep = w.chars().count().saturating_sub(strip);
    let stem: String = w.chars().take(keep).collect();
    format!("{stem}{append}")
}

/// Find the cycle reachable from some node, as `_cycle` does. `heads[0]` is
/// the root sentinel and is never part of a cycle.
fn find_cycle(heads: &[isize]) -> Option<Vec<usize>> {
    let n = heads.len();
    let mut state = vec![0u8; n];
    for start in 1..n {
        let mut path: Vec<usize> = Vec::new();
        let mut j = start as isize;
        while j > 0 && state[j as usize] == 0 {
            state[j as usize] = 1;
            path.push(j as usize);
            j = heads[j as usize];
        }
        if j > 0 && state[j as usize] == 1 {
            let at = path.iter().position(|&k| k == j as usize).unwrap_or(0);
            return Some(path[at..].to_vec());
        }
        for k in path {
            state[k] = 2;
        }
    }
    None
}

/// Chu-Liu-Edmonds over a dense `(n+1)x(n+1)` score matrix, column `0`
/// being the root. Returns each node's head column; `heads[0] == -1`.
fn chu_liu_edmonds(s: &Array2<f64>) -> Vec<isize> {
    let n = s.nrows();
    let mut heads: Vec<isize> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..n).map(|j| s[[i, j]]).collect();
            argmax_f64(&row) as isize
        })
        .collect();
    heads[0] = -1;
    let Some(cyc) = find_cycle(&heads) else {
        return heads;
    };

    let in_cycle: Vec<bool> = {
        let mut v = vec![false; n];
        for &c in &cyc {
            v[c] = true;
        }
        v
    };
    let rest: Vec<usize> = (0..n).filter(|&i| !in_cycle[i]).collect();
    let c = rest.len();

    // The contracted graph: `rest` nodes plus one node (index `c`) standing
    // for the whole cycle.
    let mut s2 = Array2::<f64>::from_elem((c + 1, c + 1), f64::NEG_INFINITY);
    for (a, &i) in rest.iter().enumerate() {
        for (b, &j) in rest.iter().enumerate() {
            s2[[a, b]] = s[[i, j]];
        }
    }
    // Entering the cycle: the gain over the arc it replaces.
    let held: Vec<f64> = cyc.iter().map(|&i| s[[i, heads[i] as usize]]).collect();
    let mut enter = Array2::<f64>::from_elem((cyc.len(), c), f64::NEG_INFINITY);
    for (k, &i) in cyc.iter().enumerate() {
        for (b, &j) in rest.iter().enumerate() {
            enter[[k, b]] = s[[i, j]] - held[k];
        }
    }
    for b in 0..c {
        s2[[c, b]] = (0..cyc.len()).fold(f64::NEG_INFINITY, |m, k| m.max(enter[[k, b]]));
    }
    // Leaving the cycle.
    let mut leave = Array2::<f64>::from_elem((c, cyc.len()), f64::NEG_INFINITY);
    for (a, &i) in rest.iter().enumerate() {
        for (k, &j) in cyc.iter().enumerate() {
            leave[[a, k]] = s[[i, j]];
        }
    }
    for a in 0..c {
        s2[[a, c]] = (0..cyc.len()).fold(f64::NEG_INFINITY, |m, k| m.max(leave[[a, k]]));
    }
    for d in 0..=c {
        s2[[d, d]] = f64::NEG_INFINITY;
    }

    let sub = chu_liu_edmonds(&s2);
    let mut out = heads.clone();
    for (k, &d) in rest.iter().enumerate() {
        if d == 0 {
            continue;
        }
        out[d] = if sub[k] == c as isize {
            let row: Vec<f64> = (0..cyc.len()).map(|t| leave[[k, t]]).collect();
            cyc[argmax_f64(&row)] as isize
        } else {
            rest[sub[k] as usize] as isize
        };
    }
    // Break the cycle at its best entry point.
    let h = sub[c];
    if h >= 0 {
        let col: Vec<f64> = (0..cyc.len()).map(|k| enter[[k, h as usize]]).collect();
        out[cyc[argmax_f64(&col)]] = rest[h as usize] as isize;
    }
    out
}

/// Chu-Liu-Edmonds, then force a single root.
///
/// A maximum spanning arborescence can attach several nodes to the root;
/// a dependency tree may not. When that happens the reference re-solves once
/// per candidate root with the other root arcs masked and keeps the
/// highest-scoring tree.
pub(crate) fn tree_heads(s: &Array2<f64>) -> Vec<isize> {
    let heads = chu_liu_edmonds(s);
    let roots: Vec<usize> = (1..heads.len()).filter(|&i| heads[i] == 0).collect();
    if roots.len() <= 1 {
        return heads;
    }
    let mut best = heads;
    let mut best_score = f64::NEG_INFINITY;
    for &r in &roots {
        let mut t = s.clone();
        for i in 1..t.nrows() {
            t[[i, 0]] = f64::NEG_INFINITY;
        }
        t[[r, 0]] = s[[r, 0]];
        let h = chu_liu_edmonds(&t);
        let score: f64 = (1..h.len())
            .map(|i| {
                let j = h[i];
                if j < 0 { 0.0 } else { t[[i, j as usize]] }
            })
            .sum();
        if score > best_score {
            best_score = score;
            best = h;
        }
    }
    best
}

/// One dependency tree per sentence over `arc [W, W+1]` (column `0` is the
/// root), returning each word's head column: `0` for a root, else the
/// 1-based window index of its head.
///
/// Words outside any sentence, or masked out, keep the raw per-row argmax —
/// the reference's behaviour, and harmless since they are never reported.
pub(crate) fn decode_trees(arc: ArrayView2<f32>, sent: &[i64], word_mask: &[bool]) -> Vec<i64> {
    let mut out: Vec<i64> = argmax_rows(arc).into_iter().map(|v| v as i64).collect();
    let lp = log_softmax(arc);

    let mut seen: Vec<i64> = Vec::new();
    for (i, &s) in sent.iter().enumerate() {
        if word_mask[i] && s >= 0 && !seen.contains(&s) {
            seen.push(s);
        }
    }
    seen.sort_unstable();

    for s in seen {
        let w: Vec<usize> = (0..sent.len())
            .filter(|&i| sent[i] == s && word_mask[i])
            .collect();
        let n = w.len();
        let mut mat = Array2::<f64>::from_elem((n + 1, n + 1), f64::NEG_INFINITY);
        for (a, &i) in w.iter().enumerate() {
            mat[[a + 1, 0]] = lp[[i, 0]] as f64;
            for (b, &j) in w.iter().enumerate() {
                mat[[a + 1, b + 1]] = lp[[i, j + 1]] as f64;
            }
        }
        for d in 0..=n {
            mat[[d, d]] = f64::NEG_INFINITY;
        }
        let h = tree_heads(&mat);
        for (a, &i) in w.iter().enumerate() {
            let head = h[a + 1];
            out[i] = if head == 0 {
                0
            } else {
                w[(head.max(1) - 1) as usize] as i64 + 1
            };
        }
    }
    out
}

/// Union-find over a word-to-antecedent map, keeping clusters of two or more
/// words. Clusters and their members are sorted, so the output is stable.
pub(crate) fn link_clusters(antecedent: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: HashMap<usize, usize> = HashMap::new();
    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        parent.entry(x).or_insert(x);
        let mut x = x;
        loop {
            let p = parent[&x];
            if p == x {
                return x;
            }
            let pp = parent[&p];
            parent.insert(x, pp);
            x = pp;
        }
    }
    for &(w, a) in antecedent {
        let rw = find(&mut parent, w);
        let ra = find(&mut parent, a);
        parent.insert(rw, ra);
    }
    let keys: Vec<usize> = parent.keys().copied().collect();
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for w in keys {
        let r = find(&mut parent, w);
        groups.entry(r).or_default().push(w);
    }
    let mut out: Vec<Vec<usize>> = groups
        .into_values()
        .filter(|g| g.len() > 1)
        .map(|mut g| {
            g.sort_unstable();
            g
        })
        .collect();
    out.sort();
    out
}

/// Every span of up to `max_width` words that lies inside one sentence and
/// starts and ends on an analysed word (`TypeCollator`).
pub(crate) fn span_candidates(
    sent: &[i64],
    word_ok: &[bool],
    max_width: usize,
) -> Vec<(usize, usize)> {
    let n = sent.len();
    let mut out = Vec::new();
    for s in 0..n {
        if !word_ok[s] || sent[s] < 0 {
            continue;
        }
        for e in (s + 1)..=(s + max_width).min(n) {
            if sent[e - 1] != sent[s] {
                break;
            }
            if word_ok[e - 1] {
                out.push((s, e));
            }
        }
    }
    out
}

/// Greedy flat span decoding: take spans by descending calibrated margin
/// while the margin is positive, skipping any that overlap one already
/// taken.
///
/// `logits` is `[N, K+1]` with column `0` the learned threshold. Returns
/// `(start, end, type_index)` sorted by position.
pub(crate) fn decode_spans(
    logits: ArrayView2<f32>,
    spans: &[(usize, usize)],
    offset: f32,
) -> Vec<(usize, usize, usize)> {
    if logits.nrows() == 0 || spans.is_empty() {
        return Vec::new();
    }
    let n_types = logits.ncols().saturating_sub(1);
    let mut best: Vec<f32> = Vec::with_capacity(logits.nrows());
    let mut which: Vec<usize> = Vec::with_capacity(logits.nrows());
    for r in 0..logits.nrows() {
        let base = logits[[r, 0]];
        let row: Vec<f32> = (0..n_types)
            .map(|k| logits[[r, k + 1]] - base - offset)
            .collect();
        let k = argmax(&row);
        best.push(row.get(k).copied().unwrap_or(f32::NEG_INFINITY));
        which.push(k);
    }

    // `np.argsort(-best, kind="stable")`: descending margin, ties by index.
    let mut order: Vec<usize> = (0..best.len()).collect();
    order.sort_by(|&a, &b| {
        best[b]
            .partial_cmp(&best[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let width = spans.iter().map(|&(_, e)| e).max().unwrap_or(0) + 1;
    let mut taken = vec![false; width];
    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for i in order {
        if best[i] <= 0.0 {
            break;
        }
        let (s, e) = spans[i];
        if taken[s..e].iter().any(|&t| t) {
            continue;
        }
        taken[s..e].iter_mut().for_each(|t| *t = true);
        out.push((s, e, which[i]));
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bio_constraints_forbid_orphan_continuations() {
        let t = tags(&["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]);
        let (allowed, start) = bio_constraints(&t);
        assert_eq!(start, [true, true, false, true, false]);
        // I-PER follows only B-PER or I-PER.
        assert!(allowed[[1, 2]] && allowed[[2, 2]]);
        assert!(!allowed[[0, 2]] && !allowed[[3, 2]] && !allowed[[4, 2]]);
        // Nothing constrains a B- or O tag.
        for i in 0..t.len() {
            assert!(allowed[[i, 0]] && allowed[[i, 1]] && allowed[[i, 3]]);
        }
    }

    #[test]
    fn viterbi_repairs_what_argmax_would_break() {
        let t = tags(&["O", "B-PER", "I-PER"]);
        let (allowed, start) = bio_constraints(&t);
        // Per-token argmax would read O, I-PER — an orphan continuation.
        let logp = array![[1.0f32, 0.0, 0.9], [0.0, 0.0, 5.0]];
        let argmaxed = argmax_rows(logp.view());
        assert_eq!(argmaxed, [0, 2]);
        let path = viterbi(logp.view(), &allowed, &start, None);
        assert_eq!(
            path.iter().map(|&k| t[k].as_str()).collect::<Vec<_>>(),
            ["B-PER", "I-PER"],
            "the constrained path must be readable as a span"
        );
    }

    #[test]
    fn viterbi_keep_mask_pins_a_tag_to_one_position() {
        let t = tags(&["O", "V", "B-ARG0"]);
        let (allowed, start) = bio_constraints(&t);
        let logp = array![[0.1f32, 5.0, 0.2], [0.1, 5.0, 0.2]];
        let mut keep = Array2::from_elem((2, 3), true);
        // V only at position 1, and position 1 is only ever V.
        keep[[0, 1]] = false;
        keep[[1, 0]] = false;
        keep[[1, 2]] = false;
        let path = viterbi(logp.view(), &allowed, &start, Some(&keep));
        assert_eq!(
            path.iter().map(|&k| t[k].as_str()).collect::<Vec<_>>(),
            ["B-ARG0", "V"]
        );
    }

    #[test]
    fn bio_spans_collapses_tags_into_half_open_spans() {
        let t = tags(&["O", "B-PER", "I-PER", "O", "B-ORG", "B-ORG"]);
        assert_eq!(
            bio_spans(&t),
            [
                ("PER".to_string(), 1, 3),
                ("ORG".to_string(), 4, 5),
                ("ORG".to_string(), 5, 6),
            ]
        );
        // An orphan I- opens a span, as the reference allows.
        assert_eq!(
            bio_spans(&tags(&["I-PER", "I-PER"])),
            [("PER".to_string(), 0, 2)]
        );
        assert!(bio_spans(&tags(&["O", "O"])).is_empty());
    }

    #[test]
    fn lemma_scripts_lowercase_strip_and_append() {
        assert_eq!(apply_script("Moved", "l|2|e"), "move");
        assert_eq!(apply_script("Lisbon", "|0|"), "Lisbon");
        assert_eq!(apply_script("mice", "l|4|mouse"), "mouse");
        // More stripped than there is: an empty stem, not a panic.
        assert_eq!(apply_script("ab", "l|9|x"), "x");
        assert_eq!(apply_script("word", "nonsense"), "word");
    }

    #[test]
    fn cle_finds_a_tree_where_row_argmax_finds_a_cycle() {
        // Three words. Row argmax makes 1 and 2 point at each other, leaving
        // no root — a cycle a per-row decode would happily emit.
        let mut s = Array2::<f64>::from_elem((4, 4), f64::NEG_INFINITY);
        // root arcs
        s[[1, 0]] = 0.0;
        s[[2, 0]] = -5.0;
        s[[3, 0]] = -9.0;
        // 1 <-> 2 strongly, 3 under 1
        s[[1, 2]] = 1.0;
        s[[2, 1]] = 1.0;
        s[[3, 1]] = 0.5;
        s[[2, 3]] = -8.0;
        s[[1, 3]] = -8.0;
        s[[3, 2]] = -8.0;
        for d in 0..4 {
            s[[d, d]] = f64::NEG_INFINITY;
        }
        let heads = tree_heads(&s);
        assert_eq!(heads[0], -1);
        // Exactly one root, and no cycles.
        let roots: Vec<usize> = (1..4).filter(|&i| heads[i] == 0).collect();
        assert_eq!(roots.len(), 1, "a dependency tree has one root: {heads:?}");
        assert!(find_cycle(&heads).is_none(), "{heads:?}");
    }

    #[test]
    fn cle_keeps_the_single_root_case_untouched() {
        let mut s = Array2::<f64>::from_elem((3, 3), f64::NEG_INFINITY);
        s[[1, 0]] = 0.0;
        s[[2, 0]] = -3.0;
        s[[2, 1]] = -0.1;
        s[[1, 2]] = -5.0;
        for d in 0..3 {
            s[[d, d]] = f64::NEG_INFINITY;
        }
        assert_eq!(tree_heads(&s), [-1, 0, 1]);
    }

    #[test]
    fn decode_trees_reports_one_based_heads_per_sentence() {
        // Two words in sentence 0, one in sentence 1. `arc` is [W, W+1].
        let arc = array![
            [0.0f32, -9.0, -1.0, -9.0],
            [-1.0, 5.0, -9.0, -9.0],
            [0.0, -9.0, -9.0, -9.0],
        ];
        let heads = decode_trees(arc.view(), &[0, 0, 1], &[true, true, true]);
        // Word 0 roots, word 1 attaches to word 0 (column 1), word 2 is its
        // own sentence's root.
        assert_eq!(heads, [0, 1, 0]);
    }

    #[test]
    fn link_clusters_merges_chains_and_drops_singletons() {
        // 5 -> 3 -> 1 is one cluster; 9 -> 9 is a self-link, so a singleton.
        let got = link_clusters(&[(5, 3), (3, 1), (9, 9)]);
        assert_eq!(got, [vec![1, 3, 5]]);
    }

    #[test]
    fn span_candidates_stay_inside_one_sentence() {
        let sent = [0i64, 0, 1, 1];
        let ok = [true, true, true, false];
        let got = span_candidates(&sent, &ok, 12);
        assert_eq!(got, [(0, 1), (0, 2), (1, 2), (2, 3)]);
    }

    #[test]
    fn decode_spans_is_greedy_by_margin_and_skips_overlaps() {
        let spans = [(0usize, 2usize), (1, 3), (3, 4)];
        // column 0 is the threshold; two types follow.
        let logits = array![
            [0.0f32, 1.0, 0.5], // margin 1.0 for type 0
            [0.0, 3.0, 0.5],    // margin 3.0 for type 0 — wins, blocks (0,2)
            [0.0, -1.0, -2.0],  // negative: dropped
        ];
        assert_eq!(decode_spans(logits.view(), &spans, 0.0), [(1, 3, 0)]);
        // A positive offset raises the bar.
        assert!(decode_spans(logits.view(), &spans, 4.0).is_empty());
    }

    #[test]
    fn log_softmax_rows_sum_to_one() {
        let x = array![[1.0f32, 2.0, 3.0]];
        let lp = log_softmax(x.view());
        let total: f32 = lp.row(0).iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        // Monotone in the input.
        assert!(lp[[0, 0]] < lp[[0, 1]] && lp[[0, 1]] < lp[[0, 2]]);
    }
}
