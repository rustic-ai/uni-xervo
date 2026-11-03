//! [`ModelTask::Rerank`](crate::api::ModelTask::Rerank) on `POST /v1/rerank`.
//!
//! The endpoint is llama.cpp's own, reachable as `/rerank`, `/reranking`,
//! `/v1/rerank` or `/v1/reranking`; this module posts to the `/v1` form so one
//! `base_url` serves every task. The server must have been started with
//! `--reranking` (or `--embedding --pooling rank`) and a cross-encoder reranker
//! such as `jina-reranker-v1-tiny-en` or `bge-reranker-v2-m3`. A server without
//! it answers HTTP 501, which surfaces as a non-retryable
//! [`RuntimeError::InferenceError`] naming the flag.
//!
//! # Options
//!
//! None beyond the shared keys documented on the [parent module](super).
//!
//! # Score domain
//!
//! `llama-server` returns a sigmoid of the cross-encoder logit — bounded to
//! `[0, 1]`, and in practice small even for a good match (0.13 for an exact
//! answer is typical). The ONNX cross-encoder reranker returns the **raw
//! logit** instead, unbounded in both directions. So rankings from the two
//! providers are comparable but absolute scores are not: a threshold tuned
//! against `local/onnx` will reject everything here. Neither score is converted
//! — a provider that quietly rescaled would make the difference harder to spot,
//! not easier.
//!
//! # `top_n` is deliberately not sent
//!
//! [`RerankerModel::rerank`] contracts to return a score for **every** input
//! document, and the server truncates to `top_n` when asked. Callers that want
//! the top *k* truncate the returned slice, which is already sorted.
//!
//! # Request shape
//!
//! The response shape depends on the *request* key: `documents` yields the Jina
//! form (`{"results": [{"index", "relevance_score"}]}`) and `texts` yields the
//! TEI form (a bare array of `{"index", "score"}`). This module always sends
//! `documents`, so the Jina form is what a real `llama-server` returns — but
//! [`parse_rerank_response`] accepts both, since a gateway in front of the
//! server may normalise one into the other.

use super::{LlamaCppCommon, post_json};
use crate::error::{Result, RuntimeError};
use crate::traits::{RerankerModel, ScoredDoc};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved rerank configuration for one alias.
///
/// Reranking adds no keys of its own, so this is the shared set verbatim. It
/// exists as a named type so the `load` dispatch reads the same way for every
/// task.
#[derive(Debug, Clone)]
pub(crate) struct RerankConfig {
    pub common: LlamaCppCommon,
}

impl RerankConfig {
    /// Parse the alias `options`. The catalog validator already enforces the
    /// same rules, but `ModelProvider::load` is public, so this is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        Ok(Self {
            common: LlamaCppCommon::from_options(options)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Response parsing (pure)
// ---------------------------------------------------------------------------

/// Read either response shape into `(index, score)` pairs, in the order the
/// server listed them.
///
/// Accepts the Jina form this module's requests produce and the TEI form a
/// proxy might, distinguished by structure rather than by any configured
/// flag — an array is TEI, an object with `results` is Jina.
fn parse_rerank_response(body: &Value) -> Result<Vec<(usize, f32)>> {
    let (entries, score_key) = match body {
        Value::Array(entries) => (entries, "score"),
        _ => {
            let entries = body
                .get("results")
                .and_then(|r| r.as_array())
                .ok_or_else(|| {
                    RuntimeError::ApiError(
                        "llama.cpp /v1/rerank response malformed: expected a 'results' array \
                         or a top-level array"
                            .to_string(),
                    )
                })?;
            (entries, "relevance_score")
        }
    };

    entries
        .iter()
        .enumerate()
        .map(|(position, entry)| {
            let index = entry.get("index").and_then(|i| i.as_u64()).ok_or_else(|| {
                RuntimeError::ApiError(format!(
                    "llama.cpp /v1/rerank response malformed: result {} has no integer \
                         'index'",
                    position
                ))
            })? as usize;
            // Accept either key regardless of the shape it arrived in: the two
            // differ only in naming, and a mismatch is not worth an error.
            // `score_key` only shapes the message when neither is present.
            let score = entry
                .get("relevance_score")
                .or_else(|| entry.get("score"))
                .and_then(|s| s.as_f64())
                .ok_or_else(|| {
                    RuntimeError::ApiError(format!(
                        "llama.cpp /v1/rerank response malformed: result {} has no numeric \
                         '{}'",
                        position, score_key
                    ))
                })? as f32;
            Ok((index, score))
        })
        .collect()
}

/// Turn raw `(index, score)` pairs into [`ScoredDoc`]s against the original
/// `docs` slice.
///
/// Every index is checked rather than trusted. A reranker's whole job is to
/// permute the caller's list, so an index past the end or seen twice would
/// silently mis-attribute a score to the wrong document — a corruption that
/// reads as a ranking bug rather than a protocol one. `expected` results must
/// all be present: a server that dropped one (by honouring a `top_n` we never
/// sent, say) would otherwise look like a document scoring zero.
///
/// `text` is always `None`: `llama-server` echoes the document text only in the
/// TEI form with `return_text`, which this module does not request, and the
/// caller already holds the originals.
fn build_scored_docs(pairs: Vec<(usize, f32)>, expected: usize) -> Result<Vec<ScoredDoc>> {
    if pairs.len() != expected {
        return Err(RuntimeError::ApiError(format!(
            "llama.cpp /v1/rerank returned {} results for {} documents",
            pairs.len(),
            expected
        )));
    }

    let mut seen = vec![false; expected];
    let mut docs = Vec::with_capacity(expected);
    for (index, score) in pairs {
        if index >= expected {
            return Err(RuntimeError::ApiError(format!(
                "llama.cpp /v1/rerank returned index {} for a batch of {} documents",
                index, expected
            )));
        }
        if std::mem::replace(&mut seen[index], true) {
            return Err(RuntimeError::ApiError(format!(
                "llama.cpp /v1/rerank returned index {} more than once",
                index
            )));
        }
        docs.push(ScoredDoc {
            index,
            score,
            text: None,
        });
    }

    // The server sorts already; re-sorting makes the trait's "typically sorted
    // by descending score" hold whatever a gateway in between did. `total_cmp`
    // rather than `partial_cmp` so a NaN score cannot panic the comparator.
    docs.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Reranker model
// ---------------------------------------------------------------------------

/// Reranker backed by a llama.cpp server.
pub struct LlamaCppRerankerModel {
    client: Client,
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    cfg: RerankConfig,
}

impl LlamaCppRerankerModel {
    pub(crate) fn new(
        client: Client,
        cb: crate::reliability::CircuitBreakerWrapper,
        model_id: String,
        cfg: RerankConfig,
    ) -> Self {
        Self {
            client,
            cb,
            model_id,
            cfg,
        }
    }

    async fn rerank_pipeline(&self, query: &str, docs: &[String]) -> Result<Vec<ScoredDoc>> {
        let body = post_json(
            &self.client,
            &self.cfg.common,
            format!("{}/rerank", self.cfg.common.base_url),
            // No `top_n`: see the module header.
            &json!({
                "model": self.model_id,
                "query": query,
                "documents": docs,
            }),
            // Reranking has no `max_input_tokens` option, so an overflow can
            // only be answered with the server's own flags.
            None,
        )
        .await?;
        build_scored_docs(parse_rerank_response(&body)?, docs.len())
    }
}

#[async_trait]
impl RerankerModel for LlamaCppRerankerModel {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        // `llama-server` answers an empty `documents` array with HTTP 400
        // ("must be a non-empty string array"), so nothing-to-rank is settled
        // here rather than turned into a spurious error.
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let query = query.to_string();
        let docs: Vec<String> = docs.iter().map(|s| s.to_string()).collect();

        // Deterministic rejections (`InferenceError`, produced only by
        // `map_llamacpp_status`) are a configuration problem rather than a
        // server outage — a reranker server started without `--reranking`
        // answers every request with 501 — so they are returned as
        // `Ok(Err(..))`, leaving the breaker closed, and flattened here. Same
        // contract as `embed` and `chat`.
        self.cb
            .call(move || async move {
                match self.rerank_pipeline(&query, &docs).await {
                    Err(e @ RuntimeError::InferenceError(_)) => Ok(Err(e)),
                    other => other.map(Ok),
                }
            })
            .await?
    }
}

impl crate::traits::ModelInfo for LlamaCppRerankerModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Tests (no HTTP)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- config -------------------------------------------------------------

    #[test]
    fn config_needs_only_the_shared_keys() {
        let cfg = RerankConfig::from_options(&json!({
            "base_url": "http://127.0.0.1:8080/v1/"
        }))
        .unwrap();
        assert_eq!(cfg.common.base_url, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn config_requires_base_url() {
        let err = RerankConfig::from_options(&json!({})).expect_err("must fail");
        assert!(err.to_string().contains("base_url"), "{err}");
    }

    // --- parsing ------------------------------------------------------------

    /// The shape a real `llama-server` returns for a `documents` request,
    /// copied from a live response.
    #[test]
    fn parses_the_jina_shape() {
        let body = json!({
            "model": "jina",
            "object": "list",
            "usage": { "prompt_tokens": 59, "total_tokens": 59 },
            "results": [
                { "index": 0, "relevance_score": 0.12988508 },
                { "index": 2, "relevance_score": 0.034857254 },
                { "index": 1, "relevance_score": 0.009407487 },
            ]
        });
        assert_eq!(
            parse_rerank_response(&body).unwrap(),
            vec![(0, 0.12988508), (2, 0.034857254), (1, 0.009407487)]
        );
    }

    /// The bare-array shape a `texts` request (or a gateway) produces.
    #[test]
    fn parses_the_tei_shape() {
        let body = json!([
            { "index": 1, "score": 0.9 },
            { "index": 0, "score": 0.1 },
        ]);
        assert_eq!(
            parse_rerank_response(&body).unwrap(),
            vec![(1, 0.9), (0, 0.1)]
        );
    }

    #[test]
    fn parses_either_score_key_in_either_shape() {
        // A gateway that keeps the Jina envelope but the TEI score key.
        let body = json!({ "results": [{ "index": 0, "score": 0.5 }] });
        assert_eq!(parse_rerank_response(&body).unwrap(), vec![(0, 0.5)]);

        // ... and the reverse.
        let body = json!([{ "index": 0, "relevance_score": 0.5 }]);
        assert_eq!(parse_rerank_response(&body).unwrap(), vec![(0, 0.5)]);
    }

    #[test]
    fn a_response_that_is_neither_shape_is_an_error() {
        for body in [json!({}), json!({ "results": "x" }), json!("nope")] {
            let err = parse_rerank_response(&body).expect_err("must fail");
            assert!(
                matches!(err, RuntimeError::ApiError(ref m) if m.contains("results")),
                "{err:?}"
            );
        }
    }

    #[test]
    fn a_result_missing_index_or_score_is_an_error() {
        for (body, needle) in [
            (json!({ "results": [{ "relevance_score": 0.5 }] }), "index"),
            (json!({ "results": [{ "index": 0 }] }), "relevance_score"),
            (
                json!({ "results": [{ "index": -1, "relevance_score": 0.5 }] }),
                "index",
            ),
        ] {
            let err = parse_rerank_response(&body).expect_err("must fail");
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    // --- index mapping ------------------------------------------------------

    #[test]
    fn scored_docs_keep_the_original_indices_and_sort_descending() {
        let docs = build_scored_docs(vec![(2, 0.1), (0, 0.9), (1, 0.5)], 3).unwrap();
        assert_eq!(docs.iter().map(|d| d.index).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(
            docs.iter().map(|d| d.score).collect::<Vec<_>>(),
            [0.9, 0.5, 0.1]
        );
        assert!(docs.iter().all(|d| d.text.is_none()));
    }

    #[test]
    fn an_unsorted_server_response_is_still_sorted() {
        let docs = build_scored_docs(vec![(0, 0.1), (1, 0.9)], 2).unwrap();
        assert_eq!(docs[0].index, 1);
        assert_eq!(docs[1].index, 0);
    }

    #[test]
    fn an_out_of_range_index_is_rejected() {
        let err = build_scored_docs(vec![(0, 0.5), (7, 0.4)], 2).expect_err("must fail");
        assert!(err.to_string().contains("index 7"), "{err}");
    }

    #[test]
    fn a_duplicate_index_is_rejected() {
        let err = build_scored_docs(vec![(1, 0.5), (1, 0.4)], 2).expect_err("must fail");
        assert!(err.to_string().contains("more than once"), "{err}");
    }

    #[test]
    fn a_short_or_long_result_list_is_rejected() {
        for pairs in [vec![(0, 0.5)], vec![(0, 0.5), (1, 0.4), (2, 0.3)]] {
            let err = build_scored_docs(pairs, 2).expect_err("must fail");
            assert!(err.to_string().contains("for 2 documents"), "{err}");
        }
    }

    /// A NaN score must not panic the comparator or lose the document.
    #[test]
    fn a_nan_score_does_not_panic_the_sort() {
        let docs = build_scored_docs(vec![(0, f32::NAN), (1, 0.5)], 2).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.index == 0));
    }
}
