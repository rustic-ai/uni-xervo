// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Reusable greedy autoregressive decoding helper for `local/onnx`.
//!
//! The function [`greedy_decode`] runs an autoregressive sampling loop on
//! top of a caller-supplied "next-step logits" callback. The callback
//! owns whatever model state is needed (ONNX session, KV cache, image
//! embeddings, attention masks, …); this module only handles the
//! decoder-loop logic so it stays independent of any specific model
//! shape, tokenizer, or ORT input-name convention.
//!
//! # Why a callback (not a session) parameter
//!
//! Different VLMs / decoder-LMs expose different input-name conventions
//! (`past_key_values.X.Y` vs `kv_cache.X.Y` vs split encoder/decoder
//! files vs joint graphs). Hardcoding any of those would block reuse.
//! A `FnMut(&[i64]) -> Result<Vec<f32>>` callback gives the model owner
//! full control over input plumbing while inheriting the loop, EOS-stop,
//! and max-token clamping for free.
//!
//! # v1 scope
//!
//! - Greedy sampling (argmax per step). Temperature, top-k, top-p,
//!   beam-search are mechanical extensions for follow-ups.
//! - Single-batch decode. Multi-batch would parallelize at the trait
//!   level (one call per request).
//! - No prompt prefill state — the callback handles prefill on its first
//!   call (when the running token list is empty).

use crate::error::Result;

/// Configuration for a [`greedy_decode`] call.
#[derive(Debug, Clone)]
pub struct AutoregConfig {
    /// Hard cap on generated token count. Decode stops at this length
    /// even if no EOS has been emitted.
    pub max_new_tokens: usize,
    /// Token ids that terminate decoding. Decode emits the EOS and then
    /// stops. An empty slice means "no EOS" (rely on `max_new_tokens`).
    pub eos_token_ids: Vec<i64>,
}

/// Greedy autoregressive decode.
///
/// On each iteration `next_logits` is invoked with the running list of
/// generated tokens (empty on the first call) and must return the logits
/// for the next position (length = vocabulary size). The function picks
/// the argmax, appends it to the output, and stops when either an
/// `eos_token_ids` token is emitted or `max_new_tokens` has been reached.
///
/// Returns the generated token sequence (including any final EOS).
///
/// # Errors
/// Returns any error produced by `next_logits` unchanged. The decode
/// loop itself does not introduce new error variants.
pub fn greedy_decode<F>(config: &AutoregConfig, mut next_logits: F) -> Result<Vec<i64>>
where
    F: FnMut(&[i64]) -> Result<Vec<f32>>,
{
    let mut output: Vec<i64> = Vec::with_capacity(config.max_new_tokens);
    for _ in 0..config.max_new_tokens {
        let logits = next_logits(&output)?;
        let next = argmax(&logits);
        output.push(next);
        if config.eos_token_ids.contains(&next) {
            break;
        }
    }
    Ok(output)
}

/// Index of the maximum value in `logits`. Ties pick the lowest index;
/// `NaN` is ignored (treated as less than any real number).
fn argmax(logits: &[f32]) -> i64 {
    let mut best_idx: usize = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeError;

    #[test]
    fn greedy_decode_stops_on_eos() {
        // 4-token vocab. Step 0 picks 1, step 1 picks 2, step 2 picks 3 (EOS).
        let scripted: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0], // never reached
        ];
        let mut step = 0;
        let cfg = AutoregConfig {
            max_new_tokens: 10,
            eos_token_ids: vec![3],
        };
        let out = greedy_decode(&cfg, |_so_far| {
            let logits = scripted[step].clone();
            step += 1;
            Ok(logits)
        })
        .unwrap();
        assert_eq!(out, vec![1, 2, 3]);
        assert_eq!(step, 3, "callback called once per output token");
    }

    #[test]
    fn greedy_decode_stops_on_max_new_tokens() {
        // Always returns the same logits — would never EOS on its own.
        let cfg = AutoregConfig {
            max_new_tokens: 5,
            eos_token_ids: vec![999],
        };
        let out = greedy_decode(&cfg, |_so_far| Ok(vec![0.1, 0.9, 0.05, 0.0])).unwrap();
        assert_eq!(out.len(), 5);
        // Every token should be 1 (the argmax of [0.1, 0.9, 0.05, 0.0]).
        assert!(out.iter().all(|&t| t == 1));
    }

    #[test]
    fn greedy_decode_picks_argmax_per_step() {
        // The callback's own logits choose: 2, 0, 3, 1 in order.
        let scripted: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let mut step = 0;
        let cfg = AutoregConfig {
            max_new_tokens: 4,
            eos_token_ids: vec![],
        };
        let out = greedy_decode(&cfg, |_so_far| {
            let logits = scripted[step].clone();
            step += 1;
            Ok(logits)
        })
        .unwrap();
        assert_eq!(out, vec![2, 0, 3, 1]);
    }

    #[test]
    fn greedy_decode_propagates_callback_errors() {
        let cfg = AutoregConfig {
            max_new_tokens: 10,
            eos_token_ids: vec![],
        };
        let res: Result<Vec<i64>> = greedy_decode(&cfg, |_so_far| {
            Err(RuntimeError::InferenceError("synthetic".to_string()))
        });
        assert!(matches!(res, Err(RuntimeError::InferenceError(_))));
    }

    #[test]
    fn greedy_decode_callback_sees_running_sequence() {
        // Step N must see exactly N tokens (the ones emitted so far).
        let mut step = 0;
        let cfg = AutoregConfig {
            max_new_tokens: 3,
            eos_token_ids: vec![],
        };
        let out = greedy_decode(&cfg, |so_far| {
            assert_eq!(so_far.len(), step, "callback sees the running sequence");
            step += 1;
            // Always emit token id == step so we can verify the running
            // list contents from outside.
            let token = step;
            let mut logits = vec![0.0; 5];
            logits[token.min(4)] = 1.0;
            Ok(logits)
        })
        .unwrap();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn argmax_picks_first_on_tie() {
        // Two equal max values — argmax keeps the first (matches the
        // strict `>` comparison in our impl).
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.5]), 1);
    }

    #[test]
    fn argmax_ignores_nan() {
        // NaN should not displace a real max.
        assert_eq!(argmax(&[0.5, f32::NAN, 0.9]), 2);
    }

    #[test]
    fn argmax_returns_zero_for_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn greedy_decode_handles_zero_max_new_tokens() {
        // Edge case: max_new_tokens=0 → empty output, callback never invoked.
        let mut calls = 0;
        let cfg = AutoregConfig {
            max_new_tokens: 0,
            eos_token_ids: vec![],
        };
        let out = greedy_decode(&cfg, |_so_far| {
            calls += 1;
            Ok(vec![1.0])
        })
        .unwrap();
        assert_eq!(out, Vec::<i64>::new());
        assert_eq!(calls, 0);
    }
}
