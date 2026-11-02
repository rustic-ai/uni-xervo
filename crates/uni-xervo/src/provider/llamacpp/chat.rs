//! [`ModelTask::Generate`](crate::api::ModelTask::Generate) on
//! `POST /v1/chat/completions`.
//!
//! `llama-server`'s chat endpoint is OpenAI-compatible, including the
//! multimodal `content` parts array. A server started with `--mmproj` loads a
//! projector alongside the language model and accepts
//! `{"type": "image_url", ...}` parts; one started without it is text-only and
//! rejects them. **Vision is therefore a property of the server process, not of
//! the alias**, which is why there is deliberately no `pipeline`-style option
//! here: point an alias at a vision-capable server and images work.
//!
//! [`LlamaCppChatClient`] is the shared request/response seam. It is public to
//! the crate so other tasks that are really "a chat call with a fixed prompt"
//! can reuse it rather than re-deriving the wire format.
//!
//! # Options
//!
//! None beyond the shared keys documented on the [parent module](super).
//!
//! # Input length
//!
//! There is no client-side truncation pre-pass here, unlike
//! [`embed`](super::embed). The embedding pre-pass works by asking the server's
//! `/tokenize` endpoint to count a prompt exactly, but that endpoint only sees
//! text: the tokens an image expands into are produced inside the multimodal
//! projector and are invisible to it, so a budget covering a vision request
//! cannot be computed. Overflow is left to the server, whose rejection becomes
//! a non-retryable [`RuntimeError::InferenceError`] naming `--ctx-size` — see
//! [`map_llamacpp_status`](super::map_llamacpp_status).
//!
//! # Reasoning models
//!
//! On a model with a thinking phase, `llama-server` splits the assistant turn
//! in two: `reasoning_content` carries the chain of thought and `content`
//! carries the answer. Only `content` becomes
//! [`GenerationResult::text`](crate::traits::GenerationResult::text) — folding
//! the reasoning in would corrupt every caller that parses the output, such as
//! document extraction.
//!
//! One consequence is worth knowing when a result comes back empty: if
//! `max_tokens` is exhausted *during* the thinking phase, the server returns
//! `finish_reason: "length"` with all the tokens in `reasoning_content` and
//! `content` empty. The generation succeeded and the usage counts are
//! non-zero; the budget simply never reached the answer. Raise `max_tokens`.
//!
//! # Unused result fields
//!
//! [`GenerationResult::images`] and [`GenerationResult::audio`] are always
//! empty: `llama-server` has no image-generation endpoint, and its speech
//! synthesis lives on a separate native `POST /tts` route rather than on chat
//! completions. [`GenerationOptions::width`] and `height` are ignored for the
//! same reason.

use super::{LlamaCppCommon, post_json};
use crate::error::{Result, RuntimeError};
use crate::provider::remote_common::build_openai_chat_messages;
use crate::traits::{
    GenerationOptions, GenerationResult, GeneratorModel, Message, ModelInfo, TokenUsage,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved generation configuration for one alias.
///
/// Generation adds no keys of its own, so this is the shared set verbatim. It
/// exists as a named type so the `load` dispatch reads the same way for every
/// task.
#[derive(Debug, Clone)]
pub(crate) struct ChatConfig {
    pub common: LlamaCppCommon,
}

impl ChatConfig {
    /// Parse the alias `options`. The catalog validator already enforces the
    /// same rules, but `ModelProvider::load` is public, so this is defensive.
    pub(crate) fn from_options(options: &Value) -> Result<Self> {
        Ok(Self {
            common: LlamaCppCommon::from_options(options)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Chat client (no circuit breaker)
// ---------------------------------------------------------------------------

/// The `/v1/chat/completions` request/response seam for one model.
///
/// Deliberately free of reliability concerns: the caller owns the circuit
/// breaker, because a task built on top of this (a per-page document
/// extraction loop, say) wants one breaker call around the whole job rather
/// than one per HTTP request.
pub(crate) struct LlamaCppChatClient {
    client: Client,
    common: LlamaCppCommon,
    model_id: String,
}

impl LlamaCppChatClient {
    pub(crate) fn new(client: Client, common: LlamaCppCommon, model_id: String) -> Self {
        Self {
            client,
            common,
            model_id,
        }
    }

    /// Run one chat completion.
    pub(crate) async fn complete(
        &self,
        messages: &[Message],
        options: &GenerationOptions,
    ) -> Result<GenerationResult> {
        let body = build_chat_payload(&self.model_id, messages, options);
        let response = post_json(
            &self.client,
            &self.common,
            format!("{}/chat/completions", self.common.base_url),
            &body,
            // Generation has no `max_input_tokens` option, so an overflow can
            // only be answered with the server's own flags.
            None,
        )
        .await?;
        parse_chat_response(&response)
    }
}

/// Build the `/v1/chat/completions` request body.
///
/// Sampling keys are omitted entirely when `None` so the server applies its
/// own defaults rather than being pinned to ours.
fn build_chat_payload(model_id: &str, messages: &[Message], options: &GenerationOptions) -> Value {
    let mut body = json!({
        "model": model_id,
        "messages": build_openai_chat_messages(messages),
        // `llama-server` can stream, but `GeneratorModel::generate` returns a
        // whole result, so ask for one response rather than reassembling SSE.
        "stream": false,
    });

    // `max_tokens` rather than OpenAI's newer `max_completion_tokens`: both
    // are accepted by current builds, but only `max_tokens` works all the way
    // back, and `llama-server` treats them identically.
    if let Some(max_tokens) = options.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = options.top_p {
        body["top_p"] = json!(top_p);
    }
    body
}

/// Read the first choice's message content and the usage block.
///
/// A response with no `choices` is a protocol violation worth surfacing, not a
/// silent empty string: callers such as the document extractor cannot tell an
/// empty page from a broken server otherwise.
fn parse_chat_response(body: &Value) -> Result<GenerationResult> {
    let choice = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| {
            RuntimeError::ApiError(
                "llama.cpp /v1/chat/completions response malformed: missing 'choices'".to_string(),
            )
        })?;

    // An assistant turn that stopped immediately has `content: null`, which is
    // a legitimately empty generation rather than a malformed response. So is
    // a reasoning model that spent its whole budget on `reasoning_content`
    // (see the module header) — deliberately not read here.
    let text = match choice.pointer("/message/content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => {
            return Err(RuntimeError::ApiError(format!(
                "llama.cpp /v1/chat/completions response malformed: \
                 message content is {} rather than a string",
                kind_of(other)
            )));
        }
    };

    Ok(GenerationResult {
        text,
        usage: parse_usage(body.get("usage")),
        images: vec![],
        audio: None,
    })
}

/// Map `usage` when the server reported it.
///
/// `total_tokens` is derived when absent so a caller asserting on it does not
/// have to special-case a server that only reports the two halves.
fn parse_usage(usage: Option<&Value>) -> Option<TokenUsage> {
    let usage = usage?;
    let field = |key: &str| usage.get(key).and_then(|v| v.as_u64()).map(|v| v as usize);
    let prompt_tokens = field("prompt_tokens").unwrap_or(0);
    let completion_tokens = field("completion_tokens").unwrap_or(0);
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: field("total_tokens").unwrap_or(prompt_tokens + completion_tokens),
    })
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

// ---------------------------------------------------------------------------
// Generator model
// ---------------------------------------------------------------------------

/// Text and vision generation backed by a llama.cpp server.
pub struct LlamaCppGeneratorModel {
    cb: crate::reliability::CircuitBreakerWrapper,
    model_id: String,
    chat: LlamaCppChatClient,
}

impl LlamaCppGeneratorModel {
    pub(crate) fn new(
        client: Client,
        cb: crate::reliability::CircuitBreakerWrapper,
        model_id: String,
        cfg: ChatConfig,
    ) -> Self {
        Self {
            cb,
            model_id: model_id.clone(),
            chat: LlamaCppChatClient::new(client, cfg.common, model_id),
        }
    }
}

#[async_trait]
impl GeneratorModel for LlamaCppGeneratorModel {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let messages = messages.to_vec();

        // Deterministic input-size rejections (`InferenceError`, produced only
        // by `map_llamacpp_status`) are a configuration problem rather than a
        // server outage, so they are returned as `Ok(Err(..))` — the breaker
        // sees a success and stays closed — and flattened here. Same contract
        // as `embed`.
        self.cb
            .call(move || async move {
                match self.chat.complete(&messages, &options).await {
                    Err(e @ RuntimeError::InferenceError(_)) => Ok(Err(e)),
                    other => other.map(Ok),
                }
            })
            .await?
    }
}

impl ModelInfo for LlamaCppGeneratorModel {
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
    use crate::traits::{ContentBlock, ImageInput, MessageRole};

    fn opts() -> Value {
        json!({ "base_url": "http://127.0.0.1:8080/v1" })
    }

    // --- config -------------------------------------------------------------

    #[test]
    fn config_needs_only_the_shared_keys() {
        let cfg = ChatConfig::from_options(&opts()).unwrap();
        assert_eq!(cfg.common.base_url, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn config_requires_base_url() {
        let err = ChatConfig::from_options(&json!({})).expect_err("must fail");
        assert!(err.to_string().contains("base_url"), "{err}");
    }

    // --- payload ------------------------------------------------------------

    #[test]
    fn payload_omits_every_unset_sampling_key() {
        let body = build_chat_payload(
            "qwen",
            &[Message::user("hi")],
            &GenerationOptions::default(),
        );
        assert_eq!(body["model"], json!("qwen"));
        assert_eq!(body["stream"], json!(false));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(body["messages"][0]["content"], json!("hi"));
        for key in ["max_tokens", "temperature", "top_p"] {
            assert!(body.get(key).is_none(), "{key} should be absent");
        }
    }

    #[test]
    fn payload_carries_the_sampling_keys_that_are_set() {
        let body = build_chat_payload(
            "qwen",
            &[Message::user("hi")],
            &GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(0.5),
                top_p: Some(0.75),
                // Diffusion-only; must not reach the wire.
                width: Some(512),
                height: Some(512),
            },
        );
        assert_eq!(body["max_tokens"], json!(64));
        assert_eq!(body["temperature"], json!(0.5));
        assert_eq!(body["top_p"], json!(0.75));
        assert!(body.get("width").is_none());
        assert!(body.get("height").is_none());
    }

    #[test]
    fn payload_sends_images_as_content_parts() {
        let body = build_chat_payload(
            "smolvlm",
            &[Message {
                role: MessageRole::User,
                content: vec![
                    ContentBlock::Text("describe".into()),
                    ContentBlock::Image(ImageInput::Bytes {
                        data: vec![1, 2, 3],
                        media_type: "image/png".into(),
                    }),
                ],
            }],
            &GenerationOptions::default(),
        );
        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], json!("text"));
        assert_eq!(parts[1]["type"], json!("image_url"));
        assert_eq!(
            parts[1]["image_url"]["url"],
            json!("data:image/png;base64,AQID")
        );
    }

    #[test]
    fn payload_preserves_role_and_message_order() {
        let body = build_chat_payload(
            "qwen",
            &[
                Message::system("be terse"),
                Message::user("hi"),
                Message::assistant("hello"),
            ],
            &GenerationOptions::default(),
        );
        let roles: Vec<_> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
    }

    // --- response -----------------------------------------------------------

    #[test]
    fn response_maps_text_and_usage() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "content": "a moon" } }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14 }
        });
        let result = parse_chat_response(&body).unwrap();
        assert_eq!(result.text, "a moon");
        assert!(result.images.is_empty());
        assert!(result.audio.is_none());
        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 14);
    }

    #[test]
    fn response_derives_a_missing_total() {
        let body = json!({
            "choices": [{ "message": { "content": "x" } }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
        });
        assert_eq!(
            parse_chat_response(&body)
                .unwrap()
                .usage
                .unwrap()
                .total_tokens,
            7
        );
    }

    /// A reasoning model's chain of thought is not the answer, and must not
    /// leak into `text` — not even when `content` is empty because the token
    /// budget ran out mid-thought.
    #[test]
    fn response_never_folds_reasoning_content_into_text() {
        let body = json!({
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Thinking Process: 1. Analyse the request",
                },
            }],
            "usage": { "prompt_tokens": 32, "completion_tokens": 64, "total_tokens": 96 }
        });
        let result = parse_chat_response(&body).unwrap();
        assert_eq!(result.text, "");
        // The call still succeeded, and usage proves the tokens were spent.
        assert_eq!(result.usage.unwrap().completion_tokens, 64);
    }

    #[test]
    fn response_without_usage_reports_none() {
        let body = json!({ "choices": [{ "message": { "content": "x" } }] });
        assert!(parse_chat_response(&body).unwrap().usage.is_none());
    }

    #[test]
    fn response_treats_null_or_absent_content_as_empty_text() {
        for message in [
            json!({ "content": Value::Null }),
            json!({ "role": "assistant" }),
        ] {
            let body = json!({ "choices": [{ "message": message }] });
            assert_eq!(parse_chat_response(&body).unwrap().text, "");
        }
    }

    #[test]
    fn response_without_choices_is_an_error() {
        for body in [
            json!({}),
            json!({ "choices": [] }),
            json!({ "choices": "x" }),
        ] {
            let err = parse_chat_response(&body).expect_err("must fail");
            assert!(
                matches!(err, RuntimeError::ApiError(_)) && err.to_string().contains("choices"),
                "{err:?}"
            );
        }
    }

    #[test]
    fn response_with_non_string_content_is_an_error() {
        let body = json!({ "choices": [{ "message": { "content": [{ "type": "text" }] } }] });
        let err = parse_chat_response(&body).expect_err("must fail");
        assert!(err.to_string().contains("an array"), "{err}");
    }
}
