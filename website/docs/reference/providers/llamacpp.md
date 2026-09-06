# remote/llamacpp

## Uni-Xervo support

- Provider ID: `remote/llamacpp`
- Feature flag: `provider-llamacpp`
- Capabilities: `embed`

Targets a llama.cpp [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
serving an embedding model (for example `ggml-org/bge-small-en-v1.5-Q8_0-GGUF`).
Use `remote/openai` for generic OpenAI-compatible servers; use this provider when
the server is llama.cpp and you need bounded embedding input.

## Why a dedicated provider

`llama-server` **never truncates embedding input**. Encoder models such as BGE
are non-causal, so the whole prompt must fit in one physical batch
(`--ubatch-size`) and inside the context (`--ctx-size`); a longer prompt is
rejected with an HTTP error (`input is too large to process` /
`exceeds the available context size`).

This provider bounds every input deterministically before embedding:

1. A text with at most `max_input_tokens - 2` characters cannot exceed the
   budget (every WordPiece token consumes at least one character), so it is
   sent as a plain string with no extra round trip.
2. Longer texts are sent to the server's native `POST /tokenize` with
   `add_special: true`, which returns the exact ids the server would embed,
   special tokens included.
3. If that sequence is longer than `max_input_tokens`, the first
   `max_input_tokens - 1` ids plus the trailing special token (`[SEP]`) are
   sent to `/v1/embeddings` as a **token array**. llama.cpp embeds integer
   arrays verbatim, so there is no lossy detokenize round trip.

Batches keep input order and return exactly one vector per input. Server-side
input-size rejections surface as `RuntimeError::InferenceError` naming
`max_input_tokens`; they are not retried and do not trip the circuit breaker.

## Authentication

None by default. `llama-server` started without `--api-key` ignores the
`Authorization` header. If the server does use `--api-key`, set `api_key_env`
to the environment variable holding the token.

## Uni-Xervo provider options

| Option | Type | Required | Description |
| --- | --- | --- | --- |
| `base_url` | string | yes | OpenAI-compatible root including `/v1`, e.g. `http://127.0.0.1:8080/v1` |
| `tokenizer_base_url` | string | no | Server root for `/tokenize` (outside `/v1`). Defaults to `base_url` with a trailing `/v1` removed |
| `max_input_tokens` | integer ≥ 3 | yes | Total token budget **including** special tokens. `512` for BGE / BERT models |
| `embedding_dimensions` | integer > 0 | yes | Expected vector width (`384` for `bge-small-en-v1.5`) |
| `api_key_env` | string | no | Env var holding a bearer token |
| `request_timeout_secs` | integer > 0 | no | Per-HTTP-request timeout, default `60` |

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/crates/uni-xervo/schemas/provider-options/llamacpp.schema.json>

## Model ID

`model_id` is sent as the `model` field of both `/tokenize` and
`/v1/embeddings`. When `llama-server` runs in router mode (`--models-dir` /
`--models-preset`) it must equal the preset section name exactly; a
single-model server accepts any value.

## Server sizing

`max_input_tokens` must not exceed the server's `--ubatch-size` or
`--ctx-size`. For BGE Small, start the server with the model's real limit so
misconfiguration fails loudly instead of silently indexing past the model's
position table:

```bash
llama-server -m bge-small-en-v1.5-q8_0.gguf --embedding -c 512 -ub 512 --port 8080
```

If the provider returns an `InferenceError` mentioning `too large to process`,
the server's batch or context is smaller than `max_input_tokens`.

## Authoritative model and config docs

- llama-server README (endpoints, router mode, presets): <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- BGE Small EN v1.5 GGUF: <https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF>

## Example catalog entry

```json
{
  "alias": "embed/default",
  "task": "embed",
  "provider_id": "remote/llamacpp",
  "model_id": "bge-small-en-v1.5",
  "options": {
    "base_url": "http://127.0.0.1:8080/v1",
    "max_input_tokens": 512,
    "embedding_dimensions": 384
  }
}
```

With a separately reachable tokenizer endpoint and an API key:

```json
{
  "alias": "embed/default",
  "task": "embed",
  "provider_id": "remote/llamacpp",
  "model_id": "rustic/bge-small-en-v1.5",
  "options": {
    "base_url": "http://127.0.0.1:8080/v1",
    "tokenizer_base_url": "http://127.0.0.1:8080",
    "max_input_tokens": 512,
    "embedding_dimensions": 384,
    "api_key_env": "LOCAL_LLM_API_KEY",
    "request_timeout_secs": 30
  }
}
```
