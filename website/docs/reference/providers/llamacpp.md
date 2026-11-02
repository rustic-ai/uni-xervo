# remote/llamacpp

## Uni-Xervo support

- Provider ID: `remote/llamacpp`
- Feature flag: `provider-llamacpp`
- Capabilities: `embed`, `generate`

Targets a llama.cpp [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
Use `remote/openai` for generic OpenAI-compatible servers; use this provider when
the server is llama.cpp — for bounded embedding input, and for text or vision
generation against a locally served model.

One `llama-server` process serves one model, so an embedding alias and a
generation alias point at two different servers (two `base_url`s).

## Why a dedicated provider

### `embed`

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

### `generate`

Generation posts to `/v1/chat/completions`, including the multimodal `content`
parts array, so **the same alias does text and vision**. There is no option to
switch between them: whether images are accepted depends on the server having
been started with `--mmproj`, which is a property of the server process rather
than something the catalog can know. `ImageInput::Bytes` becomes a
`data:` URI and `ImageInput::Url` is passed through verbatim.

`max_tokens`, `temperature` and `top_p` from `GenerationOptions` are forwarded
when set and omitted otherwise, so the server's own defaults apply. `width` and
`height` are ignored — `llama-server` has no image-generation endpoint — and
`GenerationResult.images` / `.audio` are always empty.

There is **no** truncation pre-pass for generation. The embedding pre-pass works
by having the server count a prompt exactly, but `/tokenize` only sees text: the
tokens an image expands into are produced inside the multimodal projector and
are invisible to it. Context overflow is therefore left to the server, whose
rejection becomes a non-retryable `InferenceError` naming `--ctx-size`.

!!! note "Reasoning models return the answer separately from the thinking"

    On a model with a thinking phase, `llama-server` puts the chain of thought
    in `reasoning_content` and the answer in `content`. Only `content` becomes
    `GenerationResult.text`. If `max_tokens` runs out *during* thinking, the
    call succeeds with non-zero token usage but **empty text** — the budget
    never reached the answer. Raise `max_tokens`.

## Authentication

None by default. `llama-server` started without `--api-key` ignores the
`Authorization` header. If the server does use `--api-key`, set `api_key_env`
to the environment variable holding the token.

## Uni-Xervo provider options

Shared by every task:

| Option | Type | Required | Description |
| --- | --- | --- | --- |
| `base_url` | string | yes | OpenAI-compatible root including `/v1`, e.g. `http://127.0.0.1:8080/v1` |
| `api_key_env` | string | no | Env var holding a bearer token |
| `request_timeout_secs` | integer > 0 | no | Per-HTTP-request timeout, default `60` |

`embed` only — these are **rejected** on any other task rather than ignored:

| Option | Type | Required | Description |
| --- | --- | --- | --- |
| `tokenizer_base_url` | string | no | Server root for `/tokenize` (outside `/v1`). Defaults to `base_url` with a trailing `/v1` removed |
| `max_input_tokens` | integer ≥ 3 | yes | Total token budget **including** special tokens. `512` for BGE / BERT models |
| `embedding_dimensions` | integer > 0 | yes | Expected vector width (`384` for `bge-small-en-v1.5`) |

`generate` takes no options of its own.

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/crates/uni-xervo/schemas/provider-options/llamacpp.schema.json>

## Model ID

`model_id` is sent as the `model` field of `/tokenize`, `/v1/embeddings` and
`/v1/chat/completions`. When `llama-server` runs in router mode (`--models-dir` /
`--models-preset`) it must equal the preset section name exactly; a
single-model server accepts any value.

## Server sizing

### Embedding

`max_input_tokens` must not exceed the server's `--ubatch-size` or
`--ctx-size`. For BGE Small, start the server with the model's real limit so
misconfiguration fails loudly instead of silently indexing past the model's
position table:

```bash
llama-server -m bge-small-en-v1.5-q8_0.gguf --embedding -c 512 -ub 512 --port 8080
```

If the provider returns an `InferenceError` mentioning `too large to process`,
the server's batch or context is smaller than `max_input_tokens`.

### Generation

Size `--ctx-size` for the prompt plus the generation, and pass `--mmproj` for
vision. A projector costs context: the image tokens are added to the prompt, so
a vision server wants noticeably more context than the same model serving text.

```bash
llama-server -m model-q4_0.gguf -c 4096 --port 8081
```

```bash
llama-server -m model-q4_0.gguf --mmproj model-mmproj.gguf -c 8192 --port 8081
```

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

A generation alias — note that it carries none of the embedding keys:

```json
{
  "alias": "generate/local",
  "task": "generate",
  "provider_id": "remote/llamacpp",
  "model_id": "local",
  "options": {
    "base_url": "http://127.0.0.1:8081/v1",
    "request_timeout_secs": 180
  }
}
```
