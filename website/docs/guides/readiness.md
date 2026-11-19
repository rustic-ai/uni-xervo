# Startup Readiness

A service usually wants to know, before it takes traffic, which models it can
actually serve. Uni-Xervo answers that in three steps, each more expensive than
the last, and none of them happens unless you ask:

1. **Inspect** the runtime: what is in the catalog and what has loaded. Loads
   nothing, contacts nothing.
2. **Prefetch** the aliases you need, and get a result for each one.
3. **Verify** remote aliases: one cheap authenticated request each, to tell
   "configured" from "reachable with a valid key".

## Inspect without loading

```rust
use uni_xervo::inspect::LoadState;

for alias in runtime.aliases().await {
    println!(
        "{} {:?} via {} ({}) -> {:?}",
        alias.alias, alias.task, alias.provider_id, alias.model_id, alias.state
    );
}
```

`aliases()` returns one `AliasSnapshot` per catalog alias, sorted;
`alias("embed/default")` returns one. Each carries:

- `task`, `provider_id`, `model_id`, `revision`, `warmup`, `required`;
- `state`, a `LoadState`:
  - `Registered`: nothing has been attempted yet;
  - `Loading`;
  - `Loaded`;
  - `Failed { error, at }`: the last load failed. That covers a load error, a
    timeout, or a background warmup that failed. A later successful load
    clears it.
- `execution_providers`, once loaded (for a local ONNX model, the providers
  its session attached);
- for a remote alias, `endpoint` and `credential_env` (see the next section);
- `verification`, the last verify result (see below).

Aliases that resolve to the same model (same provider, model, revision and
options) share one load, so they share one state.

`runtime.providers()` lists the registered providers and the tasks each one
supports.

A snapshot is a point-in-time view. A load that finishes while you take the
snapshot can show up in either state.

### What a snapshot never shows

An alias's `options` are never exposed. A remote alias shows:

- its `endpoint`, with any `user:password@`, query string and fragment
  removed (`https://user:pw@proxy/v1?token=x` becomes `https://proxy/v1`);
- its `credential_env`: the **name** of the environment variable holding its
  key, such as `OPENAI_API_KEY`, never the value.

## Prefetch with a result per alias

`prefetch` stops at the first failure. For a readiness report, use the
`_report` variants. They load one alias at a time, keep going past failures,
and return every alias's result in the order you asked:

```rust
for (alias, result) in runtime.prefetch_report(&["embed/default", "generate/chat"]).await {
    match result {
        Ok(()) => println!("{alias}: ready"),
        Err(e) => println!("{alias}: {e}"),
    }
}
```

`prefetch_all_report()` does the same for the whole catalog, in alias order.
An alias that isn't in the catalog reports `AliasNotFound`.

## Verify remote aliases

Building a runtime, prefetching and `health()` never contact a remote
provider: a remote alias "loads" as soon as its key's environment variable is
set. To check that the endpoint is reachable and accepts the key, ask for it:

```rust
use uni_xervo::error::RuntimeError;
use uni_xervo::inspect::VerifyErrorKind;

for (alias, result) in runtime.verify(&["generate/chat", "embed/remote"]).await {
    match result {
        Ok(outcome) => println!("{alias}: {outcome:?}"),
        Err(RuntimeError::Verification { kind: VerifyErrorKind::Unauthorized, .. }) => {
            println!("{alias}: key rejected")
        }
        Err(e) => println!("{alias}: {e}"),
    }
}
```

- Each alias is checked once, concurrently, and reported in the order given.
  `verify_all()` checks the whole catalog.
- Local aliases return `VerifyOutcome::NotApplicable` without doing anything.
- A failure is `RuntimeError::Verification { alias, provider_id, kind, detail }`.
  The message names the alias and never contains the key. `kind` is one of:
  - `Unauthorized` (HTTP 401 or 403; a 403 can also mean a valid key that
    lacks a permission);
  - `NotFound` (404: an unknown model or deployment, or an OpenAI-compatible
    server with no `/models` route);
  - `Unreachable`;
  - `Timeout`;
  - `RateLimited`;
  - `Unavailable` (5xx, or llama-server still loading its model);
  - `Config`: for example, the key's environment variable is not set.
- Each check waits at most the alias's `load_timeout`, capped at 30 seconds.
- Verification loads no model and does not touch the circuit breakers.

The result is kept. It shows up as each alias's `verification` in
`aliases()`, and `provider_health(id)` folds it in. Until one of the
provider's aliases has been verified, `provider_health(id)` returns exactly
what the provider's own `health()` reports. After that, it reads `Degraded`
if some verified aliases failed, or `Unhealthy` if all of them did.

### What each provider checks

| Provider | Request |
|---|---|
| `remote/openai` | `GET {base_url}/models` |
| `remote/anthropic` | `GET /v1/models/{model}` |
| `remote/azure-openai` | `POST` an empty body to the deployment's `embeddings` or `chat/completions` |
| `remote/cohere` | `GET /v1/models/{model}` |
| `remote/gemini` | `GET /{api_version}/models/{model}` |
| `remote/mistral` | `GET /v1/models/{model}` |
| `remote/vertexai` | `GET /v1/publishers/{publisher}/models/{model}` |
| `remote/voyageai` | `POST` an empty body to `/v1/embeddings` or `/v1/rerank` |
| `remote/llamacpp` | `GET /health`, then `GET {base_url}/models` if the alias has `api_key_env` |

Azure OpenAI has no read-only data-plane call for a deployment, and Voyage AI
has no models endpoint. So for both, verify sends an empty body to the
alias's real endpoint. The service checks the key (and, for Azure, that the
deployment exists) before it looks at the body. A `400` or `422` therefore
counts as verified, and nothing is generated or billed.

## Writing a provider

`ModelProvider::verify(&spec)` and `ModelProvider::describe(&spec)` both have
defaults (`NotApplicable`, and an empty description), so existing providers
keep compiling.

- Override `verify` to make your cheapest authenticated request. Keep keys
  out of the `VerifyFailure` detail.
- Override `describe` to return your endpoint and the name of the credential
  environment variable. It must not do I/O or read the secret itself.
