# Reliability

Uni-Xervo applies reliability controls in instrumented model wrappers returned by runtime resolution.

## Per-inference timeout

Set `ModelAliasSpec.timeout` (seconds) to bound each inference call (`embed`, `rerank`, `generate`).

Timeout expiration maps to `RuntimeError::Timeout`.

## Retry behavior

Set `ModelAliasSpec.retry` to enable retries for retryable errors:

- `RateLimited`
- `Timeout`
- `Unavailable`

Retries use exponential backoff from `initial_backoff_ms`.

## Remote circuit breaker

Remote providers use per-model circuit breakers keyed by `ModelRuntimeKey`.

- After repeated failures, breaker opens and short-circuits calls with `Unavailable`.
- After wait window, breaker allows a half-open probe call.
- Success closes breaker, failure re-opens it.

## Metrics emitted

- `model_load.duration_seconds`
- `model_load.total` (`status=success|failure`)
- `model_inference.duration_seconds` (labels include alias/task/provider)
- `model_inference.total` (`status=success|failure`)

## Operational guidance

- Use short `timeout` on latency-sensitive aliases.
- Add retries only where transient remote failures are expected.
- Keep `load_timeout` higher than `timeout` for large model initialization.
