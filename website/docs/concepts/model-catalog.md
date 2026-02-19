# Model Catalog

A catalog is a JSON array of `ModelAliasSpec` objects.

Each entry maps a stable alias (for example `embed/default`) to a provider model and reliability/warmup configuration.

## Field reference

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `alias` | `string` | Yes | - | Must be non-empty and contain `/` in `task/name` style. |
| `task` | `embed \| rerank \| generate` | Yes | - | Declares intended capability for this alias. |
| `provider_id` | `string` | Yes | - | Provider ID (for example `local/candle`, `remote/openai`). |
| `model_id` | `string` | Yes | - | Provider-specific model identifier. |
| `revision` | `string \| null` | No | `null` | Optional model revision/version selector. |
| `warmup` | `eager \| lazy \| background` | No | `lazy` | Alias-specific load strategy. |
| `required` | `bool` | No | `false` | If `true`, eager warmup failures fail runtime startup. |
| `timeout` | `u64` seconds | No | unset | Per-inference timeout for wrapper calls. |
| `load_timeout` | `u64` seconds | No | `600` | Max provider load + model warmup duration. |
| `retry` | object | No | unset | Retry config with attempts and backoff. |
| `options` | `object \| null` | No | `null` | Strict provider-specific options. |

## Retry object

```json
{
  "max_attempts": 3,
  "initial_backoff_ms": 100
}
```

Backoff doubles each retry attempt.

## Example catalog

```json
[
  {
    "alias": "embed/local",
    "task": "embed",
    "provider_id": "local/candle",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "warmup": "eager",
    "required": true,
    "options": {
      "cache_dir": ".uni_cache/candle"
    }
  },
  {
    "alias": "rerank/semantic",
    "task": "rerank",
    "provider_id": "remote/cohere",
    "model_id": "rerank-v3.5",
    "timeout": 20,
    "retry": {
      "max_attempts": 3,
      "initial_backoff_ms": 200
    },
    "options": {
      "api_key_env": "CO_API_KEY"
    }
  },
  {
    "alias": "generate/chat",
    "task": "generate",
    "provider_id": "remote/azure-openai",
    "model_id": "gpt-4o-mini",
    "load_timeout": 120,
    "options": {
      "resource_name": "my-azure-openai",
      "api_version": "2024-10-21"
    }
  }
]
```

## Validation behavior

At builder/register time Uni-Xervo rejects:

- invalid alias format,
- duplicate aliases,
- unknown providers,
- provider option type/key violations,
- zero-valued `timeout` or `load_timeout`.

See [Config Validation](../guides/config-validation.md) for schema-based CI checks.
