# Config Validation

Uni-Xervo supports two validation layers:

1. Runtime validation in Rust during runtime build/registration.
2. JSON Schema validation in CI before deployment.

## Runtime validation

The runtime rejects invalid catalogs and registrations for:

- alias format violations,
- duplicate aliases,
- unknown providers,
- tasks a bundled provider does not support,
- invalid provider options (unknown keys, keys that belong to another task, wrong types or values),
- non-positive `timeout` and `load_timeout`.

## Schema files

- `schemas/model-catalog.schema.json`
- `schemas/provider-options/*.schema.json`

The schemas use JSON Schema draft 2020-12. Each provider file holds one
options schema per task under `$defs/<task>`, and its top-level schema accepts
the keys of any of the provider's tasks. The catalog schema routes each entry
by `provider_id` and `task`: it rejects a task the provider does not support,
and checks `options` against that task's definition. Entries for providers
that are not bundled with Uni-Xervo get no `options` checks, as at runtime:
their options may be any JSON value.

Integer options may be written as whole-number floats (`128.0`), as Python's
`json` writes them: JSON Schema's `integer` accepts them, and the runtime reads
them as the integer.

A test in the crate keeps the schemas in step with runtime validation: each
schema must declare exactly the keys the runtime accepts per task, and
generated cases must get the same verdict from both.

## Schemas from Rust

The `uni_xervo::schemas` module embeds the same documents and needs no
provider feature, runtime or model load:

```rust
use uni_xervo::api::ModelTask;
use uni_xervo::schemas;

let catalog_schema: &str = schemas::MODEL_CATALOG;
let onnx_schema = schemas::provider_options_schema("local/onnx").unwrap();
let tasks = schemas::supported_tasks("remote/openai").unwrap();
let keys = schemas::option_keys("local/onnx", ModelTask::Ocr).unwrap();
schemas::validate_options("remote/openai", ModelTask::Embed, &serde_json::json!({
    "embedding_dimensions": 256
}))?;
```

The schemas returned are the ones that crate version validates against.

## CI example with Ajv

The catalog schema refers to the provider files by relative `$ref`, so pass
them with `-r`:

```bash
npm i -g ajv-cli
ajv validate --spec=draft2020 \
  -s schemas/model-catalog.schema.json \
  -r "schemas/provider-options/*.schema.json" \
  -d catalog.json
```

## Typical validation errors

- `additionalProperties` error: unknown option key for provider.
- `minimum` error: zero or negative numeric values for timeout/backoff fields.
- `enum` error: unsupported `task` or `warmup` value.
- `pattern` error: alias missing `/`.

## Best practices

- Version-control catalogs and treat them as deploy artifacts.
- Validate in CI and again at startup.
- Keep provider-specific options minimal and explicit.
