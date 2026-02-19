# Config Validation

Uni-Xervo supports two validation layers:

1. Runtime validation in Rust during runtime build/registration.
2. JSON Schema validation in CI before deployment.

## Runtime validation

The runtime rejects invalid catalogs and registrations for:

- alias format violations,
- duplicate aliases,
- unknown providers,
- invalid provider options (unknown keys, wrong types),
- non-positive `timeout` and `load_timeout`.

## Schema files

- `schemas/model-catalog.schema.json`
- `schemas/provider-options/*.schema.json`

Provider-specific option schemas are wired via `if/then` in the model catalog schema by `provider_id`.

## CI example with Ajv

```bash
npm i -g ajv-cli
ajv validate \
  -s schemas/model-catalog.schema.json \
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
