# Internals

This section is for contributors and platform engineers extending Uni-Xervo.

## Core components

- `api`: catalog types (`ModelAliasSpec`, `RetryConfig`, `WarmupPolicy`).
- `runtime`: provider registry, catalog map, keyed model instance cache, typed resolvers.
- `traits`: provider/model contracts.
- `reliability`: timeout/retry wrappers and circuit breaker implementation.
- `provider/*`: provider-specific loaders and model implementations.
- `options_validation`: strict provider option parser/validator.

## Internal guarantees

- At most one concurrent load per runtime key.
- Catalog alias uniqueness.
- Strict provider option keys and value type checks.
- Typed capability mismatch surfaced as explicit runtime errors.
