# Runtime Loading

Runtime loading has two phases: provider warmup and model warmup.

## Provider warmup (`ModelRuntimeBuilder::warmup_policy`)

- `eager`: await `provider.warmup()` for each registered provider during `build()`.
- `background`: schedule `provider.warmup()` in detached tasks.
- `lazy`: no provider warmup during build.

## Model warmup (`ModelAliasSpec.warmup`)

- `eager`: load and warm model during build.
- `background`: schedule model load after build.
- `lazy`: load on first handle resolution.

`required = true` matters only for eager warmup: if eager load fails for a required alias, startup fails.

## Deduplication and concurrency

Models are keyed by `ModelRuntimeKey`:

- task
- provider ID
- model ID
- revision
- normalized options hash

Aliases with identical runtime keys share one loaded instance.

Concurrent first-load calls for the same key are serialized with a per-key mutex so only one load happens.

## Load timeout

`load_timeout` applies to `provider.load(spec)` plus model warmup, with a runtime default of `600` seconds if not set.

A load timeout returns `RuntimeError::Timeout`.

## Prefetch APIs

- `runtime.prefetch_all().await` warms every alias.
- `runtime.prefetch(&["embed/default", "generate/chat"]).await` warms selected aliases.

These methods are useful at service startup to avoid first-request cold starts.
