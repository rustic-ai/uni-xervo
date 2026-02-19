# Contributing to Uni-Xervo

Thanks for contributing.

## Scope

This project is a Rust runtime for provider-agnostic embedding, reranking, and generation.
Contributions are welcome for:

- Bug fixes
- New providers and provider improvements
- Reliability/performance improvements
- Tests and documentation

## Before You Start

1. Open an issue for non-trivial changes (new provider, API shape changes, major refactors).
2. Confirm the change fits the project direction in `docs/first-spec.md`.
3. Keep pull requests small and reviewable.

## Development Setup

Prerequisites:

- Rust `1.85+`
- `cargo` and `rustfmt`
- Optional: `cargo-nextest` for faster test cycles

Clone and verify:

```bash
./scripts/build.sh
./scripts/test.sh
```

## Coding Standards

- Follow existing style and module structure.
- Prefer explicit error messages and typed failures.
- Add or update tests for behavior changes.
- Keep provider-specific behavior behind provider feature flags.
- Do not introduce breaking API changes without discussion.

## Testing Requirements

At minimum, run:

```bash
./scripts/test.sh
```

For provider-heavy changes, also run relevant feature checks, for example:

```bash
cargo check --locked --no-default-features --features provider-openai
cargo check --locked --no-default-features --features provider-candle,provider-openai
```

For real-provider integration checks:

```bash
./scripts/test-integration.sh
```

## Documentation Requirements

Update docs when relevant:

- `README.md` for user-facing usage/API changes
- `docs/USER_GUIDE.md` for workflow details
- `TESTING.md` for test-process changes
- `RELEASE.md` if release flow changes

## Pull Request Checklist

- [ ] Scope is focused and well described
- [ ] Tests added/updated for behavior changes
- [ ] `./scripts/test.sh` passes
- [ ] Docs updated where needed
- [ ] Changelog entry added when user-visible behavior changes

## Commit Messages

Conventional-style prefixes are recommended:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` internal restructuring
- `test:` test-only changes
- `chore:` maintenance

## Review Expectations

Maintainers prioritize correctness, reliability, and compatibility.
Be ready to revise edge cases, tests, and documentation during review.

## Licensing

By contributing, you agree your contributions are licensed under the repository license (`Apache-2.0`).
