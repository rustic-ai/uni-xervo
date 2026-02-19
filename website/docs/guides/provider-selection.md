# Provider Selection

Choose providers based on task coverage, latency profile, data governance, and operational constraints.

## Capability-first matrix

| Provider ID | Embed | Rerank | Generate | Typical use |
| --- | --- | --- | --- | --- |
| `local/candle` | Yes | No | No | Low-latency local embedding with simple deploys |
| `local/fastembed` | Yes | No | No | ONNX-backed local embedding |
| `local/mistralrs` | Yes | No | Yes | Self-hosted local embedding + generation |
| `remote/openai` | Yes | No | Yes | Hosted general-purpose embeddings and chat |
| `remote/gemini` | Yes | No | Yes | Hosted Google model family |
| `remote/vertexai` | Yes | No | Yes | GCP-native hosted models |
| `remote/mistral` | Yes | No | Yes | Hosted Mistral models |
| `remote/anthropic` | No | No | Yes | Hosted generation/chat only |
| `remote/voyageai` | Yes | Yes | No | Hosted embedding + reranking focus |
| `remote/cohere` | Yes | Yes | Yes | Hosted unified embedding/rerank/generate |
| `remote/azure-openai` | Yes | No | Yes | Azure-governed OpenAI deployments |

## Decision framework

1. Task coverage: ensure provider supports required `task`.
2. Data policy: local providers for stricter data residency/control.
3. Latency and throughput: local can reduce network latency; remote can simplify scaling.
4. Reliability posture: tune `timeout`, `retry`, and warmup strategy per alias.
5. Change management: keep alias names stable while swapping providers in catalog.

## Common patterns

- Local embed + remote generate:
  - `embed/default` -> `local/candle`
  - `generate/default` -> `remote/openai`
- Multi-provider remote fallback strategy in app layer:
  - `generate/primary` -> `remote/anthropic`
  - `generate/backup` -> `remote/gemini`
- RAG pipeline split:
  - embed via `remote/voyageai`
  - rerank via `remote/cohere`
  - generate via `remote/azure-openai`

## Developer notes

- Enable only required provider feature flags.
- Register providers explicitly in runtime builder.
- Validate catalogs in CI before deployment.
- Use the provider reference pages for official model/config links: [Provider Reference](../reference/providers/index.md).
