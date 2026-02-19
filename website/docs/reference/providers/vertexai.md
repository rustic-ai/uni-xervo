# remote/vertexai

## Uni-Xervo support

- Provider ID: `remote/vertexai`
- Feature flag: `provider-vertexai`
- Capabilities: `embed`, `generate`

## Authentication and project

Defaults used by Uni-Xervo:

- token env var: `VERTEX_AI_TOKEN`
- project fallback env var: `VERTEX_AI_PROJECT`

## Uni-Xervo provider options

- `api_token_env` (string)
- `project_id` (string)
- `location` (string)
- `publisher` (string)
- `embedding_dimensions` (integer > 0, embed task only)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/vertexai.schema.json>

## Authoritative model and config docs

- Model catalog: <https://cloud.google.com/vertex-ai/generative-ai/docs/models>
- Generation model reference/config: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference>
- Embeddings API/config: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "embed/vertex",
  "task": "embed",
  "provider_id": "remote/vertexai",
  "model_id": "text-embedding-005",
  "options": {
    "project_id": "my-gcp-project",
    "location": "us-central1",
    "api_token_env": "VERTEX_AI_TOKEN"
  }
}
```
