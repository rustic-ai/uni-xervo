# remote/voyageai

## Uni-Xervo support

- Provider ID: `remote/voyageai`
- Feature flag: `provider-voyageai`
- Capabilities: `embed`, `rerank`

## Authentication

Default key env var:

- `VOYAGE_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/voyageai.schema.json>

## Authoritative model and config docs

- Embedding models and config: <https://docs.voyageai.com/docs/embeddings>
- Rerank models and config: <https://docs.voyageai.com/docs/reranker>

## Example catalog entry

```json
{
  "alias": "rerank/voyage",
  "task": "rerank",
  "provider_id": "remote/voyageai",
  "model_id": "rerank-2",
  "options": {
    "api_key_env": "VOYAGE_API_KEY"
  }
}
```
