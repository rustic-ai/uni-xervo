# remote/cohere

## Uni-Xervo support

- Provider ID: `remote/cohere`
- Feature flag: `provider-cohere`
- Capabilities: `embed`, `rerank`, `generate`

## Authentication

Default key env var:

- `CO_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string)
- `input_type` (string, embedding requests)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/cohere.schema.json>

## Authoritative model and config docs

- Model catalog: <https://docs.cohere.com/v2/docs/models>
- Chat/generation request config: <https://docs.cohere.com/v2/reference/chat>
- Embeddings request config: <https://docs.cohere.com/v2/reference/embed>
- Rerank request config: <https://docs.cohere.com/v2/reference/rerank>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "embed/cohere",
  "task": "embed",
  "provider_id": "remote/cohere",
  "model_id": "embed-english-v3.0",
  "options": {
    "api_key_env": "CO_API_KEY",
    "input_type": "search_document"
  }
}
```
