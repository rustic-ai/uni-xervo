# remote/openai

## Uni-Xervo support

- Provider ID: `remote/openai`
- Feature flag: `provider-openai`
- Capabilities: `embed`, `generate`

## Authentication

Default key env var:

- `OPENAI_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string, optional env var override)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/openai.schema.json>

## Authoritative model and config docs

- Model catalog: <https://platform.openai.com/docs/models>
- Chat generation request params: <https://platform.openai.com/docs/api-reference/chat/create>
- Embeddings request params: <https://platform.openai.com/docs/api-reference/embeddings/create>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/chat",
  "task": "generate",
  "provider_id": "remote/openai",
  "model_id": "gpt-4o-mini",
  "options": {
    "api_key_env": "OPENAI_API_KEY"
  }
}
```
