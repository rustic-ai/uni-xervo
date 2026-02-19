# remote/gemini

## Uni-Xervo support

- Provider ID: `remote/gemini`
- Feature flag: `provider-gemini`
- Capabilities: `embed`, `generate`

## Authentication

Default key env var:

- `GEMINI_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string, optional env var override)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/gemini.schema.json>

## Authoritative model and config docs

- Model catalog: <https://ai.google.dev/gemini-api/docs/models>
- Text generation docs/config: <https://ai.google.dev/gemini-api/docs/text-generation>
- Embeddings docs/config: <https://ai.google.dev/gemini-api/docs/embeddings>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/gemini",
  "task": "generate",
  "provider_id": "remote/gemini",
  "model_id": "gemini-2.0-flash",
  "options": {
    "api_key_env": "GEMINI_API_KEY"
  }
}
```
