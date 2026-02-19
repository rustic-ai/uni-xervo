# remote/mistral

## Uni-Xervo support

- Provider ID: `remote/mistral`
- Feature flag: `provider-mistral`
- Capabilities: `embed`, `generate`

## Authentication

Default key env var:

- `MISTRAL_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string, optional env var override)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/mistral.schema.json>

## Authoritative model and config docs

- Model catalog: <https://docs.mistral.ai/getting-started/models/models_overview/>
- API request/reference docs: <https://docs.mistral.ai/api/>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/mistral",
  "task": "generate",
  "provider_id": "remote/mistral",
  "model_id": "mistral-small-latest",
  "options": {
    "api_key_env": "MISTRAL_API_KEY"
  }
}
```
