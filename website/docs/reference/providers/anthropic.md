# remote/anthropic

## Uni-Xervo support

- Provider ID: `remote/anthropic`
- Feature flag: `provider-anthropic`
- Capabilities: `generate`

## Authentication

Default key env var:

- `ANTHROPIC_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string)
- `anthropic_version` (string, defaults to `2023-06-01`)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/anthropic.schema.json>

## Authoritative model and config docs

- Model catalog: <https://docs.anthropic.com/en/docs/about-claude/models/all-models>
- Messages API request/config docs: <https://docs.anthropic.com/en/api/messages>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/claude",
  "task": "generate",
  "provider_id": "remote/anthropic",
  "model_id": "claude-sonnet-4-5",
  "options": {
    "api_key_env": "ANTHROPIC_API_KEY",
    "anthropic_version": "2023-06-01"
  }
}
```
