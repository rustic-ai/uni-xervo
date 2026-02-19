# remote/azure-openai

## Uni-Xervo support

- Provider ID: `remote/azure-openai`
- Feature flag: `provider-azure-openai`
- Capabilities: `embed`, `generate`

## Authentication

Default key env var:

- `AZURE_OPENAI_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string)
- `resource_name` (string, required)
- `api_version` (string, default `2024-10-21`)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/azure-openai.schema.json>

## Authoritative model and config docs

- Azure OpenAI models: <https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models>
- Azure OpenAI API reference/config: <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/azure",
  "task": "generate",
  "provider_id": "remote/azure-openai",
  "model_id": "gpt-4o-mini",
  "options": {
    "resource_name": "my-azure-openai-resource",
    "api_key_env": "AZURE_OPENAI_API_KEY",
    "api_version": "2024-10-21"
  }
}
```
