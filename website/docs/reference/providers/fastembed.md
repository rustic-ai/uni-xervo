# local/fastembed

## Uni-Xervo support

- Provider ID: `local/fastembed`
- Feature flag: `provider-fastembed`
- Capabilities: `embed`

## Uni-Xervo provider options

- `cache_dir` (string)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/fastembed.schema.json>

## Available model IDs

Uni-Xervo maps `model_id` values to FastEmbed's supported embedding model set.

Authoritative model references:

- FastEmbed supported models: <https://qdrant.github.io/fastembed/examples/Supported_Models/>
- FastEmbed project docs: <https://qdrant.github.io/fastembed/>
- Uni-Xervo model mapping source: <https://github.com/rustic-ai/uni-xervo/blob/main/src/provider/fastembed.rs>

## Model configuration references

For model-specific behavior and constraints, use FastEmbed docs as the source of truth.

## Example catalog entry

```json
{
  "alias": "embed/fast",
  "task": "embed",
  "provider_id": "local/fastembed",
  "model_id": "BGESmallENV15",
  "options": {
    "cache_dir": ".uni_cache/fastembed"
  }
}
```
