# local/candle

## Uni-Xervo support

- Provider ID: `local/candle`
- Feature flag: `provider-candle`
- Capabilities: `embed`

## Uni-Xervo provider options

- `cache_dir` (string)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/candle.schema.json>

## Available model IDs in Uni-Xervo

This provider currently supports these embedding model IDs:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-base-en-v1.5`

Authoritative source in code:

- <https://github.com/rustic-ai/uni-xervo/blob/main/src/provider/candle.rs>

Model cards:

- <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- <https://huggingface.co/BAAI/bge-small-en-v1.5>
- <https://huggingface.co/BAAI/bge-base-en-v1.5>

## Example catalog entry

```json
{
  "alias": "embed/default",
  "task": "embed",
  "provider_id": "local/candle",
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "options": {
    "cache_dir": ".uni_cache/candle"
  }
}
```
