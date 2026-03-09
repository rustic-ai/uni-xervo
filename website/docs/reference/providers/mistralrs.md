# local/mistralrs

## Uni-Xervo support

- Provider ID: `local/mistralrs`
- Feature flag: `provider-mistralrs`
- Capabilities: `embed`, `generate`

## Uni-Xervo provider options

- `isq` (string)
- `force_cpu` (boolean)
- `paged_attention` (boolean)
- `max_num_seqs` (integer > 0)
- `chat_template` (string)
- `tokenizer_json` (string)
- `embedding_dimensions` (integer > 0, embed task only)
- `gguf_files` (array of strings)
- `dtype` (string: `"auto"` | `"f32"` | `"f16"` | `"bf16"`, default: `"auto"`)

  Overrides the compute dtype used by the model. The default `"auto"` lets
  mistral.rs select the best dtype for the detected hardware (typically F16 on
  GPU, BF16 on newer CPUs). On CPUs **without** native F16 hardware support,
  `"auto"` may select F16 and produce all-NaN embedding vectors. Set
  `"dtype": "f32"` to avoid this:

  ```json
  { "dtype": "f32" }
  ```

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/mistralrs.schema.json>

## Available models

`local/mistralrs` delegates model support to the upstream `mistral.rs` engine.

Authoritative model/support references:

- mistral.rs docs: <https://ericlbuehler.github.io/mistral.rs/>
- mistral.rs repository: <https://github.com/EricLBuehler/mistral.rs>

## Model configuration references

Use mistral.rs docs for model-family and runtime behavior details.

Uni-Xervo generation API currently exposes:

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/local",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "options": {
    "isq": "Q4K",
    "paged_attention": true,
    "max_num_seqs": 8
  }
}
```

### CPU-only embedding example (avoids F16 NaN issue)

```json
{
  "alias": "embed/default",
  "task": "embed",
  "provider_id": "local/mistralrs",
  "model_id": "google/embeddinggemma-300m",
  "options": {
    "dtype": "f32"
  }
}
```
