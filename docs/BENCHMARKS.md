# Uni-Xervo Benchmarks

Performance metrics for the Unified Model Runtime.

## Test Environment
- OS: Linux
- Architecture: x86_64
- Runtime: Tokio (Multi-threaded)

## Core Registry Overhead

| Operation | Latency (ns) | Description |
| --------- | ------------ | ----------- |
| `runtime_init_cold` | ~300ns | Creating `ModelRuntime` with one registered provider. |
| `embed_latency_lazy` | ~370ns | `runtime.embedding()` + `model.embed()` (Hot path, Lazy). |
| `embed_latency_eager` | ~330ns | `runtime.embedding()` + `model.embed()` (Hot path, Eager). |

**Note:** The `Eager` policy provides approximately 10% faster access to model instances by ensuring they are already present in the registry, reducing synchronization and lookup overhead.

## Local Provider Warmup Impact (LocalCandleProvider)

*Experimental data based on `all-MiniLM-L6-v2`.*

| Policy | First Call Latency | Subsequent Call Latency |
| ------ | ------------------ | ----------------------- |
| `Lazy` | ~200ms - 500ms | ~20ms |
| `Eager` | ~20ms | ~20ms |

### Key Takeaways
1. **Registry Overhead is Minimal:** Sub-microsecond overhead for the abstraction layer.
2. **Eager Policy Benefits:** Pre-loading weights into memory (Eager) eliminates the 200ms+ cold-start penalty for local models.
3. **Background Policy:** Useful for improving application startup time while still ensuring models are ready shortly after boot.
