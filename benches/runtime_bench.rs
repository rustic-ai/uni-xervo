use async_trait::async_trait;
use criterion::{Criterion, criterion_group, criterion_main};
use std::sync::Arc;
use tokio::runtime::Runtime;
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::Result;
use uni_xervo::provider::candle::LocalCandleProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
};

// --- Bench Components ---

#[derive(Clone)]
struct BenchEmbeddingModel;

#[async_trait]
impl EmbeddingModel for BenchEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        // pure overhead measurement
        Ok(vec![vec![0.0; 384]; texts.len()])
    }
    fn dimensions(&self) -> u32 {
        384
    }
    fn model_id(&self) -> &str {
        "bench"
    }
}

struct BenchProvider;

#[async_trait]
impl ModelProvider for BenchProvider {
    fn provider_id(&self) -> &'static str {
        "bench"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed],
        }
    }
    async fn load(&self, _spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let model = BenchEmbeddingModel;
        // Correct wrapping for Any + Send + Sync
        let arc_model: Arc<dyn EmbeddingModel> = Arc::new(model);
        Ok(Arc::new(arc_model) as LoadedModelHandle)
    }
    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

// --- Benchmarks ---

fn bench_runtime_init(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("runtime_init_cold", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = ModelRuntime::builder()
                .register_provider(LocalCandleProvider::new())
                .build()
                .await
                .unwrap();
        })
    });
}

fn bench_embed_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // 1. Lazy Runtime (first call will be slow if it actually loaded,
    // but here BenchProvider is fast. Still, let's measure overhead)
    let runtime_lazy = rt.block_on(async {
        let provider = BenchProvider;
        let spec = ModelAliasSpec {
            alias: "bench/lazy".to_string(),
            task: ModelTask::Embed,
            provider_id: "bench".to_string(),
            model_id: "bench".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Object(serde_json::Map::new()),
        };

        ModelRuntime::builder()
            .register_provider(provider)
            .catalog(vec![spec])
            .build()
            .await
            .unwrap()
    });

    // 2. Eager Runtime
    let runtime_eager = rt.block_on(async {
        let provider = BenchProvider;
        let spec = ModelAliasSpec {
            alias: "bench/eager".to_string(),
            task: ModelTask::Embed,
            provider_id: "bench".to_string(),
            model_id: "bench".to_string(),
            revision: None,
            warmup: WarmupPolicy::Eager,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::Value::Object(serde_json::Map::new()),
        };

        ModelRuntime::builder()
            .register_provider(provider)
            .catalog(vec![spec])
            .build()
            .await
            .unwrap()
    });

    c.bench_function("embed_latency_lazy_overhead", |b| {
        b.to_async(&rt).iter(|| async {
            let model = runtime_lazy.embedding("bench/lazy").await.unwrap();
            let _ = model.embed(vec!["hello world"]).await.unwrap();
        })
    });

    c.bench_function("embed_latency_eager_overhead", |b| {
        b.to_async(&rt).iter(|| async {
            let model = runtime_eager.embedding("bench/eager").await.unwrap();
            let _ = model.embed(vec!["hello world"]).await.unwrap();
        })
    });
}

criterion_group!(benches, bench_runtime_init, bench_embed_latency);
criterion_main!(benches);
