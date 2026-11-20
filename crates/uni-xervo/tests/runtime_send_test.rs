//! Compile-time check that every public async method of `ModelRuntime`
//! returns a future that is `Send` for any lifetime.
//!
//! Callers hold these futures across `.await` inside `tokio::spawn`,
//! `pyo3_async_runtimes::tokio::future_into_py` and other places that demand
//! `Send` for every lifetime. A future that is `Send` only for one particular
//! lifetime still compiles inside the crate, so the failure shows up only
//! downstream (rustic-ai/uni-xervo#73). Each check wraps the call in an
//! `async` block that owns its inputs, as a spawned task would, and requires
//! that block to be `Send + 'static`. Nothing here runs: if the file
//! compiles, the test has passed.

use std::sync::Arc;
use uni_xervo::api::ModelAliasSpec;
use uni_xervo::runtime::{ModelRuntime, ModelRuntimeBuilder};

fn assert_send<T: Send + 'static>(_: T) {}

/// Owned alias names, borrowed as `&[&str]` inside the future, the way a
/// caller would pass them.
macro_rules! with_refs {
    ($rt:ident, $names:ident, |$refs:ident| $call:expr) => {
        assert_send({
            let $rt = $rt.clone();
            let $names = $names.clone();
            async move {
                let $refs: Vec<&str> = $names.iter().map(String::as_str).collect();
                let _ = $call.await;
            }
        })
    };
}

macro_rules! with_alias {
    ($rt:ident, $name:ident, $method:ident) => {
        assert_send({
            let $rt = $rt.clone();
            let $name = $name.clone();
            async move {
                let _ = $rt.$method(&$name).await;
            }
        })
    };
}

#[allow(dead_code)]
fn runtime_futures_are_send(rt: Arc<ModelRuntime>, spec: ModelAliasSpec) {
    let names = vec!["embed/default".to_string()];
    let name = "embed/default".to_string();

    with_refs!(rt, names, |refs| rt.verify(&refs));
    with_refs!(rt, names, |refs| rt.prefetch(&refs));
    with_refs!(rt, names, |refs| rt.prefetch_report(&refs));

    assert_send({
        let rt = rt.clone();
        async move {
            let _ = rt.verify_all().await;
            let _ = rt.prefetch_all().await;
            let _ = rt.prefetch_all_report().await;
            let _ = rt.aliases().await;
            let _ = rt.register(spec).await;
        }
    });

    with_alias!(rt, name, contains_alias);
    with_alias!(rt, name, alias);
    with_alias!(rt, name, provider_health);
    with_alias!(rt, name, embedding);
    with_alias!(rt, name, embedder);
    with_alias!(rt, name, reranker);
    with_alias!(rt, name, generator);
    with_alias!(rt, name, raw_tensor_model);
    with_alias!(rt, name, image_embedder);
    with_alias!(rt, name, audio_embedder);
    with_alias!(rt, name, multimodal_embedder);
    with_alias!(rt, name, sparse_embedder);
    with_alias!(rt, name, multi_vector_embedder);
    with_alias!(rt, name, hybrid_embedder);
    with_alias!(rt, name, nlp_model);
    with_alias!(rt, name, document_extractor);
    with_alias!(rt, name, transcriber);
    with_alias!(rt, name, ocr_model);

    assert_send(async move {
        let _ = ModelRuntimeBuilder::default().build().await;
    });
}

#[test]
fn runtime_futures_compile_as_send() {
    // The assertions are in `runtime_futures_are_send`, checked by the
    // compiler; this test exists so the file is part of the test run.
}
