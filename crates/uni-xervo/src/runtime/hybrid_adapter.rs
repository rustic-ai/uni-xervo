//! Single-head views over a [`HybridEmbeddingModel`].
//!
//! A hybrid model exposes dense, sparse, and multi-vector heads from one
//! forward pass, but the runtime stores it as `Arc<dyn HybridEmbeddingModel>`
//! — a different concrete type from the single-head trait objects. Without an
//! adapter, `runtime.sparse_embedder("embed/hybrid")` fails the downcast and
//! reports `ProviderCapabilityMissing`, even though the graph plainly has the
//! head. Callers then have to register a second alias for the same weights, or
//! hand-roll a fallback per call site.
//!
//! These adapters close that gap: each wraps the hybrid handle, requests just
//! its own head, and unwraps the matching field of [`HybridEmbedResult`].
//! Heads not requested are not post-processed, so serving one head through a
//! hybrid handle costs the shared forward pass plus that head's own work — the
//! same as the per-task model would.
//!
//! The width accessors ([`SparseEmbeddingModel::vocab_size`] and friends) are
//! answered from [`HybridEmbeddingModel::head_width`], which is why a model
//! that cannot report a width cannot be adapted.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbedResult, EmbeddingModel, HeadSet, HybridEmbeddingModel, ModelInfo, MultiVectorEmbedResult,
    MultiVectorEmbeddingModel, SparseEmbedResult, SparseEmbeddingModel,
};

/// A head was requested but the hybrid model's `embed` left it `None`.
///
/// Unreachable when `available_heads()` is honest, since the adapters only
/// exist for available heads — but the trait's contract is a convention, so
/// report it rather than panicking on the `Option`.
fn missing_head(model_id: &str, head: &str) -> RuntimeError {
    RuntimeError::InferenceError(format!(
        "Hybrid model '{model_id}' reported the {head} head as available but returned no \
         {head} output"
    ))
}

// ---------------------------------------------------------------------------
// Sparse
// ---------------------------------------------------------------------------

/// Serves [`SparseEmbeddingModel`] from a hybrid handle's sparse head.
pub(super) struct HybridAsSparse {
    pub(super) inner: Arc<dyn HybridEmbeddingModel>,
    pub(super) vocab_size: u32,
}

impl ModelInfo for HybridAsSparse {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

#[async_trait]
impl SparseEmbeddingModel for HybridAsSparse {
    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    async fn embed(&self, texts: &[&str]) -> Result<SparseEmbedResult> {
        let result = self.inner.embed(texts, HeadSet::SPARSE).await?;
        let vectors = result
            .sparse
            .ok_or_else(|| missing_head(self.inner.model_id(), "sparse"))?;
        Ok(SparseEmbedResult {
            vectors,
            usage: result.usage,
        })
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

// ---------------------------------------------------------------------------
// Multi-vector
// ---------------------------------------------------------------------------

/// Serves [`MultiVectorEmbeddingModel`] from a hybrid handle's ColBERT head.
pub(super) struct HybridAsMultiVector {
    pub(super) inner: Arc<dyn HybridEmbeddingModel>,
    pub(super) dimensions: u32,
}

impl ModelInfo for HybridAsMultiVector {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

#[async_trait]
impl MultiVectorEmbeddingModel for HybridAsMultiVector {
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    async fn embed(&self, texts: &[&str]) -> Result<MultiVectorEmbedResult> {
        let result = self.inner.embed(texts, HeadSet::MULTI_VECTOR).await?;
        let vectors = result
            .multi_vector
            .ok_or_else(|| missing_head(self.inner.model_id(), "multi-vector"))?;
        Ok(MultiVectorEmbedResult {
            vectors,
            usage: result.usage,
        })
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

// ---------------------------------------------------------------------------
// Dense
// ---------------------------------------------------------------------------

/// Serves [`EmbeddingModel`] from a hybrid handle's dense head.
pub(super) struct HybridAsDense {
    pub(super) inner: Arc<dyn HybridEmbeddingModel>,
    pub(super) dimensions: u32,
}

impl ModelInfo for HybridAsDense {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

#[async_trait]
impl EmbeddingModel for HybridAsDense {
    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    async fn embed(&self, texts: &[&str]) -> Result<EmbedResult> {
        let result = self.inner.embed(texts, HeadSet::DENSE).await?;
        let vectors = result
            .dense
            .ok_or_else(|| missing_head(self.inner.model_id(), "dense"))?;
        Ok(EmbedResult {
            vectors,
            usage: result.usage,
        })
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

/// The width to report for `head`, or an error explaining why this hybrid
/// model cannot back the corresponding single-head trait.
///
/// `head` must name exactly one flag. Callers reach this only after the handle
/// has already been identified as a hybrid model, so both failure modes —
/// head not exposed, width not reported — name the model and say which applies,
/// rather than falling back on the bare "capability missing" message.
pub(super) fn head_width_of(
    model: &Arc<dyn HybridEmbeddingModel>,
    head: HeadSet,
    alias: &str,
    provider_id: &str,
    capability: &str,
) -> Result<u32> {
    if !model.available_heads().contains(head) {
        return Err(RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: provider_id.to_string(),
            capability: format!(
                "{capability} (alias resolves to hybrid model '{}', whose graph does not \
                 expose that head)",
                model.model_id()
            ),
        });
    }
    model
        .head_width(head)
        .ok_or_else(|| RuntimeError::ProviderCapabilityMissing {
            alias: alias.to_string(),
            provider_id: provider_id.to_string(),
            capability: format!(
                "{capability} (alias resolves to hybrid model '{}', which exposes the head \
                 but does not report its width)",
                model.model_id()
            ),
        })
}
