//! Single-pass hybrid embedding: the [`HybridEmbeddingModel`] trait and its
//! [`HeadSet`] / [`HybridEmbedResult`] types.
//!
//! Some exports — notably BGE-M3 (`aapot/bge-m3-onnx`) — fuse a dense head, a
//! learned-sparse head, and a multi-vector (ColBERT) head into one graph. The
//! per-task [`EmbeddingModel`](crate::traits::EmbeddingModel),
//! [`SparseEmbeddingModel`](crate::traits::SparseEmbeddingModel), and
//! [`MultiVectorEmbeddingModel`](crate::traits::MultiVectorEmbeddingModel)
//! handles each load their own session and run their own forward pass, so
//! serving all three means loading the weights three times and running the graph
//! three times.
//!
//! This trait collapses that to one: a single shared encoder forward pass whose
//! outputs are post-processed into every requested head. It exists only for
//! graphs that actually expose multiple heads (declared by a hybrid preset);
//! single-head models keep using the per-task handles. The caller picks which
//! heads to materialize with a [`HeadSet`], mirroring how
//! [`NlpTasks`](crate::traits::NlpTasks) selects NLP heads on a single pass.

use crate::error::Result;
use crate::traits::{ModelInfo, SparseVector, TokenUsage};
use async_trait::async_trait;
use bitflags::bitflags;

bitflags! {
    /// Selects which embedding heads a [`HybridEmbeddingModel`] should populate.
    ///
    /// A head is only post-processed when its flag is set *and* the model
    /// exposes it (see [`HybridEmbeddingModel::available_heads`]); the
    /// intersection drives which fields of [`HybridEmbedResult`] are `Some`.
    /// Skipping a head skips its post-processing, not the (shared) forward pass.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct HeadSet: u32 {
        /// Pooled dense vector (one per input).
        const DENSE = 1 << 0;
        /// Learned-sparse term weights (one [`SparseVector`] per input).
        const SPARSE = 1 << 1;
        /// Per-token multi-vector / ColBERT embeddings (ragged, per input).
        const MULTI_VECTOR = 1 << 2;
        /// Convenience: all currently-defined heads.
        const ALL = Self::DENSE.bits() | Self::SPARSE.bits() | Self::MULTI_VECTOR.bits();
    }
}

/// Dense, sparse, and multi-vector embeddings from a single forward pass.
///
/// Each field is `Some` iff its head was both **requested** (set in the `heads`
/// argument to [`HybridEmbeddingModel::embed`]) and **available** (set in
/// [`HybridEmbeddingModel::available_heads`]); otherwise it is `None`. When a
/// requested-and-available head is computed over an empty input batch the field
/// is `Some` of an empty `Vec`, never `None`.
///
/// `usage` follows the [`EmbedResult`](crate::traits::EmbedResult) convention:
/// `Some` for remote providers that report token counts, `None` for local ones.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HybridEmbedResult {
    /// Pooled dense vectors, one per input. `Some` iff the dense head was
    /// requested and available.
    pub dense: Option<Vec<Vec<f32>>>,
    /// Learned-sparse vectors, one per input. `Some` iff the sparse head was
    /// requested and available.
    pub sparse: Option<Vec<SparseVector>>,
    /// Per-token (multi-vector / ColBERT) embeddings, one ragged list per input.
    /// `Some` iff the multi-vector head was requested and available.
    pub multi_vector: Option<Vec<Vec<Vec<f32>>>>,
    /// Token usage reported by the provider, if any.
    pub usage: Option<TokenUsage>,
}

/// A model that produces several embedding heads from one shared forward pass.
///
/// Backed by multi-output exports such as BGE-M3 (`aapot/bge-m3-onnx`): one
/// encoder run feeds the dense, sparse, and ColBERT post-processors, so a hybrid
/// retrieval pipeline pays for one weight load and one forward pass instead of
/// three. Select the heads to materialize with a [`HeadSet`].
#[async_trait]
pub trait HybridEmbeddingModel: ModelInfo {
    /// Embed a batch of text strings, populating the heads in
    /// `heads ∩ available_heads()` from a single forward pass.
    ///
    /// Heads outside that intersection are left `None` on the result; the others
    /// are computed and returned in input order.
    ///
    /// # Errors
    /// Returns an error if tokenization fails, the model session errors, or a
    /// requested head's output cannot be read from the graph.
    async fn embed(&self, texts: &[&str], heads: HeadSet) -> Result<HybridEmbedResult>;

    /// The heads this model's graph actually exposes.
    ///
    /// Parallels [`NlpModel::supported_tasks`](crate::traits::NlpModel::supported_tasks):
    /// requesting a head outside this set yields `None` for it rather than an error.
    fn available_heads(&self) -> HeadSet;

    /// The width of a single head, if this model knows it.
    ///
    /// What "width" means is per head, matching the accessor each head
    /// corresponds to on the single-head traits:
    ///
    /// - [`HeadSet::DENSE`] — the dense vector dimensionality
    ///   ([`EmbeddingModel::dimensions`](crate::traits::EmbeddingModel::dimensions))
    /// - [`HeadSet::SPARSE`] — the term-space size
    ///   ([`SparseEmbeddingModel::vocab_size`](crate::traits::SparseEmbeddingModel::vocab_size))
    /// - [`HeadSet::MULTI_VECTOR`] — the per-token vector dimensionality
    ///   ([`MultiVectorEmbeddingModel::dimensions`](crate::traits::MultiVectorEmbeddingModel::dimensions))
    ///
    /// Returns `None` for a head this model does not expose, for a `head`
    /// naming more than one flag, and for implementations that simply don't
    /// track it — the default. Without it, the runtime cannot serve this model
    /// through the single-head accessors, since those traits must report a
    /// width; see [`ModelRuntime::sparse_embedder`](crate::runtime::ModelRuntime::sparse_embedder).
    fn head_width(&self, head: HeadSet) -> Option<u32> {
        let _ = head;
        None
    }

    /// Optional warmup hook (e.g. load weights into memory on first access).
    /// The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Dyn-safety smoke check: fails to compile if the trait becomes non-object-safe.
    #[test]
    fn hybrid_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn HybridEmbeddingModel>) {}
    }

    #[test]
    fn head_set_membership_and_composition() {
        let ds = HeadSet::DENSE | HeadSet::SPARSE;
        assert!(ds.contains(HeadSet::DENSE));
        assert!(ds.contains(HeadSet::SPARSE));
        assert!(!ds.contains(HeadSet::MULTI_VECTOR));
        assert_eq!(
            HeadSet::DENSE | HeadSet::SPARSE | HeadSet::MULTI_VECTOR,
            HeadSet::ALL
        );
    }

    #[test]
    fn head_set_intersection_disjoint_and_empty() {
        // Requested ⊄ available collapses to the available subset.
        let requested = HeadSet::DENSE | HeadSet::MULTI_VECTOR;
        let available = HeadSet::DENSE | HeadSet::SPARSE;
        assert_eq!(requested.intersection(available), HeadSet::DENSE);

        // Fully disjoint → empty; empty request → empty.
        assert!(HeadSet::SPARSE.intersection(HeadSet::DENSE).is_empty());
        assert!(HeadSet::empty().intersection(HeadSet::ALL).is_empty());
    }
}
