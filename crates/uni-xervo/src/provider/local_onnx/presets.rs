// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Embedding-model presets for `local/onnx`.
//!
//! Maps short alias strings (and HF repo IDs) to concrete ONNX embedding
//! configuration: file layout within the repo, pooling strategy,
//! dimensionality, and tokenization shape. Catalog entries that match a
//! preset alias inherit these defaults; per-spec `options` may override.
//!
//! Presets are looked up case-sensitively via [`lookup`] against either the
//! `model_id` from a [`crate::api::ModelAliasSpec`] or any alias string in
//! the entry's `aliases` table.
//!
//! The current table mirrors the 25 text-embedding models exposed by the
//! (now-retired) `provider-fastembed` wrapper, so existing catalogs that
//! used FastEmbed alias strings (e.g. `"BGESmallENV15"`,
//! `"all-MiniLM-L6-v2"`) keep resolving without changes.
//!
//! ## How dimensions, pooling, and `token_type_ids` were chosen
//!
//! - **Dimensions** match `LocalFastEmbedProvider::dimension_for` exactly
//!   (most are 384 / 768 / 1024; BGESmallZHV15 is 512 — the architectural
//!   exception in the BGE-small family).
//! - **Pooling**: BERT-family models (BGE, MiniLM, MPNet, Mxbai, ModernBERT)
//!   use CLS; sentence-transformer-style models trained for mean-pool
//!   (E5, Paraphrase, Nomic) use Mean.
//! - **`token_type_ids`**: BERT-family models accept it; Nomic Embed (built
//!   on Nomic-BERT, no type ids) does not.
//! - **`max_seq_len`** is 512 across the board to match `fastembed-rs`'s
//!   `DEFAULT_MAX_LENGTH`. Long-context users can override via
//!   `options.max_seq_len`.
//! - **Normalization** is `true` for every preset (matches FastEmbed).

/// How a preset's hidden states are reduced to a single vector per input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolingKind {
    /// First-token (`[CLS]`) pool: take `hidden[:, 0, :]`.
    Cls,
    /// Mean pool over unmasked positions.
    Mean,
    /// Per-dimension max over unmasked positions.
    #[allow(dead_code)] // No preset uses Max today; included for completeness.
    Max,
    /// Last non-pad position pool: take the hidden state at the rightmost
    /// `attention_mask == 1` index. Convention used by decoder-style LLM
    /// embedders (Qwen3-Embedding, NV-Embed, gte-Qwen2-instruct).
    LastToken,
}

/// A canonical embedding-model configuration that matches one or more aliases.
#[derive(Debug, Clone)]
pub(super) struct EmbeddingPreset {
    /// Strings that resolve to this preset (HF repo IDs and short aliases).
    pub aliases: &'static [&'static str],
    /// HuggingFace repo to download from.
    pub hf_repo: &'static str,
    /// Path to the `.onnx` file within the repo.
    pub onnx_path: &'static str,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: &'static str,
    /// Extra files to download alongside the `.onnx` file (e.g. external-data
    /// `.onnx_data` sidecars for models too large to fit in a single protobuf,
    /// or auxiliary constant tensors). Files land in the same cache directory
    /// so ONNX Runtime can resolve them via relative paths embedded in the
    /// graph. Use an empty slice if none are required.
    pub additional_files: &'static [&'static str],
    /// Output embedding dimensionality.
    pub dimensions: u32,
    /// Pooling strategy applied to the model's last hidden state.
    pub pooling: PoolingKind,
    /// Whether outputs are L2-normalized.
    pub normalize: bool,
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
    /// Whether the model expects a `token_type_ids` input tensor.
    pub token_type_ids: bool,
    /// Graph output to read for the dense vector.
    ///
    /// `None` selects the first declared output (the norm for single-output
    /// dense exports). Set it to pin a specific head on a multi-output graph —
    /// e.g. `Some("dense_vecs")` for `aapot/bge-m3-onnx`, whose graph also emits
    /// the sparse and ColBERT heads.
    pub output_name: Option<&'static str>,
}

const PRESETS: &[EmbeddingPreset] = &[
    // ---- All-MiniLM family (sentence-transformers; mean-pool) -------------
    EmbeddingPreset {
        aliases: &["AllMiniLML6V2", "all-MiniLM-L6-v2"],
        hf_repo: "Qdrant/all-MiniLM-L6-v2-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML6V2Q"],
        hf_repo: "Xenova/all-MiniLM-L6-v2",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML12V2"],
        hf_repo: "Xenova/all-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML12V2Q"],
        hf_repo: "Xenova/all-MiniLM-L12-v2",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    // ---- All-MPNet (mean-pool; MPNet has no token_type_ids) ---------------
    EmbeddingPreset {
        aliases: &["AllMpnetBaseV2", "all-mpnet-base-v2"],
        hf_repo: "Xenova/all-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: false,
        output_name: None,
    },
    // ---- BGE English ------------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "BGESmallENV15",
            "bge-small-en-v1.5",
            "BAAI/bge-small-en-v1.5",
        ],
        hf_repo: "Xenova/bge-small-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["BGESmallENV15Q"],
        hf_repo: "Qdrant/bge-small-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["BGEBaseENV15", "bge-base-en-v1.5", "BAAI/bge-base-en-v1.5"],
        hf_repo: "Xenova/bge-base-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["BGEBaseENV15Q"],
        hf_repo: "Qdrant/bge-base-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &[
            "BGELargeENV15",
            "bge-large-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ],
        hf_repo: "Xenova/bge-large-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["BGELargeENV15Q"],
        hf_repo: "Qdrant/bge-large-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    // ---- BGE Chinese ------------------------------------------------------
    EmbeddingPreset {
        aliases: &["BGESmallZHV15", "BAAI/bge-small-zh-v1.5"],
        hf_repo: "Xenova/bge-small-zh-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        // bge-small-zh-v1.5 has hidden_size=512 (architectural exception in
        // the BGE-small family — English variant is 384).
        dimensions: 512,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["BGELargeZHV15", "BAAI/bge-large-zh-v1.5"],
        hf_repo: "Xenova/bge-large-zh-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    // ---- BGE M3 (multilingual, multifunc) ---------------------------------
    EmbeddingPreset {
        aliases: &["BGEM3", "BAAI/bge-m3"],
        hf_repo: "BAAI/bge-m3",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        // BGE-M3 ships its weights as ONNX external-data: the `.onnx_data`
        // sidecar holds the bulk tensor blob, and `Constant_7_attr__value`
        // is a separate constant tensor referenced by the graph.
        additional_files: &["onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        // BAAI's M3 ONNX export omits token_type_ids.
        token_type_ids: false,
        // BAAI's official export is dense-only — its graph emits
        // `token_embeddings` + `sentence_embedding`, and the first declared
        // output is the dense one, so no pin is needed.
        output_name: None,
    },
    // ---- BGE-M3 dense — community multi-output export (alternative to BAAI) -
    // `aapot/bge-m3-onnx` is the same multilingual BGE-M3 but exported with all
    // three heads in one graph: `dense_vecs` (#0), `sparse_vecs` (#1),
    // `colbert_vecs` (#2). This preset reads the dense head; the sparse and
    // multi-vector tasks read the other two from the same repo (the task
    // disambiguates — see SPARSE_PRESETS / MULTI_VECTOR_PRESETS). `dense_vecs`
    // is already a pooled [batch, 1024] vector, so the embedder's 2-D
    // pass-through path uses it directly and `pooling` is not applied.
    EmbeddingPreset {
        aliases: &["BGEM3Dense", "aapot/bge-m3-onnx#dense", "aapot/bge-m3-onnx"],
        hf_repo: "aapot/bge-m3-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // Same 2.1 GB external-data sidecar + constant tensor as the BGE-M3
        // sparse/ColBERT presets (one graph, three heads).
        additional_files: &["model.onnx.data", "Constant_685_attr__value"],
        dimensions: 1024,
        // Unused: `dense_vecs` is pre-pooled, so the 2-D path skips pooling.
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        // aapot graph inputs are input_ids + attention_mask only.
        token_type_ids: false,
        output_name: Some("dense_vecs"),
    },
    // ---- Nomic (mean-pool; ONNX export DOES consume token_type_ids) ------
    EmbeddingPreset {
        aliases: &["NomicEmbedTextV1", "nomic-ai/nomic-embed-text-v1"],
        hf_repo: "nomic-ai/nomic-embed-text-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &[
            "NomicEmbedTextV15",
            "nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
        ],
        hf_repo: "nomic-ai/nomic-embed-text-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["NomicEmbedTextV15Q"],
        hf_repo: "nomic-ai/nomic-embed-text-v1.5",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    // ---- Paraphrase Multilingual -----------------------------------------
    EmbeddingPreset {
        aliases: &["ParaphraseMLMiniLML12V2"],
        hf_repo: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["ParaphraseMLMiniLML12V2Q"],
        hf_repo: "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &["ParaphraseMLMpnetBaseV2"],
        hf_repo: "Xenova/paraphrase-multilingual-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // MPNet architecture: no token_type_ids.
        token_type_ids: false,
        output_name: None,
    },
    // ---- Multilingual E5 --------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Small",
            "multilingual-e5-small",
            "intfloat/multilingual-e5-small",
        ],
        hf_repo: "intfloat/multilingual-e5-small",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Base",
            "multilingual-e5-base",
            "intfloat/multilingual-e5-base",
        ],
        hf_repo: "intfloat/multilingual-e5-base",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // intfloat's E5-base ONNX export omits token_type_ids (unlike the
        // Small variant, which accepts it).
        token_type_ids: false,
        output_name: None,
    },
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Large",
            "multilingual-e5-large",
            "intfloat/multilingual-e5-large",
        ],
        hf_repo: "Qdrant/multilingual-e5-large-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // E5-large is exported with external data — model.onnx is the graph,
        // model.onnx_data is the weight blob.
        additional_files: &["model.onnx_data"],
        dimensions: 1024,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // Qdrant's E5-large ONNX export omits token_type_ids (matches
        // intfloat/multilingual-e5-base behavior; only Small accepts them).
        token_type_ids: false,
        output_name: None,
    },
    // ---- Mxbai ------------------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "MxbaiEmbedLargeV1",
            "mxbai-embed-large-v1",
            "mixedbread-ai/mxbai-embed-large-v1",
        ],
        hf_repo: "mixedbread-ai/mxbai-embed-large-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
        output_name: None,
    },
    // ---- ModernBERT (mean-pool; no token_type_ids — ModernBERT replaces
    //                  segment ids with rotary position embeddings) --------
    EmbeddingPreset {
        aliases: &["ModernBertEmbedLarge", "lightonai/modernbert-embed-large"],
        hf_repo: "lightonai/modernbert-embed-large",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: false,
        output_name: None,
    },
    // ---- Qwen3-Embedding (decoder-only LLM embedder; last-token pool;
    //                       external-data ONNX layout) ---------------------
    //
    // Qwen3-Embedding uses an instruction-prefixed input convention
    // ("Instruct: <task>\nQuery: <text>") for best retrieval quality.
    // The preset embeds the bare text, which still produces valid
    // embeddings — quality may improve with caller-side instruction
    // prefixing once an `instruct_template` option lands.
    EmbeddingPreset {
        aliases: &["Qwen3Embedding06B", "Qwen/Qwen3-Embedding-0.6B"],
        hf_repo: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &["onnx/model.onnx_data"],
        dimensions: 1024,
        pooling: PoolingKind::LastToken,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: false,
        output_name: None,
    },
];

/// Look up a preset by alias or HF repo ID. Returns `None` for unknown keys
/// so callers can fall back to pass-through (user-supplied options).
pub(super) fn lookup(key: &str) -> Option<&'static EmbeddingPreset> {
    PRESETS.iter().find(|p| p.aliases.contains(&key))
}

// ---------------------------------------------------------------------------
// Sparse (learned term-weight) presets
// ---------------------------------------------------------------------------

/// Post-processing recipe a sparse model's raw output requires.
///
/// SPLADE and BGE-M3 sparse compute term weights differently and are *not*
/// interchangeable — the method selects which path [`super::embed_sparse`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SparseMethod {
    /// SPLADE family: MLM logits over the full vocab. Apply `log(1 + ReLU(x))`,
    /// mask, then max-pool over the sequence. Term ids are vocabulary indices
    /// (the model performs term expansion).
    Mlm,
    /// BGE-M3 lexical head: one ReLU weight per token. Key each position by its
    /// input token id and take the max over duplicate ids (no term expansion).
    Lexical,
}

/// A canonical sparse-model configuration that matches one or more aliases.
#[derive(Debug, Clone)]
pub(super) struct SparsePreset {
    /// Strings that resolve to this preset (HF repo IDs and short aliases).
    pub aliases: &'static [&'static str],
    /// HuggingFace repo to download from.
    pub hf_repo: &'static str,
    /// Path to the `.onnx` file within the repo.
    pub onnx_path: &'static str,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: &'static str,
    /// Extra files to download alongside the `.onnx` graph (external-data sidecars).
    pub additional_files: &'static [&'static str],
    /// Which post-processing recipe the raw output needs.
    pub method: SparseMethod,
    /// Name of the output tensor to read, if known. `None` falls back to
    /// [`output_index`](SparsePreset::output_index).
    pub output_name: Option<&'static str>,
    /// Declared output index to read when `output_name` is `None` (for
    /// multi-output graphs like BGE-M3 where sparse is output #1).
    pub output_index: usize,
    /// Size of the term space the produced ids index into.
    pub vocab_size: u32,
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
}

const SPARSE_PRESETS: &[SparsePreset] = &[
    // ---- SPLADE++ (en v1) — first-party, in-graph MLM logits --------------
    // Single-output graph: the only output is the vocab logits tensor, so
    // selecting by index 0 is robust regardless of its exported name.
    SparsePreset {
        aliases: &["SpladePPenV1", "prithivida/Splade_PP_en_v1"],
        hf_repo: "prithivida/Splade_PP_en_v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        method: SparseMethod::Mlm,
        output_name: None,
        output_index: 0,
        // bert-base-uncased vocabulary (the SPLADE++ backbone).
        vocab_size: 30522,
        max_seq_len: 512,
    },
    // ---- SPLADE++ (en v2) — improved first-party SPLADE++, in-graph MLM ----
    // Same architecture/recipe as v1 (BERT-base MLM head) with better training;
    // the repo ships an `onnx/` export. Single vocab-logits output → index 0.
    SparsePreset {
        aliases: &["SpladePPenV2", "prithivida/Splade_PP_en_v2"],
        hf_repo: "prithivida/Splade_PP_en_v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        method: SparseMethod::Mlm,
        output_name: None,
        output_index: 0,
        vocab_size: 30522,
        max_seq_len: 512,
    },
    // ---- BGE-M3 sparse head — community multi-output export ----------------
    // `aapot/bge-m3-onnx` emits [dense, sparse, colbert]; sparse is output #1.
    // Validated end-to-end against the live repo (lexical terms produced; the
    // 2.1 GB external-data sidecar + constant tensor load correctly).
    SparsePreset {
        // The bare repo id resolves here for the sparse task; the multi-vector
        // table maps the same repo to the ColBERT head (different table, so no
        // collision). `#sparse` is an explicit alias for either path.
        aliases: &[
            "BGEM3Sparse",
            "aapot/bge-m3-onnx",
            "aapot/bge-m3-onnx#sparse",
        ],
        hf_repo: "aapot/bge-m3-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // BGE-M3 weights exceed the 2 GB protobuf limit, so the graph
        // (`model.onnx`) references an external-data sidecar plus a constant
        // tensor; both must land alongside it for ORT to load the model.
        additional_files: &["model.onnx.data", "Constant_685_attr__value"],
        method: SparseMethod::Lexical,
        output_name: None,
        output_index: 1,
        // XLM-RoBERTa vocabulary (the BGE-M3 backbone); lexical ids are input
        // token ids, which live in this space.
        vocab_size: 250002,
        max_seq_len: 512,
    },
];

/// Look up a sparse preset by alias or HF repo id.
pub(super) fn lookup_sparse(key: &str) -> Option<&'static SparsePreset> {
    SPARSE_PRESETS.iter().find(|p| p.aliases.contains(&key))
}

// ---------------------------------------------------------------------------
// Multi-vector (ColBERT / late-interaction) presets
// ---------------------------------------------------------------------------

/// A canonical multi-vector-model configuration that matches one or more aliases.
#[derive(Debug, Clone)]
pub(super) struct MultiVectorPreset {
    /// Strings that resolve to this preset (HF repo IDs and short aliases).
    pub aliases: &'static [&'static str],
    /// HuggingFace repo to download from.
    pub hf_repo: &'static str,
    /// Path to the `.onnx` file within the repo.
    pub onnx_path: &'static str,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: &'static str,
    /// Extra files to download alongside the `.onnx` graph (external-data sidecars).
    pub additional_files: &'static [&'static str],
    /// Name of the per-token output tensor, if known. `None` falls back to
    /// [`output_index`](MultiVectorPreset::output_index).
    pub output_name: Option<&'static str>,
    /// Declared output index to read when `output_name` is `None` (ColBERT is
    /// output #2 for BGE-M3's multi-output graph).
    pub output_index: usize,
    /// Dimensionality of each per-token vector.
    pub dimensions: u32,
    /// Whether to L2-normalize each per-token vector (cosine MaxSim).
    pub normalize: bool,
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
}

const MULTI_VECTOR_PRESETS: &[MultiVectorPreset] = &[
    // ---- answerai-colbert-small-v1 — first-party, in-graph 384->96 ---------
    // `vespa_colbert.onnx` applies the ColBERT projection in-graph and emits a
    // single [batch, seq, 96] per-token output (index 0).
    MultiVectorPreset {
        aliases: &[
            "AnswerAIColbertSmallV1",
            "answerdotai/answerai-colbert-small-v1",
        ],
        hf_repo: "answerdotai/answerai-colbert-small-v1",
        onnx_path: "vespa_colbert.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        output_name: None,
        output_index: 0,
        dimensions: 96,
        normalize: true,
        max_seq_len: 512,
    },
    // ---- GTE-ModernColBERT-v1 — top open ColBERT, Apache-2.0 ---------------
    // BEIR-leading late-interaction model (LightOn / PyLate, ModernBERT base,
    // 8192-token capable). Ships `model.onnx` with the 768->128 ColBERT
    // projection folded into the graph — validated end-to-end against the live
    // repo (per-token dim 128, MaxSim ranks relevant > irrelevant).
    MultiVectorPreset {
        aliases: &["GTEModernColbertV1", "lightonai/GTE-ModernColBERT-v1"],
        hf_repo: "lightonai/GTE-ModernColBERT-v1",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        output_name: None,
        output_index: 0,
        dimensions: 128,
        normalize: true,
        // ModernBERT supports long context; default cap is conservative and
        // overridable via `max_seq_len`.
        max_seq_len: 512,
    },
    // ---- mxbai-edge-colbert-v0 (32M) — small, storage-efficient -----------
    // ModernBERT/Ettin-based ColBERT with a 64-d projection — a quarter the
    // per-token storage of 256-d ColBERTv2. The `model.onnx` folds the
    // projection in (validated end-to-end), despite the separate `1_Dense`/
    // `2_Dense` modules in the repo.
    MultiVectorPreset {
        aliases: &[
            "MxbaiEdgeColbertV0_32m",
            "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        ],
        hf_repo: "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        output_name: None,
        output_index: 0,
        dimensions: 64,
        normalize: true,
        max_seq_len: 512,
    },
    // ---- mxbai-edge-colbert-v0 (17M) — smallest strong ColBERT ------------
    // 17M params, 48-d per-token vectors; beats 130M ColBERTv2 at ~1/8 the
    // size. The smallest-best late-interaction option for edge/CPU and tiny
    // MaxSim indexes. Verify ONNX output dim (see note above).
    MultiVectorPreset {
        aliases: &[
            "MxbaiEdgeColbertV0_17m",
            "mixedbread-ai/mxbai-edge-colbert-v0-17m",
        ],
        hf_repo: "mixedbread-ai/mxbai-edge-colbert-v0-17m",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        output_name: None,
        output_index: 0,
        dimensions: 48,
        normalize: true,
        max_seq_len: 512,
    },
    // ---- BGE-M3 ColBERT head — community multi-output export ---------------
    // `aapot/bge-m3-onnx` emits [dense, sparse, colbert]; colbert is output #2,
    // dim 1024. Validated end-to-end against the live repo (MaxSim ranks
    // relevant > irrelevant).
    MultiVectorPreset {
        // The bare repo id resolves here for the multi-vector task; the sparse
        // table maps the same repo to the sparse head. `#colbert` is explicit.
        aliases: &[
            "BGEM3Colbert",
            "aapot/bge-m3-onnx",
            "aapot/bge-m3-onnx#colbert",
        ],
        hf_repo: "aapot/bge-m3-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // See the sparse BGE-M3 preset: the graph references a 2.1 GB
        // external-data sidecar plus a constant tensor.
        additional_files: &["model.onnx.data", "Constant_685_attr__value"],
        output_name: None,
        output_index: 2,
        dimensions: 1024,
        normalize: true,
        max_seq_len: 512,
    },
];

/// Look up a multi-vector preset by alias or HF repo id.
pub(super) fn lookup_multi_vector(key: &str) -> Option<&'static MultiVectorPreset> {
    MULTI_VECTOR_PRESETS
        .iter()
        .find(|p| p.aliases.contains(&key))
}

// ---------------------------------------------------------------------------
// Hybrid (single-pass multi-head) presets
// ---------------------------------------------------------------------------
//
// A hybrid preset declares a multi-output graph plus the per-head recipe for
// each output the export exposes. The hybrid embedder runs the encoder once and
// post-processes every present head from that single pass — the union of the
// per-task dense / sparse / multi-vector presets for the same repo.

/// The dense head of a hybrid graph: output selection plus pooling recipe.
#[derive(Debug, Clone)]
pub(super) struct DenseHead {
    /// Name of the dense output tensor, if known. `None` falls back to
    /// [`output_index`](DenseHead::output_index).
    pub output_name: Option<&'static str>,
    /// Declared output index when `output_name` is `None`.
    pub output_index: usize,
    /// Output embedding dimensionality.
    pub dimensions: u32,
    /// Pooling strategy. Unused when the output is already a pooled
    /// `[batch, dim]` tensor (the 2-D pass-through path); set for completeness.
    pub pooling: PoolingKind,
    /// Whether to L2-normalize each row.
    pub normalize: bool,
}

/// The sparse head of a hybrid graph: output selection plus sparse recipe.
#[derive(Debug, Clone)]
pub(super) struct SparseHead {
    /// Name of the sparse output tensor, if known.
    pub output_name: Option<&'static str>,
    /// Declared output index when `output_name` is `None`.
    pub output_index: usize,
    /// Post-processing recipe (`Mlm` vs `Lexical`).
    pub method: SparseMethod,
    /// Size of the term space the emitted `term_id`s index into, matching the
    /// equivalent per-task [`SparsePreset::vocab_size`]. Needed so a hybrid
    /// handle can answer [`SparseEmbeddingModel::vocab_size`](crate::traits::SparseEmbeddingModel::vocab_size)
    /// when it is resolved through the single-head accessor.
    pub vocab_size: u32,
}

/// The multi-vector head of a hybrid graph: output selection plus per-token recipe.
#[derive(Debug, Clone)]
pub(super) struct MultiVectorHead {
    /// Name of the per-token output tensor, if known.
    pub output_name: Option<&'static str>,
    /// Declared output index when `output_name` is `None`.
    pub output_index: usize,
    /// Dimensionality of each per-token vector.
    pub dimensions: u32,
    /// Whether to L2-normalize each per-token vector.
    pub normalize: bool,
    /// Whether to strip special-token vectors.
    pub drop_special_tokens: bool,
}

/// A multi-output graph plus the recipe for each head it exposes.
///
/// Heads absent from the export are `None`; the hybrid embedder advertises only
/// the present ones via its available-head set.
#[derive(Debug, Clone)]
pub(super) struct HybridPreset {
    /// Strings that resolve to this preset (HF repo IDs and short aliases).
    pub aliases: &'static [&'static str],
    /// HuggingFace repo to download from.
    pub hf_repo: &'static str,
    /// Path to the `.onnx` file within the repo.
    pub onnx_path: &'static str,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: &'static str,
    /// Extra files to download alongside the `.onnx` graph (external-data sidecars).
    pub additional_files: &'static [&'static str],
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
    /// Dense head recipe, if the graph exposes one.
    pub dense: Option<DenseHead>,
    /// Sparse head recipe, if the graph exposes one.
    pub sparse: Option<SparseHead>,
    /// Multi-vector head recipe, if the graph exposes one.
    pub multi_vector: Option<MultiVectorHead>,
}

const HYBRID_PRESETS: &[HybridPreset] = &[
    // ---- BGE-M3 — all three heads, one community multi-output export -------
    // `aapot/bge-m3-onnx` emits [dense_vecs (0), sparse_vecs (1), colbert_vecs
    // (2)] from one XLM-RoBERTa forward pass. This is the union of the per-task
    // BGEM3Dense / BGEM3Sparse / BGEM3Colbert presets, so a hybrid pipeline pays
    // for one weight load and one pass instead of three. The bare repo id and
    // `#hybrid` resolve here only for the embed_hybrid task (separate table, so
    // no collision with the per-task tables that also list the bare id).
    HybridPreset {
        aliases: &[
            "BGEM3Hybrid",
            "aapot/bge-m3-onnx#hybrid",
            "aapot/bge-m3-onnx",
        ],
        hf_repo: "aapot/bge-m3-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // BGE-M3 weights exceed the 2 GB protobuf limit: the graph references an
        // external-data sidecar plus a constant tensor (see the per-task presets).
        additional_files: &["model.onnx.data", "Constant_685_attr__value"],
        max_seq_len: 512,
        dense: Some(DenseHead {
            // `dense_vecs` is already a pooled [batch, 1024] tensor, so the 2-D
            // pass-through path skips pooling; `pooling` is unused here.
            output_name: Some("dense_vecs"),
            output_index: 0,
            dimensions: 1024,
            pooling: PoolingKind::Cls,
            normalize: true,
        }),
        sparse: Some(SparseHead {
            output_name: None,
            output_index: 1,
            method: SparseMethod::Lexical,
            // Matches the per-task `BGEM3Sparse` preset: the XLM-RoBERTa
            // tokenizer vocabulary the lexical head's term ids index into.
            vocab_size: 250002,
        }),
        multi_vector: Some(MultiVectorHead {
            output_name: None,
            output_index: 2,
            dimensions: 1024,
            normalize: true,
            drop_special_tokens: false,
        }),
    },
];

/// Look up a hybrid (single-pass multi-head) preset by alias or HF repo id.
pub(super) fn lookup_hybrid(key: &str) -> Option<&'static HybridPreset> {
    HYBRID_PRESETS.iter().find(|p| p.aliases.contains(&key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every alias must be unique across the table — if two presets claim
    /// the same alias, lookup is order-dependent and surprising.
    #[test]
    fn aliases_are_unique() {
        let mut seen: HashSet<&'static str> = HashSet::new();
        for preset in PRESETS {
            for alias in preset.aliases {
                assert!(
                    seen.insert(alias),
                    "Duplicate alias '{}' across presets",
                    alias
                );
            }
        }
    }

    /// Spot-check: the three popular FastEmbed aliases should resolve.
    #[test]
    fn popular_aliases_resolve() {
        for alias in &[
            "BGESmallENV15",
            "bge-small-en-v1.5",
            "AllMiniLML6V2",
            "all-MiniLM-L6-v2",
            "MultilingualE5Small",
            "NomicEmbedTextV15",
        ] {
            assert!(
                lookup(alias).is_some(),
                "Expected preset for alias '{}'",
                alias
            );
        }
    }

    /// Sparse and multi-vector alias tables must also be collision-free.
    #[test]
    fn sparse_and_multi_vector_aliases_are_unique() {
        let mut seen: HashSet<&'static str> = HashSet::new();
        for alias in SPARSE_PRESETS.iter().flat_map(|p| p.aliases.iter()) {
            assert!(seen.insert(alias), "Duplicate sparse alias '{}'", alias);
        }
        let mut seen: HashSet<&'static str> = HashSet::new();
        for alias in MULTI_VECTOR_PRESETS.iter().flat_map(|p| p.aliases.iter()) {
            assert!(
                seen.insert(alias),
                "Duplicate multi-vector alias '{}'",
                alias
            );
        }
    }

    /// The sparse / multi-vector presets resolve and carry sane shapes.
    #[test]
    fn new_sparse_and_multi_vector_aliases_resolve() {
        // Sparse: SPLADE++ v1/v2 (mlm) and the BGE-M3 lexical head.
        assert!(matches!(
            lookup_sparse("prithivida/Splade_PP_en_v2").map(|p| p.method),
            Some(SparseMethod::Mlm)
        ));
        assert!(matches!(
            lookup_sparse("BGEM3Sparse").map(|p| p.method),
            Some(SparseMethod::Lexical)
        ));

        // The bare `aapot/bge-m3-onnx` repo id resolves to the dense head for the
        // embed task, the sparse head for the sparse task, and the ColBERT head
        // for the multi-vector task — the task disambiguates one repo serving
        // three outputs.
        assert!(matches!(
            lookup_sparse("aapot/bge-m3-onnx").map(|p| p.method),
            Some(SparseMethod::Lexical)
        ));
        assert_eq!(
            lookup_multi_vector("aapot/bge-m3-onnx").map(|p| p.dimensions),
            Some(1024)
        );
        // Dense head: the multi-output export pins the `dense_vecs` output and
        // carries no token_type_ids input.
        for key in ["aapot/bge-m3-onnx", "BGEM3Dense", "aapot/bge-m3-onnx#dense"] {
            let p = lookup(key).unwrap_or_else(|| panic!("expected dense preset for '{key}'"));
            assert_eq!(p.hf_repo, "aapot/bge-m3-onnx", "repo for '{key}'");
            assert_eq!(p.dimensions, 1024, "dim for '{key}'");
            assert_eq!(p.output_name, Some("dense_vecs"), "output pin for '{key}'");
            assert!(!p.token_type_ids, "token_type_ids for '{key}'");
        }
        // The official BAAI dense export stays the default `BGEM3` alias.
        assert_eq!(lookup("BGEM3").map(|p| p.hf_repo), Some("BAAI/bge-m3"));

        // Multi-vector: per-token dims match the documented projections.
        for (alias, dim) in [
            ("lightonai/GTE-ModernColBERT-v1", 128u32),
            ("mixedbread-ai/mxbai-edge-colbert-v0-32m", 64),
            ("mixedbread-ai/mxbai-edge-colbert-v0-17m", 48),
            ("answerdotai/answerai-colbert-small-v1", 96),
        ] {
            let preset = lookup_multi_vector(alias)
                .unwrap_or_else(|| panic!("expected multi-vector preset for '{alias}'"));
            assert_eq!(preset.dimensions, dim, "dim mismatch for '{alias}'");
        }
    }

    /// The hybrid alias table must be collision-free within itself.
    #[test]
    fn hybrid_aliases_are_unique() {
        let mut seen: HashSet<&'static str> = HashSet::new();
        for alias in HYBRID_PRESETS.iter().flat_map(|p| p.aliases.iter()) {
            assert!(seen.insert(alias), "Duplicate hybrid alias '{}'", alias);
        }
    }

    /// `BGEM3Hybrid` (and the bare / `#hybrid` ids) resolve to all three heads
    /// with the documented output selection and recipes — the union of the
    /// per-task BGE-M3 presets.
    #[test]
    fn hybrid_bgem3_resolves_all_three_heads() {
        for key in [
            "BGEM3Hybrid",
            "aapot/bge-m3-onnx#hybrid",
            "aapot/bge-m3-onnx",
        ] {
            let p =
                lookup_hybrid(key).unwrap_or_else(|| panic!("expected hybrid preset for '{key}'"));
            assert_eq!(p.hf_repo, "aapot/bge-m3-onnx", "repo for '{key}'");

            let dense = p.dense.as_ref().expect("dense head");
            assert_eq!(dense.output_name, Some("dense_vecs"));
            assert_eq!(dense.dimensions, 1024);

            let sparse = p.sparse.as_ref().expect("sparse head");
            assert_eq!(sparse.output_index, 1);
            assert_eq!(sparse.method, SparseMethod::Lexical);
            // Must match the per-task `BGEM3Sparse` preset: a hybrid alias
            // resolved through `sparse_embedder` reports this as `vocab_size`,
            // and downstream indexes size their term space from it.
            assert_eq!(
                sparse.vocab_size,
                lookup_sparse("BGEM3Sparse")
                    .expect("sparse preset")
                    .vocab_size,
                "hybrid and per-task sparse presets must agree on vocab size"
            );

            let mv = p.multi_vector.as_ref().expect("multi-vector head");
            assert_eq!(mv.output_index, 2);
            assert_eq!(mv.dimensions, 1024);
        }
    }

    /// A model id with no hybrid preset does not resolve — the caller falls
    /// back to the per-task handles (the "single pass when possible" contract).
    #[test]
    fn hybrid_lookup_misses_single_head_models() {
        assert!(lookup_hybrid("BAAI/bge-m3").is_none());
        assert!(lookup_hybrid("BGESmallENV15").is_none());
    }
}
