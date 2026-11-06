// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared ONNX encoder plumbing for the sparse and multi-vector embedding tasks.
//!
//! Both [`super::embed_sparse`] and [`super::embed_multi_vector`] run a text
//! encoder and then post-process *raw* per-token outputs (skipping the dense
//! embedder's pooling step). This module factors out the common load + run
//! machinery: HF download, session build, tokenization, decoder-extra inputs,
//! and a single forward pass that returns *all* floating-point outputs by name
//! so the caller can select the head it wants (SPLADE logits, BGE-M3 sparse,
//! ColBERT vectors). It reuses the helpers in [`super::embedding`].

use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::{Array2, ArrayD};
use ort::session::Session;
use tracing::info;

use super::decoder_inputs::{InputRole, InputSchema};

/// Tokenizer output for one batch: `(input_ids, attention_mask, token_type_ids?)`.
type TokenizedBatch = (Array2<i64>, Array2<i64>, Option<Array2<i64>>);
use super::decoder_inputs::build_input_schema;
use super::embedding::{build_session, download_model_files};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    EpProbe, OnnxExecutionProvider, build_execution_providers, effective_ep_list,
    parse_execution_providers_option, replace_session, verify_execution_providers,
};

/// File layout and tokenization shape needed to load an [`EncoderModel`].
pub(super) struct EncoderSpec {
    /// HF repo to download from (already resolved from preset or `model_id`).
    pub hf_repo: String,
    /// Path to the `.onnx` graph within the repo.
    pub onnx_path: String,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: String,
    /// Extra files (external-data sidecars) to fetch alongside the graph.
    pub additional_files: Vec<String>,
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
}

/// A loaded ONNX text encoder that returns raw per-token outputs.
///
/// Input tensor names are discovered from the session at load time rather than
/// hardcoded, so non-standard exports — e.g. SPLADE++'s `input_mask` /
/// `segment_ids` instead of `attention_mask` / `token_type_ids` — drive correctly.
pub(super) struct EncoderModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    alias: String,
    max_seq_len: usize,
    /// Declared name of the token-ids input (usually `input_ids`).
    input_ids_name: String,
    /// Declared name of the attention-mask input (`attention_mask` or `input_mask`).
    mask_name: String,
    /// Declared name of the token-type input, if the model has one
    /// (`token_type_ids` or `segment_ids`); `None` means don't feed it.
    token_type_name: Option<String>,
    /// Declared name of the `position_ids` input, if the model has one.
    position_ids_name: Option<String>,
    past_kv_schemas: Vec<InputSchema>,
}

#[async_trait::async_trait]
impl EpProbe for EncoderModel {
    async fn probe(&self) -> Result<()> {
        self.run(&["probe"]).map(drop)
    }
}

/// Result of one encoder forward pass.
pub(super) struct EncoderRun {
    /// Padded input token ids `[batch, seq]`. Needed by the lexical sparse head
    /// to key per-position weights by token id.
    pub input_ids: Array2<i64>,
    /// Attention mask `[batch, seq]` (1 = real token, 0 = padding).
    pub attention_mask: Array2<i64>,
    /// All floating-point session outputs, by tensor name.
    pub outputs: HashMap<String, ArrayD<f32>>,
    /// Declared output tensor names, in graph order — used for index selection.
    pub output_order: Vec<String>,
}

impl EncoderRun {
    /// Select an output tensor by name (preferred) or declared index.
    ///
    /// When `name` is `Some`, it is looked up directly. Otherwise `index`
    /// selects positionally from the declared output order (default 0). This
    /// supports multi-output graphs (BGE-M3 emits dense / sparse / ColBERT)
    /// whose tensor names may be non-descriptive.
    ///
    /// # Errors
    /// Returns an error if the requested name/index is absent or not a
    /// floating-point tensor.
    pub fn select(
        &self,
        alias: &str,
        name: Option<&str>,
        index: Option<usize>,
    ) -> Result<&ArrayD<f32>> {
        let resolved = match name {
            Some(n) => n.to_string(),
            None => {
                let idx = index.unwrap_or(0);
                self.output_order.get(idx).cloned().ok_or_else(|| {
                    RuntimeError::OnnxInvocationFailure {
                        alias: alias.to_string(),
                        cause: format!(
                            "output_index {idx} out of range (model has {} float outputs)",
                            self.output_order.len()
                        ),
                    }
                })?
            }
        };
        self.outputs
            .get(&resolved)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("missing float output tensor '{resolved}'"),
            })
    }
}

impl EncoderModel {
    /// The catalog alias this encoder was loaded for.
    pub(super) fn alias(&self) -> &str {
        &self.alias
    }

    /// Load an ONNX text encoder described by `spec` and `enc`.
    ///
    /// # Errors
    /// Returns an error if execution-provider validation, download, tokenizer
    /// parsing, or session construction fails.
    pub(super) async fn load(spec: &ModelAliasSpec, enc: EncoderSpec) -> Result<Self> {
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;

        // Validate the requested EP list before any I/O so misconfigurations
        // fail fast with a precise error (mirrors the dense embedder).
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-embed", &enc.hf_repo, &spec.options);
        let (model_path, tokenizer_path) = download_model_files(
            &spec.alias,
            &enc.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &enc.onnx_path,
            &enc.tokenizer_path,
            &enc.additional_files,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            hf_repo = %enc.hf_repo,
            model_path = %model_path.display(),
            "Loading ONNX encoder (raw per-token outputs)"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        // Discover the actual input tensor names from the session so non-standard
        // exports (renamed mask / token-type inputs) feed correctly, and so we
        // only feed token-type / position-id tensors the model actually declares.
        let schema = build_input_schema(&session, &spec.alias)?;
        let name_for = |role: InputRole| {
            schema
                .iter()
                .find(|s| s.role == role)
                .map(|s| s.name.clone())
        };
        let input_ids_name =
            name_for(InputRole::InputIds).unwrap_or_else(|| "input_ids".to_string());
        let mask_name =
            name_for(InputRole::AttentionMask).unwrap_or_else(|| "attention_mask".to_string());
        let token_type_name = name_for(InputRole::TokenTypeIds);
        let position_ids_name = name_for(InputRole::PositionIds);
        let past_kv_schemas: Vec<InputSchema> = schema
            .iter()
            .filter(|s| s.role == InputRole::PastKeyValue)
            .cloned()
            .collect();

        let model = Self {
            session: Mutex::new(session),
            tokenizer,
            alias: spec.alias.clone(),
            max_seq_len: enc.max_seq_len,
            input_ids_name,
            mask_name,
            token_type_name,
            position_ids_name,
            past_kv_schemas,
        };
        verify_execution_providers(
            &model,
            effective_ep_list(execution_providers.as_deref()),
            &spec.alias,
            &model_path,
            |m| {
                replace_session(
                    &m.session,
                    build_session(&model_path, spec, Some(&[OnnxExecutionProvider::Cpu]))?,
                    &spec.alias,
                )
            },
        )
        .await?;
        Ok(model)
    }

    /// Encode a batch of texts into padded `[batch, seq]` i64 tensors.
    fn tokenize_batch(&self, texts: &[&str]) -> Result<TokenizedBatch> {
        let batch_size = texts.len();

        let encodings: Vec<tokenizers::Encoding> = texts
            .iter()
            .map(|text| {
                self.tokenizer.encode(*text, true).map_err(|e| {
                    RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Tokenization failed: {e}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, padded_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, padded_len));
        let mut token_type_ids = if self.token_type_name.is_some() {
            Some(Array2::<i64>::zeros((batch_size, padded_len)))
        } else {
            None
        };

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let seq_len = ids.len().min(self.max_seq_len);

            for j in 0..seq_len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                if let Some(tt) = token_type_ids.as_mut() {
                    tt[[i, j]] = types[j] as i64;
                }
            }
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }

    /// Run one forward pass and return every floating-point output by name.
    ///
    /// # Errors
    /// Returns an error if tokenization, tensor construction, or inference fails.
    pub(super) fn run(&self, texts: &[&str]) -> Result<EncoderRun> {
        let (input_ids, attention_mask, token_type_ids) = self.tokenize_batch(texts)?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Session lock poisoned: {e}"),
            })?;

        let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids.clone().into_dyn())
            .map_err(|e| map_err(e, "input_ids tensor"))?;
        let attention_mask_tensor =
            ort::value::Tensor::from_array(attention_mask.clone().into_dyn())
                .map_err(|e| map_err(e, "attention_mask tensor"))?;

        let mut inputs: Vec<(String, ort::value::DynTensor)> = vec![
            (self.input_ids_name.clone(), input_ids_tensor.upcast()),
            (self.mask_name.clone(), attention_mask_tensor.upcast()),
        ];

        if let (Some(tt), Some(name)) = (token_type_ids, self.token_type_name.as_ref()) {
            let tt_tensor = ort::value::Tensor::from_array(tt.into_dyn())
                .map_err(|e| map_err(e, "token_type_ids tensor"))?;
            inputs.push((name.clone(), tt_tensor.upcast()));
        }

        if let Some(name) = self.position_ids_name.as_ref() {
            let mut position_ids = Array2::<i64>::zeros((batch_size, seq_len));
            for b in 0..batch_size {
                for s in 0..seq_len {
                    position_ids[[b, s]] = s as i64;
                }
            }
            let pos_tensor = ort::value::Tensor::from_array(position_ids.into_dyn())
                .map_err(|e| map_err(e, "position_ids tensor"))?;
            inputs.push((name.clone(), pos_tensor.upcast()));
        }

        for schema in &self.past_kv_schemas {
            let kv = super::decoder_inputs::build_empty_past_kv(schema, batch_size, &self.alias)?;
            inputs.push((schema.name.clone(), kv));
        }

        let declared: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "ONNX inference"))?;

        // Extract every floating-point output as an owned array. Non-float
        // outputs (rare for encoders) are skipped rather than failing the run.
        let mut output_map: HashMap<String, ArrayD<f32>> = HashMap::new();
        let mut output_order: Vec<String> = Vec::new();
        for name in &declared {
            if let Some(value) = outputs.get(name)
                && let Ok(view) = value.try_extract_array::<f32>()
            {
                output_map.insert(name.clone(), view.to_owned());
                output_order.push(name.clone());
            }
        }

        if output_map.is_empty() {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: "ONNX session produced no floating-point outputs".to_string(),
            });
        }

        Ok(EncoderRun {
            input_ids,
            attention_mask,
            outputs: output_map,
            output_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn run_with(outputs: &[(&str, Vec<f32>)]) -> EncoderRun {
        let mut map: HashMap<String, ArrayD<f32>> = HashMap::new();
        let mut order = Vec::new();
        for (name, data) in outputs {
            map.insert(
                (*name).to_string(),
                ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data.clone()).unwrap(),
            );
            order.push((*name).to_string());
        }
        EncoderRun {
            input_ids: Array2::<i64>::zeros((1, 1)),
            attention_mask: Array2::<i64>::ones((1, 1)),
            outputs: map,
            output_order: order,
        }
    }

    #[test]
    fn select_by_name_prefers_name_over_index() {
        let run = run_with(&[("dense", vec![1.0]), ("sparse", vec![2.0])]);
        let got = run.select("t", Some("sparse"), Some(0)).unwrap();
        assert_eq!(got.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn select_by_index_falls_back_when_no_name() {
        let run = run_with(&[("dense", vec![1.0]), ("sparse", vec![2.0])]);
        let got = run.select("t", None, Some(1)).unwrap();
        assert_eq!(got.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn select_defaults_to_first_output() {
        let run = run_with(&[("dense", vec![1.0]), ("sparse", vec![2.0])]);
        let got = run.select("t", None, None).unwrap();
        assert_eq!(got.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn select_errors_on_out_of_range_index() {
        let run = run_with(&[("dense", vec![1.0])]);
        assert!(run.select("t", None, Some(5)).is_err());
    }

    #[test]
    fn select_errors_on_missing_name() {
        let run = run_with(&[("dense", vec![1.0])]);
        assert!(run.select("t", Some("colbert"), None).is_err());
    }
}
