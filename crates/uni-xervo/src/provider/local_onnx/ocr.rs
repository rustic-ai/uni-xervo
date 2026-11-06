// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! OCR task for `local/onnx` — CRNN + CTC-style recognition.
//!
//! Targets the PaddleOCR-rec / EasyOCR family of recognition models that
//! ONNX-export with a single tensor output of shape
//! `[batch, time_steps, n_classes]`, where `n_classes` is the vocabulary
//! size plus a CTC blank token. The impl performs CTC greedy decoding:
//!
//! 1. Argmax per time-step → class index sequence.
//! 2. Collapse consecutive duplicates (CTC merge rule).
//! 3. Drop the blank token (class 0 by convention; configurable).
//! 4. Map remaining indices through the character dictionary.
//!
//! This v1 covers single-image whole-text recognition. For two-stage
//! detection + recognition (region proposals + per-region recognize),
//! callers crop their regions upstream and pass one image per region.
//! A future v2 may absorb a DBNet-style detector as a sibling `style`.

use crate::hf::HfModelRepo;
use async_trait::async_trait;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::det;
use super::image::{Normalization, fill_chw, preprocess_batch_hw, preprocess_det};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, new_session_builder,
    parse_execution_providers_option,
};
use crate::traits::{ImageInput, OcrBlock, OcrModel, OcrResult};
use image::{DynamicImage, GenericImageView};

/// Entry point for `LocalOnnxProvider::load` when `spec.task == Ocr`.
pub(super) async fn load_ocr(spec: &ModelAliasSpec) -> Result<Arc<dyn OcrModel>> {
    let model = OnnxOcrModel::load(spec).await?;
    Ok(Arc::new(model) as Arc<dyn OcrModel>)
}

/// Optional text-detection stage configuration (DBNet-style).
///
/// Present only when the OCR alias sets `det_onnx_path`. When absent, OCR runs
/// whole-image recognition (the prior behavior).
struct DetConfig {
    model_id: String,
    onnx_path: String,
    limit_side: u32,
    params: det::DetParams,
    normalization: Normalization,
    input_name: String,
    output_name: Option<String>,
}

struct OcrConfig {
    hf_repo: String,
    onnx_path: String,
    char_dict_path: String,
    image_height: u32,
    image_width: u32,
    normalization: Normalization,
    blank_class: usize,
    output_name: Option<String>,
    detector: Option<DetConfig>,
}

impl OcrConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let opts = &spec.options;
        let get_str = |key: &str| -> Option<String> {
            opts.get(key).and_then(Value::as_str).map(str::to_string)
        };

        let onnx_path = get_str("onnx_path").ok_or_else(|| {
            RuntimeError::Config(format!(
                "OCR model '{}' requires option `onnx_path` (.onnx path within the HF repo)",
                spec.alias
            ))
        })?;
        let char_dict_path = get_str("char_dict_path").ok_or_else(|| {
            RuntimeError::Config(format!(
                "OCR model '{}' requires option `char_dict_path` \
                 (character dictionary file within the HF repo, one char per line)",
                spec.alias
            ))
        })?;
        let image_height = opts
            .get("image_height")
            .and_then(Value::as_u64)
            .unwrap_or(48) as u32;
        let image_width = opts
            .get("image_width")
            .and_then(Value::as_u64)
            .unwrap_or(320) as u32;
        let normalization = match opts
            .get("normalization")
            .and_then(Value::as_str)
            .unwrap_or("imagenet")
        {
            "siglip" => Normalization::SIGLIP,
            "imagenet" => Normalization::IMAGENET,
            other => {
                return Err(RuntimeError::Config(format!(
                    "OCR model '{}' has unknown `normalization` value '{other}'; \
                     expected one of: siglip, imagenet",
                    spec.alias
                )));
            }
        };
        let blank_class = opts.get("blank_class").and_then(Value::as_u64).unwrap_or(0) as usize;
        let output_name = get_str("output_name");

        // Optional detection stage — enabled by `det_onnx_path`.
        let detector = if let Some(onnx_path) = get_str("det_onnx_path") {
            let defaults = det::DetParams::default();
            let get_f32 = |key: &str, default: f32| {
                opts.get(key)
                    .and_then(Value::as_f64)
                    .map_or(default, |v| v as f32)
            };
            let get_u32 = |key: &str, default: u32| {
                opts.get(key)
                    .and_then(Value::as_u64)
                    .map_or(default, |v| v as u32)
            };
            Some(DetConfig {
                model_id: get_str("det_model_id").unwrap_or_else(|| spec.model_id.clone()),
                onnx_path,
                limit_side: get_u32("det_limit_side", 960),
                params: det::DetParams {
                    bin_threshold: get_f32("det_bin_threshold", defaults.bin_threshold),
                    box_score_threshold: get_f32(
                        "det_box_score_threshold",
                        defaults.box_score_threshold,
                    ),
                    unclip_ratio: get_f32("det_unclip_ratio", defaults.unclip_ratio),
                    min_box_size: get_u32("det_min_box_size", defaults.min_box_size),
                },
                // DBNet detectors use ImageNet-style normalization by convention.
                normalization: Normalization::IMAGENET,
                input_name: get_str("det_input_name").unwrap_or_else(|| "x".to_string()),
                output_name: get_str("det_output_name"),
            })
        } else {
            None
        };

        Ok(Self {
            hf_repo: spec.model_id.clone(),
            onnx_path,
            char_dict_path,
            image_height,
            image_width,
            normalization,
            blank_class,
            output_name,
            detector,
        })
    }
}

/// A loaded DBNet-style text detector (the optional first OCR stage).
struct Detector {
    session: Mutex<Session>,
    limit_side: u32,
    params: det::DetParams,
    normalization: Normalization,
    input_name: String,
    output_name: Option<String>,
}

struct OnnxOcrModel {
    session: Mutex<Session>,
    alias: String,
    model_id: String,
    chars: Vec<String>,
    image_height: u32,
    image_width: u32,
    normalization: Normalization,
    blank_class: usize,
    output_name: Option<String>,
    /// When present, `recognize()` runs detect → crop → recognize → order.
    detector: Option<Detector>,
}

impl OnnxOcrModel {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = OcrConfig::resolve(spec)?;
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-ocr", &cfg.hf_repo, &spec.options);
        let (model_path, char_dict_path) = download_ocr_artifacts(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
            &cfg.char_dict_path,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            image_height = cfg.image_height,
            image_width = cfg.image_width,
            "Loading ONNX OCR recognizer"
        );

        let chars = read_char_dict(&char_dict_path)?;
        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        let detector = match &cfg.detector {
            Some(dc) => {
                info!(alias = %spec.alias, det_onnx = %dc.onnx_path, "Loading ONNX text detector");
                let det_path = download_artifact(
                    &spec.alias,
                    &dc.model_id,
                    spec.revision.as_deref(),
                    &cache_dir,
                    &dc.onnx_path,
                )
                .await?;
                let det_session = build_session(&det_path, spec, execution_providers.as_deref())?;
                Some(Detector {
                    session: Mutex::new(det_session),
                    limit_side: dc.limit_side,
                    params: dc.params,
                    normalization: dc.normalization,
                    input_name: dc.input_name.clone(),
                    output_name: dc.output_name.clone(),
                })
            }
            None => None,
        };

        Ok(Self {
            session: Mutex::new(session),
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            chars,
            image_height: cfg.image_height,
            image_width: cfg.image_width,
            normalization: cfg.normalization,
            blank_class: cfg.blank_class,
            output_name: cfg.output_name,
            detector,
        })
    }
}

impl crate::traits::ModelInfo for OnnxOcrModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[async_trait]
impl OcrModel for OnnxOcrModel {
    async fn recognize(&self, images: Vec<ImageInput>) -> Result<Vec<OcrResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if self.detector.is_some() {
            self.recognize_with_detection(&images)
        } else {
            self.recognize_whole_image(&images)
        }
    }
}

impl OnnxOcrModel {
    /// Whole-image recognition: one block per input image (the prior behavior,
    /// used when no detector is configured).
    fn recognize_whole_image(&self, images: &[ImageInput]) -> Result<Vec<OcrResult>> {
        let tensor = preprocess_batch_hw(
            images,
            self.image_height,
            self.image_width,
            self.normalization,
        )?;
        let decoded = self.run_recognition(tensor)?;
        Ok(decoded
            .into_iter()
            .map(|(text, confidence)| OcrResult {
                blocks: vec![OcrBlock {
                    text: text.clone(),
                    // Whole-image normalized box; the caller knows its size.
                    bbox: [0.0, 0.0, 1.0, 1.0],
                    confidence,
                }],
                plain_text: text,
            })
            .collect())
    }

    /// Two-stage path: detect text regions, recognize each crop, order them.
    fn recognize_with_detection(&self, images: &[ImageInput]) -> Result<Vec<OcrResult>> {
        let det = self
            .detector
            .as_ref()
            .expect("recognize_with_detection requires a configured detector");
        let mut results = Vec::with_capacity(images.len());

        for input in images {
            let img = decode_image(input, &self.alias)?;
            let (orig_w, orig_h) = img.dimensions();
            let (tensor, scale_x, scale_y) =
                preprocess_det(&img, det.limit_side, det.normalization);
            let prob = self.run_detector(det, tensor)?;

            // Detector-space boxes mapped back to original-image pixels.
            let mut boxes: Vec<det::DetBox> = det::postprocess(prob.view(), &det.params)
                .into_iter()
                .map(|b| b.scale_to(scale_x, scale_y, orig_w as f32, orig_h as f32))
                .collect();

            if boxes.is_empty() {
                // Blank / no-text page: a normal empty result (the router may
                // escalate on empty), never an error.
                results.push(OcrResult {
                    blocks: Vec::new(),
                    plain_text: String::new(),
                });
                continue;
            }

            // Cluster lines by ~half the mean box height for reading order.
            let mean_h = boxes.iter().map(det::DetBox::height).sum::<f32>() / boxes.len() as f32;
            det::sort_reading_order(&mut boxes, mean_h * 0.5);

            let mut items = Vec::with_capacity(boxes.len());
            for b in &boxes {
                let bw = (b.x1 - b.x0).floor() as u32;
                let bh = (b.y1 - b.y0).floor() as u32;
                if bw == 0 || bh == 0 {
                    continue; // degenerate region — never crop a 0-size box
                }
                let crop = img.crop_imm(b.x0 as u32, b.y0 as u32, bw, bh);
                let tensor = fill_chw(
                    &crop,
                    self.image_height,
                    self.image_width,
                    self.normalization,
                );
                let (text, confidence) = self.run_recognition(tensor)?.pop().unwrap_or_default();
                items.push((*b, text, confidence));
            }
            results.push(assemble_ocr_result(items));
        }
        Ok(results)
    }

    /// Run the recognition session on a preprocessed batch, CTC-decoding each
    /// row into `(text, mean_confidence)`.
    fn run_recognition(&self, tensor: ndarray::Array4<f32>) -> Result<Vec<(String, f32)>> {
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

        let pixel_tensor = ort::value::Tensor::from_array(tensor.into_dyn())
            .map_err(|e| map_err(e, "pixel tensor"))?;
        let inputs: Vec<(String, ort::value::DynTensor)> =
            vec![("x".to_string(), pixel_tensor.upcast())];

        let chosen_output = match self.output_name.as_deref() {
            Some(name) => name.to_string(),
            None => session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: "ONNX session has no outputs".to_string(),
                })?,
        };

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "ONNX inference"))?;
        let view = outputs
            .get(&chosen_output)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Missing output tensor '{chosen_output}'"),
            })?
            .try_extract_array::<f32>()
            .map_err(|e| map_err(e, "extract output"))?;
        let arr3 = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("OCR output expected rank 3 [batch, time, classes]: {e}"),
            })?
            .to_owned();

        let (batch, _time, n_classes) = (arr3.shape()[0], arr3.shape()[1], arr3.shape()[2]);
        if self.blank_class >= n_classes {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!(
                    "blank_class={} out of range for {n_classes} classes",
                    self.blank_class
                ),
            });
        }

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            out.push(self.decode_ctc(arr3.slice(ndarray::s![b, .., ..])));
        }
        Ok(out)
    }

    /// CTC greedy-decode one `[time, classes]` slice into `(text, mean_conf)`.
    fn decode_ctc(&self, slice: ndarray::ArrayView2<f32>) -> (String, f32) {
        let mut prev_class: Option<usize> = None;
        let mut text = String::new();
        let mut total_conf = 0.0f64;
        let mut emitted = 0usize;
        for t in 0..slice.shape()[0] {
            let row = slice.row(t);
            let class_id = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if class_id == self.blank_class {
                prev_class = None;
                continue;
            }
            if prev_class == Some(class_id) {
                continue;
            }
            prev_class = Some(class_id);

            // Index 0 is blank by convention; chars are typically 1-indexed
            // against the dictionary file. Offset by 1.
            let char_idx = class_id.saturating_sub(self.blank_class + 1);
            if let Some(c) = self.chars.get(char_idx) {
                text.push_str(c);
                total_conf += class_confidence(&row.to_vec(), class_id) as f64;
                emitted += 1;
            }
        }
        let avg_conf = if emitted > 0 {
            (total_conf / emitted as f64) as f32
        } else {
            0.0
        };
        (text, avg_conf)
    }

    /// Run the detector session and return the `[H, W]` probability map.
    ///
    /// Accepts the common DBNet output ranks ([1,1,H,W], [1,H,W], [H,W]) and
    /// squeezes to 2-D; any other rank is a typed error, not a panic.
    fn run_detector(
        &self,
        det: &Detector,
        tensor: ndarray::Array4<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        let mut session = det
            .session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Detector session lock poisoned: {e}"),
            })?;
        let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let input = ort::value::Tensor::from_array(tensor.into_dyn())
            .map_err(|e| map_err(e, "detector input tensor"))?;
        let inputs: Vec<(String, ort::value::DynTensor)> =
            vec![(det.input_name.clone(), input.upcast())];

        let chosen_output = match det.output_name.as_deref() {
            Some(name) => name.to_string(),
            None => session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: "Detector session has no outputs".to_string(),
                })?,
        };

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "detector inference"))?;
        let view = outputs
            .get(&chosen_output)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Missing detector output '{chosen_output}'"),
            })?
            .try_extract_array::<f32>()
            .map_err(|e| map_err(e, "extract detector output"))?;

        let prob = match view.ndim() {
            4 => view
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| reshape_err(&self.alias, &e))?
                .slice(ndarray::s![0, 0, .., ..])
                .to_owned(),
            3 => view
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| reshape_err(&self.alias, &e))?
                .slice(ndarray::s![0, .., ..])
                .to_owned(),
            2 => view
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| reshape_err(&self.alias, &e))?
                .to_owned(),
            n => {
                return Err(RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("detector output rank {n} unsupported (expected 2-4)"),
                });
            }
        };
        Ok(prob)
    }
}

/// Read a character dictionary file: one entry per line, in class order.
fn read_char_dict(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        RuntimeError::Config(format!(
            "Failed to read OCR char dictionary '{}': {e}",
            path.display()
        ))
    })?;
    Ok(content.lines().map(str::to_string).collect())
}

/// Confidence for the chosen class, robust to whether the model's output row is
/// raw logits or an already-normalized probability distribution.
///
/// CRNN/CTC ONNX exports differ: some emit logits (which need a softmax), while
/// others — notably PP-OCR — emit a softmax distribution already. Re-softmaxing
/// a probability row would collapse every confidence toward `1/n_classes`
/// (e.g. ~0.002 for PP-OCR's 438 classes), so when a row already looks like a
/// distribution (non-negative, sums to ~1) we read the probability directly.
fn class_confidence(row: &[f32], idx: usize) -> f32 {
    let sum: f32 = row.iter().copied().sum();
    let looks_normalized =
        (sum - 1.0).abs() < 0.05 && row.iter().all(|&v| (-0.01..=1.01).contains(&v));
    if looks_normalized {
        row.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0)
    } else {
        softmax_at(row, idx)
    }
}

/// Stable softmax probability for class `idx` over `logits`.
fn softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.get(idx).copied().unwrap_or(0.0) / sum
    } else {
        0.0
    }
}

/// Build an [`OcrResult`] from detected boxes paired with recognized text.
///
/// Boxes are expected already in reading order. Empty-text regions are dropped;
/// `plain_text` is the kept texts joined by newlines.
fn assemble_ocr_result(items: Vec<(det::DetBox, String, f32)>) -> OcrResult {
    let mut blocks = Vec::new();
    let mut texts = Vec::new();
    for (b, text, confidence) in items {
        if text.is_empty() {
            continue;
        }
        texts.push(text.clone());
        blocks.push(OcrBlock {
            text,
            bbox: [b.x0, b.y0, b.x1, b.y1],
            confidence,
        });
    }
    OcrResult {
        plain_text: texts.join("\n"),
        blocks,
    }
}

/// Decode an [`ImageInput`] to a [`DynamicImage`]; URLs are rejected (not fetched here).
fn decode_image(input: &ImageInput, alias: &str) -> Result<DynamicImage> {
    match input {
        ImageInput::Bytes { data, .. } => {
            image::load_from_memory(data).map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: alias.to_string(),
                cause: format!("Image decode failed: {e}"),
            })
        }
        ImageInput::Url(url) => Err(RuntimeError::Config(format!(
            "local/onnx OCR does not fetch URLs; fetch '{url}' upstream and pass \
             the bytes via ImageInput::Bytes"
        ))),
    }
}

/// Wrap a detector output reshape failure as a typed error.
fn reshape_err(alias: &str, e: &ndarray::ShapeError) -> RuntimeError {
    RuntimeError::OnnxInvocationFailure {
        alias: alias.to_string(),
        cause: format!("detector output reshape failed: {e}"),
    }
}

/// Download a single artifact file from a HF repo into `cache_dir`.
async fn download_artifact(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    path: &str,
) -> Result<PathBuf> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        }
    })?;
    api_repo
        .get(path)
        .await
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: format!("Could not download '{path}': {e}"),
        })
}

async fn download_ocr_artifacts(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
    char_dict_path: &str,
) -> Result<(PathBuf, PathBuf)> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        }
    })?;
    let onnx_file =
        api_repo
            .get(onnx_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download ONNX '{onnx_path}': {e}"),
            })?;
    let dict_file =
        api_repo
            .get(char_dict_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download char dict '{char_dict_path}': {e}"),
            })?;
    Ok((onnx_file, dict_file))
}

fn build_session(
    path: &Path,
    spec: &ModelAliasSpec,
    execution_providers: Option<&[OnnxExecutionProvider]>,
) -> Result<Session> {
    let builder = new_session_builder()?.map_err(|e| RuntimeError::OnnxLoadFailure {
        alias: spec.alias.clone(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    })?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;
    let dispatch = build_execution_providers(execution_providers, &spec.alias, "local/onnx")?;
    builder =
        builder
            .with_execution_providers(dispatch)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
    builder
        .commit_from_file(path)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_at_normalizes_to_unit_total() {
        let logits = vec![1.0, 2.0, 3.0];
        let s0 = softmax_at(&logits, 0);
        let s1 = softmax_at(&logits, 1);
        let s2 = softmax_at(&logits, 2);
        assert!((s0 + s1 + s2 - 1.0).abs() < 1e-5);
        assert!(s2 > s1 && s1 > s0);
    }

    #[test]
    fn class_confidence_reads_probability_rows_directly() {
        // An already-normalized distribution is read directly (no re-softmax).
        let probs = vec![0.02, 0.95, 0.03];
        assert!((class_confidence(&probs, 1) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn class_confidence_softmaxes_logit_rows() {
        // Raw logits (don't sum to ~1) fall back to softmax.
        let logits = vec![1.0, 2.0, 3.0];
        assert!((class_confidence(&logits, 2) - softmax_at(&logits, 2)).abs() < 1e-6);
        assert!(class_confidence(&logits, 2) > 0.6);
    }

    #[test]
    fn read_char_dict_parses_one_per_line() {
        let dir = std::env::temp_dir();
        let path = dir.join("uni_xervo_test_char_dict.txt");
        std::fs::write(&path, "a\nb\nc\n").unwrap();
        let chars = read_char_dict(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(chars, vec!["a", "b", "c"]);
    }

    fn ocr_spec(options: serde_json::Value) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: "ocr/test".to_string(),
            task: crate::api::ModelTask::Ocr,
            provider_id: "local/onnx".to_string(),
            model_id: "test/repo".to_string(),
            revision: None,
            warmup: crate::api::WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options,
        }
    }

    #[test]
    fn config_without_det_has_no_detector() {
        let spec = ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt"
        }));
        let cfg = OcrConfig::resolve(&spec).unwrap();
        assert!(cfg.detector.is_none(), "no det options -> whole-image path");
    }

    #[test]
    fn config_with_det_uses_defaults() {
        let spec = ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt",
            "det_onnx_path": "det.onnx"
        }));
        let det = OcrConfig::resolve(&spec)
            .unwrap()
            .detector
            .expect("detector");
        assert_eq!(det.onnx_path, "det.onnx");
        assert_eq!(det.model_id, "test/repo"); // defaults to the rec repo
        assert_eq!(det.limit_side, 960);
        assert!((det.params.bin_threshold - 0.3).abs() < 1e-6);
        assert!((det.params.unclip_ratio - 1.5).abs() < 1e-6);
        assert_eq!(det.params.min_box_size, 3);
        assert_eq!(det.input_name, "x");
    }

    #[test]
    fn config_with_det_custom_values() {
        let spec = ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt",
            "det_onnx_path": "det.onnx",
            "det_model_id": "other/det-repo",
            "det_limit_side": 1280,
            "det_bin_threshold": 0.25,
            "det_unclip_ratio": 1.8,
            "det_min_box_size": 5,
            "det_input_name": "image"
        }));
        let det = OcrConfig::resolve(&spec).unwrap().detector.unwrap();
        assert_eq!(det.model_id, "other/det-repo");
        assert_eq!(det.limit_side, 1280);
        assert!((det.params.bin_threshold - 0.25).abs() < 1e-6);
        assert!((det.params.unclip_ratio - 1.8).abs() < 1e-6);
        assert_eq!(det.params.min_box_size, 5);
        assert_eq!(det.input_name, "image");
    }

    #[test]
    fn assemble_drops_empty_text_and_preserves_order() {
        let mk = |x0: f32, text: &str| {
            (
                det::DetBox {
                    x0,
                    y0: 0.0,
                    x1: x0 + 10.0,
                    y1: 10.0,
                    score: 0.9,
                },
                text.to_string(),
                0.9f32,
            )
        };
        let items = vec![mk(0.0, "hello"), mk(20.0, ""), mk(40.0, "world")];
        let r = assemble_ocr_result(items);
        assert_eq!(r.blocks.len(), 2, "empty-text region dropped");
        assert_eq!(r.blocks[0].text, "hello");
        assert_eq!(r.blocks[1].text, "world");
        assert_eq!(r.blocks[0].bbox, [0.0, 0.0, 10.0, 10.0]);
        assert_eq!(r.plain_text, "hello\nworld");
    }

    #[test]
    fn assemble_blank_input_is_empty() {
        let r = assemble_ocr_result(Vec::new());
        assert!(r.blocks.is_empty());
        assert_eq!(r.plain_text, "");
    }
}
