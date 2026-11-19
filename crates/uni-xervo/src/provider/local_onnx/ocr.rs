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
use super::image::{Normalization, aspect_width, pack_text_lines, preprocess_det};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    BuiltSession, SessionConfig, build_session, common_execution_providers, ep_ids,
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
    use_space_char: Option<bool>,
    max_image_width: u32,
    output_name: Option<String>,
    detector: Option<DetConfig>,
}

/// Default cap on a text line's resized width (`max_image_width`).
const DEFAULT_MAX_IMAGE_WIDTH: u32 = 3200;

/// Text lines recognized per session run, grouped by similar aspect ratio so
/// little of each batch is padding. PaddleOCR's `rec_batch_num` default.
const REC_BATCH_SIZE: usize = 6;

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
        let use_space_char = opts.get("use_space_char").and_then(Value::as_bool);
        let max_image_width = opts
            .get("max_image_width")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_MAX_IMAGE_WIDTH, |v| v as u32);
        if max_image_width < image_width {
            return Err(RuntimeError::Config(format!(
                "OCR model '{}' has max_image_width={max_image_width} below \
                 image_width={image_width}",
                spec.alias
            )));
        }
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
            use_space_char,
            max_image_width,
            output_name,
            detector,
        })
    }
}

/// Whether the recognizer has a trailing space class past the dictionary.
///
/// PaddleOCR recognizers trained with `use_space_char=True` output
/// `1 + dict_len + 1` classes: the CTC blank, the dictionary entries, and a
/// space as the last class, which is not a line in the dictionary file.
/// `use_space_char` unset means detect it from the class count. A count that
/// fits neither layout is an error, since decoding would map classes to the
/// wrong characters.
fn resolve_space_class(
    alias: &str,
    dict_len: usize,
    n_classes: usize,
    use_space_char: Option<bool>,
) -> Result<bool> {
    let non_blank = n_classes.saturating_sub(1);
    let has_space = match use_space_char {
        Some(flag) => flag,
        None => non_blank == dict_len + 1,
    };
    let expected = dict_len + usize::from(has_space);
    if non_blank != expected {
        let hint = match use_space_char {
            Some(true) => " (use_space_char=true)",
            Some(false) => " (use_space_char=false)",
            None => "",
        };
        return Err(RuntimeError::Config(format!(
            "OCR model '{alias}' outputs {n_classes} classes, but its dictionary has \
             {dict_len} entries{hint}: expected {} classes (blank + dictionary) or {} \
             (blank + dictionary + space). Check that char_dict_path matches the \
             recognition model.",
            dict_len + 1,
            dict_len + 2,
        )));
    }
    Ok(has_space)
}

/// Fixed `(batch, height, width)` of the recognizer's `[N, 3, H, W]` input,
/// where the graph fixes them (`None` for a dynamic dimension).
fn rec_input_dims(session: &Session) -> (Option<u32>, Option<u32>, Option<u32>) {
    let dim = |i: usize| {
        session
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .filter(|shape| shape.len() == 4)
            .and_then(|shape| u32::try_from(shape[i]).ok())
            .filter(|&d| d > 0)
    };
    (dim(0), dim(2), dim(3))
}

/// The recognizer output's class count, when the graph fixes it.
fn rec_output_classes(session: &Session, output_name: Option<&str>) -> Option<usize> {
    let outputs = session.outputs();
    let output = match output_name {
        Some(name) => outputs.iter().find(|o| o.name() == name),
        None => outputs.first(),
    }?;
    let shape = output.dtype().tensor_shape()?;
    shape
        .last()
        .and_then(|&d| usize::try_from(d).ok())
        .filter(|&d| d > 0)
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
    /// Narrowest batch width; shorter lines are right-padded up to it.
    min_width: u32,
    /// Widest a line is resized to. Equals `min_width` when the graph fixes
    /// its input width.
    max_width: u32,
    /// Lines per session run: [`REC_BATCH_SIZE`], or the graph's fixed batch.
    batch_size: usize,
    /// Whether the graph fixes its batch, so every run must be `batch_size` rows.
    fixed_batch: bool,
    normalization: Normalization,
    blank_class: usize,
    use_space_char: Option<bool>,
    output_name: Option<String>,
    /// When present, `recognize()` runs detect → crop → recognize → order.
    detector: Option<Detector>,
    /// Reported by [`ModelInfo::active_execution_providers`](crate::traits::ModelInfo::active_execution_providers).
    execution_providers: Vec<String>,
}

impl OnnxOcrModel {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = OcrConfig::resolve(spec)?;
        let session_cfg =
            SessionConfig::from_options(&spec.options, GraphOptimizationLevel::Level3)?;
        session_cfg.check_execution_providers(&spec.alias)?;
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
        let BuiltSession {
            session,
            execution_providers: rec_eps,
        } = build_session(&model_path, &spec.alias, &session_cfg)?;

        // Catch a dictionary that doesn't match the model at load, when the
        // graph fixes its class count; otherwise each run checks it.
        if let Some(n_classes) = rec_output_classes(&session, cfg.output_name.as_deref()) {
            resolve_space_class(&spec.alias, chars.len(), n_classes, cfg.use_space_char)?;
        }
        let (fixed_batch, fixed_h, fixed_w) = rec_input_dims(&session);
        if let Some(h) = fixed_h
            && h != cfg.image_height
        {
            return Err(RuntimeError::Config(format!(
                "OCR model '{}' takes input height {h}, but image_height is {}",
                spec.alias, cfg.image_height
            )));
        }
        // A graph with a fixed input width can't take wider lines: resize to
        // it with the aspect ratio kept, padding on the right.
        // A graph exported with a fixed batch (paddle2onnx's `[1, 3, 48, 320]`
        // is common) takes that many lines per run, never more.
        let batch_size = match fixed_batch {
            Some(n) => n as usize,
            None => REC_BATCH_SIZE,
        };
        let (min_width, max_width) = match fixed_w {
            Some(w) => (w, w),
            None => (cfg.image_width, cfg.max_image_width),
        };

        let mut execution_providers = rec_eps;
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
                let det = build_session(&det_path, &spec.alias, &session_cfg)?;
                // Report what the whole pipeline runs on: a provider only one
                // stage attached does not make the alias run on it.
                execution_providers =
                    common_execution_providers(&execution_providers, &det.execution_providers);
                Some(Detector {
                    session: Mutex::new(det.session),
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
            min_width,
            max_width,
            batch_size,
            fixed_batch: fixed_batch.is_some(),
            normalization: cfg.normalization,
            blank_class: cfg.blank_class,
            use_space_char: cfg.use_space_char,
            output_name: cfg.output_name,
            detector,
            execution_providers: ep_ids(&execution_providers),
        })
    }
}

impl crate::traits::ModelInfo for OnnxOcrModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The providers the recognizer attached; with a detector, only those
    /// both stages attached, in the recognizer's order.
    fn active_execution_providers(&self) -> Vec<String> {
        self.execution_providers.clone()
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
        let decoded_images = images
            .iter()
            .map(|input| decode_image(input, &self.alias))
            .collect::<Result<Vec<_>>>()?;
        let lines: Vec<&DynamicImage> = decoded_images.iter().collect();
        let decoded = self.recognize_lines(&lines)?;
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

            let mut kept = Vec::with_capacity(boxes.len());
            let mut crops = Vec::with_capacity(boxes.len());
            for b in &boxes {
                let bw = (b.x1 - b.x0).floor() as u32;
                let bh = (b.y1 - b.y0).floor() as u32;
                if bw == 0 || bh == 0 {
                    continue; // degenerate region — never crop a 0-size box
                }
                crops.push(img.crop_imm(b.x0 as u32, b.y0 as u32, bw, bh));
                kept.push(*b);
            }
            let lines: Vec<&DynamicImage> = crops.iter().collect();
            let decoded = self.recognize_lines(&lines)?;
            let items = kept
                .into_iter()
                .zip(decoded)
                .map(|(b, (text, confidence))| (b, text, confidence))
                .collect();
            results.push(assemble_ocr_result(items));
        }
        Ok(results)
    }

    /// Recognize text-line images, returning `(text, confidence)` in input
    /// order.
    ///
    /// Lines are sorted by aspect ratio and run `batch_size` at a time,
    /// each batch as wide as its widest line (within `min_width..=max_width`),
    /// so every line keeps its aspect ratio and short lines in a batch are
    /// right-padded rather than stretched.
    fn recognize_lines(&self, lines: &[&DynamicImage]) -> Result<Vec<(String, f32)>> {
        let mut order: Vec<usize> = (0..lines.len()).collect();
        let ratio = |i: usize| {
            let (w, h) = lines[i].dimensions();
            f64::from(w) / f64::from(h.max(1))
        };
        order.sort_by(|&a, &b| ratio(a).total_cmp(&ratio(b)));

        let mut out = vec![(String::new(), 0.0f32); lines.len()];
        for chunk in order.chunks(self.batch_size) {
            let batch: Vec<&DynamicImage> = chunk.iter().map(|&i| lines[i]).collect();
            let width = batch
                .iter()
                .map(|img| {
                    let (w, h) = img.dimensions();
                    aspect_width(w, h, self.image_height, self.max_width)
                })
                .max()
                .unwrap_or(self.min_width)
                .max(self.min_width);
            let mut tensor = pack_text_lines(&batch, self.image_height, width, self.normalization);
            if self.fixed_batch && batch.len() < self.batch_size {
                // A fixed-batch graph takes exactly `batch_size` rows: pad the
                // last, short batch with blank rows and ignore their output.
                let mut padded = ndarray::Array4::<f32>::zeros((
                    self.batch_size,
                    3,
                    self.image_height as usize,
                    width as usize,
                ));
                padded
                    .slice_mut(ndarray::s![..batch.len(), .., .., ..])
                    .assign(&tensor);
                tensor = padded;
            }
            let decoded = self.run_recognition(tensor)?;
            for (&i, result) in chunk.iter().zip(decoded) {
                out[i] = result;
            }
        }
        Ok(out)
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

        let has_space = resolve_space_class(
            &self.alias,
            self.chars.len(),
            n_classes,
            self.use_space_char,
        )?;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            out.push(decode_ctc(
                arr3.slice(ndarray::s![b, .., ..]),
                &self.chars,
                self.blank_class,
                has_space,
            ));
        }
        Ok(out)
    }
}

/// CTC greedy-decode one `[time, classes]` slice into `(text, mean_conf)`.
///
/// Classes other than `blank_class` map, in order, to the dictionary entries
/// and then, when `has_space`, to a trailing space.
fn decode_ctc(
    slice: ndarray::ArrayView2<f32>,
    chars: &[String],
    blank_class: usize,
    has_space: bool,
) -> (String, f32) {
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

        if class_id == blank_class {
            prev_class = None;
            continue;
        }
        if prev_class == Some(class_id) {
            continue;
        }
        prev_class = Some(class_id);

        // Skip over the blank: classes below it map straight to the
        // dictionary, classes above it are shifted down by one.
        let char_idx = if class_id < blank_class {
            class_id
        } else {
            class_id - 1
        };
        let token = match chars.get(char_idx) {
            Some(c) => c.as_str(),
            None if has_space && char_idx == chars.len() => " ",
            None => continue,
        };
        text.push_str(token);
        total_conf += class_confidence(&row.to_vec(), class_id) as f64;
        emitted += 1;
    }
    let avg_conf = if emitted > 0 {
        (total_conf / emitted as f64) as f32
    } else {
        0.0
    };
    (text, avg_conf)
}

impl OnnxOcrModel {
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

    /// A `[time, classes]` slice whose argmax per row is `classes_by_step[t]`.
    fn one_hot(classes_by_step: &[usize], n_classes: usize) -> ndarray::Array2<f32> {
        let mut a = ndarray::Array2::<f32>::zeros((classes_by_step.len(), n_classes));
        for (t, &c) in classes_by_step.iter().enumerate() {
            a[[t, c]] = 1.0;
        }
        a
    }

    fn dict(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn decode_maps_trailing_class_to_space() {
        // PaddleOCR layout: blank, a, b, then the space class past the dict.
        let chars = dict(&["a", "b"]);
        let slice = one_hot(&[1, 0, 3, 3, 2], 4);
        let (text, conf) = decode_ctc(slice.view(), &chars, 0, true);
        assert_eq!(text, "a b");
        assert!((conf - 1.0).abs() < 1e-6);
        // Without a space class the extra index is dropped, as before.
        let (text, _) = decode_ctc(slice.view(), &chars, 0, false);
        assert_eq!(text, "ab");
    }

    #[test]
    fn decode_collapses_repeats_and_honours_blank() {
        let chars = dict(&["a", "b"]);
        // a a <blank> a b b -> "aab"
        let slice = one_hot(&[1, 1, 0, 1, 2, 2], 3);
        assert_eq!(decode_ctc(slice.view(), &chars, 0, false).0, "aab");
    }

    #[test]
    fn decode_handles_blank_as_last_class() {
        // Layout: a, b, blank. Classes below the blank map straight through.
        let chars = dict(&["a", "b"]);
        let slice = one_hot(&[0, 2, 1, 2, 0], 3);
        assert_eq!(decode_ctc(slice.view(), &chars, 2, false).0, "aba");
    }

    #[test]
    fn space_class_is_detected_from_class_count() {
        // blank + 436 + space = 438: the PP-OCRv5 English layout.
        assert!(resolve_space_class("a", 436, 438, None).unwrap());
        // blank + 436 = 437: no space class.
        assert!(!resolve_space_class("a", 436, 437, None).unwrap());
        // Anything else is a dictionary/model mismatch.
        assert!(matches!(
            resolve_space_class("a", 436, 500, None),
            Err(RuntimeError::Config(_))
        ));
    }

    #[test]
    fn explicit_space_char_must_match_class_count() {
        assert!(resolve_space_class("a", 436, 438, Some(true)).unwrap());
        assert!(!resolve_space_class("a", 436, 437, Some(false)).unwrap());
        assert!(resolve_space_class("a", 436, 437, Some(true)).is_err());
        assert!(resolve_space_class("a", 436, 438, Some(false)).is_err());
    }

    #[test]
    fn config_reads_space_char_and_width_cap() {
        let cfg = OcrConfig::resolve(&ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt"
        })))
        .unwrap();
        assert_eq!(cfg.use_space_char, None, "unset means auto-detect");
        assert_eq!(cfg.max_image_width, DEFAULT_MAX_IMAGE_WIDTH);

        let cfg = OcrConfig::resolve(&ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt",
            "use_space_char": false,
            "max_image_width": 1000
        })))
        .unwrap();
        assert_eq!(cfg.use_space_char, Some(false));
        assert_eq!(cfg.max_image_width, 1000);

        let err = OcrConfig::resolve(&ocr_spec(serde_json::json!({
            "onnx_path": "rec.onnx",
            "char_dict_path": "dict.txt",
            "image_width": 320,
            "max_image_width": 100
        })));
        assert!(matches!(err, Err(RuntimeError::Config(_))));
    }

    #[test]
    fn assemble_blank_input_is_empty() {
        let r = assemble_ocr_result(Vec::new());
        assert!(r.blocks.is_empty());
        assert_eq!(r.plain_text, "");
    }
}
