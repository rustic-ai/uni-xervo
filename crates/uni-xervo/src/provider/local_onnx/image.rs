// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared image preprocessing helpers for `local/onnx` tasks that consume
//! [`ImageInput`].
//!
//! Used by `image_embed.rs` (SigLIP-2-style ViT embedders), `ocr.rs` (PR-2b),
//! and `document_extract.rs` (PR-5). Decode → resize → normalize → emit a
//! `[batch, 3, H, W]` (NCHW) `Array4<f32>` tensor.
//!
//! The provider impls do not perform file I/O — `ImageInput::Url` is rejected
//! here with a clear error, matching the policy used elsewhere (callers fetch
//! URLs upstream and pass `Bytes`).

use crate::error::{Result, RuntimeError};
use crate::traits::ImageInput;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// Per-channel normalization constants. The two common conventions:
///
/// - **SigLIP / SigLIP-2**: `mean = std = (0.5, 0.5, 0.5)` (i.e. centre at 0,
///   scale to `[-1, 1]`).
/// - **ImageNet**: `mean = (0.485, 0.456, 0.406)`, `std = (0.224, 0.225, 0.225)`.
///
/// Callers pass whichever set their model was trained on.
#[derive(Debug, Clone, Copy)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    /// SigLIP / SigLIP-2 default — `(0.5, 0.5, 0.5)` for both mean and std.
    pub const SIGLIP: Self = Self {
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };

    /// Classic ImageNet stats — used by most ViT/CLIP-style models that
    /// weren't trained with SigLIP's normalization.
    pub const IMAGENET: Self = Self {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };
}

/// Preprocess one batch of [`ImageInput`]s into a `[batch, 3, H, W]` float
/// tensor with the given target size and normalization.
///
/// Steps per image:
/// 1. Decode bytes (PNG/JPEG/WebP per the `image` crate's defaults).
/// 2. Convert to RGB8 — alpha is dropped, paletted images are expanded.
/// 3. Resize to `target_size × target_size` with Lanczos3 (high quality).
/// 4. Scale pixels to `[0, 1]` then `(pixel - mean) / std` per channel.
///
/// # Errors
/// Returns an error if the input is `ImageInput::Url` (not supported — see
/// module doc), or if decoding fails.
pub fn preprocess_batch(
    images: &[ImageInput],
    target_size: u32,
    norm: Normalization,
) -> Result<Array4<f32>> {
    preprocess_batch_hw(images, target_size, target_size, norm)
}

/// Like [`preprocess_batch`] but to an arbitrary `height` × `width`.
///
/// Generalizes preprocessing so non-square models — notably CRNN-style OCR
/// recognition, which expects a fixed height — are not forced through a square
/// resize. Produces an `NCHW` tensor `[batch, 3, height, width]`.
///
/// Non-square inputs are stretched to the target (no letterboxing). CTC text
/// recognition should use [`pack_text_lines`] instead, which keeps the aspect
/// ratio.
///
/// # Errors
/// Returns an error if an input is `ImageInput::Url` (not fetched here) or if
/// decoding fails.
pub fn preprocess_batch_hw(
    images: &[ImageInput],
    height: u32,
    width: u32,
    norm: Normalization,
) -> Result<Array4<f32>> {
    let mut tensor = Array4::<f32>::zeros((images.len(), 3, height as usize, width as usize));

    for (i, input) in images.iter().enumerate() {
        let bytes = match input {
            ImageInput::Bytes { data, .. } => data,
            ImageInput::Url(url) => {
                return Err(RuntimeError::Config(format!(
                    "local/onnx image preprocessing does not fetch URLs; \
                     fetch '{url}' upstream and pass the bytes via \
                     ImageInput::Bytes"
                )));
            }
        };

        let img =
            image::load_from_memory(bytes).map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: "local/onnx".to_string(),
                cause: format!("Image decode failed at index {i}: {e}"),
            })?;
        write_image_into(&mut tensor, i, &img, height, width, norm);
    }

    Ok(tensor)
}

/// Preprocess a single already-decoded image into a `[1, 3, height, width]`
/// tensor.
///
/// Lets callers that already hold a [`DynamicImage`] (e.g. a cropped detection
/// region) preprocess without a decode → re-encode → decode round-trip.
#[must_use]
pub fn fill_chw(img: &DynamicImage, height: u32, width: u32, norm: Normalization) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    write_image_into(&mut tensor, 0, img, height, width, norm);
    tensor
}

/// Preprocess a page for DBNet-style detection.
///
/// Resizes preserving aspect so the longest side is at most `limit_side`, then
/// rounds each dimension to the nearest multiple of 32 (a DBNet input
/// requirement), with a 32px floor. Returns the `[1, 3, det_h, det_w]` tensor
/// plus `(scale_x, scale_y)` where `scale = det_dim / original_dim` — the
/// factors needed to map detector-space boxes back to original-image pixels.
#[must_use]
pub fn preprocess_det(
    img: &DynamicImage,
    limit_side: u32,
    norm: Normalization,
) -> (Array4<f32>, f32, f32) {
    let (orig_w, orig_h) = img.dimensions();
    let max_side = orig_w.max(orig_h) as f32;
    let ratio = if max_side > limit_side as f32 {
        limit_side as f32 / max_side
    } else {
        1.0
    };
    let det_w = round_to_multiple_of_32((orig_w as f32 * ratio).round() as u32);
    let det_h = round_to_multiple_of_32((orig_h as f32 * ratio).round() as u32);

    let scale_x = det_w as f32 / orig_w as f32;
    let scale_y = det_h as f32 / orig_h as f32;
    (fill_chw(img, det_h, det_w, norm), scale_x, scale_y)
}

/// Width a text-line image takes when resized to `height` with its aspect
/// ratio kept, capped at `max_width` and never below 1.
///
/// Mirrors PaddleOCR's `resize_norm_img`: `ceil(height * w / h)`.
#[must_use]
pub fn aspect_width(img_w: u32, img_h: u32, height: u32, max_width: u32) -> u32 {
    if img_h == 0 {
        return 1;
    }
    let w = (f64::from(height) * f64::from(img_w) / f64::from(img_h)).ceil() as u32;
    w.clamp(1, max_width.max(1))
}

/// Pack text-line images into one `[n, 3, height, width]` tensor for CTC
/// recognition, keeping each image's aspect ratio.
///
/// Each image is resized to `height` × [`aspect_width`] (capped at `width`),
/// written at the left edge, and the rest of its row is left at 0.0 — zero in
/// normalized space, which is what PaddleOCR pads with. Stretching a long line
/// to a fixed width instead leaves the recognizer fewer time steps than the
/// line has characters, and its text comes out cut short.
#[must_use]
pub fn pack_text_lines(
    images: &[&DynamicImage],
    height: u32,
    width: u32,
    norm: Normalization,
) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((images.len(), 3, height as usize, width as usize));
    for (i, img) in images.iter().enumerate() {
        let (w, h) = img.dimensions();
        let resized_w = aspect_width(w, h, height, width);
        write_image_into(&mut tensor, i, img, height, resized_w, norm);
    }
    tensor
}

/// Round up to the nearest multiple of 32, with a 32px floor.
fn round_to_multiple_of_32(n: u32) -> u32 {
    (((n + 16) / 32) * 32).max(32)
}

/// Resize, convert to RGB8, normalize, and write into `tensor[batch]`.
fn write_image_into(
    tensor: &mut Array4<f32>,
    batch: usize,
    img: &DynamicImage,
    height: u32,
    width: u32,
    norm: Normalization,
) {
    let resized = resize_to_hw(img, width, height);
    let rgb = resized.to_rgb8();
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let xi = x as usize;
        let yi = y as usize;
        // image::Rgb<u8> uses [r, g, b].
        for c in 0..3 {
            let raw = pixel.0[c] as f32 / 255.0;
            tensor[[batch, c, yi, xi]] = (raw - norm.mean[c]) / norm.std[c];
        }
    }
}

/// Resize an image to `width` × `height` using Lanczos3. Non-matching inputs are
/// stretched (no letterboxing) — matches the convention used by SigLIP / CLIP
/// exports.
fn resize_to_hw(img: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w == width && h == height {
        return img.clone();
    }
    img.resize_exact(width, height, FilterType::Lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_png() -> Vec<u8> {
        // 2x2 RGB image, alternating red/green/blue/white. Encode to PNG.
        let mut buf: image::RgbImage = image::ImageBuffer::new(2, 2);
        buf.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        buf.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        buf.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        buf.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode test PNG");
        out
    }

    #[test]
    fn preprocess_emits_correct_shape() {
        let inputs = vec![ImageInput::Bytes {
            data: make_tiny_png(),
            media_type: "image/png".to_string(),
        }];
        let tensor = preprocess_batch(&inputs, 8, Normalization::SIGLIP).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn preprocess_hw_emits_non_square_shape() {
        let inputs = vec![ImageInput::Bytes {
            data: make_tiny_png(),
            media_type: "image/png".to_string(),
        }];
        // Fixed height, wider width — the CRNN-style non-square case the
        // square helper could not express.
        let tensor = preprocess_batch_hw(&inputs, 32, 128, Normalization::SIGLIP).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 32, 128]);
    }

    fn solid_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(
            w,
            h,
            image::Rgb([128, 128, 128]),
        ))
    }

    #[test]
    fn round_to_32_floors_and_rounds() {
        assert_eq!(round_to_multiple_of_32(10), 32); // floor
        assert_eq!(round_to_multiple_of_32(64), 64); // already a multiple
        assert_eq!(round_to_multiple_of_32(960), 960);
        assert_eq!(round_to_multiple_of_32(480), 480);
    }

    #[test]
    fn preprocess_det_landscape_scale_and_shape() {
        let img = solid_image(1000, 500);
        let (tensor, sx, sy) = preprocess_det(&img, 960, Normalization::IMAGENET);
        // 1000x500 * 0.96 = 960x480, both multiples of 32.
        assert_eq!(tensor.shape(), &[1, 3, 480, 960]);
        assert!((sx - 0.96).abs() < 1e-6);
        assert!((sy - 0.96).abs() < 1e-6);
    }

    #[test]
    fn preprocess_det_portrait_long_side_is_height() {
        let img = solid_image(500, 1000);
        let (tensor, _, _) = preprocess_det(&img, 960, Normalization::IMAGENET);
        assert_eq!(tensor.shape(), &[1, 3, 960, 480]);
    }

    #[test]
    fn preprocess_det_tiny_image_clamped_to_32() {
        let img = solid_image(10, 10);
        let (tensor, sx, sy) = preprocess_det(&img, 960, Normalization::IMAGENET);
        assert_eq!(tensor.shape(), &[1, 3, 32, 32]); // never zero-size
        assert!((sx - 3.2).abs() < 1e-6);
        assert!((sy - 3.2).abs() < 1e-6);
    }

    #[test]
    fn fill_chw_shape() {
        let img = solid_image(20, 10);
        let tensor = fill_chw(&img, 16, 48, Normalization::SIGLIP);
        assert_eq!(tensor.shape(), &[1, 3, 16, 48]);
    }

    #[test]
    fn aspect_width_keeps_ratio_and_caps() {
        // 800x40 at height 48 -> 960 wide.
        assert_eq!(aspect_width(800, 40, 48, 3200), 960);
        // Capped at max_width.
        assert_eq!(aspect_width(800, 40, 48, 320), 320);
        // Rounds up, never zero.
        assert_eq!(aspect_width(1, 100, 48, 320), 1);
        assert_eq!(aspect_width(10, 0, 48, 320), 1);
    }

    #[test]
    fn pack_text_lines_pads_right_with_zeros() {
        // 40x20 -> 96 wide at height 48, inside a 200-wide tensor.
        let short = solid_image(40, 20);
        // 800x40 -> 960 wide, capped to the 200-wide tensor.
        let long = solid_image(800, 40);
        let t = pack_text_lines(&[&short, &long], 48, 200, Normalization::SIGLIP);
        assert_eq!(t.shape(), &[2, 3, 48, 200]);
        // Grey 128 under SigLIP normalization is ~0.0039, not exactly 0.
        let grey = (128.0 / 255.0 - 0.5) / 0.5;
        assert!((t[[0, 0, 10, 95]] - grey).abs() < 1e-3, "inside the image");
        assert_eq!(t[[0, 0, 10, 96]], 0.0, "padding starts after 96 px");
        assert_eq!(t[[0, 2, 47, 199]], 0.0);
        assert!(
            (t[[1, 0, 10, 199]] - grey).abs() < 1e-3,
            "long line fills the row"
        );
    }

    #[test]
    fn preprocess_normalizes_per_channel() {
        let inputs = vec![ImageInput::Bytes {
            data: make_tiny_png(),
            media_type: "image/png".to_string(),
        }];
        let tensor = preprocess_batch(&inputs, 2, Normalization::SIGLIP).unwrap();
        // Resize a 2x2 to 2x2 = identity. Pixel (0,0) is red = (255,0,0).
        // After /255 and (- 0.5) / 0.5: r=1.0, g=-1.0, b=-1.0.
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[0, 1, 0, 0]] + 1.0).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 0]] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn preprocess_rejects_url_inputs() {
        let inputs = vec![ImageInput::Url("https://example.com/x.png".to_string())];
        let res = preprocess_batch(&inputs, 8, Normalization::SIGLIP);
        assert!(matches!(res, Err(RuntimeError::Config(_))));
    }

    #[test]
    fn preprocess_handles_empty_batch() {
        let tensor = preprocess_batch(&[], 8, Normalization::SIGLIP).unwrap();
        assert_eq!(tensor.shape(), &[0, 3, 8, 8]);
    }
}
