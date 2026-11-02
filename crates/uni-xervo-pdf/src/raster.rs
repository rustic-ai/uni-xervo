//! Rasterizer seam — turn PDF pages into page images for the image tiers.
//!
//! Rasterization is the precondition for [`Ocr`](crate::Tier::Ocr) and
//! [`Vlm`](crate::Tier::Vlm): those tiers read pixels, not PDF syntax. It is
//! invoked lazily by the pipeline — only when a page actually escalates to an
//! image tier — so born-digital text pages are never rendered.
//!
//! The pure-Rust `hayro` backend ([`HayroRasterizer`], behind the `hayro`
//! feature) implements [`Rasterizer`]. The output's fidelity is an upstream
//! multiplier on every image tier's accuracy.

use uni_xervo::error::Result;
use uni_xervo::traits::ImageInput;

/// One rendered page produced by a [`Rasterizer`].
#[derive(Debug, Clone)]
pub struct RasterPage {
    /// The page's 1-based number.
    pub page_number: u32,
    /// The rendered page image.
    pub image: ImageInput,
}

/// Renders selected PDF pages to images.
///
/// Implementors render only the requested 1-based page numbers, preserving the
/// pipeline's cost-proportionality (pages that stay on the native tier are
/// never asked for).
pub trait Rasterizer: Send + Sync {
    /// Render the given 1-based `page_numbers` of `pdf` to images.
    ///
    /// # Errors
    /// Returns an error if the PDF is malformed/encrypted or a page cannot be
    /// rendered by the backend.
    fn rasterize_pages(&self, pdf: &[u8], page_numbers: &[u32]) -> Result<Vec<RasterPage>>;
}

/// Default render scale (PDF user space is 72 units/inch, so `2.0` ≈ 144 DPI).
///
/// Higher scales improve small-font OCR/VLM fidelity at the cost of larger
/// images and slower rendering; `2.0` is a balanced default. Tune via
/// [`HayroRasterizer::with_scale`].
#[cfg(feature = "hayro")]
const RASTER_SCALE_DEFAULT: f32 = 2.0;

/// Pure-Rust rasterizer backed by `hayro` (the default backend).
///
/// Renders each requested page to an RGBA pixmap on a white background and
/// encodes it as PNG. Construct with [`HayroRasterizer::new`] for the default
/// scale, or [`HayroRasterizer::with_scale`] to trade fidelity for size/speed.
#[cfg(feature = "hayro")]
#[derive(Debug, Clone, Copy)]
pub struct HayroRasterizer {
    scale: f32,
}

#[cfg(feature = "hayro")]
impl HayroRasterizer {
    /// Create a rasterizer at the default scale (≈144 DPI).
    #[must_use]
    pub fn new() -> Self {
        Self {
            scale: RASTER_SCALE_DEFAULT,
        }
    }

    /// Create a rasterizer at a specific render scale (1.0 ≈ 72 DPI).
    #[must_use]
    pub fn with_scale(scale: f32) -> Self {
        Self { scale }
    }
}

#[cfg(feature = "hayro")]
impl Default for HayroRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hayro")]
impl Rasterizer for HayroRasterizer {
    fn rasterize_pages(&self, pdf: &[u8], page_numbers: &[u32]) -> Result<Vec<RasterPage>> {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_interpret::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{PixmapSettings, RenderCache, RenderSettings, render};
        use std::collections::HashSet;
        use uni_xervo::error::RuntimeError;

        let doc = Pdf::new(pdf.to_vec()).map_err(|e| {
            RuntimeError::Config(format!("failed to parse PDF for rasterization: {e:?}"))
        })?;
        // A render cache shared across pages of one document (per hayro's design).
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        // hayro 0.8 split the output-buffer settings (scale, background) out of
        // `RenderSettings` into `PixmapSettings`. An opaque white background is
        // what `encode_png` below relies on to treat premultiplied RGBA as
        // straight RGBA.
        let render_settings = RenderSettings::default();
        let pixmap_settings = PixmapSettings {
            x_scale: self.scale,
            y_scale: self.scale,
            bg_color: WHITE,
        };

        let wanted: HashSet<u32> = page_numbers.iter().copied().collect();
        let mut out = Vec::with_capacity(page_numbers.len());
        for (idx, page) in doc.pages().iter().enumerate() {
            let page_number = idx as u32 + 1;
            if !wanted.contains(&page_number) {
                continue;
            }
            let pixmap = render(
                page,
                &cache,
                &interpreter,
                &render_settings,
                &pixmap_settings,
            );
            out.push(RasterPage {
                page_number,
                image: ImageInput::Bytes {
                    data: encode_png(&pixmap)?,
                    media_type: "image/png".to_string(),
                },
            });
        }
        Ok(out)
    }
}

/// Encode a rendered pixmap as PNG bytes.
///
/// The pixmap is rendered over an opaque white background, so its premultiplied
/// RGBA bytes equal straight RGBA and can be encoded directly.
#[cfg(feature = "hayro")]
fn encode_png(pixmap: &hayro::vello_cpu::Pixmap) -> Result<Vec<u8>> {
    use uni_xervo::error::RuntimeError;

    let width = u32::from(pixmap.width());
    let height = u32::from(pixmap.height());
    let rgba = pixmap.data_as_u8_slice().to_vec();
    let buffer = image::RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        RuntimeError::InferenceError("rasterizer produced an inconsistent pixel buffer".to_string())
    })?;
    let mut png = std::io::Cursor::new(Vec::new());
    buffer
        .write_to(&mut png, image::ImageFormat::Png)
        .map_err(|e| RuntimeError::InferenceError(format!("failed to encode page PNG: {e}")))?;
    Ok(png.into_inner())
}

#[cfg(all(test, feature = "hayro", feature = "pdf-input"))]
mod tests {
    use super::*;
    use crate::testutil::make_text_pdf;

    #[test]
    fn renders_requested_page_to_png() {
        let pdf = make_text_pdf("Hello World");
        let pages = HayroRasterizer::new().rasterize_pages(&pdf, &[1]).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page_number, 1);
        let ImageInput::Bytes { data, media_type } = &pages[0].image else {
            panic!("expected rendered bytes");
        };
        assert_eq!(media_type, "image/png");
        // PNG magic number.
        assert_eq!(
            &data[..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn skips_unrequested_pages() {
        let pdf = make_text_pdf("Hello World");
        // The document has only page 1; requesting page 2 yields nothing.
        let pages = HayroRasterizer::new().rasterize_pages(&pdf, &[2]).unwrap();
        assert!(pages.is_empty());
    }
}
