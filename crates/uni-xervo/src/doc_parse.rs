// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared document-extraction output parsers.
//!
//! Converts a document VLM's text output into a [`DocExtractResult`]. The
//! schemas differ per model family — Granite-Docling emits typed DocTags,
//! MinerU 2.5 emits structured Markdown with `$$`-delimited LaTeX, and olmOCR-2
//! emits Markdown with inline `$`-delimited LaTeX — so [`DocStyle`] selects the
//! parser. These live here (rather than under a single provider) so every
//! provider that produces such text — `local/onnx` and `remote/llamacpp` —
//! reuses the same, tested parsing without depending on the other's feature.
//!
//! Several parsers are `#[allow(dead_code)]`: which are exercised depends on the
//! enabled provider features, so they are conditionally — not unconditionally —
//! used.

use crate::traits::{DocBlock, DocBlockKind, DocExtractResult};

/// Style of VLM output. Picks the parser to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DocStyle {
    /// Granite-Docling — typed DocTags (`<heading>`, `<table>`, …) with
    /// optional bounding boxes. Smallest VLM; reference target.
    GraniteDocling,
    /// MinerU 2.5 — structured Markdown with `$$..$$` LaTeX blocks.
    Mineru,
    /// olmOCR-2 — Markdown with `$..$` inline LaTeX.
    OlmOcr,
}

/// Resolve a `style` option string into a [`DocStyle`].
///
/// Accepts `"granite-docling"`, `"mineru"`, and `"olmocr"`; returns `None` for
/// any other value.
pub(crate) fn style_from_str(s: &str) -> Option<DocStyle> {
    match s {
        "granite-docling" => Some(DocStyle::GraniteDocling),
        "mineru" => Some(DocStyle::Mineru),
        "olmocr" => Some(DocStyle::OlmOcr),
        _ => None,
    }
}

/// Parse `text` with the parser matching `style`.
#[allow(dead_code)] // Used only by providers that run a document VLM.
pub(crate) fn parse_by_style(style: DocStyle, text: &str) -> DocExtractResult {
    match style {
        DocStyle::GraniteDocling => parse_doctags(text),
        DocStyle::Mineru => parse_mineru_markdown(text),
        DocStyle::OlmOcr => parse_olmocr_markdown(text),
    }
}

/// Parse a Granite-Docling DocTags string into a [`DocExtractResult`].
///
/// DocTags is XML-style; each block is wrapped in a typed tag (e.g.
/// `<heading>`, `<table>`, `<formula>`) with an optional `loc` attribute
/// of comma-separated `x0,y0,x1,y1` floats. Inline content is the tag body.
///
/// Unknown tags are mapped to [`DocBlockKind::Text`].
#[allow(dead_code)] // Used only when a DocTags-style VLM is enabled.
pub(crate) fn parse_doctags(input: &str) -> DocExtractResult {
    let mut blocks = Vec::new();
    let mut plain_md = String::new();
    let mut reading_order = 0u32;
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] != b'<' {
            // Untagged text — gobble until next `<` and emit as text.
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            let s = input[start..i].trim();
            if !s.is_empty() {
                blocks.push(DocBlock {
                    kind: DocBlockKind::Text,
                    content: s.to_string(),
                    bbox: None,
                    reading_order,
                });
                reading_order += 1;
                plain_md.push_str(s);
                plain_md.push('\n');
            }
            continue;
        }

        // Parse opening tag `<name attr="...">`.
        let tag_start = i;
        i += 1; // skip '<'
        let name_start = i;
        while i < bytes.len() && bytes[i] != b' ' && bytes[i] != b'>' && bytes[i] != b'/' {
            i += 1;
        }
        let name_end = i;
        let tag_name = &input[name_start..name_end];

        // Find end of opening tag.
        let mut attrs_end = i;
        while attrs_end < bytes.len() && bytes[attrs_end] != b'>' {
            attrs_end += 1;
        }
        if attrs_end >= bytes.len() {
            // Malformed — bail out, drop the trailing junk.
            break;
        }
        let attr_str = &input[name_end..attrs_end];
        let bbox = parse_loc_attr(attr_str);
        i = attrs_end + 1;

        // Self-closing tags (`<x/>`) — emit empty block, continue.
        if input[..attrs_end].ends_with('/') {
            blocks.push(DocBlock {
                kind: doctag_kind(tag_name),
                content: String::new(),
                bbox,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Find matching closing `</name>`.
        let close = format!("</{tag_name}>");
        let body_start = i;
        let close_pos = input[body_start..].find(&close).map(|p| body_start + p);
        let (body_end, after_close) = match close_pos {
            Some(p) => (p, p + close.len()),
            None => {
                // Unclosed — treat the rest as body.
                (input.len(), input.len())
            }
        };
        let body = &input[body_start..body_end];
        let body_trimmed = body.trim().to_string();

        let kind = doctag_kind(tag_name);
        if !body_trimmed.is_empty() {
            plain_md.push_str(&render_block_to_markdown(kind, &body_trimmed));
            plain_md.push('\n');
        }
        blocks.push(DocBlock {
            kind,
            content: body_trimmed,
            bbox,
            reading_order,
        });
        reading_order += 1;
        i = after_close;
        let _ = tag_start;
    }

    DocExtractResult {
        blocks,
        plain_markdown: plain_md.trim_end().to_string(),
    }
}

#[allow(dead_code)] // Helper for `parse_doctags`.
fn doctag_kind(tag: &str) -> DocBlockKind {
    match tag.to_ascii_lowercase().as_str() {
        "heading" | "title" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => DocBlockKind::Heading,
        "list" | "ul" | "ol" => DocBlockKind::List,
        "table" => DocBlockKind::Table,
        "figure" | "image" => DocBlockKind::Figure,
        "formula" | "math" => DocBlockKind::Formula,
        "caption" => DocBlockKind::Caption,
        "footer" => DocBlockKind::Footer,
        "header" => DocBlockKind::Header,
        _ => DocBlockKind::Text,
    }
}

#[allow(dead_code)] // Helper for `parse_doctags`.
fn parse_loc_attr(attrs: &str) -> Option<[f32; 4]> {
    let loc_key = "loc=\"";
    let start = attrs.find(loc_key)? + loc_key.len();
    let rest = &attrs[start..];
    let end = rest.find('"')?;
    let parts: Vec<f32> = rest[..end]
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}

#[allow(dead_code)] // Helper for `parse_doctags`.
fn render_block_to_markdown(kind: DocBlockKind, content: &str) -> String {
    match kind {
        DocBlockKind::Heading => format!("## {content}"),
        DocBlockKind::List => format!("- {content}"),
        DocBlockKind::Formula => format!("$$\n{content}\n$$"),
        DocBlockKind::Caption => format!("*{content}*"),
        _ => content.to_string(),
    }
}

/// Parse MinerU 2.5's structured Markdown into a [`DocExtractResult`].
///
/// Block-detection heuristics:
/// - Lines starting with `#`, `##`, `###`, … → `Heading`.
/// - Lines starting with `- ` / `* ` / `1.` (numeric prefix) → `List`.
/// - Blocks delimited by `$$..$$` → `Formula` (LaTeX body).
/// - Blocks starting with `| ` (Markdown table syntax) → `Table`.
/// - Blocks with `![...](...)` (image markdown) → `Figure`.
/// - Everything else → `Text`.
#[allow(dead_code)] // Used only when a Markdown-style VLM is enabled.
pub(crate) fn parse_mineru_markdown(input: &str) -> DocExtractResult {
    let mut blocks = Vec::new();
    let mut reading_order = 0u32;

    // First pass: split into double-newline-separated paragraphs.
    for paragraph in input.split("\n\n") {
        let p = paragraph.trim();
        if p.is_empty() {
            continue;
        }

        // Match $$..$$ as a formula block.
        if let Some(inner) = p.strip_prefix("$$").and_then(|r| r.strip_suffix("$$")) {
            blocks.push(DocBlock {
                kind: DocBlockKind::Formula,
                content: inner.trim().to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Image: ![alt](path). Treat as figure with the alt text as content.
        if p.starts_with("![") {
            let alt = p
                .strip_prefix("![")
                .and_then(|r| r.find(']').map(|i| &r[..i]))
                .unwrap_or("");
            blocks.push(DocBlock {
                kind: DocBlockKind::Figure,
                content: alt.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Table: every line starts with '|'.
        if p.lines().all(|line| line.trim_start().starts_with('|')) {
            blocks.push(DocBlock {
                kind: DocBlockKind::Table,
                content: p.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Heading: starts with one or more '#' followed by space.
        if let Some(rest) = p.strip_prefix('#')
            && (rest.starts_with('#') || rest.starts_with(' '))
        {
            let stripped = rest.trim_start_matches('#').trim_start();
            blocks.push(DocBlock {
                kind: DocBlockKind::Heading,
                content: stripped.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // List: first line starts with "- ", "* ", or "N. ".
        let first = p.lines().next().unwrap_or("").trim_start();
        let is_list = first.starts_with("- ")
            || first.starts_with("* ")
            || first.chars().take_while(|c| c.is_ascii_digit()).count() >= 1
                && first.contains(". ");
        if is_list {
            blocks.push(DocBlock {
                kind: DocBlockKind::List,
                content: p.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Default: plain text.
        blocks.push(DocBlock {
            kind: DocBlockKind::Text,
            content: p.to_string(),
            bbox: None,
            reading_order,
        });
        reading_order += 1;
    }

    DocExtractResult {
        blocks,
        plain_markdown: input.trim().to_string(),
    }
}

/// Parse olmOCR-2's Markdown output. olmOCR-2 emits Markdown that's
/// very close to MinerU's, but inline LaTeX uses single `$..$` rather
/// than `$$..$$`. We reuse the MinerU parser since the block-level
/// heuristics still hold.
///
/// Inline `$..$` is preserved verbatim inside text blocks — no separate
/// formula block, matching the inline convention.
#[allow(dead_code)] // Used only when olmOCR-2 is enabled.
pub(crate) fn parse_olmocr_markdown(input: &str) -> DocExtractResult {
    parse_mineru_markdown(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctags_parses_typed_blocks() {
        let input = r#"<heading loc="0.1,0.1,0.9,0.2">Chapter 1</heading><text>Body.</text><formula>E = mc^2</formula>"#;
        let r = parse_doctags(input);
        assert_eq!(r.blocks.len(), 3);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[0].content, "Chapter 1");
        assert_eq!(r.blocks[0].bbox, Some([0.1, 0.1, 0.9, 0.2]));
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[2].kind, DocBlockKind::Formula);
        assert!(r.plain_markdown.contains("## Chapter 1"));
    }

    #[test]
    fn doctags_handles_unknown_tag_as_text() {
        let r = parse_doctags("<unknown>hello</unknown>");
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[0].content, "hello");
    }

    #[test]
    fn doctags_assigns_increasing_reading_order() {
        let r = parse_doctags("<heading>A</heading><text>B</text>");
        assert_eq!(r.blocks[0].reading_order, 0);
        assert_eq!(r.blocks[1].reading_order, 1);
    }

    #[test]
    fn mineru_parses_heading_text_formula_table() {
        let input = "# Title\n\nA paragraph.\n\n$$\n\\sum x_i\n$$\n\n| a | b |\n| 1 | 2 |";
        let r = parse_mineru_markdown(input);
        assert_eq!(r.blocks.len(), 4);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[0].content, "Title");
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[2].kind, DocBlockKind::Formula);
        assert_eq!(r.blocks[3].kind, DocBlockKind::Table);
    }

    #[test]
    fn mineru_parses_lists() {
        let input = "- item one\n- item two";
        let r = parse_mineru_markdown(input);
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::List);
    }

    #[test]
    fn mineru_parses_figures() {
        let r = parse_mineru_markdown("![A cat](cat.png)");
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Figure);
        assert_eq!(r.blocks[0].content, "A cat");
    }

    #[test]
    fn olmocr_reuses_mineru_block_heuristics() {
        let r = parse_olmocr_markdown("# Heading\n\nText with $x = 1$ inline math.");
        assert_eq!(r.blocks.len(), 2);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        // Inline $..$ stays verbatim in text — no separate formula block.
        assert!(r.blocks[1].content.contains("$x = 1$"));
    }

    #[test]
    fn style_from_str_maps_known_styles() {
        assert_eq!(style_from_str("olmocr"), Some(DocStyle::OlmOcr));
        assert_eq!(style_from_str("mineru"), Some(DocStyle::Mineru));
        assert_eq!(
            style_from_str("granite-docling"),
            Some(DocStyle::GraniteDocling)
        );
        assert_eq!(style_from_str("nope"), None);
    }

    #[test]
    fn parse_by_style_dispatches() {
        let r = parse_by_style(DocStyle::OlmOcr, "# H\n\nbody");
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
    }
}
