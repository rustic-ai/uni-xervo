# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-03-12

### Breaking Changes
- `GeneratorModel::generate()` signature changed: `messages: &[String]` → `messages: &[Message]`.
- `GenerationResult` now has two additional required fields: `images: Vec<GeneratedImage>`, `audio: Option<AudioOutput>`.
- All provider implementations updated accordingly.
- Migration: replace `&["text".to_string()]` with `&[Message::user("text")]`.

### Added
- **Multimodal message types**: `Message`, `MessageRole`, `ContentBlock`, `ImageInput` for structured conversation input.
- **Vision generation**: Process images + text via mistralrs vision pipeline (`"pipeline": "vision"`).
- **Image generation**: Diffusion pipeline (FLUX) via mistralrs (`"pipeline": "diffusion"`).
- **Speech synthesis**: Audio generation (Dia) via mistralrs (`"pipeline": "speech"`).
- **GGUF model support**: Load quantized GGUF models in mistralrs text pipeline.
- **dtype control**: Configure model precision (`f32`, `f16`, `bf16`, `auto`) for mistralrs pipelines.
- **ISQ quantization**: In-situ quantization support for text and vision pipelines.
- **Embedding validation**: NaN/Inf detection in embedding outputs.
- **Explicit message roles**: `System`, `User`, `Assistant` roles replace index-based role inference.
- `GenerationOptions` gains `width` and `height` fields for diffusion image sizing.

## [0.1.1] - 2026-02-23

### Changed
- Upgraded `thiserror` from 1.0 to 2, aligning with the transitive dependency ecosystem.
- Upgraded `reqwest` from 0.11 to 0.12, replacing the legacy `hyper` 0.14 / `http` 0.2 HTTP stack with the modern `hyper` 1.x stack and eliminating several duplicate transitive dependencies.

## [0.1.0] - 2026-02-19

### Added
- Provider options schema files and runtime validation for provider-specific options.
- Dedicated public API and error taxonomy improvements with clearer error variants.
- Minimal GitHub Actions workflows for CI and crates.io publishing.
- Expanded website documentation content aligned with Uni-Xervo branding.

### Changed
- Improved runtime load timeout handling and reliability behavior.
- Refined packaging metadata and include list for cleaner crates.io distribution.

### Fixed
- Correct provider capability declarations for Gemini embedding support.
- Gated mock module export for test/testing builds only.
