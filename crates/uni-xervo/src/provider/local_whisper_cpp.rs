// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Speech-to-text provider targeting whisper.cpp via `whisper-rs`.
//!
//! Loads a ggml `.bin` weights file from a HuggingFace repo (default
//! `ggml-base.bin`), decodes the input audio to 16 kHz mono f32 PCM, and
//! runs `whisper_rs::WhisperContext::full(...)` on a blocking thread via
//! `tokio::task::spawn_blocking`.
//!
//! # Audio decode (v1)
//!
//! - [`AudioInput::Pcm`] at 16 kHz, mono, f32 is passed through unchanged.
//!   Other sample rates / channel counts are rejected with a clear error;
//!   resampling is deferred to a follow-up.
//! - [`AudioInput::Bytes`] is decoded only when `media_type == "audio/wav"`
//!   (16-bit PCM mono/stereo, stereo collapsed to mono via simple averaging).
//!   Other container formats (MP3, FLAC, Opus, …) are rejected; consumers
//!   needing those decode upstream to `AudioInput::Pcm` for now.
//!
//! # Options
//!
//! - `model_path` (default `"ggml-base.bin"`) — file within the HF repo.
//! - `default_language` — ISO 639-1 code applied when
//!   [`TranscribeOptions::language`] is `None`. If both are `None`,
//!   whisper.cpp auto-detects.
//! - `cache_dir` — overrides the default download cache directory.
//!
//! # Threading
//!
//! `whisper-rs` is synchronous. Each call wraps `WhisperContext::create_state`
//! + `state.full(...)` in `spawn_blocking` so the runtime's executor isn't
//! pinned during a (typically) seconds-long inference.

use crate::api::{ModelAliasSpec, ModelTask};
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
use crate::hf::HfModelRepo;
use crate::traits::{
    AudioInput, LoadedModelHandle, ModelInfo, ModelProvider, ProviderCapabilities, ProviderHealth,
    TranscribeOptions, TranscribeResult, TranscribeSegment, TranscribeWord, TranscriptionModel,
};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// `local/whisper-cpp` provider.
pub struct LocalWhisperCppProvider;

impl Default for LocalWhisperCppProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalWhisperCppProvider {
    pub fn new() -> Self {
        Self
    }
}

const DEFAULT_MODEL_PATH: &str = "ggml-base.bin";

#[async_trait]
impl ModelProvider for LocalWhisperCppProvider {
    fn provider_id(&self) -> &'static str {
        "local/whisper-cpp"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Transcribe],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Transcribe => {
                let model_filename = spec
                    .options
                    .get("model_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or(DEFAULT_MODEL_PATH)
                    .to_string();
                let default_language = spec
                    .options
                    .get("default_language")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let cache_dir = resolve_cache_dir("whisper-cpp", &spec.model_id, &spec.options);

                let model_path = download_ggml_model(
                    &spec.alias,
                    &spec.model_id,
                    spec.revision.as_deref(),
                    &cache_dir,
                    &model_filename,
                )
                .await?;

                // Load the whisper context on a blocking thread — file I/O
                // + mmap + the initial weight checks aren't free.
                let alias = spec.alias.clone();
                let context = tokio::task::spawn_blocking(move || {
                    let params = WhisperContextParameters::default();
                    WhisperContext::new_with_params(&model_path, params).map_err(|e| {
                        RuntimeError::Load(format!(
                            "whisper-cpp failed to load ggml model for alias '{alias}': {e}"
                        ))
                    })
                })
                .await
                .map_err(|e| RuntimeError::Load(format!("whisper-cpp join error: {e}")))??;

                let model = WhisperCppModel {
                    alias: spec.alias.clone(),
                    model_id: spec.model_id.clone(),
                    context: Arc::new(context),
                    default_language,
                };
                let handle: Arc<dyn TranscriptionModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            task => Err(RuntimeError::CapabilityMismatch(format!(
                "local/whisper-cpp provider does not support task {:?}",
                task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// One loaded ggml model. The `WhisperContext` is `Send + Sync` per
/// `whisper-rs`'s contract — we wrap it in `Arc` so each `transcribe`
/// call can clone cheaply and ship the handle into `spawn_blocking`.
struct WhisperCppModel {
    alias: String,
    model_id: String,
    context: Arc<WhisperContext>,
    default_language: Option<String>,
}

impl WhisperCppModel {
    /// Transcribe a single decoded audio input.
    ///
    /// Shared by the batch [`transcribe`](TranscriptionModel::transcribe)
    /// primitive, which loops this over each input. The synchronous
    /// whisper.cpp `full()` call runs on a blocking thread.
    ///
    /// # Errors
    /// Returns an error if the audio cannot be decoded to 16 kHz mono PCM or if
    /// whisper.cpp inference fails.
    async fn transcribe_single(
        &self,
        audio: AudioInput,
        options: &TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let samples = decode_to_16khz_mono_pcm(&self.alias, audio)?;
        let language = options
            .language
            .clone()
            .or_else(|| self.default_language.clone());
        let word_timestamps = options.word_timestamps;
        let initial_prompt = options.initial_prompt.clone();
        let context = self.context.clone();
        let alias = self.alias.clone();

        tokio::task::spawn_blocking(move || {
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            if let Some(ref lang) = language {
                params.set_language(Some(lang.as_str()));
            }
            params.set_token_timestamps(word_timestamps);
            if let Some(ref prompt) = initial_prompt {
                params.set_initial_prompt(prompt);
            }
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_special(false);
            params.set_print_timestamps(false);

            let mut state = context.create_state().map_err(|e| {
                RuntimeError::InferenceError(format!(
                    "whisper-cpp create_state failed for '{alias}': {e}"
                ))
            })?;
            state.full(params, &samples).map_err(|e| {
                RuntimeError::InferenceError(format!(
                    "whisper-cpp full() failed for '{alias}': {e}"
                ))
            })?;

            let n_segments = state.full_n_segments();

            let mut segments: Vec<TranscribeSegment> = Vec::with_capacity(n_segments as usize);
            for i in 0..n_segments {
                let segment = state.get_segment(i).ok_or_else(|| {
                    RuntimeError::InferenceError(format!(
                        "whisper-cpp segment {i} out of range for '{alias}'"
                    ))
                })?;
                let text = segment.to_str().map_err(|e| {
                    RuntimeError::InferenceError(format!(
                        "whisper-cpp segment text failed for '{alias}': {e}"
                    ))
                })?;
                let t0 = segment.start_timestamp();
                let t1 = segment.end_timestamp();
                // whisper-rs reports timestamps in centiseconds (1/100 s).
                let start_ms = (t0 as u64).saturating_mul(10);
                let end_ms = (t1 as u64).saturating_mul(10);

                let mut words: Vec<TranscribeWord> = Vec::new();
                if word_timestamps {
                    for tok_i in 0..segment.n_tokens() {
                        let Some(token) = segment.get_token(tok_i) else {
                            continue;
                        };
                        let tok_text = token.to_str().unwrap_or_default().to_string();
                        let d = token.token_data();
                        words.push(TranscribeWord {
                            start_ms: (d.t0 as u64).saturating_mul(10),
                            end_ms: (d.t1 as u64).saturating_mul(10),
                            text: tok_text,
                            confidence: Some(d.p),
                        });
                    }
                }

                segments.push(TranscribeSegment {
                    start_ms,
                    end_ms,
                    text: text.trim().to_string(),
                    speaker: None, // whisper.cpp doesn't natively diarize
                    words,
                });
            }

            // Prefer the caller-supplied (or default) language; otherwise
            // surface whisper.cpp's auto-detected language id, falling back
            // to "auto" if the id doesn't map to a known code.
            let detected_language = language.unwrap_or_else(|| {
                whisper_rs::get_lang_str(state.full_lang_id_from_state())
                    .map(str::to_string)
                    .unwrap_or_else(|| "auto".to_string())
            });

            Ok::<_, RuntimeError>(TranscribeResult {
                language: detected_language,
                segments,
            })
        })
        .await
        .map_err(|e| RuntimeError::InferenceError(format!("whisper-cpp join error: {e}")))?
    }
}

#[async_trait]
impl TranscriptionModel for WhisperCppModel {
    async fn transcribe(
        &self,
        audios: Vec<AudioInput>,
        options: TranscribeOptions,
    ) -> Result<Vec<TranscribeResult>> {
        // whisper.cpp is single-stream: transcribe each input in order,
        // propagating the first error. The shared `options` apply to every
        // input.
        let mut results = Vec::with_capacity(audios.len());
        for audio in audios {
            results.push(self.transcribe_single(audio, &options).await?);
        }
        Ok(results)
    }

    fn supported_languages(&self) -> &[String] {
        // Whisper supports 99 languages plus auto-detect. Building a fresh
        // Vec<String> per call would be wasteful; this static slice keeps
        // the API stable. Callers needing the exhaustive list can consult
        // whisper-rs's `get_lang_str` / `get_lang_id` for live info.
        WHISPER_LANGUAGES.as_slice()
    }
}

impl ModelInfo for WhisperCppModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// HF download
// ---------------------------------------------------------------------------

async fn download_ggml_model(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    filename: &str,
) -> Result<std::path::PathBuf> {
    let api_repo = HfModelRepo::open(cache_dir, model_id, revision).map_err(|e| {
        RuntimeError::Load(format!("whisper-cpp HF api init failed for '{alias}': {e}"))
    })?;
    api_repo.get(filename).await.map_err(|e| {
        RuntimeError::Load(format!(
            "whisper-cpp could not download '{filename}' from '{model_id}' for alias '{alias}': {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Audio decode — minimal WAV-only path (v1)
// ---------------------------------------------------------------------------

/// Decode any [`AudioInput`] into a 16 kHz mono f32 sample buffer.
///
/// Returns a precise error for unsupported shapes so callers know exactly
/// what to fix (resample / decode upstream).
fn decode_to_16khz_mono_pcm(alias: &str, audio: AudioInput) -> Result<Vec<f32>> {
    match audio {
        AudioInput::Pcm {
            sample_rate,
            channels,
            samples,
        } => {
            if sample_rate != 16_000 {
                return Err(RuntimeError::InferenceError(format!(
                    "whisper-cpp '{alias}': PCM input must be 16000 Hz (got {sample_rate}); \
                     resample upstream"
                )));
            }
            Ok(match channels {
                1 => samples,
                2 => stereo_to_mono(&samples),
                n => {
                    return Err(RuntimeError::InferenceError(format!(
                        "whisper-cpp '{alias}': unsupported channel count {n}; \
                         pass mono or stereo PCM"
                    )));
                }
            })
        }
        AudioInput::Bytes { data, media_type } => {
            if media_type != "audio/wav"
                && media_type != "audio/wave"
                && media_type != "audio/x-wav"
            {
                return Err(RuntimeError::InferenceError(format!(
                    "whisper-cpp '{alias}': only audio/wav supported in v1 (got '{media_type}'); \
                     decode upstream to AudioInput::Pcm at 16 kHz mono"
                )));
            }
            decode_wav_to_16khz_mono(alias, &data)
        }
    }
}

fn stereo_to_mono(interleaved: &[f32]) -> Vec<f32> {
    interleaved
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

/// Decode a 16-bit PCM mono/stereo WAV byte stream to 16 kHz mono f32.
///
/// Supports the standard RIFF/WAVE/fmt /data layout produced by every
/// mainstream WAV encoder. PCM format only (no IEEE-float, no compression).
fn decode_wav_to_16khz_mono(alias: &str, data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < 44 {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': WAV input too short ({} bytes)",
            data.len()
        )));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': not a RIFF/WAVE file"
        )));
    }

    // Scan for `fmt ` and `data` chunks; some encoders insert `LIST`/`bext`
    // chunks before `data`, so we can't assume fixed offsets.
    let mut i = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_format, channels, sample_rate, bits_per_sample)
    let mut audio_bytes: Option<&[u8]> = None;
    while i + 8 <= data.len() {
        let id = &data[i..i + 4];
        let size =
            u32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]) as usize;
        let body_start = i + 8;
        let body_end = body_start.saturating_add(size).min(data.len());
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err(RuntimeError::InferenceError(format!(
                        "whisper-cpp '{alias}': fmt chunk too small ({size})"
                    )));
                }
                let audio_format = u16::from_le_bytes([data[body_start], data[body_start + 1]]);
                let channels = u16::from_le_bytes([data[body_start + 2], data[body_start + 3]]);
                let sample_rate = u32::from_le_bytes([
                    data[body_start + 4],
                    data[body_start + 5],
                    data[body_start + 6],
                    data[body_start + 7],
                ]);
                let bps = u16::from_le_bytes([data[body_start + 14], data[body_start + 15]]);
                fmt = Some((audio_format, channels, sample_rate, bps));
            }
            b"data" => {
                audio_bytes = Some(&data[body_start..body_end]);
            }
            _ => {}
        }
        // Chunks are 2-byte aligned per RIFF spec.
        let advance = if size % 2 == 1 { size + 1 } else { size };
        i = body_start + advance;
    }

    let (audio_format, channels, sample_rate, bps) = fmt.ok_or_else(|| {
        RuntimeError::InferenceError(format!("whisper-cpp '{alias}': missing fmt chunk"))
    })?;
    let audio_bytes = audio_bytes.ok_or_else(|| {
        RuntimeError::InferenceError(format!("whisper-cpp '{alias}': missing data chunk"))
    })?;

    if audio_format != 1 {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': only PCM WAV supported (audio_format={audio_format})"
        )));
    }
    if bps != 16 {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': only 16-bit PCM supported (got {bps}-bit)"
        )));
    }
    if sample_rate != 16_000 {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': WAV must be 16000 Hz (got {sample_rate}); resample upstream"
        )));
    }
    if !matches!(channels, 1 | 2) {
        return Err(RuntimeError::InferenceError(format!(
            "whisper-cpp '{alias}': WAV channel count must be 1 or 2 (got {channels})"
        )));
    }

    let mut mono: Vec<f32> = Vec::with_capacity(audio_bytes.len() / 2);
    let mut chunks = audio_bytes.chunks_exact(2);
    if channels == 1 {
        for c in chunks.by_ref() {
            let sample = i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32;
            mono.push(sample);
        }
    } else {
        // stereo — average pairs
        while let (Some(l), Some(r)) = (chunks.next(), chunks.next()) {
            let lv = i16::from_le_bytes([l[0], l[1]]) as f32 / i16::MAX as f32;
            let rv = i16::from_le_bytes([r[0], r[1]]) as f32 / i16::MAX as f32;
            mono.push((lv + rv) * 0.5);
        }
    }
    Ok(mono)
}

// ---------------------------------------------------------------------------
// Static language table for `supported_languages()`.
// ---------------------------------------------------------------------------

use std::sync::OnceLock;
static WHISPER_LANGS: OnceLock<Vec<String>> = OnceLock::new();

/// ISO 639-1 codes whisper.cpp supports. The list is stable across releases;
/// computed once on first call.
fn whisper_languages() -> &'static [String] {
    WHISPER_LANGS
        .get_or_init(|| {
            // Standard whisper language list — sourced from whisper.cpp's
            // `whisper_lang_str` / `whisper_lang_max_id`. Keeping it inline
            // is more robust than calling whisper-rs from a `&self` method
            // (which would require a per-call Vec allocation).
            [
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
                "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
                "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
                "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
                "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
                "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba",
                "jw", "su",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        })
        .as_slice()
}

// Reference the OnceLock-backed table without unsafe.
#[allow(non_upper_case_globals)]
const WHISPER_LANGUAGES: LangsHandle = LangsHandle;
struct LangsHandle;
impl LangsHandle {
    fn as_slice(&self) -> &'static [String] {
        whisper_languages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_wav_16k_mono(samples: &[i16]) -> Vec<u8> {
        let mut buf = Vec::new();
        let data_size = (samples.len() * 2) as u32;
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_size).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&16000u32.to_le_bytes());
        buf.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bps
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    #[test]
    fn wav_decode_mono_roundtrip() {
        let original: Vec<i16> = vec![0, 100, -100, 16000, -16000];
        let wav = build_wav_16k_mono(&original);
        let pcm = decode_wav_to_16khz_mono("test", &wav).unwrap();
        assert_eq!(pcm.len(), original.len());
        for (got, &want) in pcm.iter().zip(original.iter()) {
            let expected = want as f32 / i16::MAX as f32;
            assert!(
                (got - expected).abs() < 1e-5,
                "got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn wav_decode_rejects_non_riff() {
        let res = decode_wav_to_16khz_mono("test", b"not a wav file at all");
        assert!(res.is_err());
    }

    #[test]
    fn wav_decode_rejects_wrong_sample_rate() {
        let mut wav = build_wav_16k_mono(&[0, 0, 0]);
        // Patch sample rate to 44100.
        wav[24..28].copy_from_slice(&44100u32.to_le_bytes());
        let res = decode_wav_to_16khz_mono("test", &wav);
        assert!(res.is_err());
    }

    #[test]
    fn pcm_pass_through_for_16khz_mono() {
        let samples = vec![0.1, 0.2, 0.3];
        let res = decode_to_16khz_mono_pcm(
            "test",
            AudioInput::Pcm {
                sample_rate: 16000,
                channels: 1,
                samples: samples.clone(),
            },
        )
        .unwrap();
        assert_eq!(res, samples);
    }

    #[test]
    fn pcm_rejects_non_16khz() {
        let res = decode_to_16khz_mono_pcm(
            "test",
            AudioInput::Pcm {
                sample_rate: 22050,
                channels: 1,
                samples: vec![0.0; 100],
            },
        );
        assert!(res.is_err());
    }

    #[test]
    fn stereo_pcm_collapsed_to_mono() {
        let interleaved = vec![0.1, 0.3, 0.5, 0.7]; // L=0.1,R=0.3,L=0.5,R=0.7
        let res = decode_to_16khz_mono_pcm(
            "test",
            AudioInput::Pcm {
                sample_rate: 16000,
                channels: 2,
                samples: interleaved,
            },
        )
        .unwrap();
        assert_eq!(res, vec![0.2, 0.6]);
    }

    #[test]
    fn bytes_rejects_non_wav() {
        let res = decode_to_16khz_mono_pcm(
            "test",
            AudioInput::Bytes {
                data: vec![0; 100],
                media_type: "audio/mpeg".to_string(),
            },
        );
        assert!(res.is_err());
    }

    #[test]
    fn supported_languages_includes_common_codes() {
        let langs = whisper_languages();
        assert!(langs.iter().any(|s| s == "en"));
        assert!(langs.iter().any(|s| s == "es"));
        assert!(langs.iter().any(|s| s == "zh"));
    }
}
