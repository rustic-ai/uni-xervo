//! The acceptance matrix's accelerator check, shared by the real-model suites.
//!
//! A local ONNX model whose accelerator session fails (a missing cuDNN, GPU
//! memory exhausted at load) falls back to CPU and keeps working, so a suite
//! run for `cuda` or `coreml` passes whether or not the accelerator ran.
//! `scripts/acceptance.sh` therefore sets:
//!
//! - `UNI_XERVO_EXPECT_EP`: the execution provider every model should run on
//!   (`cuda`, `coreml`); unset for CPU runs, which makes the check a no-op;
//! - `UNI_XERVO_EXPECT_EP_ALLOW_CPU`: comma-separated aliases that fall back
//!   to CPU by design on that target, such as an OCR recognizer CoreML
//!   computes wrongly.
//!
//! Why a model fell back is a `warn` from the runtime, so each test calls
//! [`log_warnings`] first to print those to stderr.
//!
//! Included with `#[path = "common/accelerator.rs"] mod accelerator;` rather
//! than through `common/mod.rs`, so suites do not compile the HTTP mocks.

use uni_xervo::traits::ModelInfo;

/// Prints the runtime's warnings to stderr for the rest of the process.
///
/// Call it at the start of a test, before any model loads: an accelerator
/// fallback logs its reason (a missing library, GPU memory exhausted, a
/// failed load-time check) at `warn` while the model loads. Later calls, and
/// calls in a process that already has a subscriber, do nothing.
pub fn log_warnings() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();
}

/// The provider `UNI_XERVO_EXPECT_EP` names, if any.
fn expected_provider() -> Option<String> {
    std::env::var("UNI_XERVO_EXPECT_EP")
        .ok()
        .filter(|v| !v.is_empty())
}

/// Whether `alias` is listed in `UNI_XERVO_EXPECT_EP_ALLOW_CPU`.
fn cpu_allowed(alias: &str) -> bool {
    std::env::var("UNI_XERVO_EXPECT_EP_ALLOW_CPU")
        .map(|list| list.split(',').any(|a| a.trim() == alias))
        .unwrap_or(false)
}

/// Prints the providers `model` runs on and checks the expected one leads.
///
/// The line goes to stderr on every call, so a suite log shows where each
/// model ran even when no provider is expected.
///
/// # Panics
///
/// When `UNI_XERVO_EXPECT_EP` is set and `model`'s first active provider is
/// not that provider, unless `alias` is in `UNI_XERVO_EXPECT_EP_ALLOW_CPU`
/// and the model runs on `cpu`.
pub fn expect_accelerator<M: ModelInfo + ?Sized>(alias: &str, model: &M) {
    let active = model.active_execution_providers();
    eprintln!("execution providers for {alias}: {active:?}");
    let Some(expected) = expected_provider() else {
        return;
    };
    let first = active.first().map(String::as_str);
    if first == Some(expected.as_str()) {
        return;
    }
    if first == Some("cpu") && cpu_allowed(alias) {
        eprintln!("{alias} runs on cpu, allowed by UNI_XERVO_EXPECT_EP_ALLOW_CPU");
        return;
    }
    panic!(
        "{alias} runs on {active:?}, expected {expected} first: the accelerator session \
         failed and the model fell back to CPU (the WARN line from uni_xervo above gives \
         the reason, with ONNX Runtime's own E:onnxruntime lines)"
    );
}
