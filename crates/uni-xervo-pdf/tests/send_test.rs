//! Compile-time check that `PdfExtractor::extract` and
//! `PdfExt::pdf_extractor` return futures that are `Send` for any lifetime, so callers can spawn it (rustic-ai/uni-xervo#73).
//! Nothing here runs: if the file compiles, the test has passed.

use std::sync::Arc;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo_pdf::{DocExtractPolicy, DocInput, PdfConfig, PdfExt, PdfExtractor};

fn assert_send<T: Send + 'static>(_: T) {}

#[allow(dead_code)]
fn extract_future_is_send(extractor: Arc<PdfExtractor>, input: DocInput, policy: DocExtractPolicy) {
    assert_send(async move {
        let _ = extractor.extract(input, policy).await;
    });
}

#[allow(dead_code)]
fn pdf_extractor_future_is_send(rt: Arc<ModelRuntime>, config: PdfConfig) {
    assert_send(async move {
        let _ = rt.pdf_extractor(config).await;
    });
}

#[test]
fn pdf_futures_compile_as_send() {
    // The assertions are in the functions above, checked by the compiler.
}
