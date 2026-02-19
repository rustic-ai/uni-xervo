/// Build script that validates feature-flag / target-platform compatibility.
///
/// Currently checks that the `gpu-metal` feature is only enabled on Apple
/// platforms (macOS / iOS), since the underlying Objective-C framework crates
/// cannot compile elsewhere.
fn main() {
    // gpu-metal requires Apple platforms (macOS or iOS).
    // candle-core, fastembed, and mistralrs all depend on Objective-C framework
    // crates (objc2-foundation, candle-metal-kernels) that only compile on Apple
    // targets. Fail fast with a clear message rather than letting the linker or
    // the Objective-C runtime code surface a cryptic error.
    if std::env::var("CARGO_FEATURE_GPU_METAL").is_ok() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os != "macos" && target_os != "ios" {
            panic!(
                "The `gpu-metal` feature is only supported on macOS and iOS.\n\
                 Remove `gpu-metal` from your feature list when building for `{target_os}`."
            );
        }
    }
}
