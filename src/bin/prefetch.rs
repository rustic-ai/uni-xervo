//! `uni-prefetch` — pre-download and cache models from a catalog JSON file.
//!
//! Usage:
//!
//! ```text
//! uni-prefetch <catalog.json> [--cache-dir <path>] [--dry-run]
//! ```
//!
//! Models with remote providers (`remote/openai`, `remote/gemini`, `remote/vertexai`,
//! `remote/mistral`, `remote/anthropic`, `remote/voyageai`, `remote/cohere`,
//! `remote/azure-openai`, etc.) are skipped
//! because they have no local weights to cache.
//!
//! If a model is not pre-cached the runtime will still download it on first use —
//! this tool is purely an optimisation for pre-warming / bundling.

use std::collections::HashSet;
use std::process;
use uni_xervo::api::{ModelAliasSpec, WarmupPolicy, catalog_from_file};
use uni_xervo::runtime::ModelRuntime;

fn print_usage() {
    eprintln!("Usage: uni-prefetch <catalog.json> [OPTIONS]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <catalog.json>      Path to the model catalog JSON file");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --cache-dir <path>  Override the cache root directory");
    eprintln!("                      (also settable via UNI_CACHE_DIR env var)");
    eprintln!("  --dry-run           Show what would be downloaded without doing it");
    eprintln!("  --help              Show this message");
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    // --- Argument parsing ---------------------------------------------------
    let mut args = std::env::args().skip(1).peekable();
    let mut catalog_path: Option<String> = None;
    let mut cache_dir: Option<String> = None;
    let mut dry_run = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--dry-run" => dry_run = true,
            "--cache-dir" => {
                cache_dir = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--cache-dir requires a path argument"))?,
                );
            }
            _ if arg.starts_with('-') => {
                anyhow::bail!("Unknown option: {arg}");
            }
            _ => {
                if catalog_path.is_some() {
                    anyhow::bail!("Unexpected argument: {arg}");
                }
                catalog_path = Some(arg);
            }
        }
    }

    let catalog_path = catalog_path.ok_or_else(|| {
        print_usage();
        anyhow::anyhow!("Missing required argument: <catalog.json>")
    })?;

    // --- Cache root ---------------------------------------------------------
    if let Some(ref dir) = cache_dir {
        // SAFETY: single-threaded at this point, before tokio spawns tasks
        unsafe { std::env::set_var(uni_xervo::cache::CACHE_ROOT_ENV, dir) };
        println!("cache root : {dir}");
    } else if let Ok(dir) = std::env::var(uni_xervo::cache::CACHE_ROOT_ENV) {
        println!("cache root : {dir}  (from UNI_CACHE_DIR)");
    } else {
        println!("cache root : .uni_cache  (default)");
    }

    // --- Load catalog -------------------------------------------------------
    let all_specs = catalog_from_file(&catalog_path)
        .map_err(|e| anyhow::anyhow!("Failed to load catalog '{catalog_path}': {e}"))?;

    println!(
        "catalog    : {} model(s) from {catalog_path}\n",
        all_specs.len()
    );

    // --- Split local vs remote ----------------------------------------------
    let (local_specs, remote_specs): (Vec<_>, Vec<_>) = all_specs
        .into_iter()
        .partition(|s| s.provider_id.starts_with("local/"));

    for spec in &remote_specs {
        println!(
            "  skip  {}  ({})  — remote provider, nothing to cache",
            spec.alias, spec.provider_id
        );
    }

    if local_specs.is_empty() {
        println!("\nNo local models to prefetch.");
        return Ok(());
    }

    // --- Dry run ------------------------------------------------------------
    if dry_run {
        println!("\nDry run — would download (paths are based on spec model_id; some providers");
        println!("resolve to a canonical HF repo ID internally, so the actual path may differ):\n");
        for spec in &local_specs {
            let cache = uni_xervo::cache::resolve_cache_dir(
                spec.provider_id.trim_start_matches("local/"),
                &spec.model_id,
                &spec.options,
            );
            println!(
                "  {}  ({})  →  {}",
                spec.alias,
                spec.model_id,
                cache.display()
            );
        }
        return Ok(());
    }

    // --- Register providers (only those compiled in) ------------------------
    #[allow(unused_mut)]
    let mut builder = ModelRuntime::builder();
    #[allow(unused_mut)]
    let mut registered: HashSet<String> = HashSet::new();

    // Collect unique provider IDs so each is registered exactly once.
    let mut unique_providers: Vec<&str> = local_specs
        .iter()
        .map(|s| s.provider_id.as_str())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_providers.sort(); // deterministic order

    for provider_id in unique_providers {
        match provider_id {
            "local/candle" => {
                #[cfg(feature = "provider-candle")]
                {
                    use uni_xervo::provider::candle::LocalCandleProvider;
                    builder = builder.register_provider(LocalCandleProvider::new());
                    registered.insert(provider_id.to_string());
                }
                #[cfg(not(feature = "provider-candle"))]
                eprintln!(
                    "  warn  {provider_id}: compiled without provider-candle feature, skipping"
                );
            }
            "local/fastembed" => {
                #[cfg(feature = "provider-fastembed")]
                {
                    use uni_xervo::provider::fastembed::LocalFastEmbedProvider;
                    builder = builder.register_provider(LocalFastEmbedProvider::new());
                    registered.insert(provider_id.to_string());
                }
                #[cfg(not(feature = "provider-fastembed"))]
                eprintln!(
                    "  warn  {provider_id}: compiled without provider-fastembed feature, skipping"
                );
            }
            "local/mistralrs" => {
                #[cfg(feature = "provider-mistralrs")]
                {
                    use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
                    builder = builder.register_provider(LocalMistralRsProvider::new());
                    registered.insert(provider_id.to_string());
                }
                #[cfg(not(feature = "provider-mistralrs"))]
                eprintln!(
                    "  warn  {provider_id}: compiled without provider-mistralrs feature, skipping"
                );
            }
            other => {
                eprintln!("  warn  {other}: unknown local provider, skipping");
            }
        }
    }

    // --- Build eager catalog ------------------------------------------------
    // Filter to registered providers and force Eager so build() downloads synchronously.
    let eager_specs: Vec<ModelAliasSpec> = local_specs
        .into_iter()
        .filter(|s| registered.contains(&s.provider_id))
        .map(|mut s| {
            s.warmup = WarmupPolicy::Eager;
            s
        })
        .collect();

    if eager_specs.is_empty() {
        println!("\nNo providers available for the requested models.");
        return Ok(());
    }

    println!("Prefetching {} model(s):", eager_specs.len());
    for spec in &eager_specs {
        println!("  →  {}  ({})", spec.alias, spec.model_id);
    }
    println!();

    builder
        .catalog(eager_specs)
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Prefetch failed: {e}"))?;

    println!("\nAll models cached successfully.");
    Ok(())
}
