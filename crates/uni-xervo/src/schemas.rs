// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Machine-readable descriptions of what a catalog may contain, for
//! configuration tools that should not duplicate this crate's validation.
//!
//! Two views of the same rules, both available without building a runtime,
//! registering a provider, or enabling any provider feature:
//!
//! - **JSON Schema documents** (draft 2020-12), embedded from this crate's
//!   `schemas/` directory: [`MODEL_CATALOG`] for a whole catalog, and one
//!   document per bundled provider from [`provider_options_schema`]. In a
//!   provider document, `$defs/<task>` holds the options schema for one task
//!   (the catalog schema routes each `(provider_id, task)` pair there), and
//!   the top-level schema accepts the keys of any of the provider's tasks.
//!   The catalog schema's `$ref`s are relative, so register the documents
//!   under their `$id`s with your validator.
//! - **The `(provider, task)` table** the runtime validates against:
//!   [`bundled_providers`], [`supported_tasks`] and [`option_keys`], plus
//!   [`validate_options`] to run the runtime's own check on one alias.
//!
//! The documents are the ones this crate version validates against, so a
//! tool built against a given `uni-xervo` version gets matching schemas.
//! Tests keep the two views in step: each provider document declares exactly
//! [`option_keys`] per task, and fixture cases are checked against both the
//! schemas and the runtime validator.
//!
//! Providers registered by third parties are not described here. The runtime
//! accepts any options for them and leaves validation to the provider's own
//! `load`; the catalog schema likewise constrains `options` only for bundled
//! provider ids.

use crate::api::ModelTask;
use crate::error::Result;
use crate::options_validation::{
    BUNDLED_PROVIDERS, bundled_provider, normalize_whole_floats, validate_provider_options,
};

/// JSON Schema for a model catalog: an array of alias specs.
pub const MODEL_CATALOG: &str = include_str!("../schemas/model-catalog.schema.json");

/// Each bundled provider's options schema, keyed by provider id.
const PROVIDER_OPTION_SCHEMAS: &[(&str, &str)] = &[
    (
        "remote/openai",
        include_str!("../schemas/provider-options/openai.schema.json"),
    ),
    (
        "remote/mistral",
        include_str!("../schemas/provider-options/mistral.schema.json"),
    ),
    (
        "remote/voyageai",
        include_str!("../schemas/provider-options/voyageai.schema.json"),
    ),
    (
        "remote/gemini",
        include_str!("../schemas/provider-options/gemini.schema.json"),
    ),
    (
        "remote/anthropic",
        include_str!("../schemas/provider-options/anthropic.schema.json"),
    ),
    (
        "remote/cohere",
        include_str!("../schemas/provider-options/cohere.schema.json"),
    ),
    (
        "remote/azure-openai",
        include_str!("../schemas/provider-options/azure-openai.schema.json"),
    ),
    (
        "remote/vertexai",
        include_str!("../schemas/provider-options/vertexai.schema.json"),
    ),
    (
        "remote/llamacpp",
        include_str!("../schemas/provider-options/llamacpp.schema.json"),
    ),
    (
        "local/candle",
        include_str!("../schemas/provider-options/candle.schema.json"),
    ),
    (
        "local/onnx",
        include_str!("../schemas/provider-options/onnx.schema.json"),
    ),
    (
        "local/whisper-cpp",
        include_str!("../schemas/provider-options/whisper-cpp.schema.json"),
    ),
];

/// Ids of the providers this crate ships, whether or not their feature is
/// enabled in this build.
pub fn bundled_providers() -> impl Iterator<Item = &'static str> {
    BUNDLED_PROVIDERS.iter().map(|p| p.id)
}

/// The options JSON Schema for a bundled provider, or `None` for an id this
/// crate does not ship.
#[must_use]
pub fn provider_options_schema(provider_id: &str) -> Option<&'static str> {
    PROVIDER_OPTION_SCHEMAS
        .iter()
        .find(|(id, _)| *id == provider_id)
        .map(|(_, schema)| *schema)
}

/// The tasks a bundled provider supports, or `None` for an id this crate does
/// not ship. A catalog entry naming any other task is rejected.
#[must_use]
pub fn supported_tasks(provider_id: &str) -> Option<Vec<ModelTask>> {
    bundled_provider(provider_id).map(|p| p.tasks.iter().map(|t| t.task).collect())
}

/// Every option key a bundled provider accepts for `task`, or `None` when the
/// provider is not bundled or does not support the task. Keys outside this
/// list are rejected.
#[must_use]
pub fn option_keys(provider_id: &str, task: ModelTask) -> Option<Vec<&'static str>> {
    let provider = bundled_provider(provider_id)?;
    let task_options = provider.task(task)?;
    Some(
        provider
            .common
            .iter()
            .chain(task_options.keys)
            .copied()
            .collect(),
    )
}

/// Check one alias's `options` the way the runtime does when a catalog is
/// built or an alias registered, without building a runtime.
///
/// For a bundled provider this rejects an unsupported task, unknown or
/// misplaced keys, and bad values. Options for any other provider id are
/// accepted, as the runtime accepts them. A whole-number float such as `128.0`
/// counts as the integer `128`, as it does at runtime.
///
/// # Errors
/// [`RuntimeError::Config`](crate::error::RuntimeError::Config) naming the
/// offending task, key or value.
pub fn validate_options(
    provider_id: &str,
    task: ModelTask,
    options: &serde_json::Value,
) -> Result<()> {
    let mut options = options.clone();
    normalize_whole_floats(&mut options);
    validate_provider_options(provider_id, task, &options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_provider_has_a_schema_and_back() {
        for id in bundled_providers() {
            assert!(provider_options_schema(id).is_some(), "no schema for {id}");
        }
        for (id, _) in PROVIDER_OPTION_SCHEMAS {
            assert!(bundled_provider(id).is_some(), "schema for unknown {id}");
        }
    }

    #[test]
    fn lookups_cover_tasks_and_keys() {
        assert_eq!(
            supported_tasks("remote/anthropic"),
            Some(vec![ModelTask::Generate])
        );
        let embed = option_keys("remote/openai", ModelTask::Embed).unwrap();
        assert!(embed.contains(&"embedding_dimensions"));
        let generate = option_keys("remote/openai", ModelTask::Generate).unwrap();
        assert!(!generate.contains(&"embedding_dimensions"));
        assert!(option_keys("remote/openai", ModelTask::Rerank).is_none());
        assert!(supported_tasks("acme/custom").is_none());
    }
}
