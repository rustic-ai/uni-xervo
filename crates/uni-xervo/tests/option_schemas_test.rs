//! The shipped JSON Schemas agree with runtime option validation
//! (rustic-ai/uni-xervo#62).
//!
//! Three checks keep `schemas/` from drifting again:
//!
//! 1. **Keys and tasks.** Each provider document declares, under
//!    `$defs/<task>`, exactly the keys [`schemas::option_keys`] lists, for
//!    exactly the tasks [`schemas::supported_tasks`] lists, and the catalog
//!    schema routes every `(provider_id, task)` pair to that definition.
//! 2. **Capabilities.** Each bundled provider compiled into this build reports
//!    the same tasks from `capabilities()` as the table.
//! 3. **Values.** Generated and hand-written catalog entries get the same
//!    verdict from the catalog schema and from [`schemas::validate_options`].

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};
use uni_xervo::api::ModelTask;
use uni_xervo::schemas;

fn wire(task: ModelTask) -> String {
    serde_json::to_value(task)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

fn task_from_wire(name: &str) -> ModelTask {
    serde_json::from_value(json!(name)).unwrap()
}

fn provider_doc(id: &str) -> Value {
    serde_json::from_str(schemas::provider_options_schema(id).unwrap()).unwrap()
}

fn catalog_doc() -> Value {
    serde_json::from_str(schemas::MODEL_CATALOG).unwrap()
}

fn keys_of(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

// --- 1. keys and tasks --------------------------------------------------------

#[test]
fn provider_schemas_declare_exactly_the_validated_keys() {
    for id in schemas::bundled_providers() {
        let doc = provider_doc(id);
        let tasks = schemas::supported_tasks(id).unwrap();
        let defs = &doc["$defs"];

        let declared: BTreeSet<String> = keys_of(defs);
        let expected: BTreeSet<String> = tasks.iter().map(|t| wire(*t)).collect();
        assert_eq!(declared, expected, "{id}: $defs tasks");

        let mut union = BTreeSet::new();
        for task in &tasks {
            let keys: BTreeSet<String> = schemas::option_keys(id, *task)
                .unwrap()
                .into_iter()
                .map(str::to_string)
                .collect();
            let def = &defs[wire(*task)];
            assert_eq!(
                keys_of(&def["properties"]),
                keys,
                "{id} {}: keys",
                wire(*task)
            );
            assert_eq!(
                def["additionalProperties"],
                json!(false),
                "{id} {}",
                wire(*task)
            );
            union.extend(keys);
        }
        assert_eq!(keys_of(&doc["properties"]), union, "{id}: top-level keys");
        assert_eq!(doc["additionalProperties"], json!(false), "{id}");
    }
}

#[test]
fn catalog_schema_routes_every_provider_and_task() {
    let cat = catalog_doc();
    let item = &cat["items"];

    let all_tasks: BTreeSet<String> = item["properties"]["task"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    for name in &all_tasks {
        task_from_wire(name); // every listed name is a real task
    }

    let rules = item["allOf"].as_array().unwrap();
    let ids: BTreeSet<&str> = schemas::bundled_providers().collect();
    let routed: BTreeSet<&str> = rules
        .iter()
        .map(|r| {
            r["if"]["properties"]["provider_id"]["const"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(routed, ids, "one rule per bundled provider");

    for rule in rules {
        let id = rule["if"]["properties"]["provider_id"]["const"]
            .as_str()
            .unwrap();
        let tasks: BTreeSet<String> = schemas::supported_tasks(id)
            .unwrap()
            .into_iter()
            .map(wire)
            .collect();
        let allowed: BTreeSet<String> = rule["then"]["properties"]["task"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(allowed, tasks, "{id}: task enum");
        assert!(tasks.is_subset(&all_tasks), "{id}");

        let provider_id = provider_doc(id)["$id"].as_str().unwrap().to_string();
        let file_name = provider_id.rsplit('/').next().unwrap().to_string();
        let routed_tasks: BTreeSet<String> = rule["then"]["allOf"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let task = r["if"]["properties"]["task"]["const"].as_str().unwrap();
                let opts = &r["then"]["properties"]["options"];
                let target = opts["$ref"]
                    .as_str()
                    .or_else(|| opts["anyOf"][0]["$ref"].as_str())
                    .unwrap();
                assert_eq!(
                    target,
                    format!("./provider-options/{file_name}#/$defs/{task}"),
                    "{id} {task}"
                );
                task.to_string()
            })
            .collect();
        assert_eq!(routed_tasks, tasks, "{id}: per-task routes");
    }
}

// --- 2. capabilities ----------------------------------------------------------

#[allow(unused_mut)]
#[test]
fn table_matches_provider_capabilities() {
    use uni_xervo::traits::ModelProvider;
    let mut providers: Vec<(&str, Box<dyn ModelProvider>)> = Vec::new();
    #[cfg(feature = "provider-openai")]
    providers.push((
        "remote/openai",
        Box::new(uni_xervo::provider::RemoteOpenAIProvider::new()),
    ));
    #[cfg(feature = "provider-mistral")]
    providers.push((
        "remote/mistral",
        Box::new(uni_xervo::provider::RemoteMistralProvider::new()),
    ));
    #[cfg(feature = "provider-voyageai")]
    providers.push((
        "remote/voyageai",
        Box::new(uni_xervo::provider::RemoteVoyageAIProvider::new()),
    ));
    #[cfg(feature = "provider-gemini")]
    providers.push((
        "remote/gemini",
        Box::new(uni_xervo::provider::RemoteGeminiProvider::new()),
    ));
    #[cfg(feature = "provider-anthropic")]
    providers.push((
        "remote/anthropic",
        Box::new(uni_xervo::provider::RemoteAnthropicProvider::new()),
    ));
    #[cfg(feature = "provider-cohere")]
    providers.push((
        "remote/cohere",
        Box::new(uni_xervo::provider::RemoteCohereProvider::new()),
    ));
    #[cfg(feature = "provider-azure-openai")]
    providers.push((
        "remote/azure-openai",
        Box::new(uni_xervo::provider::RemoteAzureOpenAIProvider::new()),
    ));
    #[cfg(feature = "provider-vertexai")]
    providers.push((
        "remote/vertexai",
        Box::new(uni_xervo::provider::RemoteVertexAIProvider::new()),
    ));
    #[cfg(feature = "provider-llamacpp")]
    providers.push((
        "remote/llamacpp",
        Box::new(uni_xervo::provider::RemoteLlamaCppProvider::new()),
    ));
    #[cfg(feature = "provider-candle")]
    providers.push((
        "local/candle",
        Box::new(uni_xervo::provider::LocalCandleProvider::new()),
    ));
    #[cfg(any(feature = "provider-onnx", feature = "provider-onnx-dynamic"))]
    providers.push((
        "local/onnx",
        Box::new(uni_xervo::provider::LocalOnnxProvider::new()),
    ));
    #[cfg(feature = "provider-whisper-cpp")]
    providers.push((
        "local/whisper-cpp",
        Box::new(uni_xervo::provider::LocalWhisperCppProvider::new()),
    ));

    for (id, provider) in providers {
        assert_eq!(provider.provider_id(), id);
        let reported: BTreeSet<String> = provider
            .capabilities()
            .supported_tasks
            .into_iter()
            .map(wire)
            .collect();
        let table: BTreeSet<String> = schemas::supported_tasks(id)
            .unwrap()
            .into_iter()
            .map(wire)
            .collect();
        assert_eq!(reported, table, "{id}: capabilities vs table");
    }
}

// --- 3. values ----------------------------------------------------------------

struct Checker {
    validator: jsonschema::Validator,
}

impl Checker {
    fn new() -> Self {
        let mut registry = jsonschema::Registry::new();
        for id in schemas::bundled_providers() {
            let doc = provider_doc(id);
            let uri = doc["$id"].as_str().unwrap().to_string();
            registry = registry.add(uri, doc).unwrap();
        }
        let registry = registry.prepare().unwrap();
        let validator = jsonschema::options()
            .offline()
            .with_registry(&registry)
            .build(&catalog_doc())
            .unwrap();
        Self { validator }
    }

    /// Assert the schema and the runtime agree on one entry; returns the verdict.
    fn agree(&self, provider_id: &str, task: &str, options: Value) -> bool {
        let entry = json!([{
            "alias": "test/alias",
            "task": task,
            "provider_id": provider_id,
            "model_id": "owner/model",
            "options": options,
        }]);
        let schema_ok = self.validator.is_valid(&entry);
        let runtime = schemas::validate_options(provider_id, task_from_wire(task), &options);
        assert_eq!(
            schema_ok,
            runtime.is_ok(),
            "{provider_id} {task} {options}: schema says {}, runtime says {:?}",
            if schema_ok { "valid" } else { "invalid" },
            runtime.err().map(|e| e.to_string())
        );
        schema_ok
    }

    fn accepts(&self, provider_id: &str, task: &str, options: Value) {
        assert!(
            self.agree(provider_id, task, options.clone()),
            "{provider_id} {task} should accept {options}"
        );
    }

    fn rejects(&self, provider_id: &str, task: &str, options: Value) {
        assert!(
            !self.agree(provider_id, task, options.clone()),
            "{provider_id} {task} should reject {options}"
        );
    }
}

/// The property schema a task uses for `key`, following a local `$ref`.
fn key_schema<'a>(doc: &'a Value, task: &str, key: &str) -> &'a Value {
    let prop = &doc["$defs"][task]["properties"][key];
    match prop["$ref"].as_str() {
        Some(r) => &doc["properties"][r.trim_start_matches("#/properties/")],
        None => prop,
    }
}

/// A value the property schema accepts.
fn valid_sample(schema: &Value) -> Value {
    if let Some(values) = schema["enum"].as_array() {
        return values[0].clone();
    }
    match schema["type"].as_str().unwrap() {
        "string" if schema.get("pattern").is_some() => json!("http://127.0.0.1:8080/v1"),
        "string" => json!("value"),
        "integer" => json!(schema["minimum"].as_i64().unwrap_or(0).max(1)),
        "number" => json!(0.5),
        "boolean" => json!(true),
        "array" => json!([schema["items"]["enum"][0].clone()]),
        other => panic!("no sample for type {other}"),
    }
}

/// Values the property schema rejects: a wrong type, and a bound or enum miss.
fn invalid_samples(schema: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    match schema["type"].as_str().unwrap() {
        "string" => {
            out.push(json!(7));
            if schema.get("enum").is_some() {
                out.push(json!("not-a-listed-value"));
            }
            if schema.get("pattern").is_some() {
                out.push(json!("127.0.0.1:8080/v1"));
            }
        }
        "integer" => {
            out.push(json!("12"));
            out.push(json!(schema["minimum"].as_i64().unwrap() - 1));
        }
        "number" => {
            out.push(json!("0.5"));
            out.push(json!(0.0));
            if schema.get("exclusiveMaximum").is_some() {
                out.push(json!(1.0));
            }
        }
        "boolean" => out.push(json!("true")),
        "array" => {
            out.push(json!("cpu"));
            out.push(json!(["not-an-ep"]));
        }
        other => panic!("no invalid sample for type {other}"),
    }
    out
}

/// The smallest options object a task accepts: its required keys only.
fn base_options(doc: &Value, task: &str) -> Map<String, Value> {
    let mut base = Map::new();
    for key in doc["$defs"][task]["required"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let key = key.as_str().unwrap();
        base.insert(key.to_string(), valid_sample(key_schema(doc, task, key)));
    }
    base
}

#[test]
fn schema_and_runtime_agree_on_generated_cases() {
    let checker = Checker::new();
    let mut cases = 0usize;
    for id in schemas::bundled_providers() {
        let doc = provider_doc(id);
        let tasks = schemas::supported_tasks(id).unwrap();
        for task in &tasks {
            let t = wire(*task);
            let keys = schemas::option_keys(id, *task).unwrap();
            let base = base_options(&doc, &t);
            checker.accepts(id, &t, Value::Object(base.clone()));
            cases += 1;

            // Every key alone with a valid value, then each invalid value.
            for key in &keys {
                let schema = key_schema(&doc, &t, key);
                let mut opts = base.clone();
                opts.insert(key.to_string(), valid_sample(schema));
                checker.accepts(id, &t, Value::Object(opts));
                // A whole-number float is the same JSON number (`8.0` = `8`).
                if schema["type"] == "integer" {
                    let mut opts = base.clone();
                    let n = valid_sample(schema).as_i64().unwrap();
                    opts.insert(key.to_string(), json!(n as f64));
                    checker.accepts(id, &t, Value::Object(opts));
                    let mut opts = base.clone();
                    opts.insert(key.to_string(), json!(n as f64 + 0.5));
                    checker.rejects(id, &t, Value::Object(opts));
                    cases += 2;
                }
                for bad in invalid_samples(schema) {
                    let mut opts = base.clone();
                    opts.insert(key.to_string(), bad);
                    checker.rejects(id, &t, Value::Object(opts));
                    cases += 1;
                }
                cases += 1;
            }

            // All keys at once (kniv v1 and v7 keys cannot mix; see below).
            if t != "nlp" {
                let mut all = base.clone();
                for key in &keys {
                    all.insert(key.to_string(), valid_sample(key_schema(&doc, &t, key)));
                }
                checker.accepts(id, &t, Value::Object(all));
                cases += 1;
            }

            // A key another task of the provider owns, and a key nobody owns.
            for other in tasks.iter().filter(|o| *o != task) {
                for key in schemas::option_keys(id, *other).unwrap() {
                    if keys.contains(&key) {
                        continue;
                    }
                    let mut opts = base.clone();
                    opts.insert(
                        key.to_string(),
                        valid_sample(key_schema(&doc, &wire(*other), key)),
                    );
                    checker.rejects(id, &t, Value::Object(opts));
                    cases += 1;
                }
            }
            let mut opts = base.clone();
            opts.insert("no_such_option".into(), json!("x"));
            checker.rejects(id, &t, Value::Object(opts));
            cases += 1;
        }

        // A task the provider does not support.
        let unsupported = [
            "embed",
            "rerank",
            "generate",
            "raw",
            "transcribe",
            "embed_audio",
        ]
        .into_iter()
        .find(|name| !tasks.contains(&task_from_wire(name)))
        .unwrap();
        checker.rejects(id, unsupported, Value::Null);
        cases += 1;
    }
    assert!(cases > 500, "generated {cases} cases");
}

#[test]
fn schema_and_runtime_agree_on_cross_key_rules() {
    let c = Checker::new();

    // Null options: fine everywhere except llama.cpp, which needs base_url.
    c.accepts("remote/openai", "embed", Value::Null);
    c.accepts("local/onnx", "ocr", Value::Null);
    c.rejects("remote/llamacpp", "generate", Value::Null);
    c.rejects("remote/llamacpp", "generate", json!({}));
    c.rejects("remote/openai", "embed", json!("not an object"));

    // llama.cpp embed needs its token budget and width; 3 is the floor.
    let url = "http://127.0.0.1:8080/v1";
    c.rejects("remote/llamacpp", "embed", json!({ "base_url": url }));
    c.rejects(
        "remote/llamacpp",
        "embed",
        json!({ "base_url": url, "max_input_tokens": 512 }),
    );
    c.accepts(
        "remote/llamacpp",
        "embed",
        json!({ "base_url": url, "max_input_tokens": 3, "embedding_dimensions": 384 }),
    );
    c.rejects(
        "remote/llamacpp",
        "embed",
        json!({ "base_url": url, "max_input_tokens": 2, "embedding_dimensions": 384 }),
    );

    // URLs: an absolute http(s) URL, leading whitespace tolerated.
    c.accepts(
        "remote/openai",
        "generate",
        json!({ "base_url": "https://x.test/v1" }),
    );
    c.accepts(
        "remote/openai",
        "generate",
        json!({ "base_url": "  http://x.test" }),
    );
    c.rejects("remote/openai", "generate", json!({ "base_url": "" }));
    c.rejects(
        "remote/openai",
        "generate",
        json!({ "base_url": "ftp://x.test" }),
    );

    // embedding_dimensions is embed-only on every remote provider.
    c.rejects(
        "remote/cohere",
        "rerank",
        json!({ "embedding_dimensions": 512 }),
    );

    // ONNX: the two meanings of `style` stay with their task.
    c.accepts("local/onnx", "rerank", json!({ "style": "generative" }));
    c.rejects("local/onnx", "rerank", json!({ "style": "mineru" }));
    c.accepts(
        "local/onnx",
        "document_extract",
        json!({ "style": "mineru" }),
    );
    c.rejects(
        "local/onnx",
        "document_extract",
        json!({ "style": "generative" }),
    );

    // ONNX: an empty execution-provider list is allowed.
    c.accepts("local/onnx", "embed", json!({ "execution_providers": [] }));
    c.accepts(
        "local/onnx",
        "embed",
        json!({ "execution_providers": ["cuda", "cpu"] }),
    );

    // OCR detection thresholds sit strictly inside (0, 1).
    c.accepts("local/onnx", "ocr", json!({ "det_bin_threshold": 0.99 }));
    c.rejects("local/onnx", "ocr", json!({ "det_bin_threshold": 1 }));

    // kniv NLP: v1 and v7 keys never mix, and a declared generation rules
    // out the other's keys.
    c.accepts(
        "local/onnx",
        "nlp",
        json!({ "onnx_path": "a", "max_seq_len": 64 }),
    );
    c.accepts(
        "local/onnx",
        "nlp",
        json!({ "variant": "fp16", "generation": "v7" }),
    );
    c.rejects(
        "local/onnx",
        "nlp",
        json!({ "variant": "fp16", "onnx_path": "a" }),
    );
    c.rejects(
        "local/onnx",
        "nlp",
        json!({ "generation": "v7", "max_seq_len": 64 }),
    );
    c.rejects(
        "local/onnx",
        "nlp",
        json!({ "generation": "v1", "label_bank_path": "b" }),
    );
    c.rejects("local/onnx", "nlp", json!({ "generation": "v9" }));

    // Third-party providers: anything goes for options, objects or not.
    c.accepts("acme/custom", "embed", json!({ "whatever": [1, 2, 3] }));
    c.accepts("acme/custom", "embed", json!("opaque-string"));
    c.accepts("acme/custom", "embed", json!([1, 2]));
    c.accepts("acme/custom", "embed", Value::Null);

    // Whole-number floats count as integers, as Python's json writes them.
    c.accepts("local/onnx", "embed", json!({ "max_seq_len": 128.0 }));
    c.accepts(
        "remote/llamacpp",
        "embed",
        json!({ "base_url": url, "max_input_tokens": 512.0, "embedding_dimensions": 384.0 }),
    );
    c.rejects("local/onnx", "embed", json!({ "max_seq_len": 0.0 }));
}
