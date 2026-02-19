//! Tests for ModelAliasSpec validation and ModelRuntimeKey behavior

use uni_xervo::api::{ModelAliasSpec, ModelRuntimeKey, ModelTask, RetryConfig, WarmupPolicy};

#[test]
fn test_alias_validation_empty() {
    let spec = ModelAliasSpec {
        alias: "".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let result = spec.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot be empty"));
}

#[test]
fn test_alias_validation_no_slash() {
    let spec = ModelAliasSpec {
        alias: "nodivider".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let result = spec.validate();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("task/name' format")
    );
}

#[test]
fn test_alias_validation_valid() {
    let spec = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    assert!(spec.validate().is_ok());
}

#[test]
fn test_alias_validation_timeout_must_be_positive() {
    let spec = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: Some(0),
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let err = spec.validate();
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("Inference timeout"));
}

#[test]
fn test_alias_validation_load_timeout_must_be_positive() {
    let spec = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: Some(0),
        retry: None,
        options: serde_json::Value::Null,
    };

    let err = spec.validate();
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("Load timeout"));
}

#[test]
fn test_runtime_key_determinism() {
    let spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"key": "value"}),
    };

    let spec2 = ModelAliasSpec {
        alias: "embed/other".to_string(), // Different alias
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"key": "value"}),
    };

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);

    // Same runtime key despite different alias
    assert_eq!(key1, key2);
}

#[test]
fn test_runtime_key_option_order_independence() {
    let spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"a": "1", "b": "2"}),
    };

    let spec2 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"b": "2", "a": "1"}), // Different order
    };

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);

    // Same hash despite different JSON key order
    assert_eq!(key1, key2);
}

#[test]
fn test_runtime_key_different_tasks() {
    let spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let spec2 = ModelAliasSpec {
        alias: "rerank/test".to_string(),
        task: ModelTask::Rerank,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);

    assert_ne!(key1, key2);
}

#[test]
fn test_runtime_key_different_revisions() {
    let spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: Some("v1".to_string()),
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let spec2 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: Some("v2".to_string()),
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);

    assert_ne!(key1, key2);
}

#[test]
fn test_serde_roundtrip() {
    let spec = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "mock/test".to_string(),
        model_id: "test-model".to_string(),
        revision: Some("v1.0".to_string()),
        warmup: WarmupPolicy::Eager,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"cache_dir": "/tmp"}),
    };

    let json = serde_json::to_string(&spec).unwrap();
    let deserialized: ModelAliasSpec = serde_json::from_str(&json).unwrap();

    assert_eq!(spec, deserialized);
}

#[test]
fn test_serde_default_values() {
    let json = r#"{
        "alias": "embed/test",
        "task": "embed",
        "provider_id": "mock/test",
        "model_id": "test-model"
    }"#;

    let spec: ModelAliasSpec = serde_json::from_str(json).unwrap();

    assert_eq!(spec.warmup, WarmupPolicy::Lazy);
    assert!(!spec.required);
    // Default value for options is Null, not an empty object
    assert_eq!(spec.options, serde_json::Value::Null);
    assert!(spec.revision.is_none());
}

#[test]
fn test_task_serialization() {
    assert_eq!(
        serde_json::to_string(&ModelTask::Embed).unwrap(),
        r#""embed""#
    );
    assert_eq!(
        serde_json::to_string(&ModelTask::Rerank).unwrap(),
        r#""rerank""#
    );
    assert_eq!(
        serde_json::to_string(&ModelTask::Generate).unwrap(),
        r#""generate""#
    );
}

#[test]
fn test_warmup_policy_serialization() {
    assert_eq!(
        serde_json::to_string(&WarmupPolicy::Eager).unwrap(),
        r#""eager""#
    );
    assert_eq!(
        serde_json::to_string(&WarmupPolicy::Lazy).unwrap(),
        r#""lazy""#
    );
    assert_eq!(
        serde_json::to_string(&WarmupPolicy::Background).unwrap(),
        r#""background""#
    );
}

#[test]
fn test_retry_config_backoff() {
    let config = RetryConfig {
        max_attempts: 3,
        initial_backoff_ms: 100,
    };
    assert_eq!(config.get_backoff(1).as_millis(), 100);
    assert_eq!(config.get_backoff(2).as_millis(), 200);
    assert_eq!(config.get_backoff(3).as_millis(), 400);
}

#[test]
fn test_warmup_policy_display() {
    assert_eq!(WarmupPolicy::Eager.to_string(), "eager");
    assert_eq!(WarmupPolicy::Lazy.to_string(), "lazy");
    assert_eq!(WarmupPolicy::Background.to_string(), "background");
}

#[test]
fn test_runtime_key_different_options() {
    let spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"key": "value1"}),
    };

    let spec2 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({"key": "value2"}),
    };

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);

    assert_ne!(key1, key2);
}

#[test]
fn test_runtime_key_non_object_options_distinct() {
    let mut spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };
    let mut spec2 = spec1.clone();
    spec2.options = serde_json::json!(true);
    let mut spec3 = spec1.clone();
    spec3.options = serde_json::json!(["x", 1]);
    spec1.options = serde_json::json!("x");

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);
    let key3 = ModelRuntimeKey::new(&spec3);

    assert_ne!(key1, key2);
    assert_ne!(key1, key3);
    assert_ne!(key2, key3);
}

#[test]
fn test_runtime_key_nested_object_order_independence() {
    let mut spec1 = ModelAliasSpec {
        alias: "embed/test".to_string(),
        task: ModelTask::Embed,
        provider_id: "test".to_string(),
        model_id: "model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "outer": {
                "b": [3, 2, 1],
                "a": {"y": 2, "x": 1}
            }
        }),
    };
    let mut spec2 = spec1.clone();
    spec2.options = serde_json::json!({
        "outer": {
            "a": {"x": 1, "y": 2},
            "b": [3, 2, 1]
        }
    });

    let key1 = ModelRuntimeKey::new(&spec1);
    let key2 = ModelRuntimeKey::new(&spec2);
    assert_eq!(key1, key2);

    spec1.options = serde_json::json!({
        "outer": {
            "a": {"x": 1, "y": 999},
            "b": [3, 2, 1]
        }
    });
    let key3 = ModelRuntimeKey::new(&spec1);
    assert_ne!(key2, key3);
}
