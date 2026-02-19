//! Tests for reranker model operations

mod common;
use common::mock_support::{MockRerankerModel, runtime_with_reranker};
use uni_xervo::traits::RerankerModel;

#[tokio::test]
async fn test_rerank_returns_scored_docs() {
    let model = MockRerankerModel::new();
    let docs = vec!["doc1", "doc2", "doc3"];
    let result = model.rerank("query", &docs).await;

    assert!(result.is_ok());
    let scored = result.unwrap();
    assert_eq!(scored.len(), 3);
}

#[tokio::test]
async fn test_rerank_correct_indices() {
    let model = MockRerankerModel::new();
    let docs = vec!["first", "second", "third"];
    let result = model.rerank("query", &docs).await;

    let scored = result.unwrap();
    assert_eq!(scored[0].index, 0);
    assert_eq!(scored[1].index, 1);
    assert_eq!(scored[2].index, 2);
}

#[tokio::test]
async fn test_rerank_descending_scores() {
    let model = MockRerankerModel::new();
    let docs = vec!["doc1", "doc2", "doc3", "doc4"];
    let result = model.rerank("query", &docs).await;

    let scored = result.unwrap();
    // Scores are 1/(i+1), so 1.0, 0.5, 0.333..., 0.25
    assert_eq!(scored[0].score, 1.0);
    assert_eq!(scored[1].score, 0.5);
    assert!(scored[2].score < 0.5);
    assert!(scored[3].score < scored[2].score);
}

#[tokio::test]
async fn test_rerank_includes_text() {
    let model = MockRerankerModel::new();
    let docs = vec!["hello", "world"];
    let result = model.rerank("query", &docs).await;

    let scored = result.unwrap();
    assert_eq!(scored[0].text, Some("hello".to_string()));
    assert_eq!(scored[1].text, Some("world".to_string()));
}

#[tokio::test]
async fn test_rerank_empty_docs() {
    let model = MockRerankerModel::new();
    let result = model.rerank("query", &[]).await;

    assert!(result.is_ok());
    let scored = result.unwrap();
    assert_eq!(scored.len(), 0);
}

#[tokio::test]
async fn test_rerank_single_doc() {
    let model = MockRerankerModel::new();
    let result = model.rerank("query", &["only doc"]).await;

    assert!(result.is_ok());
    let scored = result.unwrap();
    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].score, 1.0);
}

#[tokio::test]
async fn test_failure_propagation() {
    let model = MockRerankerModel::new().with_failure(true);
    let result = model.rerank("query", &["doc"]).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Mock reranker failure")
    );
}

#[tokio::test]
async fn test_call_counting() {
    let model = MockRerankerModel::new();
    assert_eq!(model.call_count(), 0);

    let _ = model.rerank("query1", &["doc1"]).await;
    assert_eq!(model.call_count(), 1);

    let _ = model.rerank("query2", &["doc2"]).await;
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn test_end_to_end_via_runtime() {
    let runtime = runtime_with_reranker().await.unwrap();
    let model = runtime.reranker("rerank/test").await.unwrap();

    let docs = vec!["document 1", "document 2"];
    let result = model.rerank("test query", &docs).await;

    assert!(result.is_ok());
    let scored = result.unwrap();
    assert_eq!(scored.len(), 2);
    assert!(scored[0].score > scored[1].score);
}
