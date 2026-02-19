//! Reliability primitives: circuit breaker, instrumented model wrappers with
//! timeout and retry support, and metrics emission.

use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, RerankerModel, ScoredDoc,
};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Internal circuit breaker state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed,
    Open,
    HalfOpen,
}

/// Tunable parameters for the circuit breaker.
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before the breaker opens.
    pub failure_threshold: u32,
    /// Seconds to wait in the open state before allowing a probe call.
    pub open_wait_seconds: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_wait_seconds: 10,
        }
    }
}

struct Inner {
    state: State,
    failures: u32,
    last_failure: Option<Instant>,
    config: CircuitBreakerConfig,
    half_open_probe_in_flight: bool,
}

/// Thread-safe circuit breaker that tracks failures and short-circuits calls
/// when a provider is unhealthy.
///
/// State transitions: **Closed** -> (failures >= threshold) -> **Open** ->
/// (wait period elapsed) -> **HalfOpen** -> (probe succeeds) -> **Closed**
/// (or probe fails -> back to **Open**).
#[derive(Clone)]
pub struct CircuitBreakerWrapper {
    inner: Arc<Mutex<Inner>>,
}

impl CircuitBreakerWrapper {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                state: State::Closed,
                failures: 0,
                last_failure: None,
                config,
                half_open_probe_in_flight: false,
            })),
        }
    }

    /// Execute `f` through the circuit breaker.
    ///
    /// Returns [`RuntimeError::Unavailable`] immediately when the breaker is
    /// open.  In the half-open state only a single probe call is allowed;
    /// concurrent callers receive `Unavailable` until the probe completes.
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let is_probe_call;

        // 1. Check state
        {
            let mut inner = self.inner.lock().unwrap();
            match inner.state {
                State::Open => {
                    if let Some(last) = inner.last_failure {
                        if last.elapsed() >= Duration::from_secs(inner.config.open_wait_seconds) {
                            inner.state = State::HalfOpen;
                        } else {
                            return Err(RuntimeError::Unavailable);
                        }
                    }
                }
                State::HalfOpen => {
                    if inner.half_open_probe_in_flight {
                        return Err(RuntimeError::Unavailable);
                    }
                }
                State::Closed => {}
            }
            is_probe_call = inner.state == State::HalfOpen;
            if is_probe_call {
                inner.half_open_probe_in_flight = true;
            }
        }

        // 2. Execute
        let result = f().await;

        // 3. Update state
        let mut inner = self.inner.lock().unwrap();
        match result {
            Ok(val) => {
                if is_probe_call {
                    inner.state = State::Closed;
                    inner.failures = 0;
                    inner.half_open_probe_in_flight = false;
                } else if inner.state == State::Closed {
                    inner.failures = 0;
                }
                Ok(val)
            }
            Err(e) => {
                if is_probe_call {
                    inner.half_open_probe_in_flight = false;
                }
                inner.failures += 1;
                inner.last_failure = Some(Instant::now());

                if is_probe_call
                    || (inner.state == State::Closed
                        && inner.failures >= inner.config.failure_threshold)
                {
                    inner.state = State::Open;
                }
                Err(e)
            }
        }
    }
}

/// Wrapper around an [`EmbeddingModel`] that adds per-call timeout enforcement,
/// exponential-backoff retries for transient errors, and metrics emission
/// (`model_inference.duration_seconds`, `model_inference.total`).
pub struct InstrumentedEmbeddingModel {
    pub inner: Arc<dyn EmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl EmbeddingModel for InstrumentedEmbeddingModel {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.embed(texts.clone());

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying embedding call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "embed",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "embed",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

/// Wrapper around a [`GeneratorModel`] that adds timeout, retry, and metrics.
///
/// See [`InstrumentedEmbeddingModel`] for details on the instrumentation behavior.
pub struct InstrumentedGeneratorModel {
    pub inner: Arc<dyn GeneratorModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl GeneratorModel for InstrumentedGeneratorModel {
    async fn generate(
        &self,
        messages: &[String],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.generate(messages, options.clone());

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying generation call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "generate",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "generate",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

/// Wrapper around a [`RerankerModel`] that adds timeout, retry, and metrics.
///
/// See [`InstrumentedEmbeddingModel`] for details on the instrumentation behavior.
pub struct InstrumentedRerankerModel {
    pub inner: Arc<dyn RerankerModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl RerankerModel for InstrumentedRerankerModel {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.rerank(query, docs);

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying rerank call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "rerank",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "rerank",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_circuit_breaker_transitions() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            open_wait_seconds: 1,
        };
        let cb = CircuitBreakerWrapper::new(config);
        let counter = Arc::new(AtomicU32::new(0));

        // 1. Success calls - state remains Closed
        let res = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(res.is_ok());

        // 2. Failures - state transitions to Open
        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err()); // Fail 1

        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err()); // Fail 2 -> Open

        // 3. Open state - calls rejected immediately
        let res = cb
            .call(|| async {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .await;
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().to_string(), "Unavailable");
        assert_eq!(counter.load(Ordering::SeqCst), 0); // Should not have run

        // 4. Wait for HalfOpen
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // 5. HalfOpen - allow one call
        // If it fails, go back to Open
        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err());

        // Should be Open again
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().to_string(), "Unavailable");

        // 6. Wait again for HalfOpen
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // 7. Success - transition to Closed
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_ok());

        // Should be closed now, next call works
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_half_open_allows_single_probe() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_wait_seconds: 1,
        };
        let cb = CircuitBreakerWrapper::new(config);

        // Open breaker.
        let _ = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;

        tokio::time::sleep(Duration::from_millis(1100)).await;

        let started = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let finished = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let cb_probe = cb.clone();
        let started_probe = started.clone();
        let finished_probe = finished.clone();
        let probe = tokio::spawn(async move {
            cb_probe
                .call(|| async move {
                    started_probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    finished_probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok::<_, RuntimeError>(())
                })
                .await
        });

        // Allow the first probe to enter.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // A concurrent call during half-open probe should fail fast.
        let second = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(matches!(second, Err(RuntimeError::Unavailable)));

        let probe_result = probe.await.unwrap();
        assert!(probe_result.is_ok());
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(finished.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Closed again.
        let res = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(res.is_ok());
    }
}
