//! Minimal in-process HTTP/1.1 server that impersonates a llama.cpp
//! `llama-server` for the `remote/llamacpp` provider tests.
//!
//! Routes `POST /tokenize` and `POST /v1/embeddings` to configurable handlers,
//! records every request (method, path, headers, JSON body) for assertions,
//! and can hang a connection forever to exercise client timeouts. Only what
//! reqwest actually sends is parsed: a request line, headers, and a
//! `Content-Length` body. Responses always carry `Connection: close`.

#![allow(dead_code)]

use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

/// What a route handler tells the server to send back.
pub enum RouteResponse {
    /// JSON body with the given status.
    Json { status: u16, body: Value },
    /// Arbitrary text body (for malformed-JSON cases) with the given status.
    Raw { status: u16, body: String },
    /// Never respond; the connection stays open until the server shuts down.
    Hang,
}

pub type Handler = Box<dyn FnMut(&Value) -> RouteResponse + Send>;

/// One request as received by the mock.
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    /// Header names are lower-cased.
    pub headers: Vec<(String, String)>,
    /// Parsed JSON body, or `Value::Null` when the body is not JSON.
    pub body: Value,
    pub raw_body: String,
}

impl RecordedRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
}

struct State {
    tokenize: Handler,
    embeddings: Handler,
    log: Vec<RecordedRequest>,
}

pub struct MockLlamaServer {
    addr: SocketAddr,
    state: Arc<Mutex<State>>,
    shutdown: watch::Sender<bool>,
}

impl MockLlamaServer {
    /// Bind `127.0.0.1:0` and start serving with the default handlers
    /// ([`tokenize_by_chars`] and [`embeddings_echo`] with 4 dimensions).
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let state = Arc::new(Mutex::new(State {
            tokenize: tokenize_by_chars(),
            embeddings: embeddings_echo(4),
            log: Vec::new(),
        }));
        let (shutdown, mut shutdown_rx) = watch::channel(false);

        let accept_state = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break };
                        let st = accept_state.clone();
                        let rx = shutdown_rx.clone();
                        tokio::spawn(async move {
                            let _ = handle_connection(stream, st, rx).await;
                        });
                    }
                }
            }
        });

        Self {
            addr,
            state,
            shutdown,
        }
    }

    /// `http://127.0.0.1:{port}`
    pub fn root_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// `http://127.0.0.1:{port}/v1`
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.root_url())
    }

    pub fn on_tokenize(&self, f: impl FnMut(&Value) -> RouteResponse + Send + 'static) {
        self.state.lock().unwrap().tokenize = Box::new(f);
    }

    pub fn on_embeddings(&self, f: impl FnMut(&Value) -> RouteResponse + Send + 'static) {
        self.state.lock().unwrap().embeddings = Box::new(f);
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state.lock().unwrap().log.clone()
    }

    pub fn requests_to(&self, path: &str) -> Vec<RecordedRequest> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == path)
            .collect()
    }
}

impl Drop for MockLlamaServer {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<State>>,
    mut shutdown: watch::Receiver<bool>,
) -> std::io::Result<()> {
    // Read headers.
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let header_end = loop {
        let mut chunk = [0u8; 2048];
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return Ok(());
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            let v = v.trim().to_string();
            if k == "content-length" {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

    // Read body.
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let mut chunk = vec![0u8; content_length - body.len()];
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    let raw_body = String::from_utf8_lossy(&body).to_string();
    let json_body: Value = serde_json::from_str(&raw_body).unwrap_or(Value::Null);

    // Dispatch under the lock (handlers are sync), then release before I/O.
    let response = {
        let mut st = state.lock().unwrap();
        st.log.push(RecordedRequest {
            method: method.clone(),
            path: path.clone(),
            headers,
            body: json_body.clone(),
            raw_body,
        });
        match (method.as_str(), path.as_str()) {
            ("POST", "/tokenize") => (st.tokenize)(&json_body),
            ("POST", "/v1/embeddings") => (st.embeddings)(&json_body),
            _ => RouteResponse::Json {
                status: 404,
                body: json!({ "error": { "code": 404, "message": format!("no route for {method} {path}"), "type": "not_found_error" } }),
            },
        }
    };

    let (status, content_type, payload) = match response {
        RouteResponse::Hang => {
            // Keep the socket open until the server is dropped.
            let _ = shutdown.changed().await;
            return Ok(());
        }
        RouteResponse::Json { status, body } => (status, "application/json", body.to_string()),
        RouteResponse::Raw { status, body } => (status, "text/plain", body),
    };

    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        reason(status),
        content_type,
        payload.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(payload.as_bytes()).await?;
    stream.flush().await?;
    stream.shutdown().await?;
    Ok(())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Handler builders
// ---------------------------------------------------------------------------

/// BERT-like special token ids used by the builders.
pub const CLS: u32 = 101;
pub const SEP: u32 = 102;

/// Content token id for the character at position `i`.
pub fn char_token(i: usize) -> u32 {
    1000 + i as u32
}

/// `/tokenize` handler returning `[CLS, one id per char, SEP]`, so the token
/// count is `chars + 2` and boundary tests are deterministic.
pub fn tokenize_by_chars() -> Handler {
    Box::new(|req: &Value| {
        let content = req["content"].as_str().unwrap_or("");
        let mut tokens = vec![CLS];
        tokens.extend((0..content.chars().count()).map(char_token));
        tokens.push(SEP);
        RouteResponse::Json {
            status: 200,
            body: json!({ "tokens": tokens }),
        }
    })
}

/// `/v1/embeddings` handler returning one `dims`-wide vector per input
/// element whose every component equals the element's position, with `index`
/// and `usage` populated like llama.cpp does.
pub fn embeddings_echo(dims: usize) -> Handler {
    Box::new(move |req: &Value| {
        let n = req["input"].as_array().map(|a| a.len()).unwrap_or(0);
        let data: Vec<Value> = (0..n)
            .map(
                |i| json!({ "object": "embedding", "index": i, "embedding": vec![i as f32; dims] }),
            )
            .collect();
        RouteResponse::Json {
            status: 200,
            body: json!({
                "object": "list",
                "data": data,
                "model": req["model"].clone(),
                "usage": { "prompt_tokens": n * 3, "total_tokens": n * 3 }
            }),
        }
    })
}

/// Handler that always returns a fixed JSON response.
pub fn fixed_json(status: u16, body: Value) -> Handler {
    Box::new(move |_| RouteResponse::Json {
        status,
        body: body.clone(),
    })
}

/// llama.cpp's error envelope.
pub fn llama_error(code: u16, message: &str, err_type: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "type": err_type } })
}

pub const TOO_LARGE_MESSAGE: &str =
    "input is too large to process. increase the physical batch size";
pub const EXCEED_CONTEXT_MESSAGE: &str =
    "prompt exceeds the available context size. increase context size";
