// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! A minimal HTTP/1.1 stub server for remote-provider unit tests.
//!
//! Scripted by method and path (the query string is ignored for matching but
//! recorded), it answers each request with a status code and an empty JSON
//! body, or never answers at all, and records every request it sees so a test
//! can check what was sent. Unscripted routes answer 404.

use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// How the stub answers one route.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reply {
    /// Answer with this status and `{}`.
    Status(u16),
    /// Answer with this status and this JSON body.
    Json(u16, &'static str),
    /// Accept the request and never answer.
    Hang,
}

/// One request the stub received.
#[derive(Clone, Debug)]
pub(crate) struct Seen {
    pub(crate) method: String,
    /// Path including any query string, as sent.
    pub(crate) target: String,
    /// Header names lower-cased.
    pub(crate) headers: Vec<(String, String)>,
}

impl Seen {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

pub(crate) struct StubServer {
    addr: std::net::SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    task: tokio::task::JoinHandle<()>,
}

impl StubServer {
    /// Start a stub answering `routes`: `(method, path, reply)`.
    pub(crate) async fn start(routes: &[(&str, &str, Reply)]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let routes: Arc<Vec<(String, String, Reply)>> = Arc::new(
            routes
                .iter()
                .map(|(m, p, r)| (m.to_string(), p.to_string(), *r))
                .collect(),
        );
        let task = {
            let seen = seen.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        return;
                    };
                    let seen = seen.clone();
                    let routes = routes.clone();
                    tokio::spawn(async move {
                        let Some(request) = read_request(&mut socket).await else {
                            return;
                        };
                        let path = request.target.split('?').next().unwrap_or("").to_string();
                        let reply = routes
                            .iter()
                            .find(|(m, p, _)| *m == request.method && *p == path)
                            .map_or(Reply::Status(404), |(_, _, r)| *r);
                        seen.lock().unwrap().push(request);
                        let answer = |code: u16, body: &str| {
                            format!(
                                "HTTP/1.1 {code} Stub\r\ncontent-type: application/json\r\n\
                                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                                body.len()
                            )
                        };
                        match reply {
                            Reply::Status(code) => {
                                let _ = socket.write_all(answer(code, "{}").as_bytes()).await;
                            }
                            Reply::Json(code, body) => {
                                let _ = socket.write_all(answer(code, body).as_bytes()).await;
                            }
                            Reply::Hang => {
                                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                            }
                        }
                    });
                }
            })
        };
        Self { addr, seen, task }
    }

    /// `http://127.0.0.1:{port}`, with no trailing slash.
    pub(crate) fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Every request received so far.
    pub(crate) fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A base URL nothing is listening on: a port bound, then released.
pub(crate) async fn unreachable_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}")
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<Seen> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_string();
    let target = request_line.next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    // Drain the body so the client is not left mid-write.
    let length: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut have = buf.len() - (head_end + 4);
    while have < length {
        let n = socket.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        have += n;
    }
    Some(Seen {
        method,
        target,
        headers,
    })
}
