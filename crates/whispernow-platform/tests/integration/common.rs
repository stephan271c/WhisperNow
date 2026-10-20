//! A local HTTP server, fixture archives and SHA-256 digests for the download tests.
use std::{
    collections::HashMap,
    fmt::Write as _,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};
use tar::{Builder as TarBuilder, EntryType, Header};
use tempfile::{TempDir, tempdir};
use whispernow_platform::{AppPaths, DownloadProgress, DownloadTimeouts};

/// What the server answers on one path.
#[derive(Clone)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
    /// The `Content-Length` sent; `None` sends none.
    pub declared_length: Option<u64>,
    /// The body byte the response stops at, and how it goes on.
    pub pause: Option<(usize, Resume)>,
}

#[derive(Clone, Copy)]
pub enum Resume {
    After(Duration),
    /// When the test calls [`FixtureServer::release`] or drops the server.
    OnRelease,
    /// Resets the connection on release: the request is left unread, so closing makes the OS
    /// send a TCP reset instead of a clean end of stream.
    ResetOnRelease,
}

impl Response {
    /// `body` with its length.
    pub fn ok(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            declared_length: Some(body.len() as u64),
            body,
            pause: None,
        }
    }

    pub fn status(status: u16) -> Self {
        Self {
            status,
            ..Self::ok(Vec::new())
        }
    }
}

/// Serves a fixed response per request path, one connection at a time, until dropped. An
/// unknown path is answered 404.
pub struct FixtureServer {
    address: SocketAddr,
    release: Mutex<Option<mpsc::Sender<()>>>,
    stopping: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl FixtureServer {
    pub fn start<'a>(routes: impl IntoIterator<Item = (&'a str, Response)>) -> Self {
        let routes: HashMap<String, Response> = routes
            .into_iter()
            .map(|(path, response)| (path.to_owned(), response))
            .collect();
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture server bind");
        let address = listener.local_addr().expect("fixture server address");
        let (release, released) = mpsc::channel::<()>();
        let stopping = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopping);
        let worker = thread::spawn(move || {
            for stream in listener.incoming() {
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(mut stream) = stream else { return };
                let path = request_path(&stream);
                let not_found = Response::status(404);
                let response = routes.get(&path).unwrap_or(&not_found);
                respond(&mut stream, response, &released);
            }
        });
        Self {
            address,
            release: Mutex::new(Some(release)),
            stopping,
            worker: Some(worker),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    /// Lets a paused response go on.
    pub fn release(&self) {
        if let Some(release) = &*self.release.lock().unwrap_or_else(PoisonError::into_inner) {
            let _ = release.send(());
        }
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.release
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        self.stopping.store(true, Ordering::SeqCst);
        // Wakes the accept loop so it sees the stop.
        let _ = TcpStream::connect(self.address);
        if let Some(worker) = self.worker.take()
            && let Err(panic) = worker.join()
            && !thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

fn respond(stream: &mut TcpStream, response: &Response, released: &mpsc::Receiver<()>) {
    if !matches!(response.pause, Some((_, Resume::ResetOnRelease))) {
        read_request(stream);
    }
    let reason = if response.status == 200 {
        "OK"
    } else {
        "Fixture Error"
    };
    let mut headers = format!(
        "HTTP/1.1 {} {reason}\r\nConnection: close\r\n",
        response.status
    );
    if let Some(length) = response.declared_length {
        let _ = write!(&mut headers, "Content-Length: {length}\r\n");
    }
    headers.push_str("\r\n");
    let _ = stream.write_all(headers.as_bytes());
    let body = &response.body;
    let Some((at, resume)) = response.pause else {
        let _ = stream.write_all(body);
        let _ = stream.flush();
        return;
    };
    let at = at.min(body.len());
    let _ = stream.write_all(&body[..at]);
    let _ = stream.flush();
    match resume {
        Resume::After(delay) => thread::sleep(delay),
        Resume::OnRelease | Resume::ResetOnRelease => {
            let _ = released.recv();
        }
    }
    if !matches!(resume, Resume::ResetOnRelease) {
        let _ = stream.write_all(&body[at..]);
        let _ = stream.flush();
    }
}

/// The request's path, read without taking the request off the connection.
fn request_path(stream: &TcpStream) -> String {
    let mut buffer = [0_u8; 1024];
    loop {
        let count = stream.peek(&mut buffer).unwrap_or(0);
        let peeked = &buffer[..count];
        if count == 0 || count == buffer.len() || peeked.windows(2).any(|pair| pair == b"\r\n") {
            return String::from_utf8_lossy(peeked)
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_owned();
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn read_request(stream: &mut TcpStream) {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        match stream.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => request.extend_from_slice(&buffer[..count]),
        }
    }
}

/// The lowercase hex SHA-256 digest of `bytes`.
pub fn sha256(bytes: &[u8]) -> String {
    let mut encoded = String::new();
    for byte in Sha256::digest(bytes) {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

pub fn temp_paths() -> (TempDir, AppPaths) {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::under(root.path());
    (root, paths)
}

pub fn timeouts(total: Duration) -> DownloadTimeouts {
    DownloadTimeouts {
        connect: Duration::from_secs(2),
        receive_body: Duration::from_secs(2),
        total,
    }
}

pub fn recorder() -> (Arc<Mutex<Vec<DownloadProgress>>>, impl Fn(DownloadProgress)) {
    let progress = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&progress);
    (progress, move |event| {
        sink.lock().expect("progress").push(event);
    })
}

pub fn tar_file(archive: &mut TarBuilder<impl Write>, path: &str, body: &[u8]) {
    let mut header = Header::new_gnu();
    header.set_size(body.len() as u64);
    header.set_mode(0o644);
    header.set_entry_type(EntryType::file());
    header.set_cksum();
    archive
        .append_data(&mut header, path, body)
        .expect("append tar file");
}

pub fn tar_link(archive: &mut TarBuilder<impl Write>, path: &str, kind: EntryType, target: &str) {
    let mut header = Header::new_gnu();
    header.set_size(0);
    header.set_mode(0o777);
    header.set_entry_type(kind);
    header.set_link_name(target).expect("tar link target");
    header.set_cksum();
    archive
        .append_data(&mut header, path, &[][..])
        .expect("append tar link");
}
