use std::{
    fmt::Write as FmtWrite,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use bzip2::{Compression, write::BzEncoder};
use sha2::{Digest, Sha256};
use tar::{Builder as TarBuilder, EntryType, Header};
use tempfile::{TempDir, tempdir};
use whispernow_core::{CancellationToken, ResolvedModel};
use whispernow_domain::{
    ArchiveFormat, AsrCapabilities, DownloadPhase, ModelAvailability, ModelDefinition, ModelId,
    ModelLayout, ModelProfile, ModelRegistry, ModelSource, RelativeRating,
};
use whispernow_platform::{
    AppPaths, DownloadProgress, DownloadTimeouts, ModelStore, ModelStoreError,
};

const MODEL_ID: &str = "fixture-whisper";
const OTHER_MODEL_ID: &str = "fixture-other";
/// Bound for store operations that must not wait on an in-flight download.
const PROMPT: Duration = Duration::from_secs(5);
const ARTIFACTS: [&str; 3] = ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"];

#[derive(Clone)]
struct ResponsePlan {
    status: u16,
    body: Vec<u8>,
    declared_length: Option<u64>,
    split_at: Option<usize>,
    split_delay: Duration,
}

struct FixtureServer {
    url: String,
    worker: Option<thread::JoinHandle<()>>,
}

impl FixtureServer {
    fn start(plan: ResponsePlan) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture server bind");
        let address = listener.local_addr().expect("fixture server address");
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("fixture request");
            read_request(&mut stream);
            let reason = if plan.status == 200 {
                "OK"
            } else {
                "Fixture Error"
            };
            let mut headers = format!(
                "HTTP/1.1 {} {}\r\nConnection: close\r\n",
                plan.status, reason
            );
            if let Some(length) = plan.declared_length {
                let _infallible_for_string = write!(&mut headers, "Content-Length: {length}\r\n");
            }
            headers.push_str("\r\n");
            let _ = stream.write_all(headers.as_bytes());
            if let Some(split_at) = plan.split_at {
                let split_at = split_at.min(plan.body.len());
                let _ = stream.write_all(&plan.body[..split_at]);
                let _ = stream.flush();
                thread::sleep(plan.split_delay);
                let _ = stream.write_all(&plan.body[split_at..]);
            } else {
                let _ = stream.write_all(&plan.body);
            }
            let _ = stream.flush();
        });
        Self {
            url: format!("http://{address}/model.tar.bz2"),
            worker: Some(worker),
        }
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.join().expect("fixture server join");
        }
    }
}

/// How a [`HeldServer`] ends a response after the test releases it.
#[derive(Clone, Copy)]
enum HeldEnding {
    FinishBody,
    /// Closes with the request left unread, which makes the OS send a TCP
    /// reset instead of a clean end-of-stream.
    ResetConnection,
}

/// Serves a full-length `Content-Length` response that stops after `split_at`
/// body bytes until the test releases it.
struct HeldServer {
    url: String,
    release: Option<mpsc::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl HeldServer {
    fn start(body: Vec<u8>, split_at: usize, ending: HeldEnding) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("held server bind");
        let address = listener.local_addr().expect("held server address");
        let (release, released) = mpsc::channel::<()>();
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("held request");
            match ending {
                HeldEnding::FinishBody => read_request(&mut stream),
                HeldEnding::ResetConnection => {
                    let mut request_start = [0_u8; 1];
                    stream.peek(&mut request_start).expect("peek held request");
                }
            }
            let headers = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes());
            let _ = stream.write_all(&body[..split_at]);
            let _ = stream.flush();
            // A dropped sender (test failure) also releases the response.
            let _ = released.recv();
            if matches!(ending, HeldEnding::FinishBody) {
                let _ = stream.write_all(&body[split_at..]);
                let _ = stream.flush();
            }
        });
        Self {
            url: format!("http://{address}/model.tar.bz2"),
            release: Some(release),
            worker: Some(worker),
        }
    }

    fn release(&self) {
        if let Some(release) = &self.release {
            let _ = release.send(());
        }
    }
}

impl Drop for HeldServer {
    fn drop(&mut self) {
        self.release.take();
        if let Some(worker) = self.worker.take() {
            worker.join().expect("held server join");
        }
    }
}

fn read_request(stream: &mut TcpStream) {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).expect("read fixture request");
        if count == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..count]);
    }
}

fn valid_archive() -> Vec<u8> {
    archive_with_entries(&[
        (ARTIFACTS[0], b"encoder"),
        (ARTIFACTS[1], b"decoder"),
        (ARTIFACTS[2], b"tokens"),
    ])
}

fn archive_with_entries(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut archive = TarBuilder::new(encoder);
    for (relative_path, contents) in entries {
        let path = format!("{MODEL_ID}/{relative_path}");
        let mut header = Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o600);
        header.set_entry_type(EntryType::file());
        header.set_cksum();
        archive
            .append_data(&mut header, path, *contents)
            .expect("append fixture artifact");
    }
    let encoder = archive.into_inner().expect("finish tar archive");
    encoder.finish().expect("finish bzip2 archive")
}

fn archive_with_symlink() -> Vec<u8> {
    archive_with_link(EntryType::symlink())
}

fn archive_with_hardlink() -> Vec<u8> {
    archive_with_link(EntryType::hard_link())
}

fn archive_with_link(entry_type: EntryType) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut archive = TarBuilder::new(encoder);
    let mut header = Header::new_gnu();
    header.set_size(0);
    header.set_mode(0o777);
    header.set_entry_type(entry_type);
    header
        .set_link_name("../../outside")
        .expect("fixture link target");
    header.set_cksum();
    archive
        .append_data(&mut header, format!("{MODEL_ID}/escape"), &[][..])
        .expect("append fixture link");
    let encoder = archive.into_inner().expect("finish tar archive");
    encoder.finish().expect("finish bzip2 archive")
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _infallible_for_string = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn definition(url: String, archive: &[u8]) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(MODEL_ID).expect("fixture ID"),
        display_name: "Fixture Whisper".to_owned(),
        source: ModelSource {
            url,
            archive_format: ArchiveFormat::TarBz2,
            archive_size_bytes: archive.len() as u64,
            sha256: sha256(archive),
        },
        layout: ModelLayout::Whisper {
            encoder: ARTIFACTS[0].to_owned(),
            decoder: ARTIFACTS[1].to_owned(),
            tokens: ARTIFACTS[2].to_owned(),
        },
        capabilities: AsrCapabilities::default(),
        profile: ModelProfile {
            speed: RelativeRating::Medium,
            accuracy: RelativeRating::Medium,
        },
    }
}

fn make_manager(
    server: &FixtureServer,
    archive: &[u8],
) -> (tempfile::TempDir, ModelStore, ModelId) {
    make_manager_for_url(&server.url, archive)
}

fn make_manager_for_url(url: &str, archive: &[u8]) -> (tempfile::TempDir, ModelStore, ModelId) {
    make_manager_for_definition(definition(url.to_owned(), archive))
}

/// A manager for one model downloaded as a single GGUF file, like the cleanup model.
fn make_single_file_manager(url: &str, file: &[u8]) -> (tempfile::TempDir, ModelStore, ModelId) {
    let mut definition = definition(url.to_owned(), file);
    definition.source.archive_format = ArchiveFormat::SingleFile;
    definition.layout = ModelLayout::Gguf {
        model: "model-Q8_0.gguf".to_owned(),
    };
    make_manager_for_definition(definition)
}

fn make_manager_for_definition(
    definition: ModelDefinition,
) -> (tempfile::TempDir, ModelStore, ModelId) {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let id = definition.id.clone();
    let registry = ModelRegistry::new(vec![definition]);
    let manager = ModelStore::with_registry_and_timeouts(
        registry,
        paths,
        Vec::new(),
        DownloadTimeouts {
            connect: Duration::from_secs(2),
            receive_body: Duration::from_secs(2),
            total: Duration::from_secs(5),
        },
    );
    (root, manager, id)
}

#[test]
fn local_streaming_download_reports_monotonic_progress_and_installs_atomically() {
    let archive = valid_archive();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: archive.clone(),
        declared_length: Some(archive.len() as u64),
        split_at: Some(archive.len() / 2),
        split_delay: Duration::from_millis(30),
    });
    let (_root, manager, id) = make_manager(&server, &archive);
    let progress = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&progress);

    let resolved = manager
        .download(&id, &CancellationToken::default(), &move |event| {
            observed.lock().expect("progress lock").push(event);
        })
        .expect("fixture download");

    assert_eq!(resolved.availability, ModelAvailability::Downloaded);
    for artifact in ARTIFACTS {
        assert!(resolved.root.join(artifact).is_file());
    }
    let progress = progress.lock().expect("progress lock");
    assert_eq!(
        progress.last().map(|event| event.phase),
        Some(DownloadPhase::Complete)
    );
    let byte_events: Vec<_> = progress
        .iter()
        .filter(|event| event.phase == DownloadPhase::Downloading)
        .map(|event| event.downloaded_bytes)
        .collect();
    assert!(byte_events.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(byte_events.last().copied(), Some(archive.len() as u64));
    let entries = manager.list().expect("model list");
    assert_eq!(entries[0].availability, ModelAvailability::Downloaded);
}

#[test]
fn a_single_file_model_is_downloaded_verified_installed_and_removed_as_is() {
    let gguf = b"GGUF fixture weights".repeat(64);
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: gguf.clone(),
        declared_length: Some(gguf.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_single_file_manager(&server.url, &gguf);

    let resolved = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect("single-file download");

    assert_eq!(resolved.availability, ModelAvailability::Downloaded);
    assert_eq!(
        fs::read(resolved.root.join("model-Q8_0.gguf")).expect("installed model"),
        gguf
    );
    assert_eq!(
        manager.list().expect("model list")[0].availability,
        ModelAvailability::Downloaded
    );
    manager.remove(&id).expect("remove the model");
    assert!(!resolved.root.exists());
}

#[test]
fn a_single_file_model_with_the_wrong_digest_is_not_installed() {
    let gguf = b"GGUF fixture weights".repeat(64);
    let mut tampered = gguf.clone();
    tampered[0] ^= 1;
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: tampered,
        declared_length: Some(gguf.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_single_file_manager(&server.url, &gguf);

    assert!(
        manager
            .download(&id, &CancellationToken::default(), &|_| {})
            .is_err()
    );
    assert_eq!(
        manager.list().expect("model list")[0].availability,
        ModelAvailability::NotInstalled
    );
}

#[test]
fn download_succeeds_without_content_length_by_enforcing_registry_size() {
    let archive = valid_archive();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: archive.clone(),
        declared_length: None,
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_manager(&server, &archive);

    let resolved = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect("download without content length");

    assert_eq!(resolved.availability, ModelAvailability::Downloaded);
}

#[test]
fn cancellation_cleans_staging_and_does_not_install() {
    let archive = valid_archive();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: archive.clone(),
        declared_length: Some(archive.len() as u64),
        split_at: Some(1),
        split_delay: Duration::from_millis(30),
    });
    let (root, manager, id) = make_manager(&server, &archive);
    let cancellation = CancellationToken::default();
    let cancel_from_observer = cancellation.clone();

    let error = manager
        .download(&id, &cancellation, &move |event: DownloadProgress| {
            if event.downloaded_bytes > 0 {
                cancel_from_observer.cancel();
            }
        })
        .expect_err("cancelled download");

    assert_eq!(error, ModelStoreError::Cancelled);
    assert!(manager.resolve(&id).expect("resolve").is_none());
    let models_dir = root.path().join("data/WhisperNow/models");
    let leftovers = fs::read_dir(models_dir)
        .expect("models directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("models entries");
    assert!(leftovers.is_empty());
}

#[test]
fn truncated_and_http_failure_responses_are_actionable() {
    let archive = valid_archive();
    let truncated = archive[..archive.len() / 2].to_vec();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: truncated,
        declared_length: Some(archive.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_manager(&server, &archive);
    let error = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect_err("truncated response");
    assert!(matches!(error, ModelStoreError::ResponseLength { .. }));

    let server = FixtureServer::start(ResponsePlan {
        status: 503,
        body: Vec::new(),
        declared_length: Some(0),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_manager(&server, &archive);
    assert_eq!(
        manager
            .download(&id, &CancellationToken::default(), &|_| {})
            .expect_err("HTTP failure"),
        ModelStoreError::HttpStatus { status: 503 }
    );
}

#[test]
fn connection_reset_mid_body_is_a_transport_error() {
    let archive = valid_archive();
    let server = HeldServer::start(
        archive.clone(),
        archive.len() / 2,
        HeldEnding::ResetConnection,
    );
    let (_root, manager, id) = make_manager_for_url(&server.url, &archive);
    let release = server.release.clone().expect("held server release");

    let error = manager
        .download(
            &id,
            &CancellationToken::default(),
            &move |event: DownloadProgress| {
                if event.downloaded_bytes > 0 {
                    let _ = release.send(());
                }
            },
        )
        .expect_err("reset connection");

    assert_eq!(error, ModelStoreError::Transport);
}

/// A download of [`MODEL_ID`] held mid-transfer beside an installed
/// [`OTHER_MODEL_ID`].
struct BlockedDownload {
    manager: Arc<ModelStore>,
    downloading: ModelId,
    other: ModelId,
    other_root: std::path::PathBuf,
    server: HeldServer,
    worker: Option<thread::JoinHandle<Result<ResolvedModel, ModelStoreError>>>,
    _root: TempDir,
}

impl BlockedDownload {
    fn start() -> Self {
        let archive = valid_archive();
        let server = HeldServer::start(archive.clone(), archive.len() / 2, HeldEnding::FinishBody);
        let root = tempdir().expect("temporary root");
        let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
        let downloading = definition(server.url.clone(), &archive);
        let mut other = downloading.clone();
        other.id = ModelId::new(OTHER_MODEL_ID).expect("other fixture ID");
        let other_root = root
            .path()
            .join(format!("data/WhisperNow/models/{OTHER_MODEL_ID}"));
        fs::create_dir_all(&other_root).expect("other model root");
        for artifact in ARTIFACTS {
            fs::write(other_root.join(artifact), artifact).expect("other artifact");
        }
        let downloading_id = downloading.id.clone();
        let other_id = other.id.clone();
        let registry = ModelRegistry::new(vec![downloading, other]);
        let manager = Arc::new(ModelStore::with_registry_and_timeouts(
            registry,
            paths,
            Vec::new(),
            DownloadTimeouts {
                connect: Duration::from_secs(5),
                receive_body: Duration::from_mins(1),
                total: Duration::from_mins(2),
            },
        ));

        let (mid_transfer, reached_mid_transfer) = mpsc::channel();
        let downloader = Arc::clone(&manager);
        let id = downloading_id.clone();
        let worker = thread::spawn(move || {
            downloader.download(
                &id,
                &CancellationToken::default(),
                &move |event: DownloadProgress| {
                    if event.downloaded_bytes > 0 {
                        let _ = mid_transfer.send(());
                    }
                },
            )
        });
        reached_mid_transfer
            .recv_timeout(PROMPT)
            .expect("download reached mid-transfer");
        Self {
            manager,
            downloading: downloading_id,
            other: other_id,
            other_root,
            server,
            worker: Some(worker),
            _root: root,
        }
    }

    fn is_running(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
    }

    fn finish(&mut self) -> Result<ResolvedModel, ModelStoreError> {
        self.server.release();
        self.worker
            .take()
            .expect("download still held")
            .join()
            .expect("download worker join")
    }
}

#[test]
fn blocked_download_does_not_stall_resolving_listing_or_removing_other_models() {
    let mut blocked = BlockedDownload::start();
    let store = Arc::clone(&blocked.manager);
    let other = blocked.other.clone();
    let (done, completed) = mpsc::channel();
    let _operations = thread::spawn(move || {
        let resolved = store.resolve(&other);
        let listed = store.list();
        let removed = store.remove(&other);
        let _ = done.send((resolved, listed, removed));
    });

    let (resolved, listed, removed) = completed
        .recv_timeout(PROMPT)
        .expect("store operations returned while the download was held");

    assert!(blocked.is_running());
    assert_eq!(
        resolved
            .expect("resolve other")
            .map(|model| model.availability),
        Some(ModelAvailability::Downloaded)
    );
    let availability: Vec<_> = listed
        .expect("list models")
        .into_iter()
        .map(|entry| (entry.definition.id, entry.availability))
        .collect();
    assert_eq!(
        availability,
        vec![
            (blocked.downloading.clone(), ModelAvailability::NotInstalled),
            (blocked.other.clone(), ModelAvailability::Downloaded),
        ]
    );
    assert_eq!(removed, Ok(()));
    assert!(!blocked.other_root.exists());

    let installed = blocked.finish().expect("held download completes");
    assert_eq!(installed.availability, ModelAvailability::Downloaded);
    assert!(
        blocked
            .manager
            .resolve(&blocked.downloading)
            .expect("resolve downloaded")
            .is_some()
    );
}

#[test]
fn stalled_response_hits_the_configured_body_timeout() {
    let archive = valid_archive();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: archive.clone(),
        declared_length: Some(archive.len() as u64),
        split_at: Some(1),
        split_delay: Duration::from_secs(3),
    });
    let (_root, manager, id) = make_manager(&server, &archive);

    let error = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect_err("stalled response");

    assert_eq!(error, ModelStoreError::Timeout);
}

#[test]
fn same_length_archive_with_wrong_checksum_is_rejected() {
    let archive = valid_archive();
    let mut corrupted = archive.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: corrupted,
        declared_length: Some(archive.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (_root, manager, id) = make_manager(&server, &archive);

    let error = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect_err("checksum mismatch");

    assert_eq!(error, ModelStoreError::ChecksumMismatch);
}

#[test]
fn invalid_archives_missing_artifacts_and_links_are_rejected() {
    // Links are not taken as artifacts.
    let mut artifacts = ARTIFACTS.map(str::to_owned).to_vec();
    artifacts.sort();
    let missing_all = ModelStoreError::MissingArtifacts { artifacts };
    for (archive, expected) in [
        (
            b"not a bzip2 archive".to_vec(),
            ModelStoreError::InvalidArchive,
        ),
        (
            archive_with_entries(&[(ARTIFACTS[0], b"encoder")]),
            ModelStoreError::MissingArtifacts {
                artifacts: vec![ARTIFACTS[1].to_owned(), ARTIFACTS[2].to_owned()],
            },
        ),
        (archive_with_symlink(), missing_all.clone()),
        (archive_with_hardlink(), missing_all),
    ] {
        let server = FixtureServer::start(ResponsePlan {
            status: 200,
            body: archive.clone(),
            declared_length: Some(archive.len() as u64),
            split_at: None,
            split_delay: Duration::ZERO,
        });
        let (_root, manager, id) = make_manager(&server, &archive);
        let error = manager
            .download(&id, &CancellationToken::default(), &|_| {})
            .expect_err("invalid archive");
        assert_eq!(error, expected);
    }
}

#[test]
fn interrupted_upgrade_preserves_existing_valid_model() {
    let valid = valid_archive();
    let bad = b"checksum does not match".to_vec();
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: bad.clone(),
        declared_length: Some(bad.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (root, manager, id) = make_manager(&server, &valid);
    let installed = root
        .path()
        .join(format!("data/WhisperNow/models/{MODEL_ID}"));
    fs::create_dir_all(&installed).expect("existing model root");
    for artifact in ARTIFACTS {
        fs::write(installed.join(artifact), format!("old-{artifact}")).expect("existing artifact");
    }

    let error = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect_err("interrupted upgrade");

    assert!(matches!(
        error,
        ModelStoreError::ResponseLength { .. } | ModelStoreError::ChecksumMismatch
    ));
    let resolved = manager
        .resolve(&id)
        .expect("resolve existing")
        .expect("existing remains valid");
    assert_eq!(
        fs::read_to_string(resolved.root.join(ARTIFACTS[0])).expect("old artifact"),
        format!("old-{}", ARTIFACTS[0])
    );
}

#[test]
fn successful_download_replaces_an_existing_installation() {
    let archive = archive_with_entries(&[
        (ARTIFACTS[0], b"new-encoder"),
        (ARTIFACTS[1], b"new-decoder"),
        (ARTIFACTS[2], b"new-tokens"),
    ]);
    let server = FixtureServer::start(ResponsePlan {
        status: 200,
        body: archive.clone(),
        declared_length: Some(archive.len() as u64),
        split_at: None,
        split_delay: Duration::ZERO,
    });
    let (root, manager, id) = make_manager(&server, &archive);
    let installed = root
        .path()
        .join(format!("data/WhisperNow/models/{MODEL_ID}"));
    fs::create_dir_all(&installed).expect("existing model root");
    for artifact in ARTIFACTS {
        fs::write(installed.join(artifact), format!("old-{artifact}")).expect("existing artifact");
    }

    let resolved = manager
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect("successful upgrade");

    assert_eq!(
        fs::read_to_string(resolved.root.join(ARTIFACTS[0])).expect("upgraded artifact"),
        "new-encoder"
    );
}

#[cfg(unix)]
#[test]
fn resolver_rejects_intermediate_directory_symlinks_in_nested_artifacts() {
    use std::os::unix::fs::symlink;

    let archive = valid_archive();
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let mut definition = definition("http://127.0.0.1:1/model.tar.bz2".to_owned(), &archive);
    definition.layout = ModelLayout::Whisper {
        encoder: "files/encoder.int8.onnx".to_owned(),
        decoder: "files/decoder.int8.onnx".to_owned(),
        tokens: "files/tokens.txt".to_owned(),
    };
    let id = definition.id.clone();
    let registry = ModelRegistry::new(vec![definition]);
    let manager = ModelStore::with_registry_and_timeouts(
        registry,
        paths,
        Vec::new(),
        DownloadTimeouts::default(),
    );
    let outside = root.path().join("outside");
    fs::create_dir_all(&outside).expect("outside root");
    for artifact in ARTIFACTS {
        fs::write(outside.join(artifact), artifact).expect("outside artifact");
    }
    let installed = root
        .path()
        .join(format!("data/WhisperNow/models/{MODEL_ID}"));
    fs::create_dir_all(&installed).expect("model root");
    symlink(&outside, installed.join("files")).expect("intermediate symlink");

    assert!(manager.resolve(&id).expect("resolve safely").is_none());
}

#[test]
fn resolver_uses_bundled_assets_without_cwd_paths_and_remove_only_affects_downloads() {
    let archive = valid_archive();
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let bundled = root.path().join("bundled");
    let bundled_model = bundled.join(MODEL_ID);
    fs::create_dir_all(&bundled_model).expect("bundled model root");
    for artifact in ARTIFACTS {
        fs::write(bundled_model.join(artifact), artifact).expect("bundled artifact");
    }
    let definition = definition("http://127.0.0.1:1/model.tar.bz2".to_owned(), &archive);
    let id = definition.id.clone();
    let registry = ModelRegistry::new(vec![definition]);
    let manager = ModelStore::with_registry_and_timeouts(
        registry,
        paths,
        vec![bundled],
        DownloadTimeouts::default(),
    );

    let resolved = manager
        .resolve(&id)
        .expect("resolve")
        .expect("bundled model");
    assert_eq!(resolved.availability, ModelAvailability::Bundled);
    assert_eq!(manager.remove(&id), Err(ModelStoreError::NotInstalled));
    assert!(bundled_model.is_dir());
}
