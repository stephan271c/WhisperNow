use std::{
    fs,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use crate::common::{FixtureServer, Response, Resume, sha256};
use bzip2::{Compression, write::BzEncoder};
use tar::{Builder as TarBuilder, EntryType, Header};
use tempfile::{TempDir, tempdir};
use whispernow_core::{CancellationToken, ResolvedModel};
use whispernow_domain::{
    DownloadPhase, ModelAvailability, ModelDefinition, ModelId, ModelLayout, ModelProfile,
    ModelRegistry, ModelSource,
};
use whispernow_platform::{
    AppPaths, DownloadProgress, DownloadTimeouts, ModelStore, ModelStoreError,
};

const MODEL_ID: &str = "fixture-whisper";
const OTHER_MODEL_ID: &str = "fixture-other";
/// Bound for store operations that must not wait on an in-flight download.
const PROMPT: Duration = Duration::from_secs(5);
const ARTIFACTS: [&str; 3] = ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"];
const MODEL_PATH: &str = "/fixture-whisper.tar.bz2";

fn serve(response: Response) -> FixtureServer {
    FixtureServer::start([(MODEL_PATH, response)])
}

/// `body` stopping after `at` bytes until `resume`.
fn paused(body: Vec<u8>, at: usize, resume: Resume) -> Response {
    Response {
        pause: Some((at, resume)),
        ..Response::ok(body)
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
    archive_under(MODEL_ID, entries)
}

fn archive_under(root: &str, entries: &[(&str, &[u8])]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut archive = TarBuilder::new(encoder);
    for (relative_path, contents) in entries {
        let path = format!("{root}/{relative_path}");
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

fn definition(url: String, archive: &[u8]) -> ModelDefinition {
    ModelDefinition {
        id: ModelId::new(MODEL_ID).expect("fixture ID"),
        display_name: "Fixture Whisper".to_owned(),
        source: ModelSource {
            url,
            archive_size_bytes: archive.len() as u64,
            sha256: sha256(archive),
        },
        layout: ModelLayout::Whisper {
            encoder: ARTIFACTS[0].to_owned(),
            decoder: ARTIFACTS[1].to_owned(),
            tokens: ARTIFACTS[2].to_owned(),
        },
        languages: Vec::new(),
        profile: ModelProfile::rated(3, 3),
    }
}

fn make_store(server: &FixtureServer, archive: &[u8]) -> (tempfile::TempDir, ModelStore, ModelId) {
    make_store_for_url(&server.url(MODEL_PATH), archive)
}

fn make_store_for_url(url: &str, archive: &[u8]) -> (tempfile::TempDir, ModelStore, ModelId) {
    make_store_for_definition(definition(url.to_owned(), archive))
}

/// A store for one model downloaded as a single GGUF file, like the cleanup model.
fn make_single_file_store(url: &str, file: &[u8]) -> (tempfile::TempDir, ModelStore, ModelId) {
    let mut definition = definition(url.to_owned(), file);
    definition.layout = ModelLayout::Gguf {
        model: "model-Q8_0.gguf".to_owned(),
    };
    make_store_for_definition(definition)
}

fn make_store_for_definition(
    definition: ModelDefinition,
) -> (tempfile::TempDir, ModelStore, ModelId) {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let id = definition.id.clone();
    let registry = ModelRegistry::new(vec![definition]);
    let store = ModelStore::with_registry_and_timeouts(
        registry,
        paths,
        DownloadTimeouts {
            connect: Duration::from_secs(2),
            receive_body: Duration::from_secs(2),
            total: Duration::from_secs(5),
        },
    );
    (root, store, id)
}

#[test]
fn startup_removes_download_staging_and_preserves_installed_models_and_other_files() {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let models = paths.models_dir();
    let installed = models.join(MODEL_ID);
    fs::create_dir_all(&installed).expect("installed model root");
    for artifact in ARTIFACTS {
        fs::write(installed.join(artifact), artifact).expect("installed artifact");
    }
    let download = models.join(".model-download-crashed.part");
    fs::write(&download, b"incomplete archive").expect("interrupted download");
    let stage = models.join(".model-stage-crashed");
    fs::create_dir(&stage).expect("interrupted extraction root");
    fs::write(stage.join("incomplete.onnx"), b"incomplete artifact")
        .expect("interrupted extraction");
    let preserved = [
        models.join("notes.txt"),
        models.join(".model-download-retained.txt"),
        models.join(".model-download-retained.part.backup"),
        models.join(".model-stage"),
    ];
    for path in &preserved {
        fs::write(path, b"keep").expect("unrelated file");
    }
    #[cfg(unix)]
    let outside = root.path().join("outside");
    #[cfg(unix)]
    let linked_stage = models.join(".model-stage-linked");
    #[cfg(unix)]
    {
        fs::create_dir(&outside).expect("outside root");
        fs::write(outside.join("keep.txt"), b"keep").expect("outside file");
        std::os::unix::fs::symlink(&outside, &linked_stage).expect("staging symlink");
    }
    let definition = definition(
        "http://127.0.0.1:1/model.tar.bz2".to_owned(),
        &valid_archive(),
    );
    let id = definition.id.clone();

    let store = ModelStore::with_registry_and_timeouts(
        ModelRegistry::new(vec![definition]),
        paths,
        DownloadTimeouts::default(),
    );

    assert!(!download.exists());
    assert!(!stage.exists());
    assert_eq!(
        store
            .resolve(&id)
            .expect("resolve")
            .expect("installed model")
            .root,
        installed
    );
    for artifact in ARTIFACTS {
        assert_eq!(
            fs::read_to_string(installed.join(artifact)).expect("installed artifact preserved"),
            artifact
        );
    }
    for path in preserved {
        assert_eq!(fs::read(path).expect("unrelated file preserved"), b"keep");
    }
    #[cfg(unix)]
    {
        assert!(fs::symlink_metadata(linked_stage).is_err());
        assert_eq!(
            fs::read(outside.join("keep.txt")).expect("outside file preserved"),
            b"keep"
        );
    }
}

#[test]
fn local_streaming_download_reports_monotonic_progress_and_installs_atomically() {
    let archive = valid_archive();
    let server = serve(paused(
        archive.clone(),
        archive.len() / 2,
        Resume::After(Duration::from_millis(30)),
    ));
    let (_root, store, id) = make_store(&server, &archive);
    let progress = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&progress);

    let resolved = store
        .download(&id, &CancellationToken::default(), &move |event| {
            observed.lock().expect("progress lock").push(event);
        })
        .expect("fixture download");

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
    let entries = store.list().expect("model list");
    assert_eq!(entries[0].availability, ModelAvailability::Downloaded);
}

#[test]
fn a_single_file_model_is_downloaded_verified_installed_and_removed_as_is() {
    let gguf = b"GGUF fixture weights".repeat(64);
    let server = serve(Response::ok(gguf.clone()));
    let (_root, store, id) = make_single_file_store(&server.url(MODEL_PATH), &gguf);

    let resolved = store
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect("single-file download");

    assert_eq!(
        fs::read(resolved.root.join("model-Q8_0.gguf")).expect("installed model"),
        gguf
    );
    assert_eq!(
        store.list().expect("model list")[0].availability,
        ModelAvailability::Downloaded
    );
    store.remove(&id).expect("remove the model");
    assert!(!resolved.root.exists());
}

#[test]
fn archive_directory_prefixes_and_case_do_not_change_the_installed_model_id() {
    for (directory, path) in [
        (format!("./{MODEL_ID}"), MODEL_PATH),
        ("Fixture-Whisper".to_owned(), "/Fixture-Whisper.tar.bz2"),
    ] {
        let archive = archive_under(
            &directory,
            &[
                (ARTIFACTS[0], b"encoder"),
                (ARTIFACTS[1], b"decoder"),
                (ARTIFACTS[2], b"tokens"),
            ],
        );
        let server = FixtureServer::start([(path, Response::ok(archive.clone()))]);
        let (_root, store, id) = make_store_for_url(&server.url(path), &archive);
        let resolved = store
            .download(&id, &CancellationToken::default(), &|_| {})
            .expect("fixture download");
        assert!(resolved.root.ends_with(MODEL_ID), "{directory}");
        for artifact in ARTIFACTS {
            assert!(
                resolved.root.join(artifact).is_file(),
                "{directory}/{artifact}"
            );
        }
    }
}

#[test]
fn a_single_file_model_with_the_wrong_digest_is_not_installed() {
    let gguf = b"GGUF fixture weights".repeat(64);
    let mut tampered = gguf.clone();
    tampered[0] ^= 1;
    let server = serve(Response::ok(tampered));
    let (_root, store, id) = make_single_file_store(&server.url(MODEL_PATH), &gguf);

    assert!(
        store
            .download(&id, &CancellationToken::default(), &|_| {})
            .is_err()
    );
    assert_eq!(
        store.list().expect("model list")[0].availability,
        ModelAvailability::NotInstalled
    );
}

#[test]
fn download_succeeds_without_content_length_by_enforcing_registry_size() {
    let archive = valid_archive();
    let server = serve(Response {
        declared_length: None,
        ..Response::ok(archive.clone())
    });
    let (_root, store, id) = make_store(&server, &archive);

    let resolved = store
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect("download without content length");

    for artifact in ARTIFACTS {
        assert!(resolved.root.join(artifact).is_file());
    }
}

#[test]
fn cancellation_cleans_staging_and_does_not_install() {
    let archive = valid_archive();
    let server = serve(paused(
        archive.clone(),
        1,
        Resume::After(Duration::from_millis(30)),
    ));
    let (root, store, id) = make_store(&server, &archive);
    let cancellation = CancellationToken::default();
    let cancel_from_observer = cancellation.clone();

    let error = store
        .download(&id, &cancellation, &move |event: DownloadProgress| {
            if event.downloaded_bytes > 0 {
                cancel_from_observer.cancel();
            }
        })
        .expect_err("cancelled download");

    assert_eq!(error, ModelStoreError::Cancelled);
    assert!(store.resolve(&id).expect("resolve").is_none());
    let models_dir = root.path().join("data/WhisperNow/models");
    let leftovers = fs::read_dir(models_dir)
        .expect("models directory")
        .collect::<Result<Vec<_>, _>>()
        .expect("models entries");
    assert!(leftovers.is_empty());
}

#[test]
fn failed_downloads_report_why() {
    let archive = valid_archive();
    let mut corrupted = archive.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    for (case, response, expected) in [
        (
            "truncated",
            Response {
                declared_length: Some(archive.len() as u64),
                ..Response::ok(archive[..archive.len() / 2].to_vec())
            },
            ModelStoreError::ResponseLength {
                expected: archive.len() as u64,
                actual: (archive.len() / 2) as u64,
            },
        ),
        (
            "HTTP failure",
            Response::status(503),
            ModelStoreError::HttpStatus { status: 503 },
        ),
        (
            "connection reset",
            paused(archive.clone(), archive.len() / 2, Resume::ResetOnRelease),
            ModelStoreError::Transport,
        ),
        (
            // Longer than the two-second body timeout.
            "stalled",
            paused(archive.clone(), 1, Resume::After(Duration::from_secs(3))),
            ModelStoreError::Timeout,
        ),
        (
            "same length, wrong checksum",
            Response::ok(corrupted.clone()),
            ModelStoreError::ChecksumMismatch,
        ),
    ] {
        let server = serve(response);
        let (_root, store, id) = make_store(&server, &archive);
        let error = store
            .download(
                &id,
                &CancellationToken::default(),
                &|event: DownloadProgress| {
                    if event.downloaded_bytes > 0 {
                        server.release();
                    }
                },
            )
            .expect_err(case);
        assert_eq!(error, expected, "{case}");
    }
}

/// A download of [`MODEL_ID`] held mid-transfer beside an installed
/// [`OTHER_MODEL_ID`].
struct BlockedDownload {
    store: Arc<ModelStore>,
    downloading: ModelId,
    other: ModelId,
    other_root: std::path::PathBuf,
    server: FixtureServer,
    worker: Option<thread::JoinHandle<Result<ResolvedModel, ModelStoreError>>>,
    _root: TempDir,
}

impl BlockedDownload {
    fn start() -> Self {
        let archive = valid_archive();
        let server = serve(paused(
            archive.clone(),
            archive.len() / 2,
            Resume::OnRelease,
        ));
        let root = tempdir().expect("temporary root");
        let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
        let downloading = definition(server.url(MODEL_PATH), &archive);
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
        let store = Arc::new(ModelStore::with_registry_and_timeouts(
            registry,
            paths,
            DownloadTimeouts {
                connect: Duration::from_secs(5),
                receive_body: Duration::from_mins(1),
                total: Duration::from_mins(2),
            },
        ));

        let (mid_transfer, reached_mid_transfer) = mpsc::channel();
        let downloader = Arc::clone(&store);
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
            store,
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
    let store = Arc::clone(&blocked.store);
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
        resolved.expect("resolve other").map(|model| model.root),
        Some(blocked.other_root.clone())
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

    blocked.finish().expect("held download completes");
    assert!(
        blocked
            .store
            .resolve(&blocked.downloading)
            .expect("resolve downloaded")
            .is_some()
    );
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
        let server = serve(Response::ok(archive.clone()));
        let (_root, store, id) = make_store(&server, &archive);
        let error = store
            .download(&id, &CancellationToken::default(), &|_| {})
            .expect_err("invalid archive");
        assert_eq!(error, expected);
    }
}

#[test]
fn interrupted_upgrade_preserves_existing_valid_model() {
    let valid = valid_archive();
    let bad = b"checksum does not match".to_vec();
    let server = serve(Response::ok(bad.clone()));
    let (root, store, id) = make_store(&server, &valid);
    let installed = root
        .path()
        .join(format!("data/WhisperNow/models/{MODEL_ID}"));
    fs::create_dir_all(&installed).expect("existing model root");
    for artifact in ARTIFACTS {
        fs::write(installed.join(artifact), format!("old-{artifact}")).expect("existing artifact");
    }

    let error = store
        .download(&id, &CancellationToken::default(), &|_| {})
        .expect_err("interrupted upgrade");

    assert!(matches!(
        error,
        ModelStoreError::ResponseLength { .. } | ModelStoreError::ChecksumMismatch
    ));
    let resolved = store
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
    let server = serve(Response::ok(archive.clone()));
    let (root, store, id) = make_store(&server, &archive);
    let installed = root
        .path()
        .join(format!("data/WhisperNow/models/{MODEL_ID}"));
    fs::create_dir_all(&installed).expect("existing model root");
    for artifact in ARTIFACTS {
        fs::write(installed.join(artifact), format!("old-{artifact}")).expect("existing artifact");
    }

    let resolved = store
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
    let store =
        ModelStore::with_registry_and_timeouts(registry, paths, DownloadTimeouts::default());
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

    assert!(store.resolve(&id).expect("resolve safely").is_none());
}

#[test]
fn remove_only_affects_the_downloaded_model_inside_the_data_directory() {
    let archive = valid_archive();
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let outside_model = root.path().join("outside").join(MODEL_ID);
    fs::create_dir_all(&outside_model).expect("outside model root");
    for artifact in ARTIFACTS {
        fs::write(outside_model.join(artifact), artifact).expect("outside artifact");
    }
    let definition = definition("http://127.0.0.1:1/model.tar.bz2".to_owned(), &archive);
    let id = definition.id.clone();
    let registry = ModelRegistry::new(vec![definition]);
    let downloaded_model = paths.models_dir().join(MODEL_ID);
    let store =
        ModelStore::with_registry_and_timeouts(registry, paths, DownloadTimeouts::default());

    assert!(store.resolve(&id).expect("resolve").is_none());
    assert_eq!(store.remove(&id), Err(ModelStoreError::NotInstalled));
    fs::create_dir_all(&downloaded_model).expect("downloaded model root");
    for artifact in ARTIFACTS {
        fs::write(downloaded_model.join(artifact), artifact).expect("downloaded artifact");
    }
    assert_eq!(
        store
            .resolve(&id)
            .expect("resolve")
            .expect("downloaded model")
            .root,
        downloaded_model
    );
    store.remove(&id).expect("remove downloaded model");
    assert!(!downloaded_model.exists());
    assert!(store.resolve(&id).expect("resolve after removal").is_none());
    for artifact in ARTIFACTS {
        assert_eq!(
            fs::read_to_string(outside_model.join(artifact)).expect("outside artifact remains"),
            artifact
        );
    }
}
