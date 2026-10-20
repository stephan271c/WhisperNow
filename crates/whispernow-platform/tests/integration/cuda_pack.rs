//! The CUDA pack's installer, against fixture archives served locally.

use std::{
    fs,
    io::Write,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::common::{self, FixtureServer, Response};
use flate2::{Compression, write::GzEncoder};
use tar::{Builder as TarBuilder, EntryType, Header};
use tempfile::{TempDir, tempdir};
use whispernow_core::CancellationToken;
use whispernow_domain::DownloadPhase;
use whispernow_platform::{
    AppPaths, CudaPackArchive, CudaPackFormat, CudaPackRemovalOutcome, CudaPackState,
    DownloadProgress, DownloadTimeouts, ModelStore, ModelStoreError,
};
use zip::write::SimpleFileOptions;

const BACKEND: &str = "ggml-cuda-fixture";
const RUNTIME: [&str; 3] = ["cudart-fixture", "cublasLt-fixture", "cublas-fixture"];
const FILES: [&str; 4] = [RUNTIME[0], RUNTIME[1], RUNTIME[2], BACKEND];

fn sha256(bytes: &[u8]) -> &'static str {
    common::sha256(bytes).leak()
}

fn contents(name: &str) -> Vec<u8> {
    format!("fixture {name}").into_bytes()
}

/// The backend in a gzipped tar under a top directory, beside a symlink and another file.
fn backend_archive(backend: &str, body: &[u8]) -> Vec<u8> {
    let mut archive = TarBuilder::new(GzEncoder::new(Vec::new(), Compression::default()));
    for (path, body) in [
        (format!("release/{backend}"), body.to_vec()),
        ("release/llama-server".to_owned(), b"tool".to_vec()),
    ] {
        let mut header = Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o755);
        header.set_entry_type(EntryType::file());
        header.set_cksum();
        archive
            .append_data(&mut header, path, body.as_slice())
            .expect("append");
    }
    let mut link = Header::new_gnu();
    link.set_size(0);
    link.set_entry_type(EntryType::symlink());
    link.set_link_name(backend).expect("link");
    link.set_cksum();
    archive
        .append_data(&mut link, "release/backend-link", &[][..])
        .expect("append link");
    archive.into_inner().expect("tar").finish().expect("gzip")
}

/// The runtime's libraries in a zip, at its root.
fn runtime_archive(runtime: &[&str]) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for name in runtime {
        zip.start_file(*name, SimpleFileOptions::default())
            .expect("zip entry");
        zip.write_all(&contents(name)).expect("zip body");
    }
    zip.finish().expect("zip").into_inner()
}

struct Fixture {
    _root: TempDir,
    store: ModelStore,
    paths: AppPaths,
}

/// A store whose pack is the two fixture archives, the runtime's digest replaced when
/// `runtime_sha256` is given.
fn fixture(runtime_sha256: Option<&'static str>) -> (Fixture, FixtureServer, u64) {
    fixture_archives(
        backend_archive(BACKEND, &contents(BACKEND)),
        runtime_archive(&RUNTIME),
        runtime_sha256,
    )
}

fn fixture_archives(
    backend: Vec<u8>,
    runtime: Vec<u8>,
    runtime_sha256: Option<&'static str>,
) -> (Fixture, FixtureServer, u64) {
    let backend_size = backend.len() as u64;
    let runtime_size = runtime.len() as u64;
    let backend_digest = sha256(&backend);
    let runtime_digest = runtime_sha256.unwrap_or_else(|| sha256(&runtime));
    let server = FixtureServer::start([
        ("/backend.tar.gz", Response::ok(backend)),
        ("/runtime.zip", Response::ok(runtime)),
    ]);
    let backend_member: &'static str = format!("release/{BACKEND}").leak();
    let runtime_members: Vec<(&'static str, &'static str)> =
        RUNTIME.iter().map(|name| (*name, *name)).collect();
    let archives = vec![
        CudaPackArchive {
            url: server.url("/backend.tar.gz").leak(),
            size_bytes: backend_size,
            sha256: backend_digest,
            format: CudaPackFormat::TarGz,
            members: vec![(backend_member, BACKEND)].leak(),
        },
        CudaPackArchive {
            url: server.url("/runtime.zip").leak(),
            size_bytes: runtime_size,
            sha256: runtime_digest,
            format: CudaPackFormat::Zip,
            members: runtime_members.leak(),
        },
    ];
    let total = backend_size + runtime_size;
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let store = ModelStore::with_registry_and_timeouts(
        whispernow_domain::ModelRegistry::builtin(),
        paths.clone(),
        DownloadTimeouts {
            connect: Duration::from_secs(2),
            receive_body: Duration::from_secs(2),
            total: Duration::from_secs(10),
        },
    )
    .with_cuda_pack(archives.leak());
    (
        Fixture {
            _root: root,
            store,
            paths,
        },
        server,
        total,
    )
}

fn recorded() -> (Arc<Mutex<Vec<DownloadProgress>>>, impl Fn(DownloadProgress)) {
    let progress = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&progress);
    (progress, move |event| {
        sink.lock().expect("progress").push(event);
    })
}

#[test]
fn the_pack_installs_its_files_flat_from_a_tar_gz_and_a_zip_with_progress_over_both() {
    let (fixture, _server, total) = fixture(None);
    let store = &fixture.store;
    assert_eq!(store.cuda_pack_state(), CudaPackState::NotInstalled);
    assert_eq!(store.cuda_pack_size(), total);
    let (progress, observer) = recorded();
    store
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("installed");

    assert_eq!(store.cuda_pack_state(), CudaPackState::Installed);
    let directory = store.cuda_pack_dir().expect("installed directory");
    for name in FILES {
        assert_eq!(
            fs::read(directory.join(name)).expect(name),
            contents(name),
            "{name}"
        );
    }
    assert!(
        !directory.join("llama-server").exists(),
        "only the pack's files"
    );
    assert!(!directory.join("backend-link").exists());

    let progress = progress.lock().expect("progress");
    let downloaded: Vec<u64> = progress
        .iter()
        .filter(|event| event.phase == DownloadPhase::Downloading)
        .map(|event| event.downloaded_bytes)
        .collect();
    assert!(downloaded.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(
        progress
            .iter()
            .all(|event| event.total_bytes == Some(total))
    );
    assert_eq!(
        progress.last().map(|event| event.phase),
        Some(DownloadPhase::Complete)
    );
    let leftovers: Vec<_> = fs::read_dir(fixture.paths.data_dir())
        .expect("data directory")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn a_wrong_digest_installs_nothing() {
    let wrong = "0".repeat(64).leak();
    let (fixture, _server, _) = fixture(Some(wrong));
    let (_, observer) = recorded();
    let result = fixture
        .store
        .download_cuda_pack(&CancellationToken::default(), &observer);
    assert_eq!(result, Err(ModelStoreError::ChecksumMismatch));
    assert_eq!(fixture.store.cuda_pack_state(), CudaPackState::NotInstalled);
    assert!(fixture.store.cuda_pack_dir().is_none());
}

#[test]
fn a_missing_runtime_library_reports_the_archive_member_and_installs_nothing() {
    let runtime = runtime_archive(&[RUNTIME[0], RUNTIME[2]]);
    let (fixture, _server, _) =
        fixture_archives(backend_archive(BACKEND, &contents(BACKEND)), runtime, None);
    assert_eq!(
        fixture
            .store
            .download_cuda_pack(&CancellationToken::default(), &|_| {}),
        Err(ModelStoreError::MissingArtifacts {
            artifacts: vec![RUNTIME[1].to_owned()],
        })
    );
    assert!(fixture.store.cuda_pack_dir().is_none());
    assert!(!fixture.paths.cuda_pack_dir().exists());
}

#[test]
fn an_empty_backend_is_rejected_before_installation() {
    let (fixture, _server, _) = fixture_archives(
        backend_archive(BACKEND, &[]),
        runtime_archive(&RUNTIME),
        None,
    );
    assert_eq!(
        fixture
            .store
            .download_cuda_pack(&CancellationToken::default(), &|_| {}),
        Err(ModelStoreError::InvalidArchive)
    );
    assert!(!fixture.paths.cuda_pack_dir().exists());
}

#[test]
fn a_link_cannot_supply_the_required_backend() {
    let member = format!("release/{BACKEND}");
    for entry_type in [EntryType::symlink(), EntryType::hard_link()] {
        let mut archive = TarBuilder::new(GzEncoder::new(Vec::new(), Compression::default()));
        let mut header = Header::new_gnu();
        header.set_size(0);
        header.set_entry_type(entry_type);
        header.set_link_name("../../outside").expect("link target");
        header.set_cksum();
        archive
            .append_data(&mut header, &member, &[][..])
            .expect("link member");
        let backend = archive.into_inner().expect("tar").finish().expect("gzip");
        let (fixture, _server, _) = fixture_archives(backend, runtime_archive(&RUNTIME), None);
        assert_eq!(
            fixture
                .store
                .download_cuda_pack(&CancellationToken::default(), &|_| {}),
            Err(ModelStoreError::MissingArtifacts {
                artifacts: vec![member.clone()],
            })
        );
        assert!(!fixture.paths.cuda_pack_dir().exists());
    }
}

#[test]
fn startup_removes_old_pack_versions_and_interrupted_downloads_but_keeps_the_current_pack() {
    let (fixture, _server, _) = fixture(None);
    let paths = &fixture.paths;
    let current = paths.cuda_pack_dir();
    let previous = paths.data_dir().join("llama-cuda-b11258");
    let unversioned = paths.data_dir().join("llama-cuda");
    let stage = paths.data_dir().join(".llama-cuda-stage-interrupted");
    let download = paths
        .data_dir()
        .join(".llama-cuda-download-interrupted.part");
    let unrelated = paths.data_dir().join("llama-cuda-notes.txt");
    fs::create_dir_all(&previous).expect("previous build");
    fs::create_dir_all(&unversioned).expect("unversioned pack");
    fs::create_dir_all(&stage).expect("interrupted staging");
    fs::write(&download, b"partial archive").expect("interrupted download");
    fs::write(&unrelated, b"notes").expect("unrelated file");
    for directory in [&previous, &unversioned] {
        for name in FILES {
            fs::write(directory.join(name), contents(name)).expect("old library");
        }
    }
    assert_eq!(fixture.store.cuda_pack_state(), CudaPackState::NotInstalled);
    assert!(fixture.store.cuda_pack_dir().is_none());
    fs::create_dir_all(&current).expect("current build");
    for name in FILES {
        fs::write(current.join(name), contents(name)).expect("current library");
    }

    fixture.store.finish_cuda_pack_removal();

    assert!(!previous.exists());
    assert!(!unversioned.exists());
    assert!(!stage.exists());
    assert!(!download.exists());
    assert!(unrelated.exists());
    assert_eq!(fixture.store.cuda_pack_dir(), Some(current));
}

#[test]
fn a_pack_not_in_use_is_removed_at_once() {
    let (fixture, _server, _) = fixture(None);
    let (_, observer) = recorded();
    fixture
        .store
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("installed");
    let directory = fixture.store.cuda_pack_dir().expect("installed");
    assert_eq!(
        fixture.store.remove_cuda_pack(),
        Ok(CudaPackRemovalOutcome::Removed)
    );
    assert!(!directory.exists());
    assert_eq!(fixture.store.cuda_pack_state(), CudaPackState::NotInstalled);
}

#[test]
fn without_archives_there_is_no_pack() {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let store = ModelStore::new(paths).with_cuda_pack(&[]);
    assert_eq!(store.cuda_pack_state(), CudaPackState::Unsupported);
    let (_, observer) = recorded();
    assert_eq!(
        store.download_cuda_pack(&CancellationToken::default(), &observer),
        Err(ModelStoreError::Unsupported)
    );
}
