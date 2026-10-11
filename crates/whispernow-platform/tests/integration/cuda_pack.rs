//! The CUDA pack's installer, against fixture archives served locally.
#![cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
))]

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
use whispernow_domain::{ClearDataCategory, DownloadPhase};
use whispernow_platform::{
    AppPaths, CUDA_LIBRARIES, CudaPackArchive, CudaPackFormat, CudaPackRemovalOutcome,
    CudaPackState, DownloadProgress, DownloadTimeouts, ModelStore, ModelStoreError,
};
use zip::write::SimpleFileOptions;

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
    manager: ModelStore,
    paths: AppPaths,
}

/// A manager whose pack is the two fixture archives, the runtime's digest replaced when
/// `runtime_sha256` is given.
fn fixture(runtime_sha256: Option<&'static str>) -> (Fixture, FixtureServer, u64) {
    let libraries = CUDA_LIBRARIES.expect("this platform has a CUDA pack");
    fixture_archives(
        backend_archive(libraries.backend, &contents(libraries.backend)),
        runtime_archive(&libraries.runtime),
        runtime_sha256,
    )
}

fn fixture_archives(
    backend: Vec<u8>,
    runtime: Vec<u8>,
    runtime_sha256: Option<&'static str>,
) -> (Fixture, FixtureServer, u64) {
    let libraries = CUDA_LIBRARIES.expect("this platform has a CUDA pack");
    let backend_size = backend.len() as u64;
    let runtime_size = runtime.len() as u64;
    let backend_digest = sha256(&backend);
    let runtime_digest = runtime_sha256.unwrap_or_else(|| sha256(&runtime));
    let server = FixtureServer::start([
        ("/backend.tar.gz", Response::ok(backend)),
        ("/runtime.zip", Response::ok(runtime)),
    ]);
    let backend_member: &'static str = format!("release/{}", libraries.backend).leak();
    let runtime_members: Vec<(&'static str, &'static str)> = libraries
        .runtime
        .iter()
        .map(|name| (*name, *name))
        .collect();
    let archives = vec![
        CudaPackArchive {
            url: server.url("/backend.tar.gz").leak(),
            size_bytes: backend_size,
            sha256: backend_digest,
            format: CudaPackFormat::TarGz,
            members: vec![(backend_member, libraries.backend)].leak(),
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
    let manager = ModelStore::with_registry_and_timeouts(
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
            manager,
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
    let manager = &fixture.manager;
    assert_eq!(manager.cuda_pack_state(), CudaPackState::NotInstalled);
    assert_eq!(manager.cuda_pack_size(), total);
    let (progress, observer) = recorded();
    manager
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("installed");

    assert_eq!(manager.cuda_pack_state(), CudaPackState::Installed);
    let directory = manager.cuda_pack_dir().expect("installed directory");
    let libraries = CUDA_LIBRARIES.expect("pack");
    for name in libraries.files() {
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
        .manager
        .download_cuda_pack(&CancellationToken::default(), &observer);
    assert_eq!(result, Err(ModelStoreError::ChecksumMismatch));
    assert_eq!(
        fixture.manager.cuda_pack_state(),
        CudaPackState::NotInstalled
    );
    assert!(fixture.manager.cuda_pack_dir().is_none());
}

#[test]
fn a_missing_runtime_library_reports_the_archive_member_and_installs_nothing() {
    let libraries = CUDA_LIBRARIES.expect("pack");
    let runtime = runtime_archive(&[libraries.runtime[0], libraries.runtime[2]]);
    let (fixture, _server, _) = fixture_archives(
        backend_archive(libraries.backend, &contents(libraries.backend)),
        runtime,
        None,
    );
    assert_eq!(
        fixture
            .manager
            .download_cuda_pack(&CancellationToken::default(), &|_| {}),
        Err(ModelStoreError::MissingArtifacts {
            artifacts: vec![libraries.runtime[1].to_owned()],
        })
    );
    assert!(fixture.manager.cuda_pack_dir().is_none());
    assert!(!fixture.paths.cuda_pack_dir().exists());
}

#[test]
fn an_empty_backend_is_rejected_before_installation() {
    let libraries = CUDA_LIBRARIES.expect("pack");
    let (fixture, _server, _) = fixture_archives(
        backend_archive(libraries.backend, &[]),
        runtime_archive(&libraries.runtime),
        None,
    );
    assert_eq!(
        fixture
            .manager
            .download_cuda_pack(&CancellationToken::default(), &|_| {}),
        Err(ModelStoreError::InvalidArchive)
    );
    assert!(!fixture.paths.cuda_pack_dir().exists());
}

#[test]
fn a_link_cannot_supply_the_required_backend() {
    let libraries = CUDA_LIBRARIES.expect("pack");
    let member = format!("release/{}", libraries.backend);
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
        let (fixture, _server, _) =
            fixture_archives(backend, runtime_archive(&libraries.runtime), None);
        assert_eq!(
            fixture
                .manager
                .download_cuda_pack(&CancellationToken::default(), &|_| {}),
            Err(ModelStoreError::MissingArtifacts {
                artifacts: vec![member.clone()],
            })
        );
        assert!(!fixture.paths.cuda_pack_dir().exists());
    }
}

#[test]
fn a_pack_in_use_is_only_marked_for_removal_and_goes_at_the_next_start() {
    let (fixture, _server, _) = fixture(None);
    let manager = &fixture.manager;
    let (_, observer) = recorded();
    manager
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("installed");
    let directory = manager.cuda_pack_dir().expect("installed");

    assert_eq!(
        manager.remove_cuda_pack(true),
        Ok(CudaPackRemovalOutcome::Deferred)
    );
    assert!(directory.exists(), "loaded files stay");
    assert_eq!(manager.cuda_pack_state(), CudaPackState::NotInstalled);
    assert!(manager.cuda_pack_dir().is_none(), "nothing loads it again");

    let models = fixture.paths.models_dir();
    fs::create_dir_all(&models).expect("models directory");
    fs::write(models.join("weights"), b"model data").expect("model data");
    assert!(
        fixture
            .paths
            .data_locations()
            .expect("data locations")
            .iter()
            .any(|location| {
                location.category == ClearDataCategory::Models && location.path == directory
            })
    );
    let cleared = fixture
        .paths
        .clear_data_except(&[ClearDataCategory::Models], Some(&directory))
        .expect("clear other model data");
    assert_eq!(cleared.len(), 1);
    assert_eq!(cleared[0].path, models);
    assert!(cleared[0].outcome.is_ok());
    assert!(directory.exists(), "loaded files survive data clearing");
    assert!(
        manager.cuda_pack_dir().is_none(),
        "the removal marker survives"
    );

    // Downloading it again keeps the files there instead.
    let (again, observer) = recorded();
    manager
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("kept");
    assert_eq!(manager.cuda_pack_state(), CudaPackState::Installed);
    assert!(
        again
            .lock()
            .expect("progress")
            .iter()
            .all(|event| event.phase == DownloadPhase::Complete),
        "nothing downloaded"
    );

    assert_eq!(
        manager.remove_cuda_pack(true),
        Ok(CudaPackRemovalOutcome::Deferred)
    );
    manager.finish_cuda_pack_removal();
    assert!(!directory.exists());
    assert_eq!(
        manager.remove_cuda_pack(false),
        Err(ModelStoreError::NotInstalled)
    );
}

#[test]
fn a_pack_not_in_use_is_removed_at_once() {
    let (fixture, _server, _) = fixture(None);
    let (_, observer) = recorded();
    fixture
        .manager
        .download_cuda_pack(&CancellationToken::default(), &observer)
        .expect("installed");
    let directory = fixture.manager.cuda_pack_dir().expect("installed");
    assert_eq!(
        fixture.manager.remove_cuda_pack(false),
        Ok(CudaPackRemovalOutcome::Removed)
    );
    assert!(!directory.exists());
    assert_eq!(
        fixture.manager.cuda_pack_state(),
        CudaPackState::NotInstalled
    );
}

#[test]
fn without_archives_there_is_no_pack() {
    let root = tempdir().expect("temporary root");
    let paths = AppPaths::from_base_dirs(root.path().join("config"), root.path().join("data"));
    let manager = ModelStore::new(paths).with_cuda_pack(&[]);
    assert_eq!(manager.cuda_pack_state(), CudaPackState::Unsupported);
    let (_, observer) = recorded();
    assert_eq!(
        manager.download_cuda_pack(&CancellationToken::default(), &observer),
        Err(ModelStoreError::Unsupported)
    );
}
