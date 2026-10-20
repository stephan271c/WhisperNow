use std::{
    error::Error as _,
    fs::{self, File},
    io::{self, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    sync::{Mutex, MutexGuard, PoisonError},
    time::Duration,
};

use bzip2::read::BzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;
use tempfile::{Builder, NamedTempFile};
use thiserror::Error;
use whispernow_core::{CancellationToken, ResolvedModel};
use whispernow_domain::{
    DownloadPhase, ModelAvailability, ModelDefinition, ModelId, ModelLayout, ModelRegistry,
};

use crate::{AppPaths, atomic_file::sync_directory, clear_data::remove_path};

mod cuda_pack;

pub use self::cuda_pack::{CudaPackArchive, CudaPackFormat, CudaPackRemovalOutcome, CudaPackState};

const DOWNLOAD_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownloadProgress {
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelListEntry {
    pub definition: ModelDefinition,
    pub availability: ModelAvailability,
}

/// Raw URLs, native paths, response bodies, and archive contents are intentionally excluded.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ModelStoreError {
    #[error("unknown model identifier")]
    UnknownModel,
    #[error("model is not installed")]
    NotInstalled,
    #[error("model operation was cancelled")]
    Cancelled,
    #[error("model download timed out")]
    Timeout,
    #[error("model server returned HTTP status {status}")]
    HttpStatus { status: u16 },
    #[error("model download transport failed")]
    Transport,
    #[error("model response length mismatch: expected {expected}, received {actual}")]
    ResponseLength { expected: u64, actual: u64 },
    #[error("model archive checksum does not match the registry")]
    ChecksumMismatch,
    #[error("model archive is not a valid supported archive")]
    InvalidArchive,
    #[error("this platform has no CUDA pack")]
    Unsupported,
    #[error("model archive is missing required artifacts: {artifacts:?}")]
    MissingArtifacts { artifacts: Vec<String> },
    #[error("model filesystem operation failed during {operation}")]
    Filesystem { operation: &'static str },
}

#[derive(Clone, Copy, Debug)]
pub struct DownloadTimeouts {
    pub connect: Duration,
    pub receive_body: Duration,
    pub total: Duration,
}

impl Default for DownloadTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(30),
            receive_body: Duration::from_secs(30),
            total: Duration::from_hours(2),
        }
    }
}

/// Transfers and extraction run unlocked; model resolution, listing, removals and final
/// installs share the store lock.
pub struct ModelStore {
    registry: ModelRegistry,
    paths: AppPaths,
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    // Model reads must not see replacement gaps; CUDA pack mutations also share this lock.
    mutation: Mutex<()>,
    cuda_pack: &'static [CudaPackArchive],
}

impl ModelStore {
    #[must_use]
    pub fn new(paths: AppPaths) -> Self {
        Self::with_registry_and_timeouts(
            ModelRegistry::builtin(),
            paths,
            DownloadTimeouts::default(),
        )
    }

    /// Injects a registry and timeouts for fixture-server tests.
    #[must_use]
    pub fn with_registry_and_timeouts(
        registry: ModelRegistry,
        paths: AppPaths,
        timeouts: DownloadTimeouts,
    ) -> Self {
        if let Ok(entries) = fs::read_dir(paths.models_dir()) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if (name.starts_with(".model-stage-")
                    || (name.starts_with(".model-download-") && name.ends_with(".part")))
                    && let Err(error) = remove_path(&entry.path())
                {
                    tracing::warn!(%error, "model download staging could not be removed");
                }
            }
        }
        let client = reqwest::Client::builder()
            .connect_timeout(timeouts.connect)
            .read_timeout(timeouts.receive_body)
            .timeout(timeouts.total)
            .build()
            .expect("the model download client builds");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("the model download runtime builds");
        Self {
            registry,
            paths,
            runtime,
            client,
            mutation: Mutex::new(()),
            cuda_pack: cuda_pack::CUDA_PACK,
        }
    }

    fn lock_store(&self) -> MutexGuard<'_, ()> {
        self.mutation.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn definition(&self, model_id: &ModelId) -> Result<&ModelDefinition, ModelStoreError> {
        self.registry
            .get(model_id.as_str())
            .ok_or(ModelStoreError::UnknownModel)
    }

    fn resolve_unlocked(
        &self,
        definition: &ModelDefinition,
    ) -> Result<Option<ResolvedModel>, ModelStoreError> {
        let downloaded_root = self.paths.models_dir().join(definition.id.as_str());
        if validate_model_root(definition, &downloaded_root)? {
            return Ok(Some(ResolvedModel {
                definition: definition.clone(),
                root: downloaded_root,
            }));
        }
        Ok(None)
    }

    /// Extracts the required files, which the archive holds under a directory named after
    /// the model, into `staging_base`.
    fn extract_archive(
        definition: &ModelDefinition,
        archive_file: &Path,
        staging_base: &Path,
        cancellation: &CancellationToken,
    ) -> Result<PathBuf, ModelStoreError> {
        let file = File::open(archive_file).map_err(|_| filesystem("open model archive"))?;
        let archive_root = Path::new(definition.source.archive_root());
        let staged_root = staging_base.join(definition.id.as_str());
        let mut members: Vec<_> = definition
            .layout
            .required_artifacts()
            .into_iter()
            .map(|artifact| (archive_root.join(artifact), artifact))
            .collect();
        extract_tar(
            BzDecoder::new(BufReader::new(file)),
            &staged_root,
            &mut members,
            cancellation,
        )?;
        if !members.is_empty() {
            let mut artifacts: Vec<String> = members
                .into_iter()
                .map(|(_, artifact)| artifact.to_owned())
                .collect();
            artifacts.sort();
            return Err(ModelStoreError::MissingArtifacts { artifacts });
        }
        Ok(staged_root)
    }

    fn stage_single_file(
        definition: &ModelDefinition,
        download: NamedTempFile,
        staging_base: &Path,
        artifact: &str,
    ) -> Result<PathBuf, ModelStoreError> {
        let staged_root = staging_base.join(definition.id.as_str());
        let destination = staged_root.join(artifact);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|_| filesystem("create model staging directory"))?;
        }
        download
            .persist_noclobber(&destination)
            .map_err(|_| filesystem("stage model artifact"))?;
        Ok(staged_root)
    }

    fn install_staged(
        &self,
        definition: &ModelDefinition,
        staged_root: &Path,
        cancellation: &CancellationToken,
    ) -> Result<PathBuf, ModelStoreError> {
        check_cancelled(cancellation)?;
        let models_dir = self.paths.models_dir();
        let destination = models_dir.join(definition.id.as_str());
        if fs::symlink_metadata(&destination).is_ok() {
            remove_path(&destination).map_err(|_| filesystem("replace installed model"))?;
        }
        fs::rename(staged_root, &destination).map_err(|_| filesystem("install staged model"))?;
        sync_directory(&models_dir).map_err(|_| filesystem("sync models directory"))?;
        Ok(destination)
    }
}

struct PinnedFile<'a> {
    url: &'a str,
    size_bytes: u64,
    sha256: &'a str,
}

async fn download_file(
    client: &reqwest::Client,
    file: &PinnedFile<'_>,
    destination: &mut NamedTempFile,
    cancellation: &CancellationToken,
    observer: &dyn Fn(DownloadProgress),
) -> Result<(), ModelStoreError> {
    check_cancelled(cancellation)?;
    let mut response = client
        .get(file.url)
        .send()
        .await
        .map_err(|error| map_reqwest_error(&error))?;
    let status = response.status();
    if !status.is_success() {
        return Err(ModelStoreError::HttpStatus {
            status: status.as_u16(),
        });
    }
    let expected = file.size_bytes;
    let reported = response.content_length();
    if let Some(actual) = reported
        && actual != expected
    {
        return Err(ModelStoreError::ResponseLength { expected, actual });
    }

    let mut received = 0_u64;
    let mut hasher = Sha256::new();
    observer(DownloadProgress {
        phase: DownloadPhase::Downloading,
        downloaded_bytes: 0,
        total_bytes: Some(expected),
    });
    loop {
        check_cancelled(cancellation)?;
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(error) if error.is_timeout() => return Err(ModelStoreError::Timeout),
            Err(error) if ended_before_declared_length(&error) => {
                return Err(ModelStoreError::ResponseLength {
                    expected,
                    actual: received,
                });
            }
            Err(_) => return Err(ModelStoreError::Transport),
        };
        received = received.saturating_add(chunk.len() as u64);
        if received > expected {
            return Err(ModelStoreError::ResponseLength {
                expected,
                actual: received,
            });
        }
        destination
            .write_all(&chunk)
            .map_err(|_| filesystem("write download staging file"))?;
        hasher.update(&chunk);
        observer(DownloadProgress {
            phase: DownloadPhase::Downloading,
            downloaded_bytes: received,
            total_bytes: Some(expected),
        });
    }
    if received != expected {
        return Err(ModelStoreError::ResponseLength {
            expected,
            actual: received,
        });
    }
    destination
        .flush()
        .and_then(|()| destination.as_file().sync_all())
        .map_err(|_| filesystem("sync download staging file"))?;
    let actual_sha256 = hex_lower(&hasher.finalize());
    if actual_sha256 != file.sha256 {
        return Err(ModelStoreError::ChecksumMismatch);
    }
    Ok(())
}

impl ModelStore {
    pub fn resolve(&self, model_id: &ModelId) -> Result<Option<ResolvedModel>, ModelStoreError> {
        let _guard = self.lock_store();
        self.resolve_unlocked(self.definition(model_id)?)
    }

    pub fn list(&self) -> Result<Vec<ModelListEntry>, ModelStoreError> {
        let _guard = self.lock_store();
        self.registry
            .models()
            .iter()
            .map(|definition| {
                let availability = self
                    .resolve_unlocked(definition)?
                    .map_or(ModelAvailability::NotInstalled, |_| {
                        ModelAvailability::Downloaded
                    });
                Ok(ModelListEntry {
                    definition: definition.clone(),
                    availability,
                })
            })
            .collect()
    }

    pub fn remove(&self, model_id: &ModelId) -> Result<(), ModelStoreError> {
        let _guard = self.lock_store();
        let definition = self.definition(model_id)?;
        let destination = self.paths.models_dir().join(definition.id.as_str());
        remove_path(&destination).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                ModelStoreError::NotInstalled
            } else {
                filesystem("remove downloaded model")
            }
        })
    }

    pub fn download(
        &self,
        model_id: &ModelId,
        cancellation: &CancellationToken,
        observer: &dyn Fn(DownloadProgress),
    ) -> Result<ResolvedModel, ModelStoreError> {
        let definition = self.definition(model_id)?.clone();
        let report_phase = |phase| {
            observer(DownloadProgress {
                phase,
                downloaded_bytes: definition.source.archive_size_bytes,
                total_bytes: Some(definition.source.archive_size_bytes),
            });
        };
        fs::create_dir_all(self.paths.models_dir())
            .map_err(|_| filesystem("create models directory"))?;
        let mut download = Builder::new()
            .prefix(".model-download-")
            .suffix(".part")
            .tempfile_in(self.paths.models_dir())
            .map_err(|_| filesystem("create download staging file"))?;
        self.runtime.block_on(download_file(
            &self.client,
            &PinnedFile {
                url: &definition.source.url,
                size_bytes: definition.source.archive_size_bytes,
                sha256: &definition.source.sha256,
            },
            &mut download,
            cancellation,
            observer,
        ))?;
        report_phase(DownloadPhase::ValidatingArchive);
        check_cancelled(cancellation)?;
        let staging = Builder::new()
            .prefix(".model-stage-")
            .tempdir_in(self.paths.models_dir())
            .map_err(|_| filesystem("create model staging directory"))?;
        report_phase(DownloadPhase::Extracting);
        let staged_root = match &definition.layout {
            ModelLayout::Gguf { model: artifact } => {
                Self::stage_single_file(&definition, download, staging.path(), artifact)?
            }
            _ => Self::extract_archive(&definition, download.path(), staging.path(), cancellation)?,
        };
        report_phase(DownloadPhase::Installing);
        let root = {
            let _guard = self.lock_store();
            self.install_staged(&definition, &staged_root, cancellation)?
        };
        report_phase(DownloadPhase::Complete);
        Ok(ResolvedModel { definition, root })
    }
}

fn validate_model_root(definition: &ModelDefinition, root: &Path) -> Result<bool, ModelStoreError> {
    let root_metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(_) => return Err(filesystem("inspect model root")),
    };
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Ok(false);
    }
    for artifact in definition.layout.required_artifacts() {
        if !validate_artifact_path(root, Path::new(artifact))? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_artifact_path(root: &Path, relative: &Path) -> Result<bool, ModelStoreError> {
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(component) = component else {
            return Ok(false);
        };
        current.push(component);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(_) => return Err(filesystem("inspect model artifact")),
        };
        if metadata.file_type().is_symlink() {
            return Ok(false);
        }
        let is_leaf = index + 1 == components.len();
        if (is_leaf && (!metadata.is_file() || metadata.len() == 0))
            || (!is_leaf && !metadata.is_dir())
        {
            return Ok(false);
        }
    }
    Ok(!components.is_empty())
}

fn extract_tar(
    input: impl Read,
    into: &Path,
    members: &mut Vec<(PathBuf, &str)>,
    cancellation: &CancellationToken,
) -> Result<(), ModelStoreError> {
    let mut archive = Archive::new(input);
    let entries = archive
        .entries()
        .map_err(|_| ModelStoreError::InvalidArchive)?;
    for entry in entries {
        check_cancelled(cancellation)?;
        let mut entry = entry.map_err(|_| ModelStoreError::InvalidArchive)?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|_| ModelStoreError::InvalidArchive)?
            .into_owned();
        // Some upstream archives name their entries `./<model>/…`.
        let path = path.strip_prefix(".").unwrap_or(&path);
        let Some(index) = members.iter().position(|(member, _)| path == member) else {
            continue;
        };
        let (_, name) = members.swap_remove(index);
        write_member(&mut entry, &into.join(name), cancellation)?;
    }
    Ok(())
}

fn write_member(
    input: &mut impl Read,
    destination: &Path,
    cancellation: &CancellationToken,
) -> Result<(), ModelStoreError> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|_| filesystem("create model staging directory"))?;
    }
    let mut output = File::options()
        .create_new(true)
        .write(true)
        .open(destination)
        .map_err(|_| filesystem("create staged model artifact"))?;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        check_cancelled(cancellation)?;
        let count = input
            .read(&mut buffer)
            .map_err(|_| ModelStoreError::InvalidArchive)?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|_| filesystem("write staged model artifact"))?;
    }
    output
        .flush()
        .and_then(|()| output.sync_all())
        .map_err(|_| filesystem("sync staged model artifact"))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), ModelStoreError> {
    if cancellation.is_cancelled() {
        Err(ModelStoreError::Cancelled)
    } else {
        Ok(())
    }
}

fn map_reqwest_error(error: &reqwest::Error) -> ModelStoreError {
    if error.is_timeout() {
        ModelStoreError::Timeout
    } else {
        ModelStoreError::Transport
    }
}

/// A body that closes before its declared `Content-Length` surfaces as an
/// unexpected-EOF cause; any other mid-stream failure is a transport error.
fn ended_before_declared_length(error: &reqwest::Error) -> bool {
    let mut source = error.source();
    while let Some(cause) = source {
        if cause
            .downcast_ref::<io::Error>()
            .is_some_and(|io_error| io_error.kind() == io::ErrorKind::UnexpectedEof)
        {
            return true;
        }
        source = cause.source();
    }
    false
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _infallible_for_string = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn filesystem(operation: &'static str) -> ModelStoreError {
    ModelStoreError::Filesystem { operation }
}
