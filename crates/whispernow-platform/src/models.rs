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
    ModelSource,
};

use crate::{
    AppPaths,
    atomic_file::{remove_path, sync_directory},
};

const DOWNLOAD_BUFFER_BYTES: usize = 64 * 1024;
const MODEL_DOWNLOAD_PREFIX: &str = ".model-download-";
const MODEL_STAGE_PREFIX: &str = ".model-stage-";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownloadProgress {
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

impl DownloadProgress {
    const fn new(phase: DownloadPhase, downloaded_bytes: u64, total_bytes: u64) -> Self {
        Self {
            phase,
            downloaded_bytes,
            total_bytes,
        }
    }

    const fn after_transfer(phase: DownloadPhase, total_bytes: u64) -> Self {
        Self::new(phase, total_bytes, total_bytes)
    }
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
    #[error("model archive is not a valid supported archive: {kind}")]
    InvalidArchive { kind: io::ErrorKind },
    #[error("model archive is missing required artifacts: {artifacts:?}")]
    MissingArtifacts { artifacts: Vec<String> },
    #[error("model filesystem operation failed during {operation}: {kind}")]
    Filesystem {
        operation: &'static str,
        kind: io::ErrorKind,
    },
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
    // Model reads must not see replacement gaps.
    mutation: Mutex<()>,
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
                let partial_download = name.starts_with(MODEL_DOWNLOAD_PREFIX)
                    && Path::new(&*name)
                        .extension()
                        .is_some_and(|extension| extension == "part");
                if (name.starts_with(MODEL_STAGE_PREFIX) || partial_download)
                    && let Err(error) = remove_path(&entry.path())
                {
                    tracing::warn!(%error, "stale download files could not be removed");
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

    fn model_root(&self, definition: &ModelDefinition) -> PathBuf {
        self.paths.models_dir().join(definition.id.as_str())
    }

    /// Extracts the required files, which the archive holds under a directory named after
    /// the model, into `staging_base`.
    fn extract_archive(
        definition: &ModelDefinition,
        archive_file: &Path,
        staging_base: &Path,
        cancellation: &CancellationToken,
    ) -> Result<PathBuf, ModelStoreError> {
        let file = File::open(archive_file).map_err(filesystem("open model archive"))?;
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
            fs::create_dir_all(parent).map_err(filesystem("create model staging directory"))?;
        }
        download
            .persist_noclobber(&destination)
            .map_err(filesystem("stage model artifact"))?;
        Ok(staged_root)
    }

    fn install_staged(
        &self,
        definition: &ModelDefinition,
        staged_root: &Path,
        cancellation: &CancellationToken,
    ) -> Result<PathBuf, ModelStoreError> {
        check_cancelled(cancellation)?;
        let destination = self.model_root(definition);
        if fs::symlink_metadata(&destination).is_ok() {
            remove_path(&destination).map_err(filesystem("replace installed model"))?;
        }
        fs::rename(staged_root, &destination).map_err(filesystem("install staged model"))?;
        sync_directory(&self.paths.models_dir()).map_err(filesystem("sync models directory"))?;
        Ok(destination)
    }
}

async fn download_file(
    client: &reqwest::Client,
    source: &ModelSource,
    destination: &mut NamedTempFile,
    cancellation: &CancellationToken,
    observer: &dyn Fn(DownloadProgress),
) -> Result<(), ModelStoreError> {
    check_cancelled(cancellation)?;
    let mut response = client
        .get(&source.url)
        .send()
        .await
        .map_err(|error| map_reqwest_error(&error))?;
    let status = response.status();
    if !status.is_success() {
        return Err(ModelStoreError::HttpStatus {
            status: status.as_u16(),
        });
    }
    let expected = source.archive_size_bytes;
    let reported = response.content_length();
    if let Some(actual) = reported
        && actual != expected
    {
        return Err(ModelStoreError::ResponseLength { expected, actual });
    }

    let mut received = 0_u64;
    let mut hasher = Sha256::new();
    observer(DownloadProgress::new(
        DownloadPhase::Downloading,
        0,
        expected,
    ));
    loop {
        check_cancelled(cancellation)?;
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(error) if ended_before_declared_length(&error) => {
                return Err(ModelStoreError::ResponseLength {
                    expected,
                    actual: received,
                });
            }
            Err(error) => return Err(map_reqwest_error(&error)),
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
            .map_err(filesystem("write download staging file"))?;
        hasher.update(&chunk);
        observer(DownloadProgress::new(
            DownloadPhase::Downloading,
            received,
            expected,
        ));
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
        .map_err(filesystem("sync download staging file"))?;
    let actual_sha256 = hex_lower(&hasher.finalize());
    if actual_sha256 != source.sha256 {
        return Err(ModelStoreError::ChecksumMismatch);
    }
    Ok(())
}

impl ModelStore {
    pub fn resolve(&self, model_id: &ModelId) -> Result<Option<ResolvedModel>, ModelStoreError> {
        let definition = self.definition(model_id)?;
        let root = self.model_root(definition);
        let _guard = self.lock_store();
        if !validate_model_root(definition, &root)? {
            return Ok(None);
        }
        Ok(Some(ResolvedModel {
            definition: definition.clone(),
            root,
        }))
    }

    /// An installed cleanup model's GGUF; `None` while it is not installed, or the id is a
    /// speech model's.
    pub fn cleanup_model_path(
        &self,
        model_id: &ModelId,
    ) -> Result<Option<PathBuf>, ModelStoreError> {
        let Some(model) = self.resolve(model_id)? else {
            return Ok(None);
        };
        let ModelLayout::Gguf { model: file } = &model.definition.layout else {
            return Ok(None);
        };
        Ok(Some(model.root.join(file)))
    }

    pub fn list(&self) -> Result<Vec<ModelListEntry>, ModelStoreError> {
        let _guard = self.lock_store();
        self.registry
            .models()
            .iter()
            .map(|definition| {
                let installed = validate_model_root(definition, &self.model_root(definition))?;
                let availability = if installed {
                    ModelAvailability::Downloaded
                } else {
                    ModelAvailability::NotInstalled
                };
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
        remove_path(&self.model_root(definition)).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                ModelStoreError::NotInstalled
            } else {
                filesystem("remove downloaded model")(error)
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
            observer(DownloadProgress::after_transfer(
                phase,
                definition.source.archive_size_bytes,
            ));
        };
        let models_dir = self.paths.models_dir();
        fs::create_dir_all(&models_dir).map_err(filesystem("create models directory"))?;
        let mut download = Builder::new()
            .prefix(MODEL_DOWNLOAD_PREFIX)
            .suffix(".part")
            .tempfile_in(&models_dir)
            .map_err(filesystem("create download staging file"))?;
        self.runtime.block_on(download_file(
            &self.client,
            &definition.source,
            &mut download,
            cancellation,
            observer,
        ))?;
        report_phase(DownloadPhase::ValidatingArchive);
        check_cancelled(cancellation)?;
        let staging = Builder::new()
            .prefix(MODEL_STAGE_PREFIX)
            .tempdir_in(&models_dir)
            .map_err(filesystem("create model staging directory"))?;
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
        Err(error) => return Err(filesystem("inspect model root")(error)),
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
            Err(error) => return Err(filesystem("inspect model artifact")(error)),
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
    let entries = archive.entries().map_err(|error| invalid_archive(&error))?;
    for entry in entries {
        check_cancelled(cancellation)?;
        let mut entry = entry.map_err(|error| invalid_archive(&error))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|error| invalid_archive(&error))?
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
        fs::create_dir_all(parent).map_err(filesystem("create model staging directory"))?;
    }
    let mut output = File::options()
        .create_new(true)
        .write(true)
        .open(destination)
        .map_err(filesystem("create staged model artifact"))?;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        check_cancelled(cancellation)?;
        let count = input
            .read(&mut buffer)
            .map_err(|error| invalid_archive(&error))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(filesystem("write staged model artifact"))?;
    }
    output
        .flush()
        .and_then(|()| output.sync_all())
        .map_err(filesystem("sync staged model artifact"))
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

fn filesystem<E: Into<io::Error>>(operation: &'static str) -> impl FnOnce(E) -> ModelStoreError {
    move |error| ModelStoreError::Filesystem {
        operation,
        kind: error.into().kind(),
    }
}

fn invalid_archive(error: &io::Error) -> ModelStoreError {
    ModelStoreError::InvalidArchive { kind: error.kind() }
}
