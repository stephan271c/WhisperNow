//! The CUDA pack: llama.cpp's CUDA backend and NVIDIA's CUDA runtime, which run the cleanup
//! models on an NVIDIA GPU with CUDA instead of Vulkan.
//!
//! A loaded pack cannot be unloaded, and on Windows its files cannot be deleted while
//! loaded, so removing a pack in use only marks it; it is deleted when the app next starts.

use std::{
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use tempfile::Builder;
use whispernow_core::CancellationToken;
use whispernow_domain::DownloadPhase;

use super::{
    DownloadProgress, ModelStore, ModelStoreError, PinnedFile, check_cancelled, download_file,
    extract_tar, filesystem, remove_path, sync_directory, validate_artifact_path, write_member,
};
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
))]
use crate::llama::{CUDA_PACK_LIBRARIES, llama_cpp_build};

/// Left in a pack that was removed while loaded; such a pack is deleted at the next start.
const REMOVAL_MARKER: &str = ".whispernow-remove";

#[derive(Clone, Copy, Debug)]
pub struct CudaPackArchive {
    pub url: &'static str,
    pub size_bytes: u64,
    /// Lowercase SHA-256 digest of the archive.
    pub sha256: &'static str,
    pub format: CudaPackFormat,
    /// Each file's path in the archive, and its name in the pack.
    pub members: &'static [(&'static str, &'static str)],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPackFormat {
    TarGz,
    Zip,
}

/// The pinned llama.cpp build's CUDA 13.4 assets; the digests are the release's own.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) const CUDA_PACK: &[CudaPackArchive] = &[
    CudaPackArchive {
        url: concat!(
            "https://github.com/ggml-org/llama.cpp/releases/download/",
            llama_cpp_build!(),
            "/llama-",
            llama_cpp_build!(),
            "-bin-ubuntu-cuda-13.4-x64.tar.gz"
        ),
        size_bytes: 152_878_063,
        sha256: "ccd87e3754b7445a45231090e509b9142e171240f9dbc20eb9358e9f7ca9dfa6",
        format: CudaPackFormat::TarGz,
        members: &[(
            concat!("llama-", llama_cpp_build!(), "/libggml-cuda.so"),
            CUDA_PACK_LIBRARIES.backend,
        )],
    },
    CudaPackArchive {
        url: concat!(
            "https://github.com/ggml-org/llama.cpp/releases/download/",
            llama_cpp_build!(),
            "/cudart-llama-",
            llama_cpp_build!(),
            "-bin-ubuntu-cuda-13.4-x64.tar.gz"
        ),
        size_bytes: 440_236_546,
        sha256: "8beb2d945a6087810029182fd9b214d2d89b3bcc360483599625209cc136b951",
        format: CudaPackFormat::TarGz,
        members: &[
            (
                concat!(
                    "cudart-llama-",
                    llama_cpp_build!(),
                    "-bin-ubuntu-cuda-13.4-x64/libcudart.so.13"
                ),
                CUDA_PACK_LIBRARIES.runtime[0],
            ),
            (
                concat!(
                    "cudart-llama-",
                    llama_cpp_build!(),
                    "-bin-ubuntu-cuda-13.4-x64/libcublasLt.so.13"
                ),
                CUDA_PACK_LIBRARIES.runtime[1],
            ),
            (
                concat!(
                    "cudart-llama-",
                    llama_cpp_build!(),
                    "-bin-ubuntu-cuda-13.4-x64/libcublas.so.13"
                ),
                CUDA_PACK_LIBRARIES.runtime[2],
            ),
        ],
    },
];

/// The pinned llama.cpp build's CUDA 13.4 assets; the digests are the release's own.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) const CUDA_PACK: &[CudaPackArchive] = &[
    CudaPackArchive {
        url: concat!(
            "https://github.com/ggml-org/llama.cpp/releases/download/",
            llama_cpp_build!(),
            "/llama-",
            llama_cpp_build!(),
            "-bin-win-cuda-13.4-x64.zip"
        ),
        size_bytes: 153_546_058,
        sha256: "7e93d79ed0dfacb67a7a5448eab38b60511ec3ad623022259b08490d5cf01404",
        format: CudaPackFormat::Zip,
        members: &[(CUDA_PACK_LIBRARIES.backend, CUDA_PACK_LIBRARIES.backend)],
    },
    CudaPackArchive {
        url: concat!(
            "https://github.com/ggml-org/llama.cpp/releases/download/",
            llama_cpp_build!(),
            "/cudart-llama-bin-win-cuda-13.4-x64.zip"
        ),
        size_bytes: 423_535_356,
        sha256: "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        format: CudaPackFormat::Zip,
        members: &[
            (
                CUDA_PACK_LIBRARIES.runtime[0],
                CUDA_PACK_LIBRARIES.runtime[0],
            ),
            (
                CUDA_PACK_LIBRARIES.runtime[1],
                CUDA_PACK_LIBRARIES.runtime[1],
            ),
            (
                CUDA_PACK_LIBRARIES.runtime[2],
                CUDA_PACK_LIBRARIES.runtime[2],
            ),
        ],
    },
];

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
)))]
pub(super) const CUDA_PACK: &[CudaPackArchive] = &[];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPackState {
    Unsupported,
    NotInstalled,
    Installed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPackRemovalOutcome {
    Removed,
    Deferred,
}

impl ModelStore {
    /// The pack's archives, replaced by fixtures in tests.
    #[must_use]
    pub fn with_cuda_pack(mut self, archives: &'static [CudaPackArchive]) -> Self {
        self.cuda_pack = archives;
        self
    }

    #[must_use]
    pub fn cuda_pack_size(&self) -> u64 {
        self.cuda_pack
            .iter()
            .map(|archive| archive.size_bytes)
            .sum()
    }

    #[must_use]
    pub fn cuda_pack_state(&self) -> CudaPackState {
        if self.cuda_pack.is_empty() {
            CudaPackState::Unsupported
        } else if self.cuda_pack_dir().is_some() {
            CudaPackState::Installed
        } else {
            CudaPackState::NotInstalled
        }
    }

    /// The installed pack's directory; `None` while none is installed, or it is marked for
    /// removal.
    #[must_use]
    pub fn cuda_pack_dir(&self) -> Option<PathBuf> {
        let root = self.paths.cuda_pack_dir();
        (self.validate_cuda_pack(&root).is_ok() && !root.join(REMOVAL_MARKER).exists())
            .then_some(root)
    }

    fn validate_cuda_pack(&self, root: &Path) -> Result<(), ModelStoreError> {
        if self.cuda_pack.is_empty() {
            return Err(ModelStoreError::InvalidArchive);
        }
        let required = self
            .cuda_pack
            .iter()
            .flat_map(|archive| archive.members.iter());
        let mut artifacts = Vec::new();
        let mut invalid_artifact = false;
        for &(member, name) in required {
            if validate_artifact_path(root, Path::new(name))
                .map_err(|_| ModelStoreError::InvalidArchive)?
            {
                continue;
            }
            invalid_artifact = true;
            if !root.join(name).exists() {
                artifacts.push(member.to_owned());
            }
        }
        if !artifacts.is_empty() {
            artifacts.sort();
            return Err(ModelStoreError::MissingArtifacts { artifacts });
        }
        if invalid_artifact {
            return Err(ModelStoreError::InvalidArchive);
        }
        Ok(())
    }

    /// Downloads, verifies and installs the pack. A complete pack marked for removal is kept
    /// instead, without downloading it again.
    pub fn download_cuda_pack(
        &self,
        cancellation: &CancellationToken,
        observer: &dyn Fn(DownloadProgress),
    ) -> Result<(), ModelStoreError> {
        if self.cuda_pack.is_empty() {
            return Err(ModelStoreError::Unsupported);
        }
        let root = self.paths.cuda_pack_dir();
        let total_bytes = self.cuda_pack_size();
        let phase_progress = |phase| DownloadProgress {
            phase,
            downloaded_bytes: total_bytes,
            total_bytes: Some(total_bytes),
        };
        {
            let _guard = self.lock_store();
            if self.validate_cuda_pack(&root).is_ok() {
                match fs::remove_file(root.join(REMOVAL_MARKER)) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(_) => return Err(filesystem("keep the CUDA pack")),
                }
                observer(phase_progress(DownloadPhase::Complete));
                return Ok(());
            }
        }
        let data_dir = self.paths.data_dir();
        fs::create_dir_all(data_dir).map_err(|_| filesystem("create data directory"))?;
        let staging = Builder::new()
            .prefix(".llama-cuda-stage-")
            .tempdir_in(data_dir)
            .map_err(|_| filesystem("create CUDA pack staging directory"))?;
        let mut downloaded_bytes = 0_u64;
        for archive in self.cuda_pack {
            let mut download = Builder::new()
                .prefix(".llama-cuda-download-")
                .suffix(".part")
                .tempfile_in(data_dir)
                .map_err(|_| filesystem("create download staging file"))?;
            // Progress over the whole pack, not this one archive.
            let report_progress = |progress: DownloadProgress| {
                observer(DownloadProgress {
                    phase: progress.phase,
                    downloaded_bytes: downloaded_bytes.saturating_add(progress.downloaded_bytes),
                    total_bytes: Some(total_bytes),
                });
            };
            self.runtime.block_on(download_file(
                &self.client,
                &PinnedFile {
                    url: archive.url,
                    size_bytes: archive.size_bytes,
                    sha256: archive.sha256,
                },
                &mut download,
                cancellation,
                &report_progress,
            ))?;
            downloaded_bytes = downloaded_bytes.saturating_add(archive.size_bytes);
            observer(DownloadProgress {
                phase: DownloadPhase::Extracting,
                downloaded_bytes,
                total_bytes: Some(total_bytes),
            });
            extract(archive, download.path(), staging.path(), cancellation)?;
        }
        self.validate_cuda_pack(staging.path())?;
        observer(phase_progress(DownloadPhase::Installing));
        check_cancelled(cancellation)?;
        {
            let _guard = self.lock_store();
            if fs::symlink_metadata(&root).is_ok() {
                remove_path(&root).map_err(|_| filesystem("replace the CUDA pack"))?;
            }
            fs::rename(staging.path(), &root).map_err(|_| filesystem("install the CUDA pack"))?;
            // The directory moved; nothing is left for the guard to delete.
            let _moved = staging.keep();
            sync_directory(data_dir).map_err(|_| filesystem("sync models directory"))?;
        }
        observer(phase_progress(DownloadPhase::Complete));
        Ok(())
    }

    /// Removes the pack, or, while `in_use` (loaded in this process), marks it so that it is
    /// no longer used and is deleted at the next start.
    pub fn remove_cuda_pack(
        &self,
        in_use: bool,
    ) -> Result<CudaPackRemovalOutcome, ModelStoreError> {
        let _guard = self.lock_store();
        let root = self.paths.cuda_pack_dir();
        let metadata = match fs::symlink_metadata(&root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ModelStoreError::NotInstalled);
            }
            Err(_) => return Err(filesystem("inspect the CUDA pack")),
        };
        if in_use && metadata.is_dir() && !metadata.file_type().is_symlink() {
            File::create(root.join(REMOVAL_MARKER))
                .and_then(|marker| marker.sync_all())
                .map_err(|_| filesystem("mark the CUDA pack for removal"))?;
            Ok(CudaPackRemovalOutcome::Deferred)
        } else {
            remove_path(&root).map_err(|_| filesystem("remove the CUDA pack"))?;
            Ok(CudaPackRemovalOutcome::Removed)
        }
    }

    /// Deletes outdated packs, a pack marked for removal, and interrupted downloads. Run
    /// at startup, before anything loads the pack.
    pub fn finish_cuda_pack_removal(&self) {
        let _guard = self.lock_store();
        let root = self.paths.cuda_pack_dir();
        if root.join(REMOVAL_MARKER).exists()
            && let Err(error) = remove_path(&root)
        {
            tracing::warn!(%error, "the CUDA pack could not be removed");
        }
        let Ok(entries) = fs::read_dir(self.paths.data_dir()) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let stale_pack = name.starts_with("llama-cuda")
                && entry.path() != root
                && entry.file_type().is_ok_and(|kind| kind.is_dir());
            if (stale_pack
                || name.starts_with(".llama-cuda-stage-")
                || name.starts_with(".llama-cuda-download-"))
                && let Err(error) = remove_path(&entry.path())
            {
                tracing::warn!(%error, "CUDA pack files could not be removed");
            }
        }
    }
}

fn extract(
    archive: &CudaPackArchive,
    file: &Path,
    into: &Path,
    cancellation: &CancellationToken,
) -> Result<(), ModelStoreError> {
    let open = || File::open(file).map_err(|_| filesystem("open CUDA pack archive"));
    match archive.format {
        CudaPackFormat::TarGz => {
            let mut members: Vec<_> = archive
                .members
                .iter()
                .map(|&(member, name)| (PathBuf::from(member), name))
                .collect();
            extract_tar(
                GzDecoder::new(BufReader::new(open()?)),
                into,
                &mut members,
                cancellation,
            )?;
        }
        CudaPackFormat::Zip => {
            let mut zip =
                zip::ZipArchive::new(open()?).map_err(|_| ModelStoreError::InvalidArchive)?;
            for &(member, name) in archive.members {
                let mut entry = match zip.by_name(member) {
                    Ok(entry) => entry,
                    Err(zip::result::ZipError::FileNotFound) => continue,
                    Err(_) => return Err(ModelStoreError::InvalidArchive),
                };
                if !entry.is_file() {
                    continue;
                }
                write_member(&mut entry, &into.join(name), cancellation)?;
            }
        }
    }
    Ok(())
}
