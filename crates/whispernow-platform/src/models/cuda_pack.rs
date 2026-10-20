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
    DownloadProgress, ModelStore, ModelStoreError, PinnedFile, check_cancelled, extract_tar,
    filesystem, remove_path, remove_stale, replace_dir, validate_artifact_path, write_member,
};
use crate::llama::cuda_loaded;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64")
))]
use crate::llama::{CUDA_PACK_LIBRARIES, llama_cpp_build};

/// Left in a pack that was removed while loaded; such a pack is deleted at the next start.
const REMOVAL_MARKER: &str = ".whispernow-remove";
const CUDA_PACK_PREFIX: &str = "llama-cuda";
const CUDA_DOWNLOAD_PREFIX: &str = ".llama-cuda-download-";
const CUDA_STAGE_PREFIX: &str = ".llama-cuda-stage-";

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
        let report_phase = |phase| {
            observer(DownloadProgress::after_transfer(phase, total_bytes));
        };
        {
            let _guard = self.lock_store();
            if self.validate_cuda_pack(&root).is_ok() {
                match fs::remove_file(root.join(REMOVAL_MARKER)) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(_) => return Err(filesystem("keep the CUDA pack")),
                }
                report_phase(DownloadPhase::Complete);
                return Ok(());
            }
        }
        let data_dir = self.paths.data_dir();
        fs::create_dir_all(data_dir).map_err(|_| filesystem("create data directory"))?;
        let staging = Builder::new()
            .prefix(CUDA_STAGE_PREFIX)
            .tempdir_in(data_dir)
            .map_err(|_| filesystem("create CUDA pack staging directory"))?;
        let mut downloaded_bytes = 0_u64;
        for archive in self.cuda_pack {
            // Progress over the whole pack, not this one archive.
            let report_progress = |progress: DownloadProgress| {
                observer(DownloadProgress::new(
                    progress.phase,
                    downloaded_bytes.saturating_add(progress.downloaded_bytes),
                    total_bytes,
                ));
            };
            let download = self.download_to_temp(
                data_dir,
                CUDA_DOWNLOAD_PREFIX,
                &PinnedFile {
                    url: archive.url,
                    size_bytes: archive.size_bytes,
                    sha256: archive.sha256,
                },
                cancellation,
                &report_progress,
            )?;
            downloaded_bytes = downloaded_bytes.saturating_add(archive.size_bytes);
            observer(DownloadProgress::new(
                DownloadPhase::Extracting,
                downloaded_bytes,
                total_bytes,
            ));
            extract(archive, download.path(), staging.path(), cancellation)?;
        }
        self.validate_cuda_pack(staging.path())?;
        report_phase(DownloadPhase::Installing);
        check_cancelled(cancellation)?;
        {
            let _guard = self.lock_store();
            replace_dir(staging.path(), &root, data_dir)?;
            // The directory moved; nothing is left for the guard to delete.
            let _moved = staging.keep();
        }
        report_phase(DownloadPhase::Complete);
        Ok(())
    }

    /// Removes the pack, or, while it is loaded in this process, marks it so that it is no
    /// longer used and is deleted at the next start.
    pub fn remove_cuda_pack(&self) -> Result<CudaPackRemovalOutcome, ModelStoreError> {
        self.remove_cuda_pack_inner(cuda_loaded())
    }

    fn remove_cuda_pack_inner(
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
        remove_stale(self.paths.data_dir(), |name, entry| {
            let stale_pack = name.starts_with(CUDA_PACK_PREFIX)
                && entry.path() != root
                && entry.file_type().is_ok_and(|kind| kind.is_dir());
            stale_pack
                || name.starts_with(CUDA_STAGE_PREFIX)
                || name.starts_with(CUDA_DOWNLOAD_PREFIX)
        });
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use tempfile::tempdir;
    use whispernow_domain::ClearDataCategory;

    use super::*;
    use crate::AppPaths;

    const PACK: &[CudaPackArchive] = &[CudaPackArchive {
        url: "http://127.0.0.1:1/unused",
        size_bytes: 1,
        sha256: "",
        format: CudaPackFormat::TarGz,
        members: &[
            ("backend.lib", "backend.lib"),
            ("runtime.lib", "runtime.lib"),
        ],
    }];

    #[test]
    fn a_pack_in_use_is_only_marked_for_removal_and_goes_at_the_next_start() {
        let root = tempdir().expect("temporary root");
        let paths = AppPaths::under(root.path());
        let store = ModelStore::new(paths.clone()).with_cuda_pack(PACK);
        let directory = paths.cuda_pack_dir();
        fs::create_dir_all(&directory).expect("pack directory");
        for &(_, name) in PACK[0].members {
            fs::write(directory.join(name), name).expect("pack file");
        }
        assert_eq!(store.cuda_pack_state(), CudaPackState::Installed);

        assert_eq!(
            store.remove_cuda_pack_inner(true),
            Ok(CudaPackRemovalOutcome::Deferred)
        );
        assert!(directory.exists(), "loaded files stay");
        assert_eq!(store.cuda_pack_state(), CudaPackState::NotInstalled);
        assert!(store.cuda_pack_dir().is_none(), "nothing loads it again");

        let models = paths.models_dir();
        fs::create_dir_all(&models).expect("models directory");
        fs::write(models.join("weights"), b"model data").expect("model data");
        assert!(
            paths
                .data_locations()
                .expect("data locations")
                .iter()
                .any(|location| {
                    location.category == ClearDataCategory::Models && location.path == directory
                })
        );
        let cleared = paths
            .clear_plan(&[ClearDataCategory::Models])
            .expect("clear plan")
            .remove(&[ClearDataCategory::Models]);
        assert_eq!(cleared.len(), 1);
        assert_eq!(cleared[0].path, models);
        assert!(cleared[0].deleted);
        assert!(directory.exists(), "loaded files survive data clearing");
        assert!(
            store.cuda_pack_dir().is_none(),
            "the removal marker survives"
        );

        let phases = Mutex::new(Vec::new());
        store
            .download_cuda_pack(&CancellationToken::default(), &|event| {
                phases.lock().expect("phases").push(event.phase);
            })
            .expect("kept");
        assert_eq!(store.cuda_pack_state(), CudaPackState::Installed);
        assert_eq!(
            *phases.lock().expect("phases"),
            [DownloadPhase::Complete],
            "nothing downloaded"
        );

        assert_eq!(
            store.remove_cuda_pack_inner(true),
            Ok(CudaPackRemovalOutcome::Deferred)
        );
        store.finish_cuda_pack_removal();
        assert!(!directory.exists());
        assert_eq!(
            store.remove_cuda_pack_inner(false),
            Err(ModelStoreError::NotInstalled)
        );
    }
}
