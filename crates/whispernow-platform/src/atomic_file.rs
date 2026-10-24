use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use tempfile::Builder;

/// Names the temporary files of an interrupted `write_atomically`, so a sweep can find them.
pub(crate) const TEMPORARY_PREFIX: &str = ".whispernow-write-";

/// Replaces `path` with `bytes` through a synced temporary file in the same directory, so a
/// crash leaves the old or the new file. `tempfile::persist` replaces atomically on Unix and
/// Windows.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    fs::create_dir_all(parent)?;
    let mut temporary = Builder::new()
        .prefix(TEMPORARY_PREFIX)
        .tempfile_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    sync_directory(parent)
}

pub(crate) fn remove_path(path: &Path) -> io::Result<()> {
    // `symlink_metadata` keeps deletion from following the target of a symlink.
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(unix)]
pub(crate) fn sync_directory(directory: &Path) -> io::Result<()> {
    fs::File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
#[expect(clippy::unnecessary_wraps)]
pub(crate) fn sync_directory(_directory: &Path) -> io::Result<()> {
    // std cannot open a directory to sync it on Windows; the file itself was synced.
    Ok(())
}
