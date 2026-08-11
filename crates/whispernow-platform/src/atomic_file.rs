use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use tempfile::NamedTempFile;

/// Replaces `path` with `bytes` through a synced temporary file in the same directory, so a
/// crash leaves the old or the new file. `tempfile::persist` replaces atomically on Unix and
/// Windows.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    fs::create_dir_all(parent)?;
    let mut temporary = NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    sync_directory(parent)
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> io::Result<()> {
    fs::File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> io::Result<()> {
    // std cannot open a directory to sync it on Windows; the file itself was synced.
    Ok(())
}
