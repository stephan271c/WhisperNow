use std::{
    fs,
    io::{self, ErrorKind},
    path::PathBuf,
};

use crate::atomic_file::write_atomically;

use super::portal::PortalOutputError;

const MAX_TOKEN_BYTES: usize = 4 * 1024;

/// The portal's opaque `RemoteDesktop` restore token, stored as plain text and
/// never logged. On Unix the file is private (mode 0600): it is written
/// through a `tempfile` temporary that is created that way.
pub struct RemoteDesktopRestoreTokenStore {
    path: PathBuf,
}

impl RemoteDesktopRestoreTokenStore {
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Whether a valid restore token is saved, so the next `RemoteDesktop`
    /// session can skip the consent prompt.
    #[must_use]
    pub fn has_token(&self) -> bool {
        matches!(self.read(), Ok(Some(_)))
    }

    /// The saved token. One that cannot be read or is malformed counts as
    /// absent: the next consent saves a new token over it.
    pub(crate) fn load(&self) -> Option<String> {
        self.read().unwrap_or_else(|error| {
            tracing::warn!(%error, "the saved restore token is unusable; asking for consent again");
            None
        })
    }

    fn read(&self) -> io::Result<Option<String>> {
        let token = match fs::read_to_string(&self.path) {
            Ok(token) => token,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if token.is_empty() || token.len() > MAX_TOKEN_BYTES || token.chars().any(char::is_control)
        {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "the token is empty, too long or has control characters",
            ));
        }
        Ok(Some(token))
    }

    pub(crate) fn replace(&self, token: &str) -> Result<(), PortalOutputError> {
        write_atomically(&self.path, token.as_bytes())
            .inspect_err(|error| {
                tracing::warn!(kind = ?error.kind(), "the restore token could not be saved");
            })
            .map_err(|_| PortalOutputError::TokenStore)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn a_replaced_token_is_the_only_one_kept_and_is_private() {
        let root = tempdir().expect("temporary directory");
        let path = root.path().join("token");
        let store = RemoteDesktopRestoreTokenStore::new(path.clone());
        assert_eq!(store.load(), None);
        assert!(!store.has_token());

        store.replace("first-token").expect("first token");
        assert!(store.has_token());
        store.replace("rotated-token").expect("rotated token");
        assert_eq!(store.load().as_deref(), Some("rotated-token"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(&path)
                .expect("token file")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn a_damaged_token_counts_as_absent_until_a_new_one_replaces_it() {
        let root = tempdir().expect("temporary directory");
        let path = root.path().join("token");
        let store = RemoteDesktopRestoreTokenStore::new(path.clone());
        for damaged in [
            &b"bad\ntoken"[..],
            b"",
            b"\xff\xfe",
            &[b'a'; MAX_TOKEN_BYTES + 1],
        ] {
            fs::write(&path, damaged).expect("damaged token");
            assert_eq!(store.load(), None);
            assert!(!store.has_token());
        }
        fs::remove_file(&path).expect("remove damaged token");
        fs::create_dir(&path).expect("unreadable token");
        assert_eq!(store.load(), None);
        assert!(!store.has_token());
        fs::remove_dir(&path).expect("remove unreadable token");

        fs::write(&path, "bad\ntoken").expect("damaged token");
        store.replace("fresh-token").expect("replace damaged token");
        assert_eq!(store.load().as_deref(), Some("fresh-token"));
        assert!(store.has_token());
    }
}
