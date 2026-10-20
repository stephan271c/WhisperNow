use std::{fs, io::ErrorKind, path::PathBuf};

use crate::atomic_file::write_atomically;

use super::portal::WaylandPortalError;

const MAX_TOKEN_BYTES: usize = 4 * 1024;

/// The portal's opaque `RemoteDesktop` restore token, stored as plain text and
/// never logged. On Unix the file is private (mode 0600): it is written
/// through a `tempfile` temporary that is created that way.
pub struct RemoteDesktopRestoreTokenStore {
    path: PathBuf,
}

impl RemoteDesktopRestoreTokenStore {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Whether a valid restore token is saved, so the next `RemoteDesktop`
    /// session can skip the consent prompt.
    #[must_use]
    pub fn has_token(&self) -> bool {
        matches!(self.load(), Ok(Some(_)))
    }

    pub(crate) fn load(&self) -> Result<Option<String>, WaylandPortalError> {
        let token = match fs::read_to_string(&self.path) {
            Ok(token) => token,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(WaylandPortalError::TokenStore),
        };
        if token.is_empty() || token.len() > MAX_TOKEN_BYTES || token.chars().any(char::is_control)
        {
            return Err(WaylandPortalError::TokenStore);
        }
        Ok(Some(token))
    }

    pub(crate) fn replace(&self, token: &str) -> Result<(), WaylandPortalError> {
        write_atomically(&self.path, token.as_bytes()).map_err(|_| WaylandPortalError::TokenStore)
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
        let store = RemoteDesktopRestoreTokenStore::new(&path);
        assert_eq!(store.load().expect("missing token"), None);
        assert!(!store.has_token());

        store.replace("first-token").expect("first token");
        assert!(store.has_token());
        store.replace("rotated-token").expect("rotated token");
        assert_eq!(
            store.load().expect("load token").as_deref(),
            Some("rotated-token")
        );
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
    fn a_damaged_token_does_not_count_as_consent() {
        let root = tempdir().expect("temporary directory");
        let path = root.path().join("token");
        let store = RemoteDesktopRestoreTokenStore::new(&path);
        fs::write(&path, "bad\ntoken").expect("invalid token");
        assert_eq!(store.load(), Err(WaylandPortalError::TokenStore));
        assert!(!store.has_token());
    }
}
