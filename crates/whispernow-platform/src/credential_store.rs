//! Operating-system credential store for LLM provider API keys.

use std::{
    collections::BTreeMap,
    sync::{Mutex, PoisonError},
};

use thiserror::Error;
use whispernow_domain::ApiKey;

/// Content-free failure to reach or use the credential store, such as a
/// headless Linux session without a Secret Service provider or a locked keychain.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("the OS credential store is unavailable")]
pub struct CredentialStoreUnavailable;

/// Narrow secret-storage boundary keyed by a stable per-provider account name.
pub trait CredentialStore: Send + Sync {
    /// Reads a stored key; `Ok(None)` means the store has no entry.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialStoreUnavailable`] when the store cannot be read.
    fn read(&self, account: &str) -> Result<Option<ApiKey>, CredentialStoreUnavailable>;

    /// Creates or replaces a stored key.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialStoreUnavailable`] when the store cannot be written.
    fn write(&self, account: &str, key: &ApiKey) -> Result<(), CredentialStoreUnavailable>;

    /// Removes a stored key; removing an absent entry succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialStoreUnavailable`] when the store cannot be written.
    fn delete(&self, account: &str) -> Result<(), CredentialStoreUnavailable>;
}

/// The platform keychain: macOS Keychain, Windows Credential Manager, or the
/// Linux Secret Service over D-Bus.
pub(crate) struct OsCredentialStore;

const SERVICE: &str = "WhisperNow";

impl OsCredentialStore {
    fn entry(account: &str) -> Result<keyring::Entry, CredentialStoreUnavailable> {
        keyring::Entry::new(SERVICE, account).map_err(|_| unavailable("entry"))
    }
}

impl CredentialStore for OsCredentialStore {
    fn read(&self, account: &str) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        match Self::entry(account)?.get_password() {
            // A key that is no longer one line reads as no key.
            Ok(secret) => Ok(ApiKey::new(secret).ok()),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(_) => Err(unavailable("read")),
        }
    }

    fn write(&self, account: &str, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        Self::entry(account)?
            .set_password(key.expose_secret())
            .map_err(|_| unavailable("write"))
    }

    fn delete(&self, account: &str) -> Result<(), CredentialStoreUnavailable> {
        match Self::entry(account)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(_) => Err(unavailable("delete")),
        }
    }
}

/// Platform errors can name items or collections, so only the operation is logged.
fn unavailable(operation: &'static str) -> CredentialStoreUnavailable {
    tracing::warn!(
        code = "credential_store_unavailable",
        operation,
        "the OS credential store could not be used"
    );
    CredentialStoreUnavailable
}

/// A store in memory, for tests. Every call fails while it is unavailable.
#[derive(Debug, Default)]
pub struct MemoryCredentialStore {
    entries: Mutex<BTreeMap<String, ApiKey>>,
    unavailable: std::sync::atomic::AtomicBool,
}

impl MemoryCredentialStore {
    pub fn set_available(&self, available: bool) {
        self.unavailable
            .store(!available, std::sync::atomic::Ordering::SeqCst);
    }

    fn entries(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, ApiKey>>, CredentialStoreUnavailable>
    {
        if self.unavailable.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(CredentialStoreUnavailable);
        }
        Ok(self.entries.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl CredentialStore for MemoryCredentialStore {
    fn read(&self, account: &str) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        Ok(self.entries()?.get(account).cloned())
    }

    fn write(&self, account: &str, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.insert(account.to_owned(), key.clone());
        Ok(())
    }

    fn delete(&self, account: &str) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.remove(account);
        Ok(())
    }
}
