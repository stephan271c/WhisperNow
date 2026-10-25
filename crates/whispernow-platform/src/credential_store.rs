//! Operating-system credential store for LLM provider API keys.

#[cfg(any(test, feature = "test-support"))]
use std::{
    collections::BTreeMap,
    sync::{Mutex, PoisonError},
};

use thiserror::Error;
use whispernow_domain::{ApiKey, LlmProvider};

/// Failure to reach or use the credential store, such as a
/// headless Linux session without a Secret Service provider or a locked keychain.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("the OS credential store is unavailable")]
pub struct CredentialStoreUnavailable;

pub trait CredentialStore: Send + Sync {
    /// Reads a stored key; `Ok(None)` means the store has no entry.
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable>;

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable>;

    /// Removes a stored key; removing an absent entry succeeds.
    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable>;
}

/// The platform keychain: macOS Keychain, Windows Credential Manager, or the
/// Linux Secret Service over D-Bus.
pub(crate) struct OsCredentialStore;

const SERVICE: &str = "WhisperNow";

impl OsCredentialStore {
    fn entry(provider: LlmProvider) -> Result<keyring::Entry, CredentialStoreUnavailable> {
        // Saved keys are found again by this account name; changing it loses them.
        let account = format!("llm-provider:{}", provider.as_str());
        keyring::Entry::new(SERVICE, &account).map_err(|error| unavailable("entry", &error))
    }
}

impl CredentialStore for OsCredentialStore {
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        match Self::entry(provider)?.get_password() {
            // A key that is no longer one line reads as no key.
            Ok(secret) => Ok(ApiKey::new(secret).ok()),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(unavailable("read", &error)),
        }
    }

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        Self::entry(provider)?
            .set_password(key.expose_secret())
            .map_err(|error| unavailable("write", &error))
    }

    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        match Self::entry(provider)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(unavailable("delete", &error)),
        }
    }
}

/// Platform errors can name items or collections, and `BadEncoding` holds the secret's
/// bytes, so only the operation and the variant are logged.
fn unavailable(operation: &'static str, error: &keyring::Error) -> CredentialStoreUnavailable {
    let variant = match error {
        keyring::Error::PlatformFailure(_) => "PlatformFailure",
        keyring::Error::NoStorageAccess(_) => "NoStorageAccess",
        keyring::Error::NoEntry => "NoEntry",
        keyring::Error::BadEncoding(_) => "BadEncoding",
        keyring::Error::TooLong(..) => "TooLong",
        keyring::Error::Invalid(..) => "Invalid",
        keyring::Error::Ambiguous(_) => "Ambiguous",
        _ => "Other",
    };
    tracing::warn!(
        code = "credential_store_unavailable",
        operation,
        variant,
        "the OS credential store could not be used"
    );
    CredentialStoreUnavailable
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct MemoryCredentialStore {
    entries: Mutex<BTreeMap<LlmProvider, ApiKey>>,
    unavailable: std::sync::atomic::AtomicBool,
}

#[cfg(any(test, feature = "test-support"))]
impl MemoryCredentialStore {
    pub fn set_available(&self, available: bool) {
        self.unavailable
            .store(!available, std::sync::atomic::Ordering::SeqCst);
    }

    fn entries(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, BTreeMap<LlmProvider, ApiKey>>, CredentialStoreUnavailable>
    {
        if self.unavailable.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(CredentialStoreUnavailable);
        }
        Ok(self.entries.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[cfg(any(test, feature = "test-support"))]
impl CredentialStore for MemoryCredentialStore {
    fn read(&self, provider: LlmProvider) -> Result<Option<ApiKey>, CredentialStoreUnavailable> {
        Ok(self.entries()?.get(&provider).cloned())
    }

    fn write(&self, provider: LlmProvider, key: &ApiKey) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.insert(provider, key.clone());
        Ok(())
    }

    fn delete(&self, provider: LlmProvider) -> Result<(), CredentialStoreUnavailable> {
        self.entries()?.remove(&provider);
        Ok(())
    }
}
